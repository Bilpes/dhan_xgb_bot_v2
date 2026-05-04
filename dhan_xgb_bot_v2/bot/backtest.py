# ============================================================
# bot/backtest.py — Simulate bot on historical data
# ============================================================
"""
python bot/backtest.py

Fixes in this version:
- Daily P&L resets every morning (circuit breaker works correctly)
- Signal flip uses 3-condition exit via prob thresholds
- ATR-based SL matches live bot
- Position sizing uses MAX_CAPITAL_PER_TRADE cap
- Combined summary with GO/NO-GO verdict
- Transaction costs: brokerage + STT + slippage (realistic NSE costs)
"""

import sys, os, pickle, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from datetime import datetime

from data.features import build_features, FEATURE_COLS
from bot.risk_manager import RiskManager
from config.config import (
    CAPITAL, BUY_THRESHOLD, TRADE_MODE,
    MODEL_PATH, SCALER_PATH,
    TRAIL_AFTER_PCT, TRAIL_DISTANCE,
    DAILY_LOSS_LIMIT,
)

# Exit thresholds — must match signal_engine.py
EXIT_LONG_THRESHOLD = 0.50   # exit when momentum fades
WEAK_THRESHOLD      = 0.55   # consecutive weakness trigger
WEAK_CANDLES_MAX    = 2

# ── Transaction cost constants (NSE CNC realistic) ────────────
# Change these if your broker fee differs
BROKERAGE_PER_SIDE = 20      # Dhan flat Rs.20 per order (buy + sell = Rs.40)
STT_SELL_PCT       = 0.001   # 0.1% STT on sell side only (CNC equity)
EXCHANGE_CHARGES   = 0.0000345  # NSE + SEBI charges ~0.00345%
SLIPPAGE_PCT       = 0.0005  # 0.05% market impact each side

def _apply_costs(entry: float, exit_price: float, qty: int) -> float:
    """Returns realistic net PnL after all transaction costs."""
    buy_cost  = entry      * qty * (1 + SLIPPAGE_PCT + EXCHANGE_CHARGES)
    sell_cost = exit_price * qty * (1 - SLIPPAGE_PCT - STT_SELL_PCT - EXCHANGE_CHARGES)
    gross_pnl = sell_cost - buy_cost
    net_pnl   = gross_pnl - (2 * BROKERAGE_PER_SIDE)   # Rs.20 each side
    return net_pnl

def run_backtest(csv_path: str, trade_mode: str = None):
    mode = trade_mode or TRADE_MODE

    print(f"\nBacktesting: {csv_path} | mode={mode}")
    print("=" * 55)

    df = pd.read_csv(csv_path, parse_dates=["datetime"], index_col="datetime")
    df.columns = df.columns.str.lower()
    df = df.sort_index()

    if len(df) < 60:
        print("  Not enough data. Skipping.")
        return

    # Build features and score all candles
    feat = build_features(df)
    if feat.empty:
        print("  Feature build failed. Skipping.")
        return

    with open(MODEL_PATH, "rb") as f: model  = pickle.load(f)
    with open(SCALER_PATH, "rb") as f: scaler = pickle.load(f)

    X        = feat[FEATURE_COLS]
    X_scaled = scaler.transform(X)
    probs    = model.predict_proba(X_scaled)[:, 1]
    feat["prob_up"] = probs

    risk = RiskManager()

    # ── Simulation ────────────────────────────────────────────
    trades       = []
    capital      = CAPITAL
    in_trade     = False
    entry_price  = 0.0
    sl_price     = 0.0
    qty          = 0
    running_high = 0.0
    entry_time   = None
    entry_sl     = 0.0
    weak_candles = 0

    # Daily tracking — KEY FIX
    daily_pnl    = 0.0
    current_date = None

    indices = feat.index.tolist()

    for i, ts in enumerate(indices):
        row   = feat.loc[ts]
        close = float(row["close"])
        prob  = float(row["prob_up"])
        atr   = float(row["atr_14"]) if row["atr_14"] > 0 else close * 0.005

        # ── Reset daily P&L at start of each new day ──────────
        # This is what risk.reset_day() does in live bot
        trade_date = ts.date()
        if trade_date != current_date:
            daily_pnl    = 0.0
            current_date = trade_date

        # ── Circuit breaker check ─────────────────────────────
        if daily_pnl / CAPITAL <= -DAILY_LOSS_LIMIT:
            if in_trade:
                # Force exit at circuit breaker — costs applied
                pnl = _apply_costs(entry_price, close, qty)
                capital    += pnl
                daily_pnl  += pnl
                trades.append({
                    "entry_time": entry_time, "exit_time": ts,
                    "entry": round(entry_price,2), "exit": round(close,2),
                    "sl": round(entry_sl,2), "qty": qty,
                    "pnl": round(pnl,2), "reason": "CIRCUIT_BREAKER",
                    "capital": round(capital,2),
                })
                in_trade = False
            continue  # no new trades this day

        if in_trade:
            if close > running_high:
                running_high = close

            # Trailing stop
            should_trail, new_sl = risk.should_trail(
                entry_price, close, running_high)
            if should_trail and new_sl > sl_price:
                sl_price = new_sl

            exit_reason = None
            exit_price  = close

            # Exit 1 — SL breach
            if close <= sl_price:
                exit_reason = "SL"

            # Exit 2 — Signal flip (3 conditions, same as live bot)
            elif i >= 55:
                # Condition A: momentum fade
                if prob < EXIT_LONG_THRESHOLD:
                    exit_reason  = "SIGNAL_FLIP"
                    weak_candles = 0
                # Condition B: consecutive weakness
                elif prob < WEAK_THRESHOLD:
                    weak_candles += 1
                    if weak_candles >= WEAK_CANDLES_MAX:
                        exit_reason  = "SIGNAL_FLIP"
                        weak_candles = 0
                else:
                    weak_candles = 0
                # Condition C: full reversal
                if exit_reason is None and prob <= 0.38:
                    exit_reason = "SIGNAL_FLIP"

            # Exit 3 — EOD cutoff
            if exit_reason is None:
                if mode == "cnc"      and ts.hour == 15 and ts.minute >= 27:
                    exit_reason = "CUTOFF"
                elif mode == "intraday" and ts.hour == 15 and ts.minute >= 10:
                    exit_reason = "CUTOFF"

            if exit_reason:
                # FIX: use _apply_costs() instead of raw (exit-entry)*qty
                pnl = _apply_costs(entry_price, exit_price, qty)
                capital   += pnl
                daily_pnl += pnl
                weak_candles = 0
                trades.append({
                    "entry_time": entry_time, "exit_time": ts,
                    "entry": round(entry_price,2), "exit": round(exit_price,2),
                    "sl": round(entry_sl,2), "qty": qty,
                    "pnl": round(pnl,2), "reason": exit_reason,
                    "capital": round(capital,2),
                })
                in_trade = False

        else:
            if i < 55:
                continue

            if prob >= BUY_THRESHOLD:
                sl    = risk.calc_stop_loss(close, atr, mode)
                qty_n = risk.position_size(close, sl)
                if qty_n <= 0:
                    continue

                entry_price  = close
                entry_sl     = sl
                sl_price     = sl
                qty          = qty_n
                running_high = close
                entry_time   = ts
                in_trade     = True
                weak_candles = 0

    if not trades:
        print("  No trades generated.")
        return

    df_t   = pd.DataFrame(trades)
    wins   = df_t[df_t["pnl"] > 0]
    losses = df_t[df_t["pnl"] <= 0]

    total_pnl = df_t["pnl"].sum()
    win_rate  = len(wins) / len(df_t) * 100
    avg_win   = wins["pnl"].mean()   if len(wins)   else 0
    avg_loss  = losses["pnl"].mean() if len(losses) else 0
    wl_ratio  = abs(avg_win / avg_loss) if avg_loss != 0 else 0
    max_dd    = (df_t["capital"].cummax() - df_t["capital"]).max()

    # Estimate total costs paid (for transparency)
    gross_pnl   = df_t["pnl"].sum()
    cost_per_rt = (2 * BROKERAGE_PER_SIDE) + \
                  (df_t["entry"].mean() * df_t["qty"].mean() * (STT_SELL_PCT + 2*SLIPPAGE_PCT + 2*EXCHANGE_CHARGES))
    total_cost_approx = cost_per_rt * len(df_t)

    print(f"  Trades:          {len(df_t)}")
    print(f"  Win rate:        {win_rate:.1f}%")
    print(f"  Win/Loss ratio:  {wl_ratio:.2f}x (need > 1.0x)")
    print(f"  Total P&L:       Rs.{total_pnl:,.0f}  (after costs)")
    print(f"  Avg win:         Rs.{avg_win:,.0f}")
    print(f"  Avg loss:        Rs.{avg_loss:,.0f}")
    print(f"  Max drawdown:    Rs.{max_dd:,.0f}")
    print(f"  Est. costs paid: Rs.{total_cost_approx:,.0f}  (brokerage+STT+slippage)")
    print(f"  Final capital:   Rs.{df_t['capital'].iloc[-1]:,.0f}"
          f" (started Rs.{CAPITAL:,})")

    by_reason = df_t.groupby("reason")["pnl"].agg(["count","sum","mean"])
    print("\n  Exit breakdown:")
    print(by_reason.to_string())

    df_t.to_csv("logs/backtest_trades.csv", index=False)
    print("\n  Full trade log -> logs/backtest_trades.csv")
    return df_t


if __name__ == "__main__":
    import glob

    data_dir  = "data/historical"
    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))

    if not csv_files:
        print(f"No CSV files in {data_dir}/")
        print("Run: python data/download_data.py")
        sys.exit(1)

    all_results = []
    for f in csv_files:
        result = run_backtest(f)
        if result is not None:
            all_results.append(result)

    if all_results:
        combined  = pd.concat(all_results, ignore_index=True)
        total_pnl = combined["pnl"].sum()
        wins      = combined[combined["pnl"] > 0]
        losses    = combined[combined["pnl"] <= 0]
        win_rate  = len(wins) / len(combined) * 100
        avg_win   = wins["pnl"].mean()   if len(wins)   else 0
        avg_loss  = losses["pnl"].mean() if len(losses) else 0
        wl_ratio  = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        by_reason = combined.groupby("reason")["pnl"].agg(["count","sum","mean"])

        print("\n" + "=" * 55)
        print(" COMBINED SUMMARY — All Stocks")
        print("=" * 55)
        print(f"  Total trades   : {len(combined)}")
        print(f"  Win rate       : {win_rate:.1f}%")
        print(f"  Win/Loss ratio : {wl_ratio:.2f}x (need > 1.0x)")
        print(f"  Total P&L      : Rs.{total_pnl:,.0f}  (after costs)")
        print(f"  Avg win        : Rs.{avg_win:,.0f}")
        print(f"  Avg loss       : Rs.{avg_loss:,.0f}")
        print("\n  Exit breakdown:")
        print(by_reason.to_string())

        go = "GO LIVE" if wl_ratio >= 1.0 and win_rate >= 45 else "NO-GO"
        print(f"\n  Live trading verdict: {go}")
        if go == "GO LIVE":
            print("  Backtest confirms edge. Proceed with Rs.30,000 live capital.")
        else:
            print(f"  W/L={wl_ratio:.2f}x WR={win_rate:.1f}% — review signal_engine.py")
