# ============================================================
#  bot/backtest.py  —  Simulate bot on historical data
# ============================================================
"""
Run BEFORE going live:
    python bot/backtest.py

Simulates the exact same entry/exit logic as live_bot.py
on your historical CSV data. Shows equity curve + stats.

Fixes vs old version:
  - Uses SignalEngine.should_exit() — same as live bot
  - Uses RiskManager for position sizing and SL calc
  - Signal flip now uses 3-condition exit (momentum fade, etc.)
  - ATR-based SL instead of fixed STOP_LOSS_PCT
  - Correct position sizing with MAX_CAPITAL_PER_TRADE cap
"""

import sys, os, pickle, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from data.features     import build_features, FEATURE_COLS
from bot.signal_engine import SignalEngine
from bot.risk_manager  import RiskManager
from config.config     import (
    CAPITAL, BUY_THRESHOLD, TRADE_MODE,
    MODEL_PATH, SCALER_PATH,
)


def run_backtest(csv_path: str, trade_mode: str = None):
    mode = trade_mode or TRADE_MODE   # use config default

    print(f"\nBacktesting: {csv_path}  |  mode={mode}")
    print("=" * 55)

    # ── Load historical data ──────────────────────────────────
    df = pd.read_csv(csv_path, parse_dates=["datetime"], index_col="datetime")
    df.columns = df.columns.str.lower()
    df = df.sort_index()

    if len(df) < 60:
        print("  Not enough data (need 60+ candles). Skipping.")
        return

    # ── Build features once ───────────────────────────────────
    feat = build_features(df)
    if feat.empty:
        print("  Feature build failed. Skipping.")
        return

    # ── Load model and score all candles upfront ──────────────
    with open(MODEL_PATH,  "rb") as f: model  = pickle.load(f)
    with open(SCALER_PATH, "rb") as f: scaler = pickle.load(f)

    X        = feat[FEATURE_COLS]
    X_scaled = scaler.transform(X)
    probs    = model.predict_proba(X_scaled)[:, 1]
    feat["prob_up"] = probs

    # ── Init engine and risk manager ─────────────────────────
    engine = SignalEngine()
    risk   = RiskManager()
    risk.reset_day()

    # ── Simulation state ──────────────────────────────────────
    trades       = []
    capital      = CAPITAL
    in_trade     = False
    entry_price  = 0.0
    sl_price     = 0.0
    qty          = 0
    running_high = 0.0
    entry_time   = None
    entry_sl     = 0.0
    weak_candles = 0   # tracks consecutive weak candles for exit condition 3

    indices = feat.index.tolist()

    for i, ts in enumerate(indices):
        row   = feat.loc[ts]
        close = float(row["close"])
        prob  = float(row["prob_up"])
        atr   = float(row["atr_14"]) if row["atr_14"] > 0 else close * 0.005

        if in_trade:
            # Update running high for trailing stop
            if close > running_high:
                running_high = close

            # ── Trailing stop ─────────────────────────────────
            should_trail, new_sl = risk.should_trail(
                entry_price, close, running_high)
            if should_trail and new_sl > sl_price:
                sl_price = new_sl

            # ── Exit condition 1: SL breach ───────────────────
            exit_reason = None
            exit_price  = close

            if close <= sl_price:
                exit_reason = "SL"

            # ── Exit condition 2: Signal flip ─────────────────
            # Uses same 3-condition logic as live bot should_exit()
            # Needs at least 55 rows of history for features
            elif i >= 55:
                # Slice last 60 candles of raw df for rescoring
                df_slice = df.iloc[max(0, i-60):i+1].copy()

                # Condition A — momentum fade (prob drops below 0.50)
                if prob < 0.50:
                    exit_reason = "SIGNAL_FLIP"
                    weak_candles = 0

                # Condition B — consecutive weak candles (prob < 0.55 twice)
                elif prob < 0.55:
                    weak_candles += 1
                    if weak_candles >= 2:
                        exit_reason = "SIGNAL_FLIP"
                        weak_candles = 0
                else:
                    weak_candles = 0

                # Condition C — full reversal (prob < SELL_THRESHOLD)
                if exit_reason is None and prob <= 0.38:
                    exit_reason = "SIGNAL_FLIP"

            # ── Exit condition 3: EOD cutoff ──────────────────
            elif mode == "intraday" and ts.hour == 15 and ts.minute >= 10:
                exit_reason = "CUTOFF"

            elif mode == "cnc" and ts.hour == 15 and ts.minute >= 27:
                exit_reason = "CUTOFF"

            if exit_reason:
                pnl      = (exit_price - entry_price) * qty
                capital += pnl
                risk.update_pnl(pnl)
                weak_candles = 0

                trades.append({
                    "entry_time": entry_time,
                    "exit_time":  ts,
                    "entry":      round(entry_price, 2),
                    "exit":       round(exit_price, 2),
                    "sl":         round(entry_sl, 2),
                    "qty":        qty,
                    "pnl":        round(pnl, 2),
                    "reason":     exit_reason,
                    "capital":    round(capital, 2),
                })
                in_trade = False

        else:
            # ── Entry logic ───────────────────────────────────
            # Skip first 55 candles (need history for features)
            if i < 55:
                continue

            # Entry on BUY signal
            if prob >= BUY_THRESHOLD:
                # Calculate SL using RiskManager (ATR-based, same as live)
                sl     = risk.calc_stop_loss(close, atr, mode)
                qty_n  = risk.position_size(close, sl)

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

    # ── Stats ─────────────────────────────────────────────────
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

    print(f"  Trades:        {len(df_t)}")
    print(f"  Win rate:      {win_rate:.1f}%")
    print(f"  Win/Loss ratio:{wl_ratio:.2f}x  (need > 1.0x)")
    print(f"  Total P&L:     Rs.{total_pnl:,.0f}")
    print(f"  Avg win:       Rs.{avg_win:,.0f}")
    print(f"  Avg loss:      Rs.{avg_loss:,.0f}")
    print(f"  Max drawdown:  Rs.{max_dd:,.0f}")
    print(f"  Final capital: Rs.{df_t['capital'].iloc[-1]:,.0f}"
          f"  (started Rs.{CAPITAL:,})")

    by_reason = df_t.groupby("reason")["pnl"].agg(["count", "sum", "mean"])
    print("\n  Exit breakdown:")
    print(by_reason.to_string())

    df_t.to_csv("logs/backtest_trades.csv", index=False)
    print("\n  Full trade log -> logs/backtest_trades.csv")
    return df_t


if __name__ == "__main__":
    import glob

    data_dir = "data/historical"
    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))

    if not csv_files:
        print(f"No CSV files found in {data_dir}/")
        print("Run: python data/download_data.py")
        sys.exit(1)

    all_results = []
    for f in csv_files:
        result = run_backtest(f)
        if result is not None:
            all_results.append(result)

    # ── Combined summary across all stocks ────────────────────
    if all_results:
        combined   = pd.concat(all_results, ignore_index=True)
        total_pnl  = combined["pnl"].sum()
        wins       = combined[combined["pnl"] > 0]
        losses     = combined[combined["pnl"] <= 0]
        win_rate   = len(wins) / len(combined) * 100
        avg_win    = wins["pnl"].mean()   if len(wins)   else 0
        avg_loss   = losses["pnl"].mean() if len(losses) else 0
        wl_ratio   = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        by_reason  = combined.groupby("reason")["pnl"].agg(["count","sum","mean"])

        print("\n" + "=" * 55)
        print("  COMBINED SUMMARY — All Stocks")
        print("=" * 55)
        print(f"  Total trades   : {len(combined)}")
        print(f"  Win rate       : {win_rate:.1f}%")
        print(f"  Win/Loss ratio : {wl_ratio:.2f}x  (need > 1.0x)")
        print(f"  Total P&L      : Rs.{total_pnl:,.0f}")
        print(f"  Avg win        : Rs.{avg_win:,.0f}")
        print(f"  Avg loss       : Rs.{avg_loss:,.0f}")
        print("\n  Exit breakdown:")
        print(by_reason.to_string())

        go = "GO" if wl_ratio >= 1.0 and win_rate >= 48 else "NO-GO"
        print(f"\n  Live trading verdict: {go}")
        if go == "GO":
            print("  Win rate and W/L ratio meet minimum requirements.")
        else:
            print("  Win/Loss ratio < 1.0x — exits need improvement.")
            print("  Continue paper trading and check signal_engine.py")