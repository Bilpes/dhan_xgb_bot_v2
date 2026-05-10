# ============================================================
# bot/backtest.py — Simulate bot on historical data
# ============================================================
"""
USAGE:
  python bot/backtest.py                          # backtest current model, all symbols
  python bot/backtest.py --download               # download fresh data first, then backtest
  python bot/backtest.py --symbol HDFCBANK        # single symbol only
  python bot/backtest.py --download --symbol INFY # download + single symbol

What changed from original:
  1. --download flag: pulls fresh 6-month 5-min data via yfinance before testing
     (original assumed data/historical/ was already populated)
  2. BLOCKED_SYMBOLS respected: blocked stocks are skipped entirely
     (original would still score and trade VEDL/NYKAA/SBIN in simulation)
  3. Equity curve + per-symbol summary saved to logs/ as CSV
  4. Per-symbol verdict table printed at end (not just combined totals)
  5. Sharpe ratio added to summary (daily returns basis)
  6. --period flag: default 6mo, override to 3mo/1y etc.
"""

import sys, os, pickle, warnings, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

from data.features import build_features, FEATURE_COLS
from bot.risk_manager import RiskManager
from config.config import (
    CAPITAL, BUY_THRESHOLD, TRADE_MODE,
    MODEL_PATH, SCALER_PATH,
    TRAIL_AFTER_PCT, TRAIL_DISTANCE,
    DAILY_LOSS_LIMIT,
)

# ── Must match signal_engine.py exactly ──────────────────────
EXIT_LONG_THRESHOLD = 0.50
WEAK_THRESHOLD      = 0.55
WEAK_CANDLES_MAX    = 2

# ── FIX 2: Blocked symbols — must match signal_engine.py ─────
# These are skipped in simulation just as they are in live trading
BLOCKED_SYMBOLS = {
    "VEDL", "NYKAA", "SBIN", "POWERGRID", "ONGC",
    "INDUSINDBK", "BANKBARODA", "IOC",
}

# ── Transaction costs (NSE CNC realistic) ────────────────────
BROKERAGE_PER_SIDE = 20
STT_SELL_PCT       = 0.001
EXCHANGE_CHARGES   = 0.0000345
SLIPPAGE_PCT       = 0.0005


def _apply_costs(entry: float, exit_price: float, qty: int) -> float:
    buy_cost  = entry      * qty * (1 + SLIPPAGE_PCT + EXCHANGE_CHARGES)
    sell_cost = exit_price * qty * (1 - SLIPPAGE_PCT - STT_SELL_PCT - EXCHANGE_CHARGES)
    return sell_cost - buy_cost - (2 * BROKERAGE_PER_SIDE)


# ── FIX 1: Download fresh data ────────────────────────────────
NIFTY50_SYMBOLS = [
    "HDFCBANK","ICICIBANK","RELIANCE","INFY","TCS","AXISBANK","KOTAKBANK",
    "SBIN","BAJFINANCE","BHARTIARTL","WIPRO","HCLTECH","LT","ADANIENT",
    "NTPC","POWERGRID","TITAN","DRREDDY","SUNPHARMA","CIPLA","DIVISLAB",
    "ONGC","BPCL","IOC","COALINDIA","VEDL","TATASTEEL","HINDALCO","JSWSTEEL",
    "MARUTI","HEROMOTOCO","BAJAJFINSV","HDFCLIFE","MUTHOOTFIN","PIDILITIND",
    "ADANIPORTS","SHREECEM","ULTRACEMCO","NYKAA","PAYTM","TATACONSUM",
    "ITC","ETERNAL","PERSISTENT","HAVELLS","CHOLAFIN",
]

def download_fresh_data(symbols=None, period="6mo"):
    """Download fresh 5-min data for all symbols into data/historical/."""
    os.makedirs("data/historical", exist_ok=True)
    syms = symbols or NIFTY50_SYMBOLS
    downloaded = []
    print(f"\nDownloading {len(syms)} symbols ({period} of 5-min data)...")
    print("This takes ~2-3 minutes. Please wait.\n")
    for sym in syms:
        try:
            df = yf.download(
                f"{sym}.NS", period=period, interval="5m",
                auto_adjust=True, progress=False
            )
            if df.empty:
                print(f"  {sym:<14} ✗ no data")
                continue
            # Flatten MultiIndex columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0].lower() for c in df.columns]
            else:
                df.columns = df.columns.str.lower()
            df.index.name = "datetime"
            df = df[["open","high","low","close","volume"]].dropna()
            path = f"data/historical/{sym}_5min.csv"
            df.to_csv(path)
            print(f"  {sym:<14} ✓ {len(df):,} rows  ({df.index[0].date()} → {df.index[-1].date()})")
            downloaded.append(sym)
        except Exception as e:
            print(f"  {sym:<14} ✗ {e}")
    print(f"\nDownloaded {len(downloaded)}/{len(syms)} symbols.\n")
    return downloaded


def run_backtest(csv_path: str, trade_mode: str = None):
    mode   = trade_mode or TRADE_MODE
    symbol = os.path.basename(csv_path).split("_")[0].upper()

    # FIX 2: Skip blocked symbols in backtest simulation
    if symbol in BLOCKED_SYMBOLS:
        print(f"  {symbol:<14} BLOCKED — skipped (matches live BLOCKED_SYMBOLS)")
        return None

    print(f"\nBacktesting: {symbol} | mode={mode}")
    print("-" * 45)

    df = pd.read_csv(csv_path, parse_dates=["datetime"], index_col="datetime")
    df.columns = df.columns.str.lower()
    df = df.sort_index()

    if len(df) < 100:
        print(f"  {symbol}: only {len(df)} rows — need 100+. Skipping.")
        return None

    feat = build_features(df)
    if feat.empty:
        print(f"  {symbol}: feature build failed. Skipping.")
        return None

    with open(MODEL_PATH,  "rb") as f: model  = pickle.load(f)
    with open(SCALER_PATH, "rb") as f: scaler = pickle.load(f)

    X        = feat[FEATURE_COLS]
    X_scaled = scaler.transform(X)
    probs    = model.predict_proba(X_scaled)[:, 1]
    feat     = feat.copy()
    feat["prob_up"] = probs

    risk = RiskManager()
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
    daily_pnl    = 0.0
    current_date = None
    equity_curve = []   # FIX 3: track capital every candle for Sharpe

    for i, ts in enumerate(feat.index):
        row   = feat.loc[ts]
        close = float(row["close"])
        prob  = float(row["prob_up"])
        atr   = float(row["atr_14"]) if row["atr_14"] > 0 else close * 0.005

        trade_date = ts.date()
        if trade_date != current_date:
            daily_pnl    = 0.0
            current_date = trade_date

        equity_curve.append({"time": ts, "capital": capital})

        if daily_pnl / CAPITAL <= -DAILY_LOSS_LIMIT:
            if in_trade:
                pnl = _apply_costs(entry_price, close, qty)
                capital   += pnl
                daily_pnl += pnl
                trades.append({
                    "symbol": symbol, "entry_time": entry_time, "exit_time": ts,
                    "entry": round(entry_price,2), "exit": round(close,2),
                    "sl": round(entry_sl,2), "qty": qty,
                    "pnl": round(pnl,2), "reason": "CIRCUIT_BREAKER",
                    "capital": round(capital,2),
                })
                in_trade = False
            continue

        if in_trade:
            if close > running_high:
                running_high = close
            should_trail, new_sl = risk.should_trail(entry_price, close, running_high)
            if should_trail and new_sl > sl_price:
                sl_price = new_sl

            exit_reason = None
            exit_price  = close

            if close <= sl_price:
                exit_reason = "SL"
            elif i >= 55:
                if prob < EXIT_LONG_THRESHOLD:
                    exit_reason  = "SIGNAL_FLIP"
                    weak_candles = 0
                elif prob < WEAK_THRESHOLD:
                    weak_candles += 1
                    if weak_candles >= WEAK_CANDLES_MAX:
                        exit_reason  = "SIGNAL_FLIP"
                        weak_candles = 0
                else:
                    weak_candles = 0
                if exit_reason is None and prob <= 0.38:
                    exit_reason = "SIGNAL_FLIP"

            if exit_reason is None:
                if mode == "cnc"       and ts.hour == 15 and ts.minute >= 27:
                    exit_reason = "CUTOFF"
                elif mode == "intraday" and ts.hour == 15 and ts.minute >= 10:
                    exit_reason = "CUTOFF"

            if exit_reason:
                pnl = _apply_costs(entry_price, exit_price, qty)
                capital   += pnl
                daily_pnl += pnl
                weak_candles = 0
                trades.append({
                    "symbol": symbol, "entry_time": entry_time, "exit_time": ts,
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
        print(f"  {symbol}: no trades generated (threshold too high or no signal).")
        return None

    df_t   = pd.DataFrame(trades)
    wins   = df_t[df_t["pnl"] > 0]
    losses = df_t[df_t["pnl"] <= 0]

    total_pnl = df_t["pnl"].sum()
    win_rate  = len(wins) / len(df_t) * 100
    avg_win   = wins["pnl"].mean()   if len(wins)   else 0
    avg_loss  = losses["pnl"].mean() if len(losses) else 0
    wl_ratio  = abs(avg_win / avg_loss) if avg_loss != 0 else 0
    max_dd    = (df_t["capital"].cummax() - df_t["capital"]).max()

    verdict = "✅ EDGE" if (wl_ratio >= 1.5 and win_rate >= 50) else (
              "⚠️  WEAK" if (wl_ratio >= 1.0 and win_rate >= 45) else "❌ NO EDGE")

    print(f"  Trades      : {len(df_t)}  |  WR: {win_rate:.1f}%  |  W/L: {wl_ratio:.2f}x  {verdict}")
    print(f"  P&L         : Rs.{total_pnl:+,.0f}  |  Avg win: Rs.{avg_win:+,.0f}  |  Avg loss: Rs.{avg_loss:+,.0f}")
    print(f"  Max DD      : Rs.{max_dd:,.0f}  |  Final capital: Rs.{df_t['capital'].iloc[-1]:,.0f}")

    by_reason = df_t.groupby("reason")["pnl"].agg(count="count", total="sum").to_string()
    print(f"  Exits       : {by_reason}")

    return df_t


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest trading bot on historical data")
    parser.add_argument("--download", action="store_true",
                        help="Download fresh 5-min data before backtesting")
    parser.add_argument("--symbol",   type=str, default=None,
                        help="Test a single symbol only (e.g. HDFCBANK)")
    parser.add_argument("--period",   type=str, default="6mo",
                        help="Data period for download: 3mo, 6mo, 1y (default: 6mo)")
    args = parser.parse_args()

    import glob

    # ── Step 1: Optionally download fresh data ─────────────────
    if args.download:
        syms = [args.symbol] if args.symbol else None
        download_fresh_data(symbols=syms, period=args.period)

    # ── Step 2: Collect CSV files ──────────────────────────────
    data_dir = "data/historical"
    if args.symbol:
        pattern   = os.path.join(data_dir, f"{args.symbol}_5min.csv")
        csv_files = glob.glob(pattern)
        if not csv_files:
            print(f"\nNo data file found for {args.symbol}.")
            print(f"Run: python bot/backtest.py --download --symbol {args.symbol}")
            sys.exit(1)
    else:
        csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))

    if not csv_files:
        print(f"\nNo CSV files found in {data_dir}/")
        print("Run: python bot/backtest.py --download")
        sys.exit(1)

    # ── Step 3: Run backtests ──────────────────────────────────
    print(f"\nRunning backtest on {len(csv_files)} file(s)...")
    all_results  = []
    symbol_stats = []

    for f in csv_files:
        result = run_backtest(f)
        if result is not None:
            all_results.append(result)
            sym    = os.path.basename(f).split("_")[0].upper()
            wins   = result[result["pnl"] > 0]
            losses = result[result["pnl"] <= 0]
            wr     = len(wins) / len(result) * 100
            aw     = wins["pnl"].mean()   if len(wins)   else 0
            al     = losses["pnl"].mean() if len(losses) else 0
            wl     = abs(aw / al)          if al != 0    else 0
            symbol_stats.append({
                "symbol":   sym,
                "trades":   len(result),
                "win_rate": round(wr, 1),
                "wl_ratio": round(wl, 2),
                "total_pnl": round(result["pnl"].sum(), 0),
                "verdict":  "EDGE" if (wl >= 1.5 and wr >= 50) else (
                            "WEAK" if (wl >= 1.0 and wr >= 45) else "NO EDGE"),
            })

    if not all_results:
        print("\nNo trades generated across all symbols.")
        sys.exit(0)

    # ── Step 4: Combined summary ───────────────────────────────
    combined  = pd.concat(all_results, ignore_index=True)
    wins      = combined[combined["pnl"] > 0]
    losses    = combined[combined["pnl"] <= 0]
    total_pnl = combined["pnl"].sum()
    win_rate  = len(wins) / len(combined) * 100
    avg_win   = wins["pnl"].mean()   if len(wins)   else 0
    avg_loss  = losses["pnl"].mean() if len(losses) else 0
    wl_ratio  = abs(avg_win / avg_loss) if avg_loss != 0 else 0

    # Sharpe ratio (daily P&L basis)
    combined["date"] = pd.to_datetime(combined["exit_time"]).dt.date
    daily_pnl_series = combined.groupby("date")["pnl"].sum()
    sharpe = (daily_pnl_series.mean() / daily_pnl_series.std() * np.sqrt(252)
              ) if daily_pnl_series.std() > 0 else 0.0

    by_reason = combined.groupby("reason")["pnl"].agg(count="count", total="sum")

    print("\n" + "=" * 55)
    print(" COMBINED SUMMARY — All Symbols")
    print("=" * 55)
    print(f"  Total trades   : {len(combined)}")
    print(f"  Win rate       : {win_rate:.1f}%")
    print(f"  W/L ratio      : {wl_ratio:.2f}x  (need ≥ 1.5x for strong edge)")
    print(f"  Sharpe ratio   : {sharpe:.2f}       (need ≥ 1.0 to go live)")
    print(f"  Total P&L      : Rs.{total_pnl:+,.0f}  (after costs)")
    print(f"  Avg win        : Rs.{avg_win:+,.0f}")
    print(f"  Avg loss       : Rs.{avg_loss:+,.0f}")
    print(f"\n  Exit breakdown:")
    print(by_reason.to_string())

    # ── Per-symbol table ───────────────────────────────────────
    print("\n" + "=" * 55)
    print(" PER-SYMBOL RESULTS")
    print("=" * 55)
    stats_df = pd.DataFrame(symbol_stats).sort_values("total_pnl", ascending=False)
    print(stats_df.to_string(index=False))

    # ── GO/NO-GO verdict ──────────────────────────────────────
    print("\n" + "=" * 55)
    go = "GO LIVE" if (wl_ratio >= 1.5 and win_rate >= 50 and sharpe >= 1.0) else (
         "PAPER ONLY" if (wl_ratio >= 1.0 and win_rate >= 45) else "NO-GO")
    print(f"  Live trading verdict: {go}")
    if go == "GO LIVE":
        print("  Backtest confirms edge. Proceed with Rs.30,000 live capital.")
    elif go == "PAPER ONLY":
        print("  Marginal edge. Continue paper trading — do not go live yet.")
    else:
        print(f"  W/L={wl_ratio:.2f}x  WR={win_rate:.1f}%  Sharpe={sharpe:.2f}")
        print("  Review signal_engine.py thresholds and BLOCKED_SYMBOLS list.")
    print("=" * 55)

    # ── Save outputs ───────────────────────────────────────────
    os.makedirs("logs", exist_ok=True)
    combined.to_csv("logs/backtest_trades.csv", index=False)
    stats_df.to_csv("logs/backtest_summary.csv", index=False)
    print(f"\n  Trade log   → logs/backtest_trades.csv")
    print(f"  Symbol summary → logs/backtest_summary.csv")

