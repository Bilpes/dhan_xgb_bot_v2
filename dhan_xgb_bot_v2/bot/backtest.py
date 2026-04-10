# ============================================================
#  bot/backtest.py  —  Simulate bot on historical data
# ============================================================
"""
Run BEFORE going live:
    python bot/backtest.py

Simulates the exact same entry/exit logic as live_bot.py
on your historical CSV data. Shows equity curve + stats.
"""

import sys, os, pickle, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from data.features  import build_features, FEATURE_COLS
from config.config  import (
    MODEL_PATH, SCALER_PATH, CAPITAL,
    BUY_THRESHOLD, STOP_LOSS_PCT, TRAIL_AFTER_PCT, TRAIL_DISTANCE
)


def run_backtest(csv_path: str, trade_mode: str = "intraday"):
    print(f"\nBacktesting: {csv_path}  |  mode={trade_mode}")
    print("=" * 55)

    # Load
    df = pd.read_csv(csv_path, parse_dates=["datetime"], index_col="datetime")
    df.columns = df.columns.str.lower()
    df = df.sort_index()

    # Features
    feat = build_features(df)

    # Load model
    with open(MODEL_PATH,  "rb") as f: model  = pickle.load(f)
    with open(SCALER_PATH, "rb") as f: scaler = pickle.load(f)

    X = feat[FEATURE_COLS]
    X_scaled = scaler.transform(X)
    probs = model.predict_proba(X_scaled)[:, 1]
    feat["prob_up"] = probs

    # Simulate
    trades      = []
    capital     = CAPITAL
    in_trade    = False
    entry_price = 0
    sl_price    = 0
    qty         = 0
    running_high= 0

    for i, (ts, row) in enumerate(feat.iterrows()):
        close = row["close"]
        prob  = row["prob_up"]
        atr   = row["atr_14"]

        if in_trade:
            # Update high
            if close > running_high:
                running_high = close

            # Trailing stop activation
            profit_pct = (close - entry_price) / entry_price
            if profit_pct >= TRAIL_AFTER_PCT:
                new_sl = running_high * (1 - TRAIL_DISTANCE)
                if new_sl > sl_price:
                    sl_price = new_sl

            # Exit conditions
            exit_reason = None
            exit_price  = close

            if close <= sl_price:
                exit_reason = "SL"
            elif prob <= 0.38:              # signal flip
                exit_reason = "SIGNAL_FLIP"
            elif trade_mode == "intraday" and ts.hour == 15 and ts.minute >= 10:
                exit_reason = "CUTOFF"

            if exit_reason:
                pnl    = (exit_price - entry_price) * qty
                capital += pnl
                trades.append({
                    "entry_time": entry_time,
                    "exit_time":  ts,
                    "entry":      entry_price,
                    "exit":       exit_price,
                    "sl":         sl_price,
                    "qty":        qty,
                    "pnl":        round(pnl, 2),
                    "reason":     exit_reason,
                    "capital":    round(capital, 2),
                })
                in_trade = False

        else:
            # Entry signal
            if prob >= BUY_THRESHOLD:
                risk_per_share = close * STOP_LOSS_PCT
                sl_price   = close * (1 - STOP_LOSS_PCT)
                risk_amount= capital * 0.03
                qty        = max(1, int(risk_amount / risk_per_share))
                qty        = min(qty, int(capital * 0.95 / close))
                entry_price= close
                running_high= close
                entry_time = ts
                in_trade   = True

    # ── Stats ────────────────────────────────────────────────
    if not trades:
        print("No trades generated. Lower BUY_THRESHOLD or check data.")
        return

    df_t    = pd.DataFrame(trades)
    wins    = df_t[df_t["pnl"] > 0]
    losses  = df_t[df_t["pnl"] <= 0]

    total_pnl   = df_t["pnl"].sum()
    win_rate    = len(wins) / len(df_t) * 100
    avg_win     = wins["pnl"].mean() if len(wins) else 0
    avg_loss    = losses["pnl"].mean() if len(losses) else 0
    max_dd      = (df_t["capital"].cummax() - df_t["capital"]).max()

    print(f"  Trades:        {len(df_t)}")
    print(f"  Win rate:      {win_rate:.1f}%")
    print(f"  Total P&L:     ₹{total_pnl:,.0f}")
    print(f"  Avg win:       ₹{avg_win:,.0f}")
    print(f"  Avg loss:      ₹{avg_loss:,.0f}")
    print(f"  Max drawdown:  ₹{max_dd:,.0f}")
    print(f"  Final capital: ₹{df_t['capital'].iloc[-1]:,.0f}  "
          f"(started ₹{CAPITAL:,})")

    by_reason = df_t.groupby("reason")["pnl"].agg(["count","sum","mean"])
    print("\n  Exit breakdown:")
    print(by_reason.to_string())

    df_t.to_csv("logs/backtest_trades.csv", index=False)
    print("\n  Full trade log → logs/backtest_trades.csv")


if __name__ == "__main__":
    # Run on all CSVs in data/historical/
    data_dir = "data/historical"
    for f in os.listdir(data_dir):
        if f.endswith(".csv"):
            run_backtest(os.path.join(data_dir, f))
