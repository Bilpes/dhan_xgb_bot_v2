# diagnostics.py — dhan_xgb_bot_v3
# Run daily after market close to monitor calibration + reject patterns

import pandas as pd
import config as cfg


def calibration_report():
    """
    Probability calibration check.
    Healthy: prob=0.65 -> actual win ~55-65%.
    If prob=0.90 -> actual win ~25%: leakage remains in features/labels.
    """
    try:
        t = pd.read_csv(cfg.TRADE_LOG_PATH)
        e = t[t["action"] == "EXIT"].copy()
        e["pnl"] = pd.to_numeric(e["pnl"], errors="coerce")
        e["win"] = e["pnl"] > 0

        bins   = [0, .55, .60, .65, .70, .75, .80, .85, .90, .95, 1.01]
        labels = ["<.55", ".55", ".60", ".65", ".70", ".75", ".80", ".85", ".90", ".95+"]
        e["bin"] = pd.cut(e["prob"], bins=bins, labels=labels)

        cal = e.groupby("bin", observed=True).agg(
            n=("win", "count"),
            win_rate=("win", "mean"),
        )
        cal["win_rate"] = (cal["win_rate"] * 100).round(1)

        print("\n=== CALIBRATION ===")
        print("(If win_rate << prob bucket: leakage still present)")
        print(cal.to_string())
        print(f"\nTotal PnL : \u20b9{e['pnl'].sum():.2f}")
        print(f"Win rate  : {e['win'].mean():.1%}")
        print(f"Trades    : {len(e)}")

        er = e.groupby("exit_reason").agg(
            n=("pnl", "count"),
            pnl=("pnl", "sum"),
            wr=("win", "mean"),
        )
        er["wr"] = (er["wr"] * 100).round(1)
        print("\n=== EXIT REASONS ===")
        print(er.sort_values("n", ascending=False).to_string())

    except Exception as ex:
        print(f"No trade log yet or parse error: {ex}")


def scan_report():
    """
    Signal scan summary — shows which filter is killing most signals.
    """
    try:
        df = pd.read_csv(cfg.SIGNAL_LOG_PATH)
        total = len(df)
        buys  = (df["action"] == "BUY").sum()
        print(f"\n=== SCAN REPORT ===")
        print(f"Total scans : {total}")
        print(f"BUY signals : {buys} ({buys/total:.1%})")
        print("\nTop reject reasons:")
        rejects = df[df["action"] != "BUY"]["reject_reason"].value_counts().head(15)
        print(rejects.to_string())

        # Symbol-level breakdown
        sym_buys = (
            df[df["action"] == "BUY"]["symbol"]
            .value_counts()
            .head(10)
        )
        print("\nTop symbols by BUY count:")
        print(sym_buys.to_string())

    except Exception as ex:
        print(f"No scan log yet or parse error: {ex}")


def daily_pnl_summary():
    """Quick daily P&L curve from trade log."""
    try:
        t = pd.read_csv(cfg.TRADE_LOG_PATH, parse_dates=["time"])
        e = t[t["action"] == "EXIT"].copy()
        e["pnl"] = pd.to_numeric(e["pnl"], errors="coerce")
        e["date"] = e["time"].dt.date
        daily = e.groupby("date")["pnl"].sum()
        print("\n=== DAILY P&L ===")
        print(daily.to_string())
        print(f"Cumulative: \u20b9{daily.sum():.2f} | Avg/day: \u20b9{daily.mean():.2f}")
    except Exception as ex:
        print(f"Error: {ex}")


if __name__ == "__main__":
    calibration_report()
    scan_report()
    daily_pnl_summary()
