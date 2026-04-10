# ============================================================
#  data/download_data.py  —  Download Nifty 50 historical data
# ============================================================
"""
Downloads free historical OHLCV data using yfinance.
Run once before training:
    pip install yfinance
    python data/download_data.py

Saves CSVs to data/historical/SYMBOL_5min.csv
"""

import os
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    print("Run:  pip install yfinance")
    raise

NIFTY50_SYMBOLS = [
    "RELIANCE", "HDFCBANK", "INFY", "TCS", "ICICIBANK",
    "AXISBANK", "SBIN", "KOTAKBANK", "LT", "BAJFINANCE",
    "HINDUNILVR", "ITC", "ASIANPAINT", "MARUTI", "TITAN",
    "WIPRO", "ULTRACEMCO", "NESTLEIND", "TATAMOTORS", "POWERGRID",
]

OUTPUT_DIR = "data/historical"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Downloading 60-day 5-min OHLCV data for Nifty 50 stocks...")
print("(yfinance free tier: max 60 days of intraday data)\n")

for sym in NIFTY50_SYMBOLS:
    ticker = f"{sym}.NS"
    try:
        df = yf.download(
            ticker,
            period="60d",
            interval="5m",
            auto_adjust=True,
            progress=False,
        )
        if df.empty:
            print(f"  {sym:<15} NO DATA")
            continue

        df.index.name = "datetime"
        df.columns    = ["open","high","low","close","volume"]
        df = df.dropna()

        out = os.path.join(OUTPUT_DIR, f"{sym}_5min.csv")
        df.to_csv(out)
        print(f"  {sym:<15} {len(df):>5} rows  →  {out}")

    except Exception as e:
        print(f"  {sym:<15} ERROR: {e}")

print(f"\nDone. CSVs saved to {OUTPUT_DIR}/")
print("Next: python models/train.py")
