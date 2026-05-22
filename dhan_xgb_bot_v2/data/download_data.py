# ============================================================
#  data/download_data.py  —  Download historical OHLCV data
# ============================================================
"""
Downloads free historical OHLCV data using yfinance.

Run once before training:
    pip install yfinance
    python data/download_data.py

What changed:
  1. Reads symbols dynamically from config/watchlist.json
  2. So load_instruments.py is now the single source of truth
  3. Supports Yahoo-specific ticker overrides for tricky NSE symbols

Saves CSVs to:
  data/historical/SYMBOL_5min.csv

Nifty50 index saved to:
  data/raw/NIFTY50.csv
  data/historical/NIFTY50_5min.csv
"""

import os
import json
import time
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    print("Run: pip install yfinance")
    raise

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
WATCHLIST_JSON = os.path.join(BASE_DIR, "config", "watchlist.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "historical")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)

# Yahoo Finance overrides for NSE symbols that do not always follow plain SYMBOL.NS
YF_OVERRIDES = {
    "M&M": "M&M.NS",
    "BAJAJ-AUTO": "BAJAJ-AUTO.NS",
    "ETERNAL": "ETERNAL.NS",
    "ADANIENSOL": "ADANIENSOL.NS",
    "MAZDOCK": "MAZDOCK.NS",
    "ICICIPRULI": "ICICIPRULI.NS",
    "AARTIIND": "AARTIIND.NS",
    "LTM": "LTM.NS",
    "JIOFIN": "JIOFIN.NS",
    "TMCV": "TMCV.NS",
    "RVNL": "RVNL.NS",
}

# Optional aliases in case your watchlist still carries old logical names
YF_ALIASES = {
    "TATAMOTORS": "TMCV",
    "LTIM": "LTM",
    "ADANIENERGY": "ADANIENSOL",
    "MAZAGONDOCK": "MAZDOCK",
    "ICICPRULI": "ICICIPRULI",
    "AARTI": "AARTIIND",
    "JIOFINANCE": "JIOFIN",
    "RAILVIKAS": "RVNL",
}


def load_watchlist_symbols():
    if not os.path.exists(WATCHLIST_JSON):
        raise FileNotFoundError(
            f"watchlist.json not found at: {WATCHLIST_JSON}\n"
            f"Run first: python data/load_instruments.py"
        )

    with open(WATCHLIST_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    watchlist = data.get("WATCHLIST", {})
    if not watchlist:
        raise ValueError(
            "WATCHLIST is empty in config/watchlist.json\n"
            "Run again: python data/load_instruments.py"
        )

    return list(watchlist.keys())


def to_yf_ticker(symbol: str) -> tuple[str, str]:
    resolved = YF_ALIASES.get(symbol, symbol)
    ticker = YF_OVERRIDES.get(resolved, f"{resolved}.NS")
    return resolved, ticker


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    rename_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "close",
        "Volume": "volume",
    }
    df = df.rename(columns=rename_map)

    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    df = df[required].copy()
    df = df.dropna()
    df.index.name = "datetime"
    return df


symbols = load_watchlist_symbols()
print(f"Downloading 60-day 5-min OHLCV for {len(symbols)} stocks...")
print("(yfinance free tier: max 60 days intraday)\n")

success, failed = 0, 0
failed_symbols = []

for symbol in symbols:
    resolved_symbol, yf_ticker = to_yf_ticker(symbol)
    try:
        df = yf.download(
            yf_ticker,
            period="60d",
            interval="5m",
            auto_adjust=True,
            progress=False,
            threads=False,
        )

        if df.empty:
            print(f"  {symbol:<15} NO DATA ({yf_ticker})")
            failed += 1
            failed_symbols.append(symbol)
            continue

        df = normalize_columns(df)
        out = os.path.join(OUTPUT_DIR, f"{symbol}_5min.csv")
        df.to_csv(out)

        alias_note = f" [{resolved_symbol}->{yf_ticker}]" if symbol != resolved_symbol or yf_ticker != f"{symbol}.NS" else ""
        print(f"  {symbol:<15} {len(df):>5} rows  ->  {out}{alias_note}")
        success += 1
        time.sleep(0.15)

    except Exception as e:
        print(f"  {symbol:<15} ERROR: {e}")
        failed += 1
        failed_symbols.append(symbol)

print("\nDownloading Nifty50 index candles (^NSEI)...")
try:
    nifty_df = yf.download(
        "^NSEI",
        period="60d",
        interval="5m",
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if nifty_df.empty:
        print("  WARNING: Nifty50 returned empty — Nifty features will be neutral (0)")
    else:
        nifty_df = normalize_columns(nifty_df)
        nifty_df.to_csv(os.path.join(OUTPUT_DIR, "NIFTY50_5min.csv"))
        nifty_df.to_csv(os.path.join(RAW_DIR, "NIFTY50.csv"))
        print(f"  NIFTY50         {len(nifty_df):>5} rows  ->  {os.path.join(RAW_DIR, 'NIFTY50.csv')}")
except Exception as e:
    print(f"  ERROR: {e}")

print(f"\nDone. {success} saved, {failed} failed.")
print(f"CSVs in {OUTPUT_DIR}")
if failed_symbols:
    print("Failed symbols:", ", ".join(failed_symbols))
print("\nNext: python models/train.py")
