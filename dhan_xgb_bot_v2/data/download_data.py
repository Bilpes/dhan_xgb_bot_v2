# ============================================================
#  data/download_data.py  —  Download historical OHLCV data
# ============================================================
"""
Downloads free historical OHLCV data using yfinance.
Run once before training:
    pip install yfinance
    python data/download_data.py

Saves CSVs to data/historical/SYMBOL_5min.csv
Nifty50 index saved to data/raw/NIFTY50.csv
"""

import os
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    print("Run:  pip install yfinance")
    raise


# ── Full watchlist — must match config/config.py WATCHLIST ───
# yfinance needs .NS suffix for NSE stocks
# Special cases: M&M -> MM.NS, BAJAJ-AUTO -> BAJAJ-AUTO.NS
WATCHLIST_YF = {
    # Banking
    "HDFCBANK":   "HDFCBANK.NS",
    "ICICIBANK":  "ICICIBANK.NS",
    "SBIN":       "SBIN.NS",
    "KOTAKBANK":  "KOTAKBANK.NS",
    "AXISBANK":   "AXISBANK.NS",
    "INDUSINDBK": "INDUSINDBK.NS",
    "BANKBARODA": "BANKBARODA.NS",
    # IT
    "TCS":        "TCS.NS",
    "INFY":       "INFY.NS",
    "WIPRO":      "WIPRO.NS",
    "HCLTECH":    "HCLTECH.NS",
    "TECHM":      "TECHM.NS",
    "PERSISTENT": "PERSISTENT.NS",
    "COFORGE":    "COFORGE.NS",
    # Energy / Power
    "RELIANCE":   "RELIANCE.NS",
    "ONGC":       "ONGC.NS",
    "NTPC":       "NTPC.NS",
    "POWERGRID":  "POWERGRID.NS",
    "COALINDIA":  "COALINDIA.NS",
    "BPCL":       "BPCL.NS",
    "IOC":        "IOC.NS",
    # FMCG
    "HINDUNILVR": "HINDUNILVR.NS",
    "ITC":        "ITC.NS",
    "NESTLEIND":  "NESTLEIND.NS",
    "BRITANNIA":  "BRITANNIA.NS",
    "TATACONSUM": "TATACONSUM.NS",
    # Auto
    "MARUTI":     "MARUTI.NS",
    "M&M":        "M&M.NS",           # special: & not supported
    "BAJAJ-AUTO": "BAJAJ-AUTO.NS",
    "EICHERMOT":  "EICHERMOT.NS",
    "HEROMOTOCO": "HEROMOTOCO.NS",
    # Pharma / Healthcare
    "SUNPHARMA":  "SUNPHARMA.NS",
    "DRREDDY":    "DRREDDY.NS",
    "CIPLA":      "CIPLA.NS",
    "DIVISLAB":   "DIVISLAB.NS",
    "TORNTPHARM": "TORNTPHARM.NS",
    "AUROPHARMA": "AUROPHARMA.NS",
    "APOLLOHOSP": "APOLLOHOSP.NS",
    # Infra / Cement
    "LT":         "LT.NS",
    "ADANIPORTS": "ADANIPORTS.NS",
    "ADANIENT":   "ADANIENT.NS",
    "SIEMENS":    "SIEMENS.NS",
    "HAVELLS":    "HAVELLS.NS",
    "SHREECEM":   "SHREECEM.NS",
    # Finance / NBFC / Insurance
    "BAJFINANCE": "BAJFINANCE.NS",
    "BAJAJFINSV": "BAJAJFINSV.NS",
    "SHRIRAMFIN": "SHRIRAMFIN.NS",
    "MUTHOOTFIN": "MUTHOOTFIN.NS",
    "CHOLAFIN":   "CHOLAFIN.NS",
    "ICICIGI":    "ICICIGI.NS",
    "HDFCLIFE":   "HDFCLIFE.NS",
    "SBILIFE":    "SBILIFE.NS",
    # Defence
    "HAL":        "HAL.NS",
    "BEL":        "BEL.NS",
    # Telecom
    "BHARTIARTL": "BHARTIARTL.NS",
    # Consumer / Retail
    "TITAN":      "TITAN.NS",
    "ASIANPAINT": "ASIANPAINT.NS",
    "TRENT":      "TRENT.NS",
    "DMART":      "DMART.NS",
    "NYKAA":      "NYKAA.NS",
    "ETERNAL":    "ETERNAL.NS",
    "VOLTAS":     "VOLTAS.NS",
    # Metals
    "JSWSTEEL":   "JSWSTEEL.NS",
    "TATASTEEL":  "TATASTEEL.NS",
    "HINDALCO":   "HINDALCO.NS",
    "VEDL":       "VEDL.NS",
    # Chemicals / Others
    "PIDILITIND": "PIDILITIND.NS",
    # Fintech
    "PAYTM":      "PAYTM.NS",
    # Others
    "ULTRACEMCO": "ULTRACEMCO.NS",
    "LT":         "LT.NS",
    "WIPRO":      "WIPRO.NS",
}

OUTPUT_DIR = "data/historical"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("data/raw", exist_ok=True)


# ── Download stocks ───────────────────────────────────────────
print(f"Downloading 60-day 5-min OHLCV for {len(WATCHLIST_YF)} stocks...")
print("(yfinance free tier: max 60 days intraday)\n")

success, failed = 0, 0

for symbol, yf_ticker in WATCHLIST_YF.items():
    try:
        df = yf.download(
            yf_ticker,
            period    = "60d",
            interval  = "5m",
            auto_adjust = True,
            progress  = False,
        )
        if df.empty:
            print(f"  {symbol:<15} NO DATA")
            failed += 1
            continue

        df.index.name = "datetime"
        df.columns    = ["open", "high", "low", "close", "volume"]
        df = df.dropna()

        out = os.path.join(OUTPUT_DIR, f"{symbol}_5min.csv")
        df.to_csv(out)
        print(f"  {symbol:<15} {len(df):>5} rows  ->  {out}")
        success += 1

    except Exception as e:
        print(f"  {symbol:<15} ERROR: {e}")
        failed += 1


# ── Download Nifty50 index ────────────────────────────────────
# ^NSEI = Nifty50 index on yfinance
# Needed for: nifty_roc5, rs_vs_nifty, nifty_trend features
print("\nDownloading Nifty50 index candles (^NSEI)...")
try:
    nifty_df = yf.download(
        "^NSEI",
        period      = "60d",
        interval    = "5m",
        auto_adjust = True,
        progress    = False,
    )
    if nifty_df.empty:
        print("  WARNING: Nifty50 returned empty — Nifty features will be neutral (0)")
    else:
        nifty_df.index.name = "datetime"
        nifty_df.columns    = ["open", "high", "low", "close", "volume"]
        nifty_df = nifty_df.dropna()

        # Save to both locations
        nifty_df.to_csv(os.path.join(OUTPUT_DIR, "NIFTY50_5min.csv"))
        nifty_df.to_csv(os.path.join("data", "raw", "NIFTY50.csv"))
        print(f"  NIFTY50        {len(nifty_df):>5} rows  ->  data/raw/NIFTY50.csv")

except Exception as e:
    print(f"  ERROR: {e}")


# ── Summary ───────────────────────────────────────────────────
print(f"\nDone. {success} saved, {failed} failed.")
print(f"CSVs in {OUTPUT_DIR}/")
print("\nNext: python models/train.py")