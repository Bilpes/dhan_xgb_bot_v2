# paste this in a new file: debug_labels.py
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, ".")

from data.features import build_features, FEATURE_COLS
from bot.trade_policy import ATR_TP_MULT, ATR_SL_MULT, HORIZON

df = pd.read_csv("data/historical/HDFCBANK_5min.csv", parse_dates=["datetime"])
df["symbol"] = "HDFCBANK"

feat = build_features(df)
print(f"\nATR_TP_MULT={ATR_TP_MULT}  ATR_SL_MULT={ATR_SL_MULT}  HORIZON={HORIZON}")
print(f"\nSample ATR values (last 10):")
print(feat["atr_14"].tail(10).round(4).to_string())
print(f"\nSample close values (last 10):")
print(feat["close"].tail(10).round(2).to_string())
print(f"\nATR as % of price (should be 0.2% - 0.8% for NSE 5min):")
atr_pct = (feat["atr_14"] / feat["close"] * 100)
print(f"  Mean: {atr_pct.mean():.3f}%  |  Min: {atr_pct.min():.3f}%  |  Max: {atr_pct.max():.3f}%")
print(f"\nWith ATR_TP_MULT=5.0 → avg TP distance: {(atr_pct.mean() * 5.0):.2f}% from entry")
print(f"With HORIZON=8 → must hit TP in 40 minutes")
# debug_labels.py — verify first
TP_PCT = 0.025
SL_PCT = 0.010
entry = feat["close"].values
tp = entry * (1 + TP_PCT)
sl = entry * (1 - SL_PCT)
print(f"\nWith 2.5% TP / 1.0% SL on HDFCBANK:")
print(f"  Avg TP level: ₹{(entry * TP_PCT).mean():.2f} above entry")
print(f"  Avg SL level: ₹{(entry * SL_PCT).mean():.2f} below entry")