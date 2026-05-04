# check_features.py
import pandas as pd
from bot.dhan_api import DhanBroker
from config.config import WATCHLIST
from data.features import build_features      # ← correct import

b = DhanBroker()
sym, sec_id = list(WATCHLIST.items())[0]
print(f"Fetching candles for {sym}...")

df = b.get_candles(sec_id, sym, days_back=10)
print(f"Candles fetched: {len(df)} rows")

feat = build_features(df)
print(f"\nTotal features: {len(feat.columns)}")
print("\nFeature columns:")
for i, col in enumerate(feat.columns.tolist(), 1):
    print(f"  {i:>2}. {col}")