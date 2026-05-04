# test_nifty.py — run once to verify Nifty fetch works
# python test_nifty.py

import os
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join("config", ".env"))

from bot.dhan_api import DhanBroker

broker = DhanBroker()

print("Fetching Nifty50 candles (security_id=13, IDX_I)...")
df = broker.get_candles(
    security_id = "13",
    symbol      = "NIFTY50",
    days_back   = 3
)

if df.empty:
    print("❌ FAILED — empty dataframe returned")
    print("   Check: exchange segment in get_candles() — must use IDX_I not NSE")
else:
    print(f"✅ SUCCESS — {len(df)} candles fetched")
    print(f"   First row : {df.index[0]}  close={df['close'].iloc[0]}")
    print(f"   Last row  : {df.index[-1]}  close={df['close'].iloc[-1]}")
    print(f"   Columns   : {list(df.columns)}")
    print("\nSample (last 5 candles):")
    print(df.tail(5)[["open","high","low","close","volume"]].to_string())