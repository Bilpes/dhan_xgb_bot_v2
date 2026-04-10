#!/usr/bin/env python3
# ============================================================
#  test_telegram.py  —  Verify Telegram alerts work on both phones
# ============================================================
"""
Run this FIRST after filling in config/config.py:
    python test_telegram.py

Both phones should receive all 5 test messages within seconds.
If one phone doesn't get it, check that person did /start on your bot.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))

from bot.telegram_alert import (
    alert_test, alert_entry, alert_exit,
    alert_trail_update, alert_daily_summary,
)

print("Sending test alerts to both Telegram numbers...")
print("(Check your phones after each step)\n")

print("1. Sending connection test...")
alert_test()
time.sleep(2)

print("2. Sending sample BUY alert...")
alert_entry(
    symbol     = "HDFCBANK",
    buy_price  = 1723.50,
    stop_loss  = 1680.75,
    target     = 1810.00,
    quantity   = 52,
    prob_up    = 0.673,
    trade_mode = "intraday",
    invested   = 89622,
)
time.sleep(2)

print("3. Sending sample trailing stop update...")
alert_trail_update(
    symbol          = "HDFCBANK",
    new_sl          = 1748.20,
    ltp             = 1762.50,
    unrealised_pnl  = 2028,
)
time.sleep(2)

print("4. Sending sample SELL alert (profit)...")
alert_exit(
    symbol     = "HDFCBANK",
    buy_price  = 1723.50,
    sell_price = 1805.20,
    quantity   = 52,
    pnl        = 4246,
    reason     = "SIGNAL_FLIP",
    trade_mode = "intraday",
)
time.sleep(2)

print("5. Sending sample daily summary...")
alert_daily_summary(
    total_pnl = 4246,
    trades    = [
        {"symbol": "HDFCBANK", "entry": 1723.50, "exit": 1805.20, "qty": 52, "pnl": 4246},
        {"symbol": "INFY",     "entry": 1842.00, "exit": 1821.30, "qty": 38, "pnl": -786},
    ],
    capital   = 54246,
)

print("\nAll 5 test messages sent!")
print("Both phones should have received them.")
print("\nIf missing:")
print("  - Make sure both users sent /start to your bot on Telegram")
print("  - Double-check TELEGRAM_BOT_TOKEN and CHAT_IDs in config/config.py")
print("  - Visit https://api.telegram.org/bot<TOKEN>/getUpdates to find chat_ids")
