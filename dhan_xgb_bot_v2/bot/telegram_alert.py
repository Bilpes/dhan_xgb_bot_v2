# ============================================================
#  bot/telegram_alert.py  —  Send trade alerts to 2 Telegram numbers
# ============================================================
"""
SETUP (one time):
  1. Open Telegram → search @BotFather → send /newbot
  2. Give your bot a name → BotFather gives you a BOT_TOKEN
  3. Search your new bot on Telegram → send it /start
  4. Visit: https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates
     → copy your "chat_id" from the response
  5. Do steps 3–4 for the second person too (they must /start the bot)
  6. Fill BOT_TOKEN, CHAT_ID_1, CHAT_ID_2 in config/config.py

Install:
  pip install requests
"""

import requests
import logging
from datetime import datetime
from config.config import (
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHAT_ID_1,
    TELEGRAM_CHAT_ID_2,
)

log = logging.getLogger("telegram")

RECIPIENTS = [TELEGRAM_CHAT_ID_1, TELEGRAM_CHAT_ID_2]


# ── Core sender ──────────────────────────────────────────────

def _send(message: str):
    """Send message to all configured recipients."""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

    for chat_id in RECIPIENTS:
        if not chat_id:
            continue
        try:
            resp = requests.post(url, json={
                "chat_id":    chat_id,
                "text":       message,
                "parse_mode": "HTML",        # enables <b>, <i>, <code> tags
            }, timeout=10)

            if resp.status_code != 200:
                log.error("Telegram send failed to %s: %s", chat_id, resp.text)
            else:
                log.debug("Telegram alert sent to %s", chat_id)

        except Exception as e:
            log.error("Telegram error for %s: %s", chat_id, e)


# ── Alert types ──────────────────────────────────────────────

def alert_entry(symbol: str, buy_price: float, stop_loss: float,
                target: float, quantity: int, prob_up: float,
                trade_mode: str, invested: float):
    """
    Sent when bot places a BUY order.

    Example message:
    ────────────────────────
    🟢 BUY ORDER PLACED
    ────────────────────────
    🏢 Company   : HDFCBANK
    📈 Buy price : ₹1,723.50
    🎯 Target    : ₹1,810.00
    🛑 Stop-loss : ₹1,681.00
    📦 Quantity  : 52 shares
    💰 Invested  : ₹89,622
    📊 Confidence: 67.3%
    🕐 Mode      : INTRADAY
    ⏰ Time      : 09:32 AM
    """
    time_str = datetime.now().strftime("%I:%M %p")
    mode_str = "INTRADAY" if trade_mode == "intraday" else "SWING (1-2 days)"

    msg = (
        f"🟢 <b>BUY ORDER PLACED</b>\n"
        f"{'─' * 28}\n"
        f"🏢 <b>Company</b>    : <b>{symbol}</b>\n"
        f"📈 <b>Buy price</b>  : ₹{buy_price:,.2f}\n"
        f"🎯 <b>Target</b>     : ₹{target:,.2f}\n"
        f"🛑 <b>Stop-loss</b>  : ₹{stop_loss:,.2f}\n"
        f"📦 <b>Quantity</b>   : {quantity} shares\n"
        f"💰 <b>Invested</b>   : ₹{invested:,.0f}\n"
        f"📊 <b>Confidence</b> : {prob_up*100:.1f}%\n"
        f"🕐 <b>Mode</b>       : {mode_str}\n"
        f"⏰ <b>Time</b>       : {time_str}"
    )
    _send(msg)


def alert_exit(symbol: str, buy_price: float, sell_price: float,
               quantity: int, pnl: float, reason: str, trade_mode: str):
    """
    Sent when bot exits a position.

    Example:
    ────────────────────────
    🔴 POSITION CLOSED
    ────────────────────────
    🏢 Company   : HDFCBANK
    📈 Buy price : ₹1,723.50
    📉 Sell price: ₹1,805.20
    📦 Quantity  : 52 shares
    💵 P&L       : +₹4,246  ✅ PROFIT
    📋 Reason    : Signal flip (model bearish)
    ⏰ Time      : 11:47 AM
    """
    time_str  = datetime.now().strftime("%I:%M %p")
    pnl_sign  = "+" if pnl >= 0 else ""
    pnl_emoji = "✅ PROFIT" if pnl >= 0 else "❌ LOSS"
    top_emoji = "🔴" if pnl < 0 else "💚"

    reason_map = {
        "SL_HIT":          "Stop-loss hit",
        "SIGNAL_FLIP":     "Signal flip (model turned bearish)",
        "INTRADAY_CUTOFF": "3:10 PM square-off",
        "TRAIL_STOP":      "Trailing stop triggered",
        "MANUAL":          "Manual exit",
    }
    reason_str = reason_map.get(reason, reason)

    msg = (
        f"{top_emoji} <b>POSITION CLOSED</b>\n"
        f"{'─' * 28}\n"
        f"🏢 <b>Company</b>    : <b>{symbol}</b>\n"
        f"📈 <b>Buy price</b>  : ₹{buy_price:,.2f}\n"
        f"📉 <b>Sell price</b> : ₹{sell_price:,.2f}\n"
        f"📦 <b>Quantity</b>   : {quantity} shares\n"
        f"💵 <b>P&L</b>        : {pnl_sign}₹{abs(pnl):,.0f}  {pnl_emoji}\n"
        f"📋 <b>Reason</b>     : {reason_str}\n"
        f"⏰ <b>Time</b>       : {time_str}"
    )
    _send(msg)


def alert_trail_update(symbol: str, new_sl: float, ltp: float, unrealised_pnl: float):
    """Sent when trailing stop moves up (locking in profit)."""
    msg = (
        f"🔒 <b>TRAILING STOP UPDATED</b>\n"
        f"{'─' * 28}\n"
        f"🏢 <b>Company</b>     : <b>{symbol}</b>\n"
        f"📍 <b>Current price</b>: ₹{ltp:,.2f}\n"
        f"🛑 <b>New stop-loss</b>: ₹{new_sl:,.2f}\n"
        f"💵 <b>Unrealised P&L</b>: +₹{unrealised_pnl:,.0f} (locked in)\n"
        f"⏰ <b>Time</b>         : {datetime.now().strftime('%I:%M %p')}"
    )
    _send(msg)


def alert_daily_summary(
    pnl: float,
    trades: list,
    capital: float,
    total_trades: int = 0,
    wins: int = 0,
    losses: int = 0,
):
    """
    Sent at end of day (3:30 PM).
    Shows all trades with buy/sell prices.
    """

    time_str = datetime.now().strftime("%d %b %Y")

    # fallback safety
    if not wins and not losses:
        wins = len([t for t in trades if t["pnl"] > 0])
        losses = len([t for t in trades if t["pnl"] <= 0])

    pnl_sign = "+" if pnl >= 0 else ""
    summary_emoji = "🏆" if pnl >= 0 else "📉"

    trade_lines = ""

    for t in trades:
        p_sign = "+" if t["pnl"] >= 0 else ""
        emoji  = "✅" if t["pnl"] >= 0 else "❌"

        trade_lines += (
            f"\n{emoji} <b>{t['symbol']}</b>  "
            f"Buy ₹{t['entry']:,.1f} → Sell ₹{t['exit']:,.1f}  "
            f"Qty {t['qty']}  <b>{p_sign}₹{t['pnl']:,.0f}</b>"
        )

    msg = (
        f"{summary_emoji} <b>DAILY SUMMARY — {time_str}</b>\n"
        f"{'─' * 28}\n"
        f"📊 Total trades  : {total_trades}\n"
        f"✅ Wins          : {wins}\n"
        f"❌ Losses        : {losses}\n"
        f"💰 Net P&L       : {pnl_sign}₹{abs(pnl):,.0f}\n"
        f"🏦 Capital now   : ₹{capital:,.0f}\n"
        f"{'─' * 28}"
        f"{trade_lines if trade_lines else chr(10) + 'No trades today.'}"
    )

    _send(msg)


def alert_circuit_breaker(daily_loss: float, capital: float):
    """Sent when daily loss limit is hit and bot shuts down."""
    msg = (
        f"🚨 <b>CIRCUIT BREAKER TRIGGERED</b>\n"
        f"{'─' * 28}\n"
        f"⛔ Bot has <b>stopped trading</b> for today.\n"
        f"📉 Daily loss    : -₹{abs(daily_loss):,.0f}\n"
        f"🏦 Capital left  : ₹{capital:,.0f}\n"
        f"⏰ Time          : {datetime.now().strftime('%I:%M %p')}\n\n"
        f"Bot will resume automatically tomorrow at 9:15 AM."
    )
    _send(msg)


def alert_bot_started(mode: str, capital: float, trade_mode: str, max_trades: int = 4):
    _send(
        f"🤖 <b>BOT STARTED</b>\n"
        f"Mode: <b>{mode.upper()}</b>\n"
        f"Capital: ₹{capital:,}\n"
        f"Trade mode: {trade_mode.upper()}\n"
        f"Max open trades: {max_trades}\n"
        f"Time: {datetime.now().strftime('%d %b %Y %H:%M:%S')}"
    )

def alert_test():
    """Send a test message to verify setup. Run this first."""
    msg = (
        f"✅ <b>Telegram alert working!</b>\n"
        f"{'─' * 28}\n"
        f"Your Dhan XGBoost trading bot is connected.\n"
        f"You will receive alerts for:\n"
        f"  📈 Every BUY order (company, price, qty)\n"
        f"  📉 Every SELL order (profit/loss)\n"
        f"  🔒 Trailing stop updates\n"
        f"  📊 Daily summary at 3:30 PM\n"
        f"  🚨 Circuit breaker if triggered\n\n"
        f"⏰ Test sent at {datetime.now().strftime('%I:%M %p, %d %b %Y')}"
    )
    _send(msg)
