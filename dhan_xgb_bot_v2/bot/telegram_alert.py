# ============================================================
#  bot/telegram_alert.py  —  Send trade alerts to Telegram
#                            with TRUE net P&L after all Dhan
#                            intraday charges (brokerage.py)
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
# FIX-1: Single canonical import at module level — removed duplicate
#         `from bot.brokerage import calculate_charges` inside alert_entry.
from bot.brokerage import calculate_charges, ChargeBreakdown  # noqa: F401

log = logging.getLogger("telegram")

RECIPIENTS = [TELEGRAM_CHAT_ID_1, TELEGRAM_CHAT_ID_2]


# ── Core sender ──────────────────────────────────────────────

def _send(message: str) -> None:
    """Send message to all configured recipients. Never raises — errors are logged."""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

    for chat_id in RECIPIENTS:
        if not chat_id:
            continue
        try:
            resp = requests.post(
                url,
                json={
                    "chat_id":    chat_id,
                    "text":       message,
                    "parse_mode": "HTML",
                },
                timeout=10,
            )
            if resp.status_code != 200:
                log.error("Telegram send failed to %s: %s", chat_id, resp.text)
            else:
                log.debug("Telegram alert sent to %s", chat_id)
        except Exception as exc:
            log.error("Telegram error for %s: %s", chat_id, exc)


# ── Alert types ──────────────────────────────────────────────

def alert_entry(
    symbol: str,
    buy_price: float,
    stop_loss: float,
    target: float,
    quantity: int,
    prob_up: float,
    trade_mode: str,
    invested: float,
) -> None:
    """
    Telegram BUY alert — includes breakeven price so trader
    knows the minimum move needed to cover all Dhan charges.

    FIX-1: Removed duplicate `from bot.brokerage import calculate_charges`
    that previously lived inside this function body.
    """
    time_str = datetime.now().strftime("%I:%M %p")

    rr       = (target - buy_price) / max(buy_price - stop_loss, 0.01)
    risk_pct = (abs(buy_price - stop_loss) / buy_price) * 100
    mode_str = "INTRADAY" if trade_mode.lower() == "intraday" else "SWING"

    # Breakeven at entry: calculate charges with sell == buy
    # so be_sell shows the exact price needed to cover all charges.
    charges_est = calculate_charges(buy_price, buy_price, quantity)
    be_sell     = charges_est.breakeven_sell

    msg = (
        f"🟢 <b>BUY ORDER PLACED</b>\n"
        f"{'─' * 30}\n"
        f"🏢 <b>Company</b>      : <b>{symbol}</b>\n"
        f"📈 <b>Buy Price</b>    : ₹{buy_price:,.2f}\n"
        f"🎯 <b>Target</b>       : ₹{target:,.2f}\n"
        f"🛑 <b>Stop Loss</b>    : ₹{stop_loss:,.2f}\n"
        f"⚖️ <b>Breakeven</b>    : ₹{be_sell:,.2f}  <i>(after charges)</i>\n"
        f"📦 <b>Quantity</b>     : {quantity:,} shares\n"
        f"💰 <b>Invested</b>     : ₹{invested:,.0f}\n"
        f"⚖️ <b>Risk/Reward</b>  : {rr:.2f}\n"
        f"📏 <b>Risk</b>         : {risk_pct:.2f}%\n"
        f"🤖 <b>Model Score</b>  : {prob_up:.3f}\n"
        f"📊 <b>Confidence</b>   : {prob_up * 100:.1f}%\n"
        f"🕐 <b>Mode</b>         : {mode_str}\n"
        f"⏰ <b>Time</b>         : {time_str}"
    )
    _send(msg)


def alert_exit(
    symbol: str,
    buy_price: float,
    sell_price: float,
    quantity: int,
    pnl: float,        # gross pnl from live_bot — kept for backward compat
    reason: str,
    trade_mode: str,   # noqa: ARG001 — kept for caller compat, not used in message
) -> None:
    """
    Exit alert with FULL Dhan charge breakdown and true net P&L.

    The `pnl` parameter is the gross P&L forwarded from live_bot.
    We always recalculate via brokerage.py so every charge line is shown.

    FIX-2: Added explicit \\n after the Time line so the P&L breakdown
    block starts on a new line and does not visually merge with Time.
    """
    time_str = datetime.now().strftime("%I:%M %p")

    charges   = calculate_charges(buy_price, sell_price, quantity)
    net       = charges.net_pnl
    net_sign  = "+" if net >= 0 else ""
    top_emoji = "💚" if net >= 0 else "🔴"
    result    = "✅ PROFIT" if net >= 0 else "❌ LOSS"

    reason_map = {
        "SL_HIT":          "Stop-loss hit",
        "SIGNAL_FLIP":     "Signal flip (model turned bearish)",
        "INTRADAY_CUTOFF": "3:10 PM square-off",
        "TRAIL_STOP":      "Trailing stop triggered",
        "TARGET_HIT":      "Target hit",
        "MANUAL":          "Manual exit",
    }
    reason_str = reason_map.get(reason, reason)

    log.info(
        "[EXIT] %s  buy=₹%.2f sell=₹%.2f qty=%d  %s",
        symbol, buy_price, sell_price, quantity,
        charges.to_log_string(),
    )

    msg = (
        f"{top_emoji} <b>POSITION CLOSED</b>\n"
        f"{'─' * 30}\n"
        f"🏢 <b>Company</b>      : <b>{symbol}</b>\n"
        f"📈 <b>Buy price</b>    : ₹{buy_price:,.2f}\n"
        f"📉 <b>Sell price</b>   : ₹{sell_price:,.2f}\n"
        f"📦 <b>Quantity</b>     : {quantity} shares\n"
        f"📋 <b>Reason</b>       : {reason_str}\n"
        # FIX-2: \n added here so to_telegram_lines() starts on its own line
        f"⏰ <b>Time</b>         : {time_str}\n"
        f"{charges.to_telegram_lines()}\n"
        f"{'─' * 30}\n"
        f"<b>Result: {result}  {net_sign}₹{net:,.2f}</b>"
    )
    _send(msg)


def alert_trail_update(
    symbol: str,
    new_sl: float,
    ltp: float,
    unrealised_gross: float,
    buy_price: float,
    quantity: int,
) -> None:
    """
    Trailing stop update — shows unrealised net P&L if stopped out at LTP.

    FIX-3: Unrealised net sign is now dynamic — was hardcoded '+' which
    rendered as '+₹-45.20' on gap-down candles. Now uses conditional sign.
    """
    charges_if_exit = calculate_charges(buy_price, ltp, quantity)
    unrealised_net  = charges_if_exit.net_pnl

    # FIX-3: dynamic signs — never show '+' when value is negative
    gross_sign = "+" if unrealised_gross >= 0 else ""
    net_sign   = "+" if unrealised_net   >= 0 else ""

    msg = (
        f"🔒 <b>TRAILING STOP UPDATED</b>\n"
        f"{'─' * 28}\n"
        f"🏢 <b>Company</b>          : <b>{symbol}</b>\n"
        f"📍 <b>Current price</b>    : ₹{ltp:,.2f}\n"
        f"🛑 <b>New stop-loss</b>    : ₹{new_sl:,.2f}\n"
        f"💵 <b>Unrealised gross</b> : {gross_sign}₹{unrealised_gross:,.2f}\n"
        f"💵 <b>Unrealised net</b>   : {net_sign}₹{unrealised_net:,.2f}  "
        f"<i>(after charges)</i>\n"
        f"⏰ <b>Time</b>             : {datetime.now().strftime('%I:%M %p')}"
    )
    _send(msg)


def alert_daily_summary(
    pnl: float,            # gross pnl sum — kept for caller compat
    trades: list,
    capital: float,
    total_trades: int = 0,
    wins: int = 0,
    losses: int = 0,
) -> None:
    """
    End-of-day summary with TRUE net P&L per trade and overall total.

    Each dict in `trades` must contain:
      symbol  — ticker string
      entry   — buy fill price (float)
      exit    — sell fill price (float)
      qty     — number of shares (int)
      pnl     — gross pnl (float, used only as fallback display)

    FIX-4: Win/loss recount is now unconditional. The old guard
    `if not wins and not losses` skipped the recount on all-win days
    (losses == 0 evaluated falsy), potentially leaving stale caller-
    supplied values. We always recount from net P&L so the figures
    are always accurate regardless of what the caller passed.
    """
    time_str = datetime.now().strftime("%d %b %Y")

    # ── Recalculate net P&L per trade from actual fills ──────
    enriched_trades: list[dict] = []
    total_gross     = 0.0
    total_charges   = 0.0
    total_net       = 0.0

    for t in trades:
        c = calculate_charges(
            buy_price  = t["entry"],
            sell_price = t["exit"],
            quantity   = t["qty"],
        )
        enriched_trades.append({**t, "_charges": c})
        total_gross   += c.gross_pnl
        total_charges += c.total_charges
        total_net     += c.net_pnl

    # FIX-4: Always recount from net — no conditional guard
    wins   = sum(1 for t in enriched_trades if t["_charges"].net_pnl > 0)
    losses = sum(1 for t in enriched_trades if t["_charges"].net_pnl <= 0)

    net_sign      = "+" if total_net >= 0 else ""
    summary_emoji = "🏆" if total_net >= 0 else "📉"

    # ── Per-trade lines ───────────────────────────────────────
    trade_lines = ""
    for t in enriched_trades:
        c      = t["_charges"]
        p_sign = "+" if c.net_pnl   >= 0 else ""
        g_sign = "+" if c.gross_pnl >= 0 else ""
        emoji  = "✅" if c.net_pnl  >  0 else "❌"
        trade_lines += (
            f"\n{emoji} <b>{t['symbol']}</b>  "
            f"₹{t['entry']:,.1f}→₹{t['exit']:,.1f}  "
            f"×{t['qty']}  "
            f"gross {g_sign}₹{c.gross_pnl:.0f}  "
            f"chg -₹{c.total_charges:.2f}  "
            f"<b>net {p_sign}₹{c.net_pnl:.2f}</b>"
        )

    msg = (
        f"{summary_emoji} <b>DAILY SUMMARY — {time_str}</b>\n"
        f"{'─' * 32}\n"
        f"📊 Total trades   : {total_trades or len(trades)}\n"
        f"✅ Wins (net)     : {wins}\n"
        f"❌ Losses (net)   : {losses}\n"
        f"{'─' * 32}\n"
        f"💹 Gross P&L      : {'+' if total_gross >= 0 else ''}₹{total_gross:,.2f}\n"
        f"🏦 Total charges  : -₹{total_charges:,.2f}\n"
        f"💰 <b>Net P&L</b>       : <b>{net_sign}₹{total_net:,.2f}</b>\n"
        f"🏦 Capital now    : ₹{capital:,.0f}\n"
        f"{'─' * 32}"
        f"{trade_lines if trade_lines else chr(10) + 'No trades today.'}"
    )
    _send(msg)

    log.info(
        "[DAILY] trades=%d wins=%d losses=%d "
        "gross=₹%.2f charges=₹%.2f net=₹%.2f capital=₹%.0f",
        total_trades or len(trades), wins, losses,
        total_gross, total_charges, total_net, capital,
    )


def alert_circuit_breaker(daily_loss: float, capital: float) -> None:
    """Sent when the daily loss limit is hit and the bot stops trading."""
    msg = (
        f"🚨 <b>CIRCUIT BREAKER TRIGGERED</b>\n"
        f"{'─' * 28}\n"
        f"⛔ Bot has <b>stopped trading</b> for today.\n"
        f"📉 Daily net loss : -₹{abs(daily_loss):,.2f}\n"
        f"🏦 Capital left   : ₹{capital:,.0f}\n"
        f"⏰ Time           : {datetime.now().strftime('%I:%M %p')}\n\n"
        f"Bot will resume automatically tomorrow at 9:15 AM."
    )
    _send(msg)


def alert_bot_started(
    mode: str,
    capital: float,
    trade_mode: str,
    max_trades: int = 5,
) -> None:
    """Sent once when the bot process starts."""
    _send(
        f"🤖 <b>BOT STARTED</b>\n"
        f"Mode: <b>{mode.upper()}</b>\n"
        f"Capital: ₹{capital:,}\n"
        f"Trade mode: {trade_mode.upper()}\n"
        f"Max open trades: {max_trades}\n"
        f"Time: {datetime.now().strftime('%d %b %Y %H:%M:%S')}"
    )


def alert_test() -> None:
    """
    Send a test message to verify the full Telegram + brokerage setup.
    Run once after configuring BOT_TOKEN and CHAT_IDs:

        python -c "from bot.telegram_alert import alert_test; alert_test()"
    """
    c = calculate_charges(buy_price=1000, sell_price=1005, quantity=10)
    msg = (
        f"✅ <b>Telegram alert working!</b>\n"
        f"{'─' * 28}\n"
        f"Your Dhan XGBoost trading bot is connected.\n"
        f"Charges are calculated with EXACT Dhan MIS formula.\n\n"
        f"<b>Example: HDFCBANK  ₹1000→₹1005  ×10 shares</b>"
        f"{c.to_telegram_lines()}\n\n"
        f"⏰ Test sent at {datetime.now().strftime('%I:%M %p, %d %b %Y')}"
    )
    _send(msg)
