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
    CAPITAL,
    TRADE_MODE,
    MAX_OPEN_TRADES,
)
from bot.brokerage import calculate_charges, ChargeBreakdown  # noqa: F401

log = logging.getLogger("telegram")

RECIPIENTS = [TELEGRAM_CHAT_ID_1, TELEGRAM_CHAT_ID_2]

SEP30 = "─" * 30
SEP28 = "─" * 28
SEP32 = "─" * 32


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
    symbol:   str,
    entry:    float,          # live_bot passes 'entry'
    sl:       float,          # live_bot passes 'sl'
    target:   float,
    qty:      int,            # live_bot passes 'qty'
    prob:     float,          # live_bot passes 'prob'
    rr:       float   = 0.0,  # live_bot passes 'rr'
    be_sell:  float   = 0.0,  # live_bot passes 'be_sell' (already computed)
    **_kwargs,                # absorb any legacy kwargs silently
) -> None:
    """
    Telegram BUY alert.

    live_bot._enter_paper / _enter_live calls:
        alert_entry(symbol, entry, sl, target, qty, prob, rr, be_sell)

    be_sell is pre-computed by live_bot via calculate_charges(entry, entry, qty).
    If not supplied (e.g. fallback call), we compute it here.
    """
    if be_sell <= 0:
        be_sell = calculate_charges(entry, entry, qty).breakeven_sell

    time_str = datetime.now().strftime("%I:%M %p")
    risk_pct = (abs(entry - sl) / entry) * 100
    invested = entry * qty
    mode_str = "INTRADAY" if TRADE_MODE.lower() == "intraday" else "SWING"

    msg = (
        f"🟢 <b>BUY ORDER PLACED</b>\n"
        f"{SEP30}\n"
        f"🏢 <b>Company</b>      : <b>{symbol}</b>\n"
        f"📈 <b>Buy Price</b>    : ₹{entry:,.2f}\n"
        f"🎯 <b>Target</b>       : ₹{target:,.2f}\n"
        f"🛑 <b>Stop Loss</b>    : ₹{sl:,.2f}\n"
        f"⚖️ <b>Breakeven</b>    : ₹{be_sell:,.2f}  <i>(after charges)</i>\n"
        f"📦 <b>Quantity</b>     : {qty:,} shares\n"
        f"💰 <b>Invested</b>     : ₹{invested:,.0f}\n"
        f"⚖️ <b>Risk/Reward</b>  : {rr:.2f}\n"
        f"📏 <b>Risk</b>         : {risk_pct:.2f}%\n"
        f"🤖 <b>Model Score</b>  : {prob:.3f}\n"
        f"📊 <b>Confidence</b>   : {prob * 100:.1f}%\n"
        f"🕐 <b>Mode</b>         : {mode_str}\n"
        f"⏰ <b>Time</b>         : {time_str}"
    )
    _send(msg)


def alert_exit(
    symbol:         str,
    exit_price:     float,          # live_bot passes 'exit_price'
    pnl:            float,          # live_bot passes net_pnl
    reason:         str,
    hold_minutes:   float   = 0.0,
    charges_detail: str     = "",   # live_bot passes charges.to_telegram_lines()
    entry:          float   = 0.0,  # NOT passed by live_bot — kept for compat only
    qty:            int     = 0,    # NOT passed by live_bot — kept for compat only
    **_kwargs,
) -> None:
    """
    Exit alert.

    live_bot._exit_trade calls:
        alert_exit(symbol, exit_price, pnl, reason, hold_minutes, charges_detail)

    pnl is already net P&L (after charges) computed by brokerage.py in live_bot.
    charges_detail is charges.to_telegram_lines() — the full breakdown string.
    If charges_detail is empty (fallback path), we just show the net pnl.
    """
    time_str  = datetime.now().strftime("%I:%M %p")
    net       = pnl
    net_sign  = "+" if net >= 0 else ""
    top_emoji = "💚" if net >= 0 else "🔴"
    result    = "✅ PROFIT" if net >= 0 else "❌ LOSS"

    reason_map = {
        "SL_HIT":                 "Stop-loss hit",
        "TARGET_HIT":             "Target hit 🎯",
        "SIGNAL_FLIP":            "Signal flip (model turned bearish)",
        "INTRADAY_CUTOFF":        "3:15 PM square-off",
        "TRAIL_STOP":             "Trailing stop triggered",
        "AUTO_EXIT_EOD":          "Auto-exit (15:15 weak position)",
        "MOMENTUM_FAILURE":       "Momentum failure exit",
        "ROTATION_BETTER_SIGNAL": "Rotated to better signal",
        "EOD_AUTO_SQUARE_OFF":    "EOD auto square-off",
        "CIRCUIT_BREAKER":        "Circuit breaker triggered",
        "KEYBOARD_INTERRUPT":     "Bot stopped manually",
        "CLOSED_ON_DHAN":         "Closed on Dhan (external)",
        "MANUAL":                 "Manual exit",
    }
    reason_str = reason_map.get(reason, reason)
    hold_str   = f"{hold_minutes:.0f} min" if hold_minutes else ""

    msg = (
        f"{top_emoji} <b>POSITION CLOSED</b>\n"
        f"{SEP30}\n"
        f"🏢 <b>Company</b>      : <b>{symbol}</b>\n"
        f"📉 <b>Exit price</b>   : ₹{exit_price:,.2f}\n"
        f"📋 <b>Reason</b>       : {reason_str}\n"
        + (f"⏱️ <b>Held</b>          : {hold_str}\n" if hold_str else "")
        + (charges_detail if charges_detail else
           f"\n💰 <b>Net P&L</b>       : {net_sign}₹{net:,.2f}")
        + f"\n{SEP30}\n"
        f"⏰ <b>Time</b>         : {time_str}\n"
        f"<b>Result: {result}  {net_sign}₹{net:,.2f}</b>"
    )
    _send(msg)


def alert_trail_update(
    symbol:  str,
    new_sl:  float,
    ltp:     float,
    **_kwargs,   # absorb buy_price, quantity, unrealised_gross if passed
) -> None:
    """
    Trailing stop update.

    live_bot._monitor_positions calls:
        alert_trail_update(symbol=symbol, new_sl=new_sl, ltp=ltp)
    """
    msg = (
        f"🔒 <b>TRAILING STOP UPDATED</b>\n"
        f"{SEP28}\n"
        f"🏢 <b>Company</b>          : <b>{symbol}</b>\n"
        f"📍 <b>Current price</b>    : ₹{ltp:,.2f}\n"
        f"🛑 <b>New stop-loss</b>    : ₹{new_sl:,.2f}\n"
        f"⏰ <b>Time</b>             : {datetime.now().strftime('%I:%M %p')}"
    )
    _send(msg)


def alert_daily_summary(
    pnl:          float,        # net daily pnl from risk.daily_pnl
    total_trades: int   = 0,
    wins:         int   = 0,
    losses:       int   = 0,
    capital:      float = 0.0,
    trades:       list  = None,  # optional per-trade list for detail lines
    **_kwargs,
) -> None:
    """
    End-of-day summary.

    live_bot._eod_reset calls:
        alert_daily_summary(pnl, total_trades, wins, losses, capital)
    """
    if capital <= 0:
        capital = CAPITAL

    time_str      = datetime.now().strftime("%d %b %Y")
    net_sign      = "+" if pnl >= 0 else ""
    summary_emoji = "🏆" if pnl >= 0 else "📉"

    trade_lines = ""
    if trades:
        for t in trades:
            try:
                exit_val = t.get("exit", t.get("exit_price", t["entry"]))
                c        = calculate_charges(t["entry"], exit_val, t["qty"])
                p_sign   = "+" if c.net_pnl >= 0 else ""
                emoji    = "✅" if c.net_pnl > 0 else "❌"
                trade_lines += (
                    f"\n{emoji} <b>{t['symbol']}</b>  "
                    f"₹{t['entry']:,.1f}→₹{exit_val:,.1f}  "
                    f"×{t['qty']}  "
                    f"<b>net {p_sign}₹{c.net_pnl:.2f}</b>"
                )
            except Exception:
                pass

    msg = (
        f"{summary_emoji} <b>DAILY SUMMARY — {time_str}</b>\n"
        f"{SEP32}\n"
        f"📊 Total trades   : {total_trades}\n"
        f"✅ Wins (net)     : {wins}\n"
        f"❌ Losses (net)   : {losses}\n"
        f"{SEP32}\n"
        f"💰 <b>Net P&L</b>       : <b>{net_sign}₹{pnl:,.2f}</b>\n"
        f"🏦 Capital now    : ₹{capital:,.0f}\n"
        f"{SEP32}"
        f"{trade_lines if trade_lines else chr(10) + 'No trades today.'}"
    )
    _send(msg)

    log.info(
        "[DAILY] trades=%d wins=%d losses=%d net=₹%.2f capital=₹%.0f",
        total_trades, wins, losses, pnl, capital,
    )


def alert_circuit_breaker(
    daily_pnl:  float = 0.0,   # live_bot passes 'daily_pnl'
    reason:     str   = "",    # live_bot passes 'reason'
    daily_loss: float = 0.0,   # legacy compat
    capital:    float = 0.0,   # legacy compat
    **_kwargs,
) -> None:
    """
    Circuit breaker alert.

    live_bot calls:
        alert_circuit_breaker(daily_pnl=self.risk.daily_pnl,
                              reason='daily_loss_or_consecutive_sl')
    """
    loss_amt = abs(daily_pnl or daily_loss)
    cap      = capital if capital > 0 else CAPITAL
    msg = (
        f"🚨 <b>CIRCUIT BREAKER TRIGGERED</b>\n"
        f"{SEP28}\n"
        f"⛔ Bot has <b>stopped trading</b> for today.\n"
        f"📉 Daily net loss : -₹{loss_amt:,.2f}\n"
        f"🏦 Capital left   : ₹{cap:,.0f}\n"
        f"📝 Reason         : {reason}\n"
        f"⏰ Time           : {datetime.now().strftime('%I:%M %p')}\n\n"
        f"Bot will resume automatically tomorrow at 9:15 AM."
    )
    _send(msg)


def alert_bot_started(
    mode:    str,
    capital: float,
    **_kwargs,   # absorb trade_mode, max_trades if passed
) -> None:
    """
    Startup alert.

    live_bot calls:
        alert_bot_started(mode=BOT_MODE, capital=CAPITAL)
    """
    _send(
        f"🤖 <b>BOT STARTED</b>\n"
        f"Mode: <b>{mode.upper()}</b>\n"
        f"Capital: ₹{capital:,}\n"
        f"Trade mode: {TRADE_MODE.upper()}\n"
        f"Max open trades: {MAX_OPEN_TRADES}\n"
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
        f"{SEP28}\n"
        f"Your Dhan XGBoost trading bot is connected.\n"
        f"Charges are calculated with EXACT Dhan MIS formula.\n\n"
        f"<b>Example: HDFCBANK  ₹1000→₹1005  ×10 shares</b>"
        f"{c.to_telegram_lines()}\n\n"
        f"⏰ Test sent at {datetime.now().strftime('%I:%M %p, %d %b %Y')}"
    )
    _send(msg)
