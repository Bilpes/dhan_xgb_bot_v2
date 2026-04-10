# ============================================================
#  bot/health_check.py  —  Pre-market check at 9:00 AM
# ============================================================
"""
Runs at 9:00 AM every trading day via Task Scheduler.
Checks everything is ready before market opens at 9:15 AM.
Sends Telegram: green = all good, red = something broken.
"""

import os, sys, pickle, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from datetime   import datetime
from bot.telegram_alert import _send
from config.config      import (
    MODEL_PATH, SCALER_PATH, CAPITAL,
    DHAN_CLIENT_ID, DHAN_ACCESS_TOKEN, TRADE_MODE
)

log = logging.getLogger("health_check")
logging.basicConfig(level=logging.INFO)


def check_model_exists():
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        size = os.path.getsize(MODEL_PATH)
        return True, f"Model OK ({size//1024} KB)"
    return False, "model.pkl missing — run models/train.py first"


def check_api_credentials():
    if DHAN_CLIENT_ID == "YOUR_CLIENT_ID":
        return False, "Dhan credentials not set in config.py"
    if DHAN_ACCESS_TOKEN == "YOUR_ACCESS_TOKEN":
        return False, "Dhan token not set — paste today's token"
    if len(DHAN_ACCESS_TOKEN) < 20:
        return False, "Dhan token looks invalid"
    return True, "API credentials set"


def check_api_connection():
    try:
        import requests
        headers = {
            "access-token": DHAN_ACCESS_TOKEN,
            "client-id":    DHAN_CLIENT_ID,
            "Content-Type": "application/json",
        }
        # Correct Dhan v2 endpoint: GET /fundlimit
        resp = requests.get(
            "https://api.dhan.co/v2/fundlimit",
            headers=headers, timeout=8
        )
        if resp.status_code == 200:
            try:
                data      = resp.json()
                available = (
                    data.get("data", {}).get("availabelBalance")
                    or data.get("availabelBalance")
                    or data.get("availableBalance")
                    or "check app"
                )
                return True, f"Dhan connected — available ₹{available}"
            except Exception:
                return True, "Dhan API connected successfully"
        return False, f"Dhan API error: {resp.status_code} — regenerate token"
    except Exception as e:
        return False, f"Dhan connection failed: {e}"


def check_market_holiday():
    from bot.nse_holidays import is_holiday_today
    if is_holiday_today():
        return False, "TODAY IS NSE HOLIDAY — bot will not trade"
    return True, "Trading day confirmed"


def check_logs_dir():
    os.makedirs("logs", exist_ok=True)
    return True, "Logs directory ready"


def check_data_freshness():
    data_dir = "data/historical"
    if not os.path.exists(data_dir):
        return False, "No historical data — run data/download_data.py"
    files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    if not files:
        return False, "No CSV files in data/historical/"
    newest = max(
        os.path.getmtime(os.path.join(data_dir, f)) for f in files
    )
    from datetime import datetime
    days_old = (datetime.now().timestamp() - newest) / 86400
    if days_old > 8:
        return False, f"Data is {days_old:.0f} days old — run download_data.py"
    return True, f"Data is {days_old:.0f} days old — OK"


def run_health_check():
    checks = [
        ("Model file",       check_model_exists),
        ("API credentials",  check_api_credentials),
        ("Dhan connection",  check_api_connection),
        ("Market holiday",   check_market_holiday),
        ("Log directory",    check_logs_dir),
        ("Data freshness",   check_data_freshness),
    ]

    results  = []
    all_good = True

    for name, fn in checks:
        try:
            ok, msg = fn()
        except Exception as e:
            ok, msg = False, str(e)

        results.append((name, ok, msg))
        if not ok:
            all_good = False
        log.info("  [%s] %s — %s", "OK" if ok else "FAIL", name, msg)

    # Build Telegram message
    date_str = datetime.now().strftime("%d %b %Y, %I:%M %p")
    lines    = []
    for name, ok, msg in results:
        emoji = "✅" if ok else "❌"
        lines.append(f"{emoji} {name}: {msg}")

    if all_good:
        header = f"🟢 <b>BOT HEALTH CHECK — ALL GOOD</b>\n{date_str}"
        footer = "\n\n⏰ Market opens in 15 minutes. Bot is ready."
    else:
        header = f"🔴 <b>BOT HEALTH CHECK — ACTION NEEDED</b>\n{date_str}"
        footer = "\n\n⚠️ Fix the issues above before market opens at 9:15 AM."

    msg = header + "\n" + "─" * 28 + "\n" + "\n".join(lines) + footer
    _send(msg)

    return all_good


if __name__ == "__main__":
    ok = run_health_check()
    sys.exit(0 if ok else 1)