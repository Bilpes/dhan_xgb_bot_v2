# ============================================================
#  bot/token_refresh.py  —  Verify Dhan token is valid
# ============================================================
"""
Checks if today's token in config/.env is working.
Run at 8:55 AM before starting the bot.

    python -m bot.token_refresh

If it shows OK  -> proceed to health_check and live_bot
If it shows FAIL -> get fresh token from Dhan app and repaste
"""

import os, sys, logging, requests

# Force reload .env fresh every time — no caching
from dotenv import load_dotenv, dotenv_values

log = logging.getLogger("token_refresh")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")

ENV_PATH = os.path.join("config", ".env")


def read_credentials():
    """Read directly from file — bypasses any dotenv caching."""
    if not os.path.exists(ENV_PATH):
        return None, None

    values = dotenv_values(ENV_PATH)   # reads fresh every time
    token     = values.get("DHAN_ACCESS_TOKEN", "").strip()
    client_id = values.get("DHAN_CLIENT_ID", "").strip()
    return token, client_id


def verify_token(token: str, client_id: str) -> tuple:
    """
    Verify token against Dhan v2 fundlimit endpoint.
    Returns (is_valid: bool, message: str)
    """
    if not token or len(token) < 20:
        return False, "Token is empty or too short in config/.env"

    if token == "paste_todays_token_here":
        return False, "Token not updated — still has placeholder text"

    headers = {
        "access-token": token,
        "client-id":    client_id,
        "Content-Type": "application/json",
    }

    try:
        # Correct Dhan v2 endpoint
        resp = requests.get(
            "https://api.dhan.co/v2/fundlimit",
            headers=headers,
            timeout=10,
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
                return True, f"Token valid — available balance: Rs.{available}"
            except Exception:
                return True, "Token valid (connected to Dhan)"

        elif resp.status_code == 401:
            return False, "Token expired or invalid — generate a new one from Dhan app"

        elif resp.status_code == 429:
            return True, "Token likely valid — rate limited (429), try again in 2 seconds"

        else:
            return False, f"Dhan returned HTTP {resp.status_code} — check token"

    except requests.exceptions.ConnectionError:
        return False, "No internet connection"
    except requests.exceptions.Timeout:
        return False, "Dhan API timeout — try again"
    except Exception as e:
        return False, f"Error: {e}"


def run_token_refresh():
    try:
        from bot.telegram_alert import _send
        telegram_available = True
    except Exception:
        telegram_available = False

    log.info("Reading token from %s ...", ENV_PATH)
    token, client_id = read_credentials()

    if not token:
        msg = "config/.env not found or DHAN_ACCESS_TOKEN missing"
        log.error(msg)
        if telegram_available:
            _send(f"FAIL - {msg}")
        return False

    log.info("Verifying token against Dhan v2 API...")
    is_valid, message = verify_token(token, client_id)

    if is_valid:
        log.info("OK - %s", message)
        if telegram_available:
            _send(f"Token OK - {message}")
        return True
    else:
        log.error("FAIL - %s", message)
        if telegram_available:
            _send(
                f"TOKEN FAIL\n"
                f"{message}\n\n"
                f"Fix: Dhan app -> Profile -> Data APIs -> Copy token\n"
                f"Paste in config/.env on DHAN_ACCESS_TOKEN line"
            )
        return False


if __name__ == "__main__":
    ok = run_token_refresh()
    sys.exit(0 if ok else 1)