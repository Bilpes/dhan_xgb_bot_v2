# ============================================================
#  config/config.py  —  Infrastructure, paths, API keys,
#                        timing, filters, Redis, Telegram.
#
#  THIS FILE OWNS: credentials, paths, candle settings,
#                  timing windows, entry quality filters,
#                  Redis TTLs, Telegram IDs.
#
#  THIS FILE DOES NOT OWN: BUY_THRESHOLD, ATR_SL_MULT,
#  ATR_TP_MULT, HORIZON, MAX_OPEN_POSITIONS, MAX_DAILY_LOSS,
#  MIN_RR_RATIO, or any numeric trading parameter.
#  → All of those live exclusively in bot/trade_policy.py
#
#  CREDENTIALS: Never stored here — read from .env file.
#  See config/.env.example for setup instructions.
#
# Fix log:
#   2026-05-25: BUG-A/B/C/D fixes (see git history)
#   2026-06-28: Removed all duplicate trading params.
#   2026-06-28: Bulletproof .env loader (4 candidate paths).
#   2026-06-28: Fixed _load_watchlist() — was calling
#               data.get('WATCHLIST') but watchlist.json uses
#               keys tier_a / tier_b / SECURITY_IDS.
#               Now builds {symbol: security_id} correctly.
#   2026-06-29: Added DAILY_LOSS_LIMIT — was missing, caused
#               ImportError in risk_manager.py on live_bot start.
# ============================================================

import os
import json as _json
from pathlib import Path
from dotenv import load_dotenv


# ── Bulletproof .env loader ──────────────────────────────────
def _find_and_load_env() -> bool:
    """
    Try loading .env from 4 candidate locations.
    Priority:
        1. config/.env  (relative to this file — canonical)
        2. ../.env      (project root, one level above config/)
        3. cwd/config/.env
        4. cwd/.env     (last resort)
    """
    _this_dir = Path(__file__).resolve().parent
    _project  = _this_dir.parent
    _cwd      = Path.cwd()

    candidates = [
        _this_dir  / ".env",
        _project   / ".env",
        _cwd       / "config" / ".env",
        _cwd       / ".env",
    ]

    for path in candidates:
        if path.exists():
            load_dotenv(dotenv_path=str(path), override=True)
            print(f"[config] ✅  Loaded .env from: {path}")
            return True

    print(
        "\n[config] ⚠️  WARNING: No .env file found.\n"
        "  Expected location: config/.env\n"
        "  Copy config/.env.example → config/.env and fill in your credentials.\n"
        "  Tried:\n" +
        "\n".join(f"    • {p}" for p in candidates) + "\n"
    )
    return False


_ENV_LOADED = _find_and_load_env()


DHAN_CLIENT_ID    = os.getenv("DHAN_CLIENT_ID",    "")
DHAN_ACCESS_TOKEN = os.getenv("DHAN_ACCESS_TOKEN",  "")
TELEGRAM_BOT_TOKEN= os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID_1= os.getenv("TELEGRAM_CHAT_ID_1", "")
TELEGRAM_CHAT_ID_2= os.getenv("TELEGRAM_CHAT_ID_2", "")


# ── Capital & position sizing ────────────────────────────────
CAPITAL               = 400_000
MAX_RISK_PCT          = 0.01        # 1% of capital max risk per trade
MAX_CAPITAL_PER_TRADE = 0.25        # 25% of capital max per single position
MAX_PER_SECTOR        = 2
DAILY_LOSS_LIMIT      = 0.04        # 4% of CAPITAL = ₹16,000 daily circuit breaker
                                    # matches MAX_RISK_PCT×4 max open positions
                                    # warning fires at 75% = 3% (see risk_manager SYNC-3)


# ── Trade mode ──────────────────────────────────────────────
TRADE_MODE      = "intraday"
ALLOW_SHORTS    = False


# ── Timing ──────────────────────────────────────────────────
CANDLE_INTERVAL      = "5"
LOOKBACK_CANDLES     = 60
NO_NEW_TRADE_BEFORE  = "09:20"
NO_NEW_TRADE_AFTER   = "15:00"
INTRADAY_CUTOFF      = "15:15"
MARKET_OPEN          = "09:15"
MARKET_CLOSE         = "15:30"


# ── Entry quality filters ───────────────────────────────────
STOP_LOSS_PCT                = 0.025
MIN_VOLUME_RATIO             = 0.60
MIN_VOLUME_RATIO_CONFIRM     = 0.75
MIN_ATR_PCT                  = 0.0007
MIN_CANDLE_BODY_PCT          = 0.0005
MAX_DISTANCE_FROM_EMA20      = 0.06
MAX_DISTANCE_FROM_VWAP       = 0.05
REQUIRE_BREAKOUT_CONFIRMATION= False
REQUIRE_VWAP_CONFIRM         = False
TREND_STRENGTH_ENABLED       = True


# ── Re-entry protection ─────────────────────────────────────
NO_REENTRY_MINUTES = 60


# ── Lunch hours ─────────────────────────────────────────────
AVOID_LUNCH_HOURS  = False
LUNCH_START        = "12:30"
LUNCH_END          = "13:00"


# ── Trailing stop ───────────────────────────────────────────
TRAIL_AFTER_PCT  = 0.015
TRAIL_DISTANCE   = 0.012


# ── Position rotation ───────────────────────────────────────
ROTATION_ENABLED   = True
ROTATION_MIN_PROFIT= 0.005
ROTATION_MIN_EDGE  = 0.05


# ── Watchlist ───────────────────────────────────────────────
NIFTY50_SECURITY_ID = 13

_WATCHLIST_FILE = os.path.join(os.path.dirname(__file__), "watchlist.json")


def _load_watchlist():
    """
    Build WATCHLIST as {symbol: security_id} from watchlist.json.

    watchlist.json structure:
        tier_a       : ["ICICIBANK", ...]
        tier_b       : ["HDFCBANK", ...]
        SECURITY_IDS : {"ICICIBANK": "4963", ...}
        SECTOR_MAP   : {"ICICIBANK": "BANKING", ...}

    Returns:
        WATCHLIST   : dict {symbol: security_id_str}
        SECTOR_MAP  : dict {symbol: sector_str}
    """
    if not os.path.exists(_WATCHLIST_FILE):
        print(
            "\n[CONFIG WARNING] config/watchlist.json not found.\n"
            "Expected at: " + _WATCHLIST_FILE + "\n"
        )
        return {}, {}

    with open(_WATCHLIST_FILE) as f:
        data = _json.load(f)

    sec_ids    = data.get("SECURITY_IDS", {})
    sector_map = data.get("SECTOR_MAP",   {})

    # Combine tier_a + tier_b into active universe
    tier_a  = data.get("tier_a", [])
    tier_b  = data.get("tier_b", [])
    symbols = tier_a + tier_b

    if not symbols:
        # Fallback: if old-style WATCHLIST key exists, use it directly
        old_style = data.get("WATCHLIST", {})
        if old_style:
            print("[config] Using legacy WATCHLIST key from watchlist.json")
            return old_style, sector_map
        print("[CONFIG WARNING] watchlist.json has no tier_a/tier_b/WATCHLIST keys.")
        return {}, sector_map

    # Build {symbol: security_id}
    watchlist = {}
    missing   = []
    for sym in symbols:
        sid = sec_ids.get(sym)
        if sid:
            watchlist[sym] = str(sid)
        else:
            missing.append(sym)

    if missing:
        print(f"[config] ⚠️  {len(missing)} symbols have no SECURITY_ID — skipped: {missing}")

    print(f"[config] ✅  Watchlist loaded: {len(watchlist)} symbols "
          f"({len(tier_a)} Tier-A + {len(tier_b)} Tier-B)")
    return watchlist, sector_map


WATCHLIST, SECTOR_MAP = _load_watchlist()


# ── Model paths ─────────────────────────────────────────────
MODEL_PATH         = "models/xgb_model.pkl"
SCALER_PATH        = "models/scaler.pkl"
BACKUP_MODEL_PATH  = "models/xgb_model_backup.pkl"
BACKUP_SCALER_PATH = "models/scaler_backup.pkl"


# ── Logging ─────────────────────────────────────────────────
LOG_FILE      = "logs/bot.log"
TRADE_LOG     = "logs/trades.csv"
RETRAIN_LOG   = "logs/retrain.log"
SIGNAL_LOG    = "logs/signal_scan.csv"


# ── Retraining schedule ─────────────────────────────────────
RETRAIN_EVERY_DAYS  = 7
EMBARGO_DAYS        = 14
MIN_TRAIN_SAMPLES   = 3000
WALK_FORWARD_FOLDS  = 5
MIN_ACCURACY        = 0.52
MIN_AUC             = 0.56
MIN_PRECISION       = 0.52
