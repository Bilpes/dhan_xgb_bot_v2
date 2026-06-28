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
#   2026-06-28: Removed all duplicate trading params that
#               conflicted with trade_policy.py (thresholds,
#               ATR mults, HORIZON, MAX_OPEN_TRADES,
#               DAILY_LOSS_LIMIT, MIN_RR_RATIO).
#               config.py is now infra-only.
# ============================================================


import os
import json as _json
from dotenv import load_dotenv


# ── Load credentials from .env file ─────────────────────────
# .env file lives at:  config/.env
# Never touch config.py for credentials — only .env
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))


DHAN_CLIENT_ID    = os.getenv("DHAN_CLIENT_ID",    "")
DHAN_ACCESS_TOKEN = os.getenv("DHAN_ACCESS_TOKEN",  "")
TELEGRAM_BOT_TOKEN= os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID_1= os.getenv("TELEGRAM_CHAT_ID_1", "")
TELEGRAM_CHAT_ID_2= os.getenv("TELEGRAM_CHAT_ID_2", "")


# ── Capital & position sizing ────────────────────────────────
CAPITAL               = 400_000    # update as you scale
MAX_RISK_PCT          = 0.01       # 1% risk per trade
MAX_CAPITAL_PER_TRADE = 0.25       # never more than 25% in one trade
MAX_PER_SECTOR        = 2


# ── Trade mode ──────────────────────────────────────────────
TRADE_MODE      = "intraday"   # MIS on Dhan
ALLOW_SHORTS    = False


# ── Timing ──────────────────────────────────────────────────
CANDLE_INTERVAL      = "5"
LOOKBACK_CANDLES     = 60
NO_NEW_TRADE_BEFORE  = "09:20"
NO_NEW_TRADE_AFTER   = "15:00"
INTRADAY_CUTOFF      = "15:15"    # hard MIS exit deadline
MARKET_OPEN          = "09:15"
MARKET_CLOSE         = "15:30"


# ── Entry quality filters ───────────────────────────────────
# These are hard gates in signal_engine.py BEFORE the model score.
# Numeric trading params (thresholds, ATR mults) are in trade_policy.py.

STOP_LOSS_PCT                = 0.025   # hard cap: SL never more than 2.5% away
MIN_VOLUME_RATIO             = 0.60    # early-session floor
MIN_VOLUME_RATIO_CONFIRM     = 0.75    # confirmation gate after model passes
MIN_ATR_PCT                  = 0.0007  # minimum ATR as % of price
MIN_CANDLE_BODY_PCT          = 0.0005
MAX_DISTANCE_FROM_EMA20      = 0.06
MAX_DISTANCE_FROM_VWAP       = 0.05
REQUIRE_BREAKOUT_CONFIRMATION= False
REQUIRE_VWAP_CONFIRM         = False   # VWAP is already a model feature
TREND_STRENGTH_ENABLED       = True


# ── Re-entry protection ─────────────────────────────────────
NO_REENTRY_MINUTES = 60


# ── Lunch hours ─────────────────────────────────────────────
AVOID_LUNCH_HOURS  = False
LUNCH_START        = "12:30"
LUNCH_END          = "13:00"


# ── Trailing stop ───────────────────────────────────────────
TRAIL_AFTER_PCT  = 0.015    # activate after +1.5%
TRAIL_DISTANCE   = 0.012    # trail 1.2% below running high


# ── Position rotation ───────────────────────────────────────
ROTATION_ENABLED   = True
ROTATION_MIN_PROFIT= 0.005
ROTATION_MIN_EDGE  = 0.05


# ── Watchlist ───────────────────────────────────────────────
NIFTY50_SECURITY_ID = 13

_WATCHLIST_FILE = os.path.join(os.path.dirname(__file__), "watchlist.json")

def _load_watchlist():
    if not os.path.exists(_WATCHLIST_FILE):
        print(
            "\n[CONFIG WARNING] config/watchlist.json not found.\n"
            "Run:  python data/load_instruments.py\n"
            "This downloads Security IDs from Dhan automatically.\n"
        )
        return {}, {}
    with open(_WATCHLIST_FILE) as f:
        data = _json.load(f)
    return data.get("WATCHLIST", {}), data.get("SECTOR_MAP", {})

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
