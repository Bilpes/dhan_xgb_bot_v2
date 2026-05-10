# ============================================================
#  config/config.py  —  All settings in one place
#  CREDENTIALS: Never stored here — read from .env file
#  See config/.env.example for setup instructions
# ============================================================

import os
from dotenv import load_dotenv

# ── Load credentials from .env file ─────────────────────────
# .env file lives at:  config/.env
# You NEVER touch config.py for credentials — only .env
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

DHAN_CLIENT_ID    = os.getenv("DHAN_CLIENT_ID",    "")
DHAN_ACCESS_TOKEN = os.getenv("DHAN_ACCESS_TOKEN",  "")
TELEGRAM_BOT_TOKEN= os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID_1= os.getenv("TELEGRAM_CHAT_ID_1", "")
TELEGRAM_CHAT_ID_2= os.getenv("TELEGRAM_CHAT_ID_2", "")

# ── Capital & risk ──────────────────────────────────────────
CAPITAL              = 100000      # update as you scale
MAX_RISK_PCT         = 0.01        # 2% risk per trade (Rs.1,000 on Rs.50k)
MAX_CAPITAL_PER_TRADE= 0.25        # never put more than 25% in one trade
DAILY_LOSS_LIMIT     = 0.04        # circuit breaker: halt if -4% on the day

# ── Trade mode ──────────────────────────────────────────────
TRADE_MODE           = "cnc"       # CNC delivery (shares go to demat)
AUTO_EXIT_IF_DOWN    = True        # sell before close if position in loss
AUTO_EXIT_THRESHOLD  = -0.01       # exit if position is -1% at 2:45 PM
AUTO_EXIT_TIME       = "15:15"     # time to check for same-day exit
MAX_OPEN_TRADES      = 3          # max concurrent trades (1 now, 2 at month 3, 3 at month 5)
NO_NEW_TRADE_BEFORE  = "09:30"   # wait 15 min for market to settle
ALLOW_SHORTS         = False
# ── XGBoost signal thresholds ───────────────────────────────
BUY_THRESHOLD        = 0.65
SELL_THRESHOLD       = 0.38

# ── Stop-loss settings ──────────────────────────────────────
STOP_LOSS_PCT        = 0.025       # hard cap: SL never more than 2.5% away
ATR_MULTIPLIER_CNC   = 2.5         # wider stop for overnight holds
ATR_MULTIPLIER_INTRA = 1.5         # tighter stop for intraday

# ── Trailing stop ───────────────────────────────────────────
TRAIL_AFTER_PCT      = 0.01       # activate after +1.5% profit
TRAIL_DISTANCE       = 0.005       # trail 1% below running high

# ── position rotation ───────────────────────────────────────────
ROTATION_ENABLED     = True    # allow switching to better opportunity
ROTATION_MIN_PROFIT  = 0.005   # only rotate if current position up 0.5%+
ROTATION_MIN_EDGE    = 0.05    # new signal must be 5% more confident

# ===== ML =====
MODEL_THRESHOLD = 0.60
MIN_VOLUME_RATIO = 1.2
MIN_ATR_PCT = 0.003

# ── Nifty 50 watchlist ──────────────────────────────────────
# Auto-loaded from config/watchlist.json
# Generate this file by running:  python data/load_instruments.py
# That script downloads Security IDs live from Dhan's master CSV
# Never hardcode IDs here — always use load_instruments.py
NIFTY50_SECURITY_ID=13

import json as _json

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

# ── Candle settings ─────────────────────────────────────────
CANDLE_INTERVAL      = "5"
LOOKBACK_CANDLES     = 60

# ── Model paths ─────────────────────────────────────────────
MODEL_PATH           = "models/xgb_model.pkl"
SCALER_PATH          = "models/scaler.pkl"
BACKUP_MODEL_PATH    = "models/xgb_model_backup.pkl"
BACKUP_SCALER_PATH   = "models/scaler_backup.pkl"

# ── Market hours IST ────────────────────────────────────────
MARKET_OPEN          = "09:15"
MARKET_CLOSE         = "15:30"
NO_NEW_TRADE_AFTER   = "15:00"
INTRADAY_CUTOFF      = "15:22"  # was "15:10" — CNC can hold until 3:22 PM

# ── Logging ─────────────────────────────────────────────────
LOG_FILE             = "logs/bot.log"
TRADE_LOG            = "logs/trades.csv"
RETRAIN_LOG          = "logs/retrain.log"
