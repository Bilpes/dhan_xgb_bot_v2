# config.py — dhan_xgb_bot_v2 / v3
# All tunable parameters in one place.
# ──────────────────────────────────────────────────────────────
# AUDIT FIXES applied (2026-06-28):
#   1. BUY_THRESHOLD_DEFAULT 0.65 → 0.55  (was killing valid signals)
#   2. BUY_THRESHOLD_WEAK    0.72 → 0.62
#   3. ATR_SL_MULT 2.2 → 1.2             (must match ATR_LABEL_SL_MULT)
#   4. ATR_TP_MULT 3.5 → 2.2             (must match ATR_LABEL_TP_MULT)
#   5. MAX_OPEN_POSITIONS 4 → 6
#   6. MAX_PER_SECTOR      2 → 3
#   7. WATCHLIST_JSON_PATH added
#   8. WatchlistManager tuning constants added
# ──────────────────────────────────────────────────────────────

import os
from datetime import time

# ── Dhan credentials (set via environment or .env) ──────────
DHAN_CLIENT_ID     = os.getenv("DHAN_CLIENT_ID",    "")
DHAN_ACCESS_TOKEN  = os.getenv("DHAN_ACCESS_TOKEN", "")

# ── Telegram ─────────────────────────────────────────────────
TELEGRAM_TOKEN    = os.getenv("TELEGRAM_TOKEN",    "")
TELEGRAM_CHAT_ID  = os.getenv("TELEGRAM_CHAT_ID",  "")

# ── paths ─────────────────────────────────────────────────────
BASE_DIR          = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH        = os.path.join(BASE_DIR, "models", "xgb_model.pkl")
SCALER_PATH       = os.path.join(BASE_DIR, "models", "scaler.pkl")
FEATURE_PATH      = os.path.join(BASE_DIR, "models", "features.pkl")
SIGNAL_LOG_PATH   = os.path.join(BASE_DIR, "logs",   "signals.csv")
TRADE_LOG_PATH    = os.path.join(BASE_DIR, "logs",   "trades.csv")
WATCHLIST_JSON_PATH = os.path.join(BASE_DIR, "watchlist.json")

# ── market hours ──────────────────────────────────────────────
NO_NEW_TRADE_BEFORE = time(9, 20)    # catch opening momentum
NO_NEW_TRADE_AFTER  = time(15, 10)   # hard close before 15:15
AVOID_LUNCH_HOURS   = False          # lunch noise handled by model

# ── signal thresholds ─────────────────────────────────────────
# CRITICAL: must align with label construction in train.py
# ATR_LABEL_SL_MULT = 1.2, ATR_LABEL_TP_MULT = 2.0
# Execution SL/TP must match — otherwise RR gate fires on valid signals.
BUY_THRESHOLD_DEFAULT = 0.55    # FIX: was 0.65 — too aggressive
BUY_THRESHOLD_WEAK    = 0.62    # FIX: was 0.72

ATR_SL_MULT   = 1.2             # FIX: was 2.2 — now matches label SL
ATR_TP_MULT   = 2.2             # FIX: was 3.5 — now matches label TP
MIN_RR_RATIO  = 1.2             # achievable with 2.2/1.2 = 1.83 theoretical RR
MAX_SL_PCT    = 0.035           # cap SL at 3.5% below entry
MIN_SL_PCT    = 0.004           # floor SL at 0.4% below entry

# ── label construction (train.py must match these) ────────────
ATR_LABEL_TP_MULT    = 2.0
ATR_LABEL_SL_MULT    = 1.2
LABEL_ENTRY_SHIFT    = 1        # entry = open[t+1], not close[t]
HORIZON              = 8        # 40 min forward window
EMBARGO_DAYS         = 14       # temporal train/val separation

# ── position sizing ────────────────────────────────────────────
MAX_OPEN_POSITIONS   = 6        # FIX: was 4
MAX_PER_SECTOR       = 3        # FIX: was 2
RISK_PCT_PER_TRADE   = 0.01     # 1% of capital per trade
MIN_STOCK_PRICE      = 50.0

# ── volume gates ───────────────────────────────────────────────
MIN_VOLUME_RATIO         = 0.50  # relaxed from 0.60 for early session
MIN_VOLUME_RATIO_CONFIRM = 0.65

# ── trade mode ─────────────────────────────────────────────────
TRADE_MODE   = "intraday"   # MIS — no overnight carry
PAPER_TRADE  = True         # set False for live

# ── auto-exit ──────────────────────────────────────────────────
AUTO_EXIT_TIME        = time(15, 10)
TRAILING_SL_TRIGGER   = 0.007   # activate trailing SL after 0.7% gain
TRAILING_SL_DISTANCE  = 0.004   # trail 0.4% below peak

# ── retrain schedule ───────────────────────────────────────────
RETRAIN_HOUR          = 8       # 8:00 AM pre-market
RETRAIN_INTERVAL_DAYS = 7       # weekly retrain
MIN_ACC   = 0.52
MIN_AUC   = 0.56
MIN_PREC  = 0.52

# ── Redis cache (optional) ─────────────────────────────────────
REDIS_HOST    = os.getenv("REDIS_HOST",    "localhost")
REDIS_PORT    = int(os.getenv("REDIS_PORT", "6379"))
REDIS_ENABLED = os.getenv("REDIS_ENABLED", "false").lower() == "true"

# ── WatchlistManager OODA tuning ──────────────────────────────
WM_ADD_THRESHOLD      = 0.60    # min prob to add a new stock
WM_PRUNE_THRESHOLD    = 0.45    # avg prob below which stock is pruned
WM_SCAN_INTERVAL_MIN  = 5       # OODA tick frequency
WM_UNIVERSE_RESCAN_MIN= 30      # full-universe rescan frequency
WM_MIN_DAILY_VOL_CR   = 200.0   # minimum daily turnover in Cr
WM_ATR_MIN_PCT        = 0.005   # minimum ATR% (not too flat)
WM_ATR_MAX_PCT        = 0.060   # maximum ATR% (circuit breaker risk)
WM_MAX_WATCHLIST_SIZE = 40      # hard cap on dynamic watchlist
WM_PRUNE_SCORE_WINDOW = 5       # rolling window for prune avg
WM_PRUNE_COOLDOWN_BARS= 24      # 24×5min = 2hr cooldown after prune
WM_MAX_CONSEC_LOSSES  = 4       # prune after N consecutive losses
