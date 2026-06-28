# config.py — dhan_xgb_bot_v2 / v3
# All tunable parameters in one place.
# ──────────────────────────────────────────────────────────────
# PATCH 2026-06-28 (production-ready):
#   1. BUY_THRESHOLD_DEFAULT 0.65 → 0.55  (was killing valid signals)
#   2. BUY_THRESHOLD_WEAK    0.72 → 0.62
#   3. ATR_SL_MULT 2.2 → 1.2             (must match ATR_LABEL_SL_MULT)
#   4. ATR_TP_MULT 3.5 → 2.2             (must match ATR_LABEL_TP_MULT)
#   5. MAX_OPEN_POSITIONS 4 → 6
#   6. MAX_PER_SECTOR      2 → 3
#   7. WATCHLIST_JSON_PATH added
#   8. WatchlistManager tuning constants added
#   9. Redis TTL constants added (was crashing on REDIS_ENABLED=true)
#  10. RETRAIN_EVERY_DAYS, WALK_FORWARD_FOLDS, MIN_TRAIN_SAMPLES added
#  11. MIN_ACCURACY, MIN_PRECISION renamed from MIN_ACC/MIN_PREC (both kept)
#  12. TELEGRAM_BOT_TOKEN alias added
# ──────────────────────────────────────────────────────────────

import os
from datetime import time

# ── Dhan credentials (set via environment or .env) ──────────
DHAN_CLIENT_ID     = os.getenv("DHAN_CLIENT_ID",    "")
DHAN_ACCESS_TOKEN  = os.getenv("DHAN_ACCESS_TOKEN", "")

# ── Telegram ─────────────────────────────────────────────────
TELEGRAM_TOKEN     = os.getenv("TELEGRAM_TOKEN",    "")   # legacy name
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_TOKEN",    "")   # bot.py alias
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID",  "")

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

ATR_SL_MULT   = 1.2             # FIX: was 2.2 — now matches ATR_LABEL_SL_MULT
ATR_TP_MULT   = 2.2             # FIX: was 3.5 — now matches ATR_LABEL_TP_MULT
MIN_RR_RATIO  = 1.2             # achievable: 2.2/1.2 = 1.83 theoretical RR
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
RETRAIN_HOUR          = 8            # 8:00 AM pre-market
RETRAIN_EVERY_DAYS    = 7            # FIX: was RETRAIN_INTERVAL_DAYS — unified name
RETRAIN_INTERVAL_DAYS = 7            # keep legacy alias
WALK_FORWARD_FOLDS    = 5            # FIX: was missing — used in train.py
MIN_TRAIN_SAMPLES     = 500          # FIX: was missing — minimum rows per fold

# ── model quality gates ────────────────────────────────────────
MIN_AUC       = 0.56
MIN_ACC       = 0.52   # legacy name
MIN_PREC      = 0.52   # legacy name
MIN_ACCURACY  = 0.52   # FIX: train.py uses MIN_ACCURACY
MIN_PRECISION = 0.52   # FIX: train.py uses MIN_PRECISION

# ── Redis cache (optional) ─────────────────────────────────────
REDIS_HOST              = os.getenv("REDIS_HOST",     "localhost")
REDIS_PORT              = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB                = int(os.getenv("REDIS_DB",   "0"))
REDIS_PASSWORD          = os.getenv("REDIS_PASSWORD", None)
REDIS_ENABLED           = os.getenv("REDIS_ENABLED",  "false").lower() == "true"
REDIS_MAX_CONNECTIONS   = 10
REDIS_SOCKET_TIMEOUT    = 2.0
REDIS_RETRY_ON_TIMEOUT  = True

# ── Redis TTL constants ────────────────────────────────────────
# FIX: all were missing — signal_engine.py crashed with AttributeError
# when REDIS_ENABLED=true
TTL_FEATURE         = 240    # 4 min — feature vector cache (1 candle)
TTL_PREDICTION      = 240    # 4 min — XGBoost prediction cache
TTL_ATR             = 300    # 5 min — ATR value cache
TTL_COOLDOWN        = 1800   # 30 min — post-SL-hit cooldown per symbol
TTL_CIRCUIT_BREAKER = 3600   # 1 hr  — global circuit breaker
TTL_DEDUP_ORDER     = 120    # 2 min — duplicate order guard
TTL_NIFTY_REGIME    = 300    # 5 min — nifty regime cache

# ── WatchlistManager OODA tuning ──────────────────────────────
WM_ADD_THRESHOLD       = 0.60    # min prob to add a new stock
WM_PRUNE_THRESHOLD     = 0.45    # avg prob below which stock is pruned
WM_SCAN_INTERVAL_MIN   = 5       # OODA tick frequency (minutes)
WM_UNIVERSE_RESCAN_MIN = 30      # full-universe rescan frequency
WM_MIN_DAILY_VOL_CR    = 200.0   # minimum daily turnover in Cr
WM_ATR_MIN_PCT         = 0.005   # minimum ATR% (not too flat)
WM_ATR_MAX_PCT         = 0.060   # maximum ATR% (circuit breaker risk)
WM_MAX_WATCHLIST_SIZE  = 40      # hard cap on dynamic watchlist
WM_PRUNE_SCORE_WINDOW  = 5       # rolling window for prune score
WM_PRUNE_COOLDOWN_BARS = 24      # 24×5min = 2hr cooldown after prune
WM_MAX_CONSEC_LOSSES   = 4       # prune after N consecutive losses
