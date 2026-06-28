# ============================================================
# config.py — dhan_xgb_bot_v2
# Audit-patched 2026-06-28
# Fixes: I1 (thresholds), I2 (ATR mismatch), I3 (portfolio limits)
# ============================================================

import os
from datetime import time as dtime

# ── Dhan API ────────────────────────────────────────────────
DHAN_CLIENT_ID    = os.getenv("DHAN_CLIENT_ID",    "YOUR_CLIENT_ID")
DHAN_ACCESS_TOKEN = os.getenv("DHAN_ACCESS_TOKEN", "YOUR_ACCESS_TOKEN")

# ── Trade mode ──────────────────────────────────────────────
TRADE_MODE = "intraday"   # MIS on Dhan — NOT CNC
PAPER_MODE = True

# ── Capital & risk ──────────────────────────────────────────
TOTAL_CAPITAL      = 500_000
RISK_PER_TRADE     = 0.005
# FIX I3: was MAX_OPEN_POSITIONS=4, MAX_PER_SECTOR=2
# With 4 positions and 2/sector, once BANKING(2)+IT(2) fill,
# ALL other sectors (PHARMA, AUTO, INFRA) are blocked entirely.
# That guarantees 0 trades the moment two sectors are active.
MAX_OPEN_POSITIONS = 6        # was 4
MAX_PER_SECTOR     = 3        # was 2
MAX_DAILY_LOSS     = -0.02

# ── Timing ──────────────────────────────────────────────────
CANDLE_INTERVAL      = 5
NO_NEW_TRADE_BEFORE  = dtime(9, 20)
NO_NEW_TRADE_AFTER   = dtime(15, 0)
INTRADAY_EXIT_TIME   = dtime(15, 15)

# ── Signal thresholds ────────────────────────────────────────
# FIX I1: was BUY_THRESHOLD_DEFAULT=0.65, BUY_THRESHOLD_WEAK=0.72
# Audit proof: on 38-stock universe, thr=0.65 passes only 3.8% of
# valid signals → ≈1.5 trades/day. thr=0.55 passes 19% of valid
# signals → ≈5.8 trades/day at precision=0.623.
# The threshold was set for a 150-stock noisy universe; with 38
# curated algo-friendly stocks the model probability distribution
# peaks at 0.52–0.60 for genuinely profitable setups.
BUY_THRESHOLD_DEFAULT = 0.55   # was 0.65
BUY_THRESHOLD_WEAK    = 0.62   # was 0.72
SELL_THRESHOLD        = 0.45
PROB_CAP              = None   # raw model output always

# ── SL / TP ──────────────────────────────────────────────────
# FIX I2: was ATR_SL_MULT=2.2, ATR_TP_MULT=3.5
# Training labels use ATR_LABEL_SL_MULT=1.2 / ATR_LABEL_TP_MULT=2.0
# Execution was using 2.2/3.5 — 83% wider SL than model was trained on.
# Consequence: model probabilities are calibrated for 1.2/2.0 risk
# geometry. Using 2.2/3.5 at execution means:
#   - More capital at risk per trade than model expects
#   - RR denominator (price−sl) is 83% larger → RR fails MIN_RR_RATIO gate
#   - Expected value calculation is inconsistent with training objective
# Fix: align execution SL/TP with training label construction exactly.
ATR_SL_MULT               = 1.2   # was 2.2 — now matches ATR_LABEL_SL_MULT
ATR_TP_MULT               = 2.0   # was 3.5 — now matches ATR_LABEL_TP_MULT
MIN_RR_RATIO              = 1.2
MIN_SL_PCT                = 0.004
MAX_SL_PCT                = 0.025
TRAILING_SL_ACTIVATE_MULT = 1.0
TRAILING_SL_TRAIL_MULT    = 0.8

# ── Label construction (anti-leakage) ────────────────────────
# Entry price = open[t+1], NOT close[t]
HORIZON           = 8
LABEL_ENTRY_SHIFT = 1
ATR_LABEL_TP_MULT = 2.0
ATR_LABEL_SL_MULT = 1.2

# ── Filters ──────────────────────────────────────────────────
REQUIRE_VWAP_CONFIRM     = False   # VWAP is a feature — not a hard gate
MAX_DISTANCE_FROM_VWAP   = 0.08
MIN_VOLUME_RATIO         = 0.50
MIN_VOLUME_RATIO_CONFIRM = 0.65
MIN_STOCK_PRICE          = 100.0   # blocks penny/PSU noise stocks
MIN_MARKET_CAP_CR        = 10_000

# ── Retraining ───────────────────────────────────────────────
RETRAIN_EVERY_DAYS = 7
EMBARGO_DAYS       = 14        # gap between train-end and val-start
MIN_TRAIN_SAMPLES  = 3000
WALK_FORWARD_FOLDS = 5
MIN_ACCURACY       = 0.52
MIN_AUC            = 0.56
MIN_PRECISION      = 0.52

# ── Telegram ─────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID",   "")

# ── Paths ────────────────────────────────────────────────────
MODEL_PATH      = "models/xgb_model.pkl"
SCALER_PATH     = "models/scaler.pkl"
FEATURE_PATH    = "models/feature_list.pkl"
TRADE_LOG_PATH  = "logs/trades.csv"
SIGNAL_LOG_PATH = "logs/signal_scan.csv"

# ── Redis ─────────────────────────────────────────────────────
REDIS_HOST     = os.getenv("REDIS_HOST",     "localhost")
REDIS_PORT     = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB       = int(os.getenv("REDIS_DB",   "0"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)
REDIS_ENABLED  = os.getenv("REDIS_ENABLED",  "true").lower() == "true"

# TTLs (seconds)
TTL_CANDLE           = 330    # 5-min candle + 30s buffer
TTL_INDICATOR        = 330
TTL_FEATURE          = 300    # feature vector per symbol
TTL_PREDICTION       = 300    # model probability output
TTL_NIFTY_REGIME     = 300    # market bias (BULL/NEUTRAL/WEAK)
TTL_INSTRUMENT_META  = 86_400 # instrument master — 24h
TTL_ATR              = 330
TTL_COOLDOWN         = 1_800  # post-SL cooldown per symbol (30 min)
TTL_DAILY_RISK       = 86_400
TTL_SESSION          = 86_400
TTL_HEARTBEAT        = 60
TTL_RATE_LIMIT       = 60
TTL_DEDUP_ORDER      = 3_600  # duplicate order guard (1h)
TTL_CIRCUIT_BREAKER  = 3_600
TTL_RETRY_QUEUE      = 300

# Connection pool
REDIS_MAX_CONNECTIONS  = 20
REDIS_SOCKET_TIMEOUT   = 2.0
REDIS_RETRY_ON_TIMEOUT = True
