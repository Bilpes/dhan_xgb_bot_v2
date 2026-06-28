# ============================================================
#  bot/trade_policy.py — Single source of truth for all
#  trade execution parameters (ATR, SL, TP, thresholds)
#
#  auto_retrain.py, signal_engine.py, live_bot.py ALL import
#  from here so train-time and live-time logic are IDENTICAL.
#
#  config/config.py owns ONLY: infra, paths, API keys, Redis,
#  Telegram, timing, filters (VWAP/volume/ATR floor).
#  ALL numeric trading parameters live here — never in config.
# ============================================================


# ─────────────────────────────────────────────────────────────
# Label design (train-time targets)  — used by auto_retrain.py
# ─────────────────────────────────────────────────────────────
# Goal: predict strong intraday momentum continuation.
# Entry price = open[t+1], NOT close[t]  (anti-leakage).

HORIZON           = 12    # 12 x 5 min = 60 min forward window
LABEL_ENTRY_SHIFT = 1     # entry at next-candle open — prevents look-ahead bias
ATR_LABEL_TP_MULT = 2.0   # TP multiplier used during label generation
ATR_LABEL_SL_MULT = 1.2   # SL multiplier used during label generation


# ─────────────────────────────────────────────────────────────
# LIVE trade execution settings
# ─────────────────────────────────────────────────────────────
# ATR adapts to volatility dynamically.
# Intentionally different from label multipliers — train label
# defines what to predict; live ATR defines how to execute it.

ATR_SL_MULT = 1.5
ATR_TP_MULT = 3.0
ATR_PERIOD  = 14


# ─────────────────────────────────────────────────────────────
# Entry filters
# ─────────────────────────────────────────────────────────────

BUY_THRESHOLD_DEFAULT = 0.55   # F1-optimal for AUC=0.803 balanced dataset
BUY_THRESHOLD_WEAK    = 0.60   # Weak Nifty days — slightly tighter

# Minimum reward:risk required
MIN_RR_RATIO = 1.2


# ─────────────────────────────────────────────────────────────
# Exit / deterioration logic
# ─────────────────────────────────────────────────────────────

# Hard probability exits
# NOTE: EXIT_LONG_THRESHOLD must be meaningfully below BUY_THRESHOLD_DEFAULT
# (gap >= 0.06) to avoid instant exit on normal prob fluctuation after entry.
EXIT_LONG_THRESHOLD  = 0.48
EXIT_SHORT_THRESHOLD = 0.60

# Soft / weakening regime exit
# WEAK_THRESHOLD must be below BUY_THRESHOLD_DEFAULT or soft-exit fires on entry.
WEAK_THRESHOLD    = 0.52
WEAK_CANDLES_MAX  = 5

# Momentum failure exit
# Exit if position still negative after N completed candles (70 min).
MOMENTUM_EXIT_CANDLES = 14


# ─────────────────────────────────────────────────────────────
# Risk controls
# ─────────────────────────────────────────────────────────────

# Maximum open trades simultaneously
MAX_OPEN_POSITIONS = 5

# Daily circuit breaker — halt if cumulative loss exceeds this fraction
MAX_DAILY_LOSS_PCT = 0.02     # stop trading after -2% on the day

# Consecutive SL protection
MAX_CONSECUTIVE_LOSSES = 5


# ─────────────────────────────────────────────────────────────
# Volatility-adjusted position sizing
# ─────────────────────────────────────────────────────────────

# Reduce size in highly volatile stocks
MAX_ATR_RISK_MULTIPLIER = 0.015

# Never reduce below 35% normal size
MIN_POSITION_SCALE = 0.35


# ─────────────────────────────────────────────────────────────
# Blocked symbols
# ─────────────────────────────────────────────────────────────

BLOCKED_SYMBOLS = {
    "ADANIENT",
    "ADANIPORTS",
}
