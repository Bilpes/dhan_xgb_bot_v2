# ============================================================
#  bot/trade_policy.py — Single source of truth for all
#  trade execution parameters (ATR, SL, TP, thresholds)
#
#  auto_retrain.py, signal_engine.py, live_bot.py ALL import
#  from here so train-time and live-time logic are IDENTICAL.
# ============================================================

# ── Label generation (auto_retrain.py uses these) ───────────
# bot/trade_policy.py — REPLACE the ATR label section with this:

# ─────────────────────────────────────────────────────────────
# Label design (train-time targets)
# ─────────────────────────────────────────────────────────────
# Goal:
# Train model to predict strong intraday momentum continuation.
# Labels are percentage-based for train/live consistency.

#TP_PCT   = 0.025    # 2.5% target
#SL_PCT   = 0.010    # 1.0% stop
HORIZON  = 12       # 24 x 5m candles ≈ 2 trading hours


# ─────────────────────────────────────────────────────────────
# LIVE trade execution settings
# ─────────────────────────────────────────────────────────────
# ATR adapts to volatility dynamically.
# Used only during live execution, NOT label generation.

ATR_SL_MULT = 1.5
ATR_TP_MULT = 3.0
ATR_PERIOD  = 14


# ─────────────────────────────────────────────────────────────
# Entry filters
# ─────────────────────────────────────────────────────────────

BUY_THRESHOLD_DEFAULT = 0.55
BUY_THRESHOLD_WEAK    = 0.60   # Weak Nifty days — slightly tighter
# Minimum reward:risk required
MIN_RR_RATIO = 1.2


# ─────────────────────────────────────────────────────────────
# Exit / deterioration logic
# ─────────────────────────────────────────────────────────────

# Hard probability exits
EXIT_LONG_THRESHOLD  = 0.48
EXIT_SHORT_THRESHOLD = 0.60

# Weakening regime
WEAK_THRESHOLD    = 0.52        
WEAK_CANDLES_MAX  = 5

# Momentum failure exit
# Good trades should work quickly.
# Exit if still negative after N completed candles.

MOMENTUM_EXIT_CANDLES = 14


# ─────────────────────────────────────────────────────────────
# Risk controls
# ─────────────────────────────────────────────────────────────

# Maximum open trades simultaneously
MAX_OPEN_POSITIONS = 5

# Daily circuit breaker
MAX_DAILY_LOSS_PCT = 0.03     # stop trading after -3%

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