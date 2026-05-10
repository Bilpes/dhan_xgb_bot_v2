# ============================================================
#  bot/trade_policy.py — Single source of truth for all
#  trade execution parameters (ATR, SL, TP, thresholds)
#
#  auto_retrain.py, signal_engine.py, live_bot.py ALL import
#  from here so train-time and live-time logic are IDENTICAL.
# ============================================================

# ── Label generation (auto_retrain.py uses these) ───────────
# bot/trade_policy.py — REPLACE the ATR label section with this:

# ── Label design (percentage-based — train/live consistent) ──
TP_PCT   = 0.025    # 2.5% TP  — HDFCBANK ₹780 → must reach ₹799.50
SL_PCT   = 0.010    # 1.0% SL  — HDFCBANK ₹780 → stop at ₹772.20
HORIZON  = 24       # max 24 candles (2 hours) to hit TP — realistic window

# ── ATR still used for LIVE SL/TP (not labels) ───────────────
ATR_SL_MULT = 1.5   # live trade SL (SignalEngine uses this)
ATR_TP_MULT = 3.0   # live trade TP (SignalEngine uses this)
ATR_PERIOD  = 14

# ── Signal engine thresholds ────────────────────────────────
BUY_THRESHOLD_DEFAULT  = 0.65   # minimum prob_up to fire BUY
MIN_RR_RATIO           = 1.8    # R:R gate

# ── Exit / deterioration thresholds ─────────────────────────
EXIT_LONG_THRESHOLD    = 0.42   # hard exit if prob_up drops below this
EXIT_SHORT_THRESHOLD   = 0.58   # hard exit for short
WEAK_THRESHOLD         = 0.52   # "weakening" zone — starts candle counter
WEAK_CANDLES_MAX       = 3      # exit after N consecutive weak candles

# ── Blocked symbols ──────────────────────────────────────────
BLOCKED_SYMBOLS = {
    "ADANIENT",
    "ADANIPORTS",
}