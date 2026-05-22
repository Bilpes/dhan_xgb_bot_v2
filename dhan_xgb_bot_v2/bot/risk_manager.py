# ============================================================
# bot/risk_manager.py — Position sizing, trailing SL,
#                        daily P&L, circuit breaker
#
# Aligned with:
#   config/config.py  — CAPITAL, MAX_RISK_PCT, MAX_CAPITAL_PER_TRADE,
#                        DAILY_LOSS_LIMIT, TRAIL_AFTER_PCT, TRAIL_DISTANCE
#   signal_engine.py  — SL and target are owned by SignalEngine;
#                       risk_manager only does sizing + trailing + halts
#   live_bot.py       — calls position_size(), should_trail(),
#                       update_pnl(), is_halted(), reset_daily()
#
# Sync log (2026-05-14):
#   SYNC-1: reset_day() renamed to reset_daily() to match live_bot._eod_reset() call
#   SYNC-2: Consecutive-loss circuit breaker added (MAX_CONSECUTIVE_LOSSES from trade_policy)
#   SYNC-3: Warning threshold made dynamic: fires at 75% of DAILY_LOSS_LIMIT
#   SYNC-4: should_trail() respects TRAIL_AFTER_PCT and TRAIL_DISTANCE from config
#   SYNC-5: Deprecated shims (calc_stop_loss, calc_target) removed — owned by SignalEngine
#   SYNC-6: daily_loss_pct property exposed for live_bot.NEW_TRADE_LOSS_PAUSE gate
# ============================================================

import logging

from config.config import (
    CAPITAL,
    MAX_RISK_PCT,
    MAX_CAPITAL_PER_TRADE,
    DAILY_LOSS_LIMIT,
    TRAIL_AFTER_PCT,
    TRAIL_DISTANCE,
)
from bot.trade_policy import (
    MAX_CONSECUTIVE_LOSSES,
    MAX_ATR_RISK_MULTIPLIER,
    MIN_POSITION_SCALE,
)

log = logging.getLogger("risk")

# Warning fires at this fraction of the daily loss limit.
# e.g. 0.75 × 4% = 3% — matches the old hardcoded "-0.03" check.
_WARNING_FRACTION = 0.75



class RiskManager:
    def __init__(self):
        self.daily_pnl         = 0.0
        self.trade_count       = 0
        self.circuit_open      = False
        self._consecutive_loss = 0      # SYNC-2
        self._warned_drawdown  = False  # SYNC-3: unified warning flag
        self.win_count  = 0
        self.loss_count = 0

    # ── Daily loss % — exposed for live_bot NEW_TRADE_LOSS_PAUSE gate ──
    @property
    def daily_loss_pct(self) -> float:
        """SYNC-6: Returns signed daily P&L as fraction of CAPITAL.
        Negative means loss. live_bot reads this directly instead of
        computing `self.risk.daily_pnl / CAPITAL` inline.
        """
        return self.daily_pnl / CAPITAL

    # ── Position sizing ──────────────────────────────────────
    def position_size(self, entry: float, stop_loss: float, atr_pct: float = 0.0,) -> int:
        """
        Returns quantity such that:
          1) max loss per trade <= MAX_RISK_PCT * CAPITAL
          2) capital deployed   <= MAX_CAPITAL_PER_TRADE * CAPITAL

        SL comes from SignalEngine (ATR-based) — risk_manager just sizes.
        """
        if entry <= 0 or stop_loss <= 0:
            log.warning(
                "Invalid pricing: entry=%.2f SL=%.2f",
                entry,
                stop_loss,
            )
            return 0

        if atr_pct < 0:
            log.warning(
                "Invalid atr_pct: %.4f",
                atr_pct,
            )
            atr_pct = 0.0

        risk_per_share = entry - stop_loss
        if risk_per_share <= 0:
            log.warning(
                "SL >= entry for LONG trade: entry=%.2f SL=%.2f", entry, stop_loss
            )
            return 0

        risk_amount = CAPITAL * MAX_RISK_PCT
        # ── Volatility-adjusted sizing ──────────────────
        volatility_scale = 1.0

        if atr_pct > 0:

            volatility_scale = min(
                1.0,
                MAX_ATR_RISK_MULTIPLIER / max(atr_pct, 1e-9)
            )

            volatility_scale = max(
                MIN_POSITION_SCALE,
                volatility_scale
            )
        risk_amount *= volatility_scale

        max_capital = CAPITAL * MAX_CAPITAL_PER_TRADE

        qty_risk = int(risk_amount / risk_per_share)
        qty_cap  = int(max_capital / entry)
        qty      = min(qty_risk, qty_cap)

        if qty <= 0:
            log.warning(
                "Position size = 0: risk_amount=%.0f risk/share=%.2f max_cap=%.0f",
                risk_amount, risk_per_share, max_capital,
            )
            return 0

        log.info(
            "Size: entry=%.2f SL=%.2f | risk/share=%.2f | "
            "qty_risk=%d qty_cap=%d scale=%.2f → qty=%d (₹%.0f invested)",
            entry, stop_loss, risk_per_share,
            qty_risk, qty_cap,volatility_scale, qty, qty * entry,
        )
        return qty

    # ── Trailing stop ────────────────────────────────────────
    def should_trail(
        self, entry: float, current: float, running_high: float
    ) -> tuple[bool, float]:
        """
        Returns (should_update: bool, new_sl: float).

        Activates trailing only after profit >= TRAIL_AFTER_PCT.
        Trails at TRAIL_DISTANCE below the running high.
        Never trails to a level at or below entry (breakeven floor).

        Config alignment:
          TRAIL_AFTER_PCT = 0.01  → activate after +1% profit
          TRAIL_DISTANCE  = 0.005 → trail 0.5% below running high

        Note: these values mean a trailed exit at +1% running high
        yields a new SL at +0.5%. Consider widening TRAIL_AFTER_PCT
        to 0.015 and TRAIL_DISTANCE to 0.01 if winners are being
        cut too early on normal post-breakout consolidation.
        """
        if entry <= 0 or current <= 0 or running_high <= 0:
            return False, 0.0

        profit_pct = (current - entry) / entry
        if profit_pct < TRAIL_AFTER_PCT:
            return False, 0.0

        # Adaptive trailing:
        # stronger trends get more room

        adaptive_trail = TRAIL_DISTANCE

        if profit_pct > 0.02:
            adaptive_trail *= 1.5

        if profit_pct > 0.03:
            adaptive_trail *= 2.0

        new_sl = round(
            running_high * (1 - adaptive_trail),
            2,
        )

        # Never trail below breakeven — preserves capital on parabolic
        # moves that snap back through entry before trailing fires.
        if new_sl <= entry:
            return False, 0.0

        return True, new_sl

    # ── Daily P&L + circuit breakers ─────────────────────────
    def update_pnl(self, pnl: float):
        """
        Call after every exit with realised P&L (positive = profit).

        Two independent circuit breakers:
          1. Daily loss limit  (DAILY_LOSS_LIMIT from config)
          2. Consecutive SL streak (MAX_CONSECUTIVE_LOSSES from trade_policy)

        SYNC-2: Consecutive loss tracker added.
        SYNC-3: Early warning fires dynamically at 75% of DAILY_LOSS_LIMIT,
                replacing the old hardcoded -3% threshold.
        """
        self.daily_pnl   += pnl
        self.trade_count += 1

        # ── Consecutive loss tracking (SYNC-2) ──────────────
        if pnl < 0:
            self._consecutive_loss += 1
            self.loss_count += 1
        else:
            self._consecutive_loss = 0
            self.win_count += 1

        loss_pct = self.daily_pnl / CAPITAL

        # ── Early warning at 75% of limit (SYNC-3) ──────────
        warn_threshold = -DAILY_LOSS_LIMIT * _WARNING_FRACTION  # e.g. -0.03
        if loss_pct <= warn_threshold and not self._warned_drawdown:
            self._warned_drawdown = True
            try:
                from bot.telegram_alert import _send
                _send(
                    f"⚠️ <b>DRAWDOWN WARNING</b>\n"
                    f"Daily loss : ₹{abs(self.daily_pnl):,.0f} "
                    f"({abs(loss_pct) * 100:.1f}%)\n"
                    f"Approaching circuit breaker limit "
                    f"({DAILY_LOSS_LIMIT * 100:.0f}%)."
                )
            except Exception:
                pass
            
        # ── Regime degradation detection ───────────────────
        total_trades = (
            self.win_count
            + self.loss_count
        )

        if total_trades >= 10:

            win_rate = (
                self.win_count / total_trades
            )

            if (
                win_rate < 0.30
                and self.loss_count >= 7
            ):
                self.circuit_open = True

                log.critical(
                    "CIRCUIT BREAKER [REGIME FAILURE] "
                    "WinRate=%.2f%% over %d trades",
                    win_rate * 100,
                    total_trades,
                )    

        # ── Daily loss circuit breaker ───────────────────────
        if loss_pct <= -DAILY_LOSS_LIMIT and not self.circuit_open:
            self.circuit_open = True
            log.critical(
                "CIRCUIT BREAKER [DAILY LOSS] — %.2f%% ≥ limit %.2f%% | Bot halted.",
                abs(loss_pct) * 100, DAILY_LOSS_LIMIT * 100,
            )

        # ── Consecutive SL circuit breaker (SYNC-2) ─────────
        if (
            self._consecutive_loss >= MAX_CONSECUTIVE_LOSSES
            and not self.circuit_open
        ):
            self.circuit_open = True
            log.critical(
                "CIRCUIT BREAKER [CONSECUTIVE SL] — %d losses in a row "
                "(limit=%d) | Bot halted.",
                self._consecutive_loss, MAX_CONSECUTIVE_LOSSES,
            )
            try:
                from bot.telegram_alert import _send
                _send(
                    f"🛑 <b>CIRCUIT BREAKER — Consecutive SL</b>\n"
                    f"{self._consecutive_loss} stop-losses in a row "
                    f"(limit: {MAX_CONSECUTIVE_LOSSES}).\n"
                    f"Bot halted for the day."
                )
            except Exception:
                pass
            
        # ── End-of-day summary ─────────────────────────
    def daily_summary(self) -> dict:

        total_trades = (
            self.win_count
            + self.loss_count
        )

        return {
            "pnl": self.daily_pnl,
            "trades": [],  # live_bot can later pass closed trades if needed
            "capital": CAPITAL + self.daily_pnl,
            "total_trades": total_trades,
            "wins": self.win_count,
            "losses": self.loss_count,
        }

    def is_halted(self) -> bool:
        return self.circuit_open

    def reset_daily(self):
        """
        SYNC-1: Renamed from reset_day() to reset_daily() to match
        live_bot._eod_reset() which calls self.risk.reset_daily().
        Call at EOD to clear all daily state.
        """
        self.daily_pnl         = 0.0
        self.trade_count       = 0
        self.circuit_open      = False
        self._consecutive_loss = 0
        self._warned_drawdown  = False
        self.win_count  = 0
        self.loss_count = 0
        log.info("RiskManager: daily state reset.")

    # ── Back-compat alias — remove after confirming no other callers ──
    def reset_day(self):
        """Deprecated alias for reset_daily(). Will be removed."""
        log.warning(
            "RiskManager.reset_day() is deprecated — use reset_daily(). "
            "Update the caller."
        )
        self.reset_daily()
