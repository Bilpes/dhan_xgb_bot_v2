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
#                       update_pnl(), is_halted(), reset_day()
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

log = logging.getLogger("risk")


class RiskManager:
    def __init__(self):
        self.daily_pnl    = 0.0
        self.trade_count  = 0
        self.circuit_open = False

    # ── Position sizing ──────────────────────────────────────
    def position_size(self, entry: float, stop_loss: float) -> int:
        """
        Returns quantity such that:
          1) max loss per trade <= MAX_RISK_PCT * CAPITAL
          2) capital deployed   <= MAX_CAPITAL_PER_TRADE * CAPITAL

        SL comes from SignalEngine (ATR-based) — risk_manager just sizes.
        """
        if entry <= 0 or stop_loss <= 0:
            log.warning("Invalid pricing: entry=%.2f SL=%.2f", entry, stop_loss)
            return 0

        risk_per_share = entry - stop_loss
        if risk_per_share <= 0:
            log.warning("SL >= entry for LONG trade: entry=%.2f SL=%.2f", entry, stop_loss)
            return 0

        risk_amount = CAPITAL * MAX_RISK_PCT
        max_capital = CAPITAL * MAX_CAPITAL_PER_TRADE

        qty_risk = int(risk_amount  / risk_per_share)
        qty_cap  = int(max_capital  / entry)
        qty      = min(qty_risk, qty_cap)

        if qty <= 0:
            log.warning(
                "Position size = 0: risk_amount=%.0f risk/share=%.2f max_cap=%.0f",
                risk_amount, risk_per_share, max_capital,
            )
            return 0

        log.info(
            "Size: entry=%.2f SL=%.2f | risk/share=%.2f | "
            "qty_risk=%d qty_cap=%d → qty=%d (₹%.0f invested)",
            entry, stop_loss, risk_per_share,
            qty_risk, qty_cap, qty, qty * entry,
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
        Never trails below breakeven.
        """
        if entry <= 0 or current <= 0 or running_high <= 0:
            return False, 0.0

        profit_pct = (current - entry) / entry
        if profit_pct < TRAIL_AFTER_PCT:
            return False, 0.0

        new_sl = round(running_high * (1 - TRAIL_DISTANCE), 2)

        if new_sl <= entry:
            return False, 0.0

        return True, new_sl

    # ── Daily P&L + circuit breaker ─────────────────────────
    def update_pnl(self, pnl: float):
        """Call after every exit with realised P&L (positive = profit)."""
        self.daily_pnl   += pnl
        self.trade_count += 1
        loss_pct = self.daily_pnl / CAPITAL

        # Early warning at -3%
        if -0.04 < loss_pct <= -0.03 and not getattr(self, "_warned_3pct", False):
            self._warned_3pct = True
            try:
                from bot.telegram_alert import _send
                _send(
                    f"⚠️ <b>DRAWDOWN WARNING</b>\n"
                    f"Daily loss : ₹{abs(self.daily_pnl):,.0f} "
                    f"({abs(loss_pct) * 100:.1f}%)\n"
                    f"Approaching circuit breaker limit ({DAILY_LOSS_LIMIT*100:.0f}%)."
                )
            except Exception:
                pass

        # Circuit breaker
        if loss_pct <= -DAILY_LOSS_LIMIT:
            self.circuit_open = True
            log.critical(
                "CIRCUIT BREAKER — daily loss %.2f%% ≥ limit %.2f%% | Bot halted.",
                abs(loss_pct) * 100, DAILY_LOSS_LIMIT * 100,
            )

    def is_halted(self) -> bool:
        return self.circuit_open

    def reset_day(self):
        """Call at EOD reset to clear all daily state."""
        self.daily_pnl    = 0.0
        self.trade_count  = 0
        self.circuit_open = False
        self._warned_3pct = False
        log.info("RiskManager: daily state reset.")

    # ── Compatibility shims (kept for older callers) ─────────
    def calc_stop_loss(
        self, entry: float, atr: float = None, trade_mode: str = "cnc"
    ) -> float:
        """
        Deprecated shim — SL should come from SignalEngine.
        Kept to avoid breaking any legacy code paths.
        """
        if entry <= 0:
            return 0.0
        if atr and atr > 0:
            mult = 1.5 if trade_mode == "intraday" else 2.5
            return round(max(entry - mult * atr, 0.01), 2)
        return round(max(entry * 0.975, 0.01), 2)

    def calc_target(
        self, entry: float, stop_loss: float, rr_ratio: float = 2.0
    ) -> float:
        """
        Deprecated shim — target should come from SignalEngine.
        """
        if entry <= 0 or stop_loss <= 0 or stop_loss >= entry:
            return 0.0
        return round(entry + (entry - stop_loss) * rr_ratio, 2)
