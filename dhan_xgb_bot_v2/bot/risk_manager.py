# ============================================================
#  bot/risk_manager.py  —  Position sizing, SL, trail logic
# ============================================================

import logging
from config.config import (
    CAPITAL, MAX_RISK_PCT, STOP_LOSS_PCT,
    TRAIL_AFTER_PCT, TRAIL_DISTANCE, DAILY_LOSS_LIMIT
)

log = logging.getLogger("risk")


class RiskManager:

    def __init__(self):
        self.daily_pnl      = 0.0          # running P&L for today
        self.trade_count    = 0
        self.circuit_open   = False         # True = bot halted for day

    # ── Position sizing ──────────────────────────────────────

    def position_size(self, entry: float, stop_loss: float) -> int:
        """
        How many shares to buy so that max loss = MAX_RISK_PCT of capital.
        E.g. capital=50k, risk=3%, entry=1000, SL=975
          → risk per share = 25
          → qty = (50000 * 0.03) / 25 = 60 shares
        """
        risk_amount   = CAPITAL * MAX_RISK_PCT      # ₹1,500 on ₹50k
        risk_per_share = entry - stop_loss

        if risk_per_share <= 0:
            log.warning("Invalid SL: entry=%.2f  SL=%.2f", entry, stop_loss)
            return 0

        qty = int(risk_amount / risk_per_share)
        max_qty = int(CAPITAL * 0.95 / entry)       # never use >95% capital
        qty = min(qty, max_qty)

        log.info(
            "Position size: entry=%.2f SL=%.2f | risk/share=%.2f | qty=%d",
            entry, stop_loss, risk_per_share, qty
        )
        return max(qty, 0)

    # ── Stop-loss price ──────────────────────────────────────

    def calc_stop_loss(self, entry: float, atr: float = None,
                       trade_mode: str = "intraday") -> float:
        """
        ATR-based SL with mode-aware multiplier:
          intraday → 1.5× ATR  (tighter, squared off by 3:10)
          swing    → 2.0× ATR  (wider, holds overnight gaps)
        Hard cap: never more than STOP_LOSS_PCT below entry.
        """
        multiplier = 1.5 if trade_mode == "intraday" else 2.0

        if atr and atr > 0:
            sl = entry - (multiplier * atr)
        else:
            sl = entry * (1 - STOP_LOSS_PCT)        # flat % fallback

        # Hard cap: SL never more than STOP_LOSS_PCT away
        sl = max(sl, entry * (1 - STOP_LOSS_PCT))
        log.debug("SL calc | mode=%s | multiplier=%.1f | ATR=%.2f | SL=%.2f",
                  trade_mode, multiplier, atr or 0, sl)
        return round(sl, 2)

    # ── Target price ─────────────────────────────────────────

    def calc_target(self, entry: float, stop_loss: float,
                    rr_ratio: float = 2.0) -> float:
        """
        Default: 2× risk-reward (if risk = 2.5%, target = 5%).
        No hard cap — model decides when to exit via signal flip.
        This is just the bracket order target for auto-fill.
        """
        import math
        risk    = entry - stop_loss
        target  = entry + (risk * rr_ratio)
        return float(math.floor(target)) # round DOWN to whole number

    # ── Trailing stop ────────────────────────────────────────

    def should_trail(self, entry: float, current: float,
                     running_high: float) -> tuple[bool, float]:
        """
        Returns (should_update_sl, new_sl_price).
        Activates only after profit >= TRAIL_AFTER_PCT.
        """
        profit_pct = (current - entry) / entry

        if profit_pct < TRAIL_AFTER_PCT:
            return False, 0.0                       # not yet in profit zone

        new_sl = running_high * (1 - TRAIL_DISTANCE)
        return True, round(new_sl, 2)

    # ── Daily circuit breaker ────────────────────────────────

    def update_pnl(self, pnl: float):
        self.daily_pnl += pnl
        loss_pct = self.daily_pnl / CAPITAL

        # Drawdown warning at 3%
        if -0.04 < loss_pct <= -0.03 and not hasattr(self, "_warned_3pct"):
            self._warned_3pct = True
            from bot.telegram_alert import _send
            _send(
                f"⚠️ <b>DRAWDOWN WARNING</b>\n"
                f"Daily loss has reached <b>₹{abs(self.daily_pnl):,.0f}</b> "
                f"({abs(loss_pct)*100:.1f}% of capital).\n"
                f"Approaching circuit breaker limit."
            )

        if loss_pct <= -DAILY_LOSS_LIMIT:
            self.circuit_open = True
            log.critical(
                "CIRCUIT BREAKER TRIGGERED — daily loss %.2f%% | "
                "Bot halted for today.", loss_pct * 100
            )

    def is_halted(self) -> bool:
        return self.circuit_open

    def reset_day(self):
        """Call at market open each day."""
        self.daily_pnl   = 0.0
        self.trade_count = 0
        self.circuit_open= False
        log.info("Risk manager reset for new trading day.")
