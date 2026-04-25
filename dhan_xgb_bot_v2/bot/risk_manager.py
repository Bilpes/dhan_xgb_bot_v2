# ============================================================
#  bot/risk_manager.py  —  Position sizing, SL, trail logic
# ============================================================

import logging
from config.config import (
    CAPITAL, MAX_RISK_PCT, STOP_LOSS_PCT,
    TRAIL_AFTER_PCT, TRAIL_DISTANCE, DAILY_LOSS_LIMIT,
    MAX_CAPITAL_PER_TRADE,
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
        Shares to buy so max loss = MAX_RISK_PCT of capital.
        Also capped at MAX_CAPITAL_PER_TRADE of capital.

        Example: capital=60k, risk=2%, entry=812, SL=809
          risk/share = 3.0
          qty_risk   = 1200 / 3.0 = 400
          max_invest = 60000 * 0.40 = 24000
          qty_cap    = 24000 / 812 = 29
          final qty  = min(400, 29) = 29
        """
        risk_amount    = CAPITAL * MAX_RISK_PCT
        risk_per_share = entry - stop_loss

        if risk_per_share <= 0:
            log.warning("Invalid SL: entry=%.2f SL=%.2f", entry, stop_loss)
            return 0

        qty_risk = int(risk_amount / risk_per_share)
        qty_cap  = int(CAPITAL * MAX_CAPITAL_PER_TRADE / entry)
        qty      = min(qty_risk, qty_cap)

        log.info(
            "Position size: entry=%.2f SL=%.2f | risk/share=%.2f | "
            "qty=%d | invested=Rs.%.0f",
            entry, stop_loss, risk_per_share, qty, qty * entry
        )
        return max(qty, 0)

    # ── Stop-loss price ──────────────────────────────────────

    def calc_stop_loss(self, entry: float, atr: float = None,
                       trade_mode: str = "cnc") -> float:
        """
        ATR-based SL with mode multiplier:
          cnc/swing  -> 2.5x ATR  (wider — CNC needs room for intraday noise)
          intraday   -> 1.5x ATR  (tighter)
        Hard cap: never more than STOP_LOSS_PCT (2.5%) below entry.
        """
        # FIX: was 2.0 for CNC — increased to 2.5 to reduce noise-triggered SL hits
        if trade_mode == "intraday":
            multiplier = 1.5
        else:
            multiplier = 2.5   # cnc / swing

        if atr and atr > 0:
            sl = entry - (multiplier * atr)
        else:
            sl = entry * (1 - STOP_LOSS_PCT)

        # Hard cap — SL never more than STOP_LOSS_PCT away
        sl = max(sl, entry * (1 - STOP_LOSS_PCT))

        log.debug("SL: mode=%s | mult=%.1f | ATR=%.2f | SL=%.2f | gap=%.2f%%",
                  trade_mode, multiplier, atr or 0, sl,
                  (entry - sl) / entry * 100)
        return round(sl, 2)

    # ── Target price ─────────────────────────────────────────

    def calc_target(self, entry: float, stop_loss: float,
                    rr_ratio: float = 2.0) -> float:
        """
        Target = entry + (risk * RR ratio).
        Rounded DOWN to whole number for clean price levels.
        Bot does NOT auto-sell at target — trailing stop handles exits.
        Target is only used for Telegram display and bracket orders.
        """
        import math
        risk   = entry - stop_loss
        target = entry + (risk * rr_ratio)
        return float(math.floor(target))

    # ── Trailing stop ────────────────────────────────────────

    def should_trail(self, entry: float, current: float,
                     running_high: float) -> tuple:
        """
        Returns (should_update_sl: bool, new_sl: float).

        Activates after profit >= TRAIL_AFTER_PCT (1.0%).
        Trails TRAIL_DISTANCE (0.7%) below the running high.

        Key fix: returns False if new_sl <= entry (never trail below breakeven).
        This ensures trailing stop only moves UP and locks in profit.

        Example:
          entry=812, running_high=830 (+2.2%)
          new_sl = 830 * (1 - 0.007) = 824.19
          If current SL was 809.76 -> update to 824.19 (locking Rs.12 profit)
        """
        profit_pct = (current - entry) / entry

        # Not yet in profit zone — don't trail
        if profit_pct < TRAIL_AFTER_PCT:
            return False, 0.0

        new_sl = round(running_high * (1 - TRAIL_DISTANCE), 2)

        # Never trail below entry — always lock in at least breakeven
        if new_sl <= entry:
            return False, 0.0

        return True, new_sl

    # ── Daily P&L and circuit breaker ────────────────────────

    def update_pnl(self, pnl: float):
        self.daily_pnl += pnl
        loss_pct = self.daily_pnl / CAPITAL

        # Drawdown warning at -3%
        if -0.04 < loss_pct <= -0.03 and not hasattr(self, "_warned_3pct"):
            self._warned_3pct = True
            try:
                from bot.telegram_alert import _send
                _send(
                    f"DRAWDOWN WARNING\n"
                    f"Daily loss: Rs.{abs(self.daily_pnl):,.0f} "
                    f"({abs(loss_pct)*100:.1f}%)\n"
                    f"Approaching circuit breaker limit.\n"
                    f"No new trades after next SL hit."
                )
            except Exception:
                pass

        # Circuit breaker at DAILY_LOSS_LIMIT (6%)
        if loss_pct <= -DAILY_LOSS_LIMIT:
            self.circuit_open = True
            log.critical(
                "CIRCUIT BREAKER TRIGGERED — daily loss %.2f%% | "
                "Bot halted for today.", loss_pct * 100
            )

    def is_halted(self) -> bool:
        return self.circuit_open

    def reset_day(self):
        """Call at start of each trading day."""
        self.daily_pnl    = 0.0
        self.trade_count  = 0
        self.circuit_open = False
        if hasattr(self, "_warned_3pct"):
            del self._warned_3pct
        log.info("Risk manager reset for new trading day.")