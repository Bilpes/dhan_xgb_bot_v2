"""trade_manager.py — full execution layer for dhan_xgb_bot_v2/v3

Responsibilities
----------------
* ATR-aligned SL/TP computation (must mirror label construction in train.py)
* Kelly-fraction position sizing with hard caps
* Daily-loss circuit breaker
* Sector-spread enforcement
* Trailing-SL update (called every candle by scheduler)
* Paper-trade mode (logs orders without sending to Dhan)
* Trade log (CSV append, thread-safe)

All thresholds imported from config.py — no magic numbers here.
"""

from __future__ import annotations

import csv
import logging
import math
import os
import threading
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, Optional

import config as cfg

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Position:
    symbol:     str
    sector:     str
    entry:      float
    qty:        int
    sl:         float          # initial SL
    tp:         float
    atr:        float
    peak:       float = 0.0    # highest observed price since entry
    trailing_active: bool = False
    current_sl: float = 0.0   # may trail upward over time
    opened_at:  datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        if self.current_sl == 0.0:
            self.current_sl = self.sl
        if self.peak == 0.0:
            self.peak = self.entry

    @property
    def unrealised_pnl(self) -> float:
        """Requires last_price to be set externally before calling."""
        return (self._last_price - self.entry) * self.qty

    def set_last_price(self, price: float) -> None:
        self._last_price = price
        if price > self.peak:
            self.peak = price


# ---------------------------------------------------------------------------
# TradeManager
# ---------------------------------------------------------------------------

class TradeManager:
    """Central execution and position-management layer."""

    def __init__(self, dhan_client=None, watchlist_manager=None):
        self._dhan      = dhan_client       # None in paper mode
        self._wm        = watchlist_manager # WatchlistManager instance
        self._lock      = threading.Lock()

        self.capital: float = cfg.TOTAL_CAPITAL
        self.positions: Dict[str, Position] = {}

        # Daily P&L tracking
        self._today: date         = date.today()
        self._realised_pnl: float = 0.0
        self._daily_cb_tripped    = False

        # Ensure log directory exists
        os.makedirs(os.path.dirname(cfg.TRADE_LOG_PATH), exist_ok=True)
        self._log_path = cfg.TRADE_LOG_PATH
        self._log_lock = threading.Lock()
        self._ensure_header()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def can_trade(self) -> bool:
        """Returns False if daily-loss CB is tripped or position limit reached."""
        self._reset_daily_if_needed()
        if self._daily_cb_tripped:
            log.warning("[TradeManager] Daily-loss circuit breaker ACTIVE — no new entries.")
            return False
        with self._lock:
            if len(self.positions) >= cfg.MAX_OPEN_POSITIONS:
                log.debug("[TradeManager] MAX_OPEN_POSITIONS=%d reached.", cfg.MAX_OPEN_POSITIONS)
                return False
        return True

    def can_enter_sector(self, sector: str) -> bool:
        """Returns False if sector already has MAX_PER_SECTOR open positions."""
        with self._lock:
            count = sum(1 for p in self.positions.values() if p.sector == sector)
        if count >= cfg.MAX_PER_SECTOR:
            log.debug("[TradeManager] Sector '%s' at limit %d.", sector, cfg.MAX_PER_SECTOR)
            return False
        return True

    def compute_sl_tp(
        self, entry: float, atr: float
    ) -> tuple[float, float, float]:
        """Returns (sl, tp, rr).

        Uses cfg.ATR_SL_MULT and cfg.ATR_TP_MULT which MUST match the label
        construction multipliers in train.py (ATR_LABEL_SL_MULT / TP_MULT).
        Applies MAX_SL_PCT / MIN_SL_PCT caps.
        """
        raw_sl = entry - cfg.ATR_SL_MULT * atr
        raw_tp = entry + cfg.ATR_TP_MULT * atr

        # Percentage caps
        sl_by_pct_floor = entry * (1 - cfg.MAX_SL_PCT)   # max risk per share
        sl_by_pct_ceil  = entry * (1 - cfg.MIN_SL_PCT)   # min meaningful SL

        sl = max(raw_sl, sl_by_pct_floor)   # don't blow out on high-ATR
        sl = min(sl, sl_by_pct_ceil)         # don't make SL trivially tight

        sl_pts = entry - sl
        tp     = entry + sl_pts * (cfg.ATR_TP_MULT / cfg.ATR_SL_MULT)
        rr     = (tp - entry) / max(entry - sl, 1e-6)

        return round(sl, 2), round(tp, 2), round(rr, 4)

    def compute_qty(self, entry: float, sl: float) -> int:
        """Kelly-fraction position sizing.

        qty = floor( RISK_PER_TRADE * capital / sl_pts )
        Hard cap: no single leg > 20% of capital.
        """
        sl_pts = entry - sl
        if sl_pts <= 0:
            log.warning("[TradeManager] SL >= entry for %s — skipping.", entry)
            return 0

        risk_amount = cfg.RISK_PER_TRADE * self.capital
        qty = math.floor(risk_amount / sl_pts)

        # Hard cap
        max_qty = math.floor(self.capital * 0.20 / entry)
        qty = min(qty, max_qty)
        return max(qty, 1)  # minimum 1 share

    def enter_trade(
        self,
        symbol:    str,
        entry:     float,
        atr:       float,
        prob:      float,
        rr:        Optional[float] = None,
    ) -> bool:
        """Attempt to open a position.

        Returns True if order was placed (or paper-logged), False if blocked.
        """
        if not self.can_trade():
            return False

        sector = self._get_sector(symbol)
        if not self.can_enter_sector(sector):
            return False

        with self._lock:
            if symbol in self.positions:
                log.debug("[TradeManager] Already in %s.", symbol)
                return False

        sl, tp, computed_rr = self.compute_sl_tp(entry, atr)
        actual_rr = rr if rr is not None else computed_rr

        if actual_rr < cfg.MIN_RR_RATIO:
            log.debug(
                "[TradeManager] %s RR=%.2f < MIN_RR_RATIO=%.2f — skipped.",
                symbol, actual_rr, cfg.MIN_RR_RATIO,
            )
            return False

        qty = self.compute_qty(entry, sl)
        if qty <= 0:
            return False

        pos = Position(
            symbol=symbol,
            sector=sector,
            entry=entry,
            qty=qty,
            sl=sl,
            tp=tp,
            atr=atr,
        )

        if cfg.PAPER_TRADE:
            log.info(
                "[PAPER] BUY %s qty=%d entry=%.2f sl=%.2f tp=%.2f rr=%.2f prob=%.3f",
                symbol, qty, entry, sl, tp, actual_rr, prob,
            )
        else:
            success = self._place_order(symbol, qty, entry)
            if not success:
                return False

        with self._lock:
            self.positions[symbol] = pos

        self._log_trade(
            action="ENTER", symbol=symbol, qty=qty, price=entry,
            sl=sl, tp=tp, rr=actual_rr, prob=prob, pnl=0.0,
        )
        return True

    def exit_trade(
        self,
        symbol: str,
        price:  float,
        reason: str = "SIGNAL",
    ) -> float:
        """Close a position. Returns realised P&L."""
        with self._lock:
            pos = self.positions.pop(symbol, None)

        if pos is None:
            log.warning("[TradeManager] exit_trade called for unknown symbol %s.", symbol)
            return 0.0

        pnl = (price - pos.entry) * pos.qty

        if cfg.PAPER_TRADE:
            log.info(
                "[PAPER] SELL %s qty=%d price=%.2f pnl=%.2f reason=%s",
                symbol, pos.qty, price, pnl, reason,
            )
        else:
            self._place_order(symbol, pos.qty, price, side="SELL")

        self._realised_pnl += pnl
        self._check_daily_cb()

        self._log_trade(
            action=f"EXIT:{reason}", symbol=symbol, qty=pos.qty,
            price=price, sl=pos.current_sl, tp=pos.tp,
            rr=0.0, prob=0.0, pnl=pnl,
        )
        return pnl

    def update_trailing_sl(self, symbol: str, last_price: float) -> Optional[float]:
        """Update trailing SL for a position given latest price.

        Called every 5-min candle by the scheduler.
        Returns new SL if updated, None otherwise.
        """
        with self._lock:
            pos = self.positions.get(symbol)
            if pos is None:
                return None

            pos.set_last_price(last_price)

            gain_atr = (last_price - pos.entry) / pos.atr if pos.atr > 0 else 0.0

            # Activate trailing SL once gain exceeds ACTIVATE_MULT × ATR
            if not pos.trailing_active:
                if gain_atr >= cfg.TRAILING_SL_ACTIVATE_MULT:
                    pos.trailing_active = True
                    log.info(
                        "[TradeManager] Trailing SL ACTIVATED for %s at %.2f (gain=%.2f ATR).",
                        symbol, last_price, gain_atr,
                    )

            if pos.trailing_active:
                new_sl = pos.peak - cfg.TRAILING_SL_TRAIL_MULT * pos.atr
                new_sl = round(new_sl, 2)
                if new_sl > pos.current_sl:
                    log.info(
                        "[TradeManager] Trailing SL %s: %.2f → %.2f (peak=%.2f).",
                        symbol, pos.current_sl, new_sl, pos.peak,
                    )
                    pos.current_sl = new_sl
                    return new_sl

        return None

    def check_sl_tp(
        self, symbol: str, last_price: float
    ) -> Optional[str]:
        """Check if SL or TP is triggered. Returns 'SL', 'TP', or None."""
        with self._lock:
            pos = self.positions.get(symbol)
            if pos is None:
                return None

        if last_price <= pos.current_sl:
            self.exit_trade(symbol, last_price, reason="SL")
            return "SL"
        if last_price >= pos.tp:
            self.exit_trade(symbol, last_price, reason="TP")
            return "TP"
        return None

    def exit_all(self, prices: Dict[str, float], reason: str = "EOD") -> float:
        """Exit all open positions. Returns total P&L."""
        total = 0.0
        for symbol in list(self.positions.keys()):
            price = prices.get(symbol, 0.0)
            if price <= 0:
                log.warning("[TradeManager] No price for %s on EOD exit.", symbol)
                continue
            total += self.exit_trade(symbol, price, reason=reason)
        return total

    @property
    def open_symbols(self) -> list[str]:
        with self._lock:
            return list(self.positions.keys())

    @property
    def daily_pnl(self) -> float:
        return self._realised_pnl

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_sector(self, symbol: str) -> str:
        if self._wm is not None:
            return self._wm.get_sector(symbol)
        return "UNKNOWN"

    def _reset_daily_if_needed(self) -> None:
        today = date.today()
        if today != self._today:
            self._today           = today
            self._realised_pnl    = 0.0
            self._daily_cb_tripped = False
            log.info("[TradeManager] Daily P&L reset for %s.", today)

    def _check_daily_cb(self) -> None:
        """Trip the daily circuit breaker if loss exceeds threshold."""
        threshold = cfg.MAX_DAILY_LOSS * self.capital  # e.g. -0.02 * 100000 = -2000
        if self._realised_pnl <= threshold:
            self._daily_cb_tripped = True
            log.warning(
                "[TradeManager] Daily-loss circuit breaker TRIPPED: P&L=%.2f threshold=%.2f.",
                self._realised_pnl, threshold,
            )

    def _place_order(self, symbol: str, qty: int, price: float, side: str = "BUY") -> bool:
        """Send order to Dhan. Returns True on success."""
        if self._dhan is None:
            log.error("[TradeManager] Dhan client not initialised — cannot place live order.")
            return False
        try:
            resp = self._dhan.place_order(
                security_id=symbol,
                exchange_segment="NSE_EQ",
                transaction_type=side,
                quantity=qty,
                order_type="MARKET",
                product_type="INTRADAY",
                price=0,
            )
            log.info("[TradeManager] Order response: %s", resp)
            return True
        except Exception as exc:
            log.error("[TradeManager] Order failed for %s: %s", symbol, exc)
            return False

    # ------------------------------------------------------------------
    # Trade log
    # ------------------------------------------------------------------

    _COLS = [
        "ts", "action", "symbol", "qty", "price",
        "sl", "tp", "rr", "prob", "pnl",
    ]

    def _ensure_header(self) -> None:
        with self._log_lock:
            if not os.path.exists(self._log_path):
                with open(self._log_path, "w", newline="") as f:
                    csv.DictWriter(f, fieldnames=self._COLS).writeheader()

    def _log_trade(self, **kwargs) -> None:
        row = {"ts": datetime.now().isoformat(timespec="seconds")}
        row.update(kwargs)
        with self._log_lock:
            with open(self._log_path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=self._COLS, extrasaction="ignore")
                w.writerow(row)
