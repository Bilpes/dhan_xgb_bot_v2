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

Patches
-------
ISSUE-13: set_watchlist_manager() added — bot.py __init__ was calling it
          and getting AttributeError before first scan.
ISSUE-14: force_exit() added as public alias — bot.py EOD block calls
          tm.force_exit(sym, ltp, reason); was named exit_trade() internally.
ISSUE-15: daily_loss_breached property added — bot.py scan() guards on
          `if self.tm.daily_loss_breached` but only _daily_cb_tripped existed.
ISSUE-16: check_exits(symbol, candle_dict) added — bot.py passes the last
          candle as a dict; check_sl_tp() only accepted a scalar price.
          New method extracts low/high from the candle for realistic SL/TP.
ISSUE-17: reset_daily() public method added — bot.run() calls it on startup.
ISSUE-18: enter() public alias added for enter_trade() — bot.py calls
          tm.enter(sym, sig) where sig is the dict from signal_engine.
ISSUE-19: Position.entry_price property alias — bot.py EOD fallback accesses
          pos.entry_price; field is named `entry`.
ISSUE-20: _get_sector() now uses wm.get_sector() correctly after WM patch.

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

    # ISSUE-19 FIX: bot.py EOD fallback reads pos.entry_price (not pos.entry)
    @property
    def entry_price(self) -> float:
        """Alias for self.entry — bot.py uses this name."""
        return self.entry

    @property
    def unrealised_pnl(self) -> float:
        """Requires set_last_price() to have been called first."""
        return (getattr(self, '_last_price', self.entry) - self.entry) * self.qty

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
    # ISSUE-13 FIX: bot.py __init__ calls tm.set_watchlist_manager(wm)
    # ------------------------------------------------------------------
    def set_watchlist_manager(self, wm) -> None:
        """
        Wire in a WatchlistManager so that every trade exit
        automatically calls wm.record_trade_result(symbol, pnl).
        Called from bot.py after both tm and wm are initialised.
        """
        self._wm = wm
        log.info("[TradeManager] WatchlistManager linked.")

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
    ) -> tuple:
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
            log.warning("[TradeManager] SL >= entry — skipping qty computation.")
            return 0

        risk_amount = cfg.RISK_PER_TRADE * self.capital
        qty = math.floor(risk_amount / sl_pts)

        # Hard cap
        max_qty = math.floor(self.capital * 0.20 / entry)
        qty = min(qty, max_qty)
        return max(qty, 1)  # minimum 1 share

    # ------------------------------------------------------------------
    # ISSUE-18 FIX: bot.py calls tm.enter(sym, sig) where sig is a dict
    # ------------------------------------------------------------------
    def enter(self, symbol: str, sig: dict) -> bool:
        """
        Convenience wrapper called by bot.py scan().
        sig is the dict returned by SignalEngine.get_signal():
          {'action':'BUY', 'entry':..., 'sl':..., 'target':...,
           'prob':..., 'atr':..., 'rr_ratio':...}
        Delegates to enter_trade() with unpacked fields.
        """
        return self.enter_trade(
            symbol = symbol,
            entry  = sig.get("entry",  0.0),
            atr    = sig.get("atr",    0.0),
            prob   = sig.get("prob",   0.0),
            rr     = sig.get("rr_ratio", None),
        )

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

        # ISSUE-13 FIX: notify WatchlistManager of trade result
        if self._wm is not None:
            try:
                self._wm.record_trade_result(symbol, pnl)
            except Exception as e:
                log.debug("[TradeManager] wm.record_trade_result error: %s", e)

        self._log_trade(
            action=f"EXIT:{reason}", symbol=symbol, qty=pos.qty,
            price=price, sl=pos.current_sl, tp=pos.tp,
            rr=0.0, prob=0.0, pnl=pnl,
        )
        return pnl

    # ------------------------------------------------------------------
    # ISSUE-14 FIX: bot.py EOD block calls tm.force_exit(sym, ltp, reason)
    # ------------------------------------------------------------------
    def force_exit(self, symbol: str, price: float, reason: str = "FORCE") -> float:
        """
        Unconditional exit — used by bot.py EOD sweep and emergency stop.
        Delegates to exit_trade(); the `reason` string is preserved in the log.
        Returns realised P&L (0.0 if position not found).
        """
        return self.exit_trade(symbol, price, reason)

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

            # Activate trailing SL once gain exceeds ACTIVATE_MULT x ATR
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
                        "[TradeManager] Trailing SL %s: %.2f -> %.2f (peak=%.2f).",
                        symbol, pos.current_sl, new_sl, pos.peak,
                    )
                    pos.current_sl = new_sl
                    return new_sl

        return None

    # ------------------------------------------------------------------
    # ISSUE-16 FIX: bot.py calls tm.check_exits(sym, candle_dict)
    # check_sl_tp() only accepted a scalar price; candles have high/low
    # ------------------------------------------------------------------
    def check_exits(self, symbol: str, candle: dict) -> Optional[str]:
        """
        Check if SL or TP is triggered using the candle's high AND low.
        This is more realistic than checking close price only:
          - SL uses candle low  (worst-case fill within the bar)
          - TP uses candle high (best-case fill within the bar)

        Returns 'SL', 'TP', or None.  Calls exit_trade() when triggered.
        Accepts a candle dict with keys: open, high, low, close.
        Also accepts a scalar float (last_price) for backward compatibility.
        """
        with self._lock:
            pos = self.positions.get(symbol)
            if pos is None:
                return None

        # Accept both dict (from bot.py) and scalar (legacy calls)
        if isinstance(candle, dict):
            low  = float(candle.get("low",   candle.get("close", 0)))
            high = float(candle.get("high",  candle.get("close", 0)))
        else:
            low = high = float(candle)

        # SL check (uses low — worst price seen in the bar)
        if low <= pos.current_sl:
            self.exit_trade(symbol, pos.current_sl, reason="SL")
            return "SL"

        # TP check (uses high — target may have been hit during the bar)
        if high >= pos.tp:
            self.exit_trade(symbol, pos.tp, reason="TP")
            return "TP"

        return None

    # Legacy scalar interface — kept for any direct callers
    def check_sl_tp(self, symbol: str, last_price: float) -> Optional[str]:
        """Backward-compatible scalar version — delegates to check_exits."""
        return self.check_exits(symbol, last_price)

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
    def open_symbols(self) -> list:
        with self._lock:
            return list(self.positions.keys())

    @property
    def daily_pnl(self) -> float:
        return self._realised_pnl

    # ------------------------------------------------------------------
    # ISSUE-15 FIX: bot.py guards on `if self.tm.daily_loss_breached`
    # ------------------------------------------------------------------
    @property
    def daily_loss_breached(self) -> bool:
        """
        True if the daily-loss circuit breaker has been tripped.
        bot.py checks this before every scan to avoid new entries
        after the daily P&L limit is hit.
        """
        self._reset_daily_if_needed()
        return self._daily_cb_tripped

    # ------------------------------------------------------------------
    # ISSUE-17 FIX: bot.run() calls tm.reset_daily() on startup
    # ------------------------------------------------------------------
    def reset_daily(self) -> None:
        """
        Explicit daily reset — called by bot.run() at startup so that
        any stale in-process state from a previous session is cleared.
        Also called automatically by _reset_daily_if_needed() on date change.
        """
        self._today            = date.today()
        self._realised_pnl     = 0.0
        self._daily_cb_tripped = False
        log.info("[TradeManager] Daily state reset for %s.", self._today)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_sector(self, symbol: str) -> str:
        # ISSUE-20 FIX: wm.get_sector() is now defined on WatchlistManager.
        # Fallback chain: wm.get_sector() -> SECTOR_MAP import -> 'UNKNOWN'
        if self._wm is not None:
            try:
                return self._wm.get_sector(symbol)
            except Exception:
                pass
        try:
            from watchlist import SECTOR_MAP
            return SECTOR_MAP.get(symbol, "UNKNOWN")
        except Exception:
            return "UNKNOWN"

    def _reset_daily_if_needed(self) -> None:
        today = date.today()
        if today != self._today:
            self.reset_daily()

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
