# trade_manager.py — dhan_xgb_bot_v2
# =============================================================
# OODA wiring added 2026-06-28:
#   set_watchlist_manager(wm)  — inject WatchlistManager ref
#   _exit_position() calls wm.record_trade_result(symbol, pnl)
#   on every exit: SL_HIT, TARGET_HIT, and EOD/force_exit
#
# Previous fixes retained:
#   Fix I7: CSV header checked at write time via _needs_header()
#   Fix I8: compute_qty capped at 25% of TOTAL_CAPITAL
# =============================================================

import logging
import os
import csv
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Dict

import config as cfg

log = logging.getLogger("trade_manager")


@dataclass
class Position:
    symbol:             str
    entry_price:        float
    qty:                int
    sl:                 float
    target:             float
    atr:                float
    prob:               float
    entry_time:         datetime
    mode:               str  = "INTRADAY"
    trailing_sl_active: bool = False
    peak_price:         float = 0.0
    trailing_sl:        float = 0.0


class TradeManager:
    def __init__(self, dhan_client=None):
        self.dhan      = dhan_client
        self.positions: Dict[str, Position] = {}
        self.daily_pnl = 0.0
        self._wm       = None   # injected via set_watchlist_manager()
        os.makedirs(os.path.dirname(cfg.TRADE_LOG_PATH), exist_ok=True)
        # Fix I7: header checked at write time, not here

    # ── OODA: WatchlistManager injection ─────────────────────
    def set_watchlist_manager(self, wm):
        """
        Inject WatchlistManager after construction.

        Setter pattern avoids circular imports:
          TradeManager is instantiated BEFORE WatchlistManager
          in DhanXGBBot.__init__, so passing wm as a constructor
          argument would require importing watchlist_manager at
          module level inside trade_manager.py, creating a cycle:
            watchlist_manager → trade_manager → watchlist_manager
          The setter keeps both modules fully independent.
        """
        self._wm = wm
        log.info("WatchlistManager wired into TradeManager")

    def reset_daily(self):
        """Call at session start to clear daily P&L counter."""
        self.daily_pnl = 0.0

    # ── Risk checks ───────────────────────────────────────────
    @property
    def daily_loss_breached(self) -> bool:
        return (self.daily_pnl / cfg.TOTAL_CAPITAL) <= cfg.MAX_DAILY_LOSS

    @property
    def open_count(self) -> int:
        return len(self.positions)

    def sector_count(self, sector: str) -> int:
        from watchlist import SECTOR_MAP
        return sum(
            1 for sym in self.positions
            if SECTOR_MAP.get(sym) == sector
        )

    def can_open(self, symbol: str) -> bool:
        if self.open_count >= cfg.MAX_OPEN_POSITIONS:
            return False
        from watchlist import SECTOR_MAP
        sec = SECTOR_MAP.get(symbol, "OTHER")
        if self.sector_count(sec) >= cfg.MAX_PER_SECTOR:
            return False
        return True

    # ── Sizing ────────────────────────────────────────────────
    def compute_qty(self, entry: float, sl: float) -> int:
        """
        Risk-based position sizing.
        Fix I8: capped at 25% of TOTAL_CAPITAL per position.
        """
        risk_per_share     = max(entry - sl, entry * cfg.MIN_SL_PCT)
        raw_qty            = int(cfg.TOTAL_CAPITAL * cfg.RISK_PER_TRADE / risk_per_share)
        max_position_value = cfg.TOTAL_CAPITAL * 0.25
        max_qty_by_capital = int(max_position_value / max(entry, 1))
        qty = min(raw_qty, max_qty_by_capital)
        return max(qty, 1)

    # ── Entry ─────────────────────────────────────────────────
    def enter(self, symbol: str, signal: dict) -> Optional[Position]:
        if self.daily_loss_breached:
            log.warning("Daily loss breached — blocking new entry")
            return None
        if not self.can_open(symbol):
            log.info(f"Position/sector limit reached — skip {symbol}")
            return None

        entry, sl = signal["entry"], signal["sl"]
        qty = self.compute_qty(entry, sl)

        pos = Position(
            symbol      = symbol,
            entry_price = entry,
            qty         = qty,
            sl          = sl,
            target      = signal["target"],
            atr         = signal["atr"],
            prob        = signal["prob"],
            entry_time  = datetime.now(),
            peak_price  = entry,
            trailing_sl = sl,
        )

        if cfg.PAPER_TRADE:
            log.info(
                f"[PAPER] BUY {symbol} qty={qty} entry={entry:.2f} "
                f"sl={sl:.2f} tp={signal['target']:.2f} prob={signal['prob']:.3f}"
            )
        else:
            try:
                self.dhan.place_order(
                    security_id      = symbol,
                    exchange_segment = "NSE_EQ",
                    transaction_type = "BUY",
                    quantity         = qty,
                    order_type       = "MARKET",
                    product_type     = "INTRADAY",
                    price            = 0,
                )
            except Exception as e:
                log.error(f"Order failed {symbol}: {e}")
                return None

        self.positions[symbol] = pos
        self._log(symbol, "ENTRY", qty, entry, sl, signal["target"], signal["prob"])
        return pos

    # ── Trailing SL ───────────────────────────────────────────
    def update_trailing_sl(self, symbol: str, price: float):
        pos = self.positions.get(symbol)
        if not pos:
            return
        pos.peak_price = max(pos.peak_price, price)
        if pos.peak_price >= pos.entry_price + cfg.TRAILING_SL_ACTIVATE_MULT * pos.atr:
            pos.trailing_sl_active = True
        if pos.trailing_sl_active:
            pos.trailing_sl = max(
                pos.trailing_sl,
                pos.peak_price - cfg.TRAILING_SL_TRAIL_MULT * pos.atr,
            )

    # ── Exit check on each candle ─────────────────────────────
    def check_exits(self, symbol: str, candle: dict) -> Optional[str]:
        pos = self.positions.get(symbol)
        if not pos:
            return None
        active_sl = pos.trailing_sl if pos.trailing_sl_active else pos.sl
        if candle["low"] <= active_sl:
            self._exit_position(symbol, active_sl, "SL_HIT")
            return "SL_HIT"
        if candle["high"] >= pos.target:
            self._exit_position(symbol, pos.target, "TARGET_HIT")
            return "TARGET_HIT"
        return None

    # ── Force exit (EOD / manual) ─────────────────────────────
    def force_exit(self, symbol: str, ltp: float, reason: str = "EOD"):
        self._exit_position(symbol, ltp, reason)

    # ── Internal exit — OODA feedback injected here ───────────
    def _exit_position(self, symbol: str, exit_price: float, reason: str):
        pos = self.positions.pop(symbol, None)
        if not pos:
            return
        pnl = (exit_price - pos.entry_price) * pos.qty
        self.daily_pnl += pnl

        if not cfg.PAPER_TRADE:
            try:
                self.dhan.place_order(
                    security_id      = symbol,
                    exchange_segment = "NSE_EQ",
                    transaction_type = "SELL",
                    quantity         = pos.qty,
                    order_type       = "MARKET",
                    product_type     = "INTRADAY",
                    price            = 0,
                )
            except Exception as e:
                log.error(f"Exit order failed {symbol}: {e}")

        log.info(
            f"EXIT {symbol} reason={reason} exit={exit_price:.2f} "
            f"pnl=₹{pnl:.2f} daily=₹{self.daily_pnl:.2f}"
        )
        self._log(
            symbol, "EXIT", pos.qty, exit_price,
            pos.sl, pos.target, pos.prob, pnl, reason,
        )

        # ── OODA feedback loop ────────────────────────────────
        # Record P&L with WatchlistManager so the consecutive-loss
        # prune gate has real trade outcomes to act on.
        # Fires for ALL exit paths: SL_HIT, TARGET_HIT, EOD.
        # try/except: a WM error must never crash a trade exit.
        # _wm is None until set_watchlist_manager() is called,
        # so this is safe even when WM is not wired in.
        if self._wm is not None:
            try:
                self._wm.record_trade_result(symbol=symbol, pnl=pnl)
            except Exception as e:
                log.debug("wm.record_trade_result error: %s", e)

    # ── CSV header helper ─────────────────────────────────────
    @staticmethod
    def _needs_header(path: str) -> bool:
        """
        Fix I7: check file size at write time, not at init.
        Returns True if file does not exist or is empty.
        """
        try:
            return not os.path.exists(path) or os.path.getsize(path) == 0
        except OSError:
            return True

    # ── CSV trade log ─────────────────────────────────────────
    def _log(
        self,
        symbol, action, qty, price, sl, target, prob,
        pnl=None, reason=None,
    ):
        row = {
            "time":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "mode":        "PAPER" if cfg.PAPER_TRADE else "LIVE",
            "symbol":      symbol,
            "action":      action,
            "qty":         qty,
            "price":       price,
            "sl":          sl,
            "target":      target,
            "prob":        round(prob, 4),
            "pnl":         round(pnl, 2) if pnl is not None else "",
            "exit_reason": reason or "",
            "daily_pnl":   round(self.daily_pnl, 2),
        }
        with open(cfg.TRADE_LOG_PATH, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=row.keys())
            if self._needs_header(cfg.TRADE_LOG_PATH):
                w.writeheader()
            w.writerow(row)
