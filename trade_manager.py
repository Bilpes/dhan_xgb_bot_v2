# trade_manager.py — dhan_xgb_bot_v2
# Audit-patched 2026-06-28
# Fix I7: CSV header now checked at write time via _needs_header() — not at init
#         Prevents headerless rows if file is deleted/rotated mid-session
# Fix I8: compute_qty now caps single position at 25% of TOTAL_CAPITAL
#         Prevents capital overflow for high-price stocks (MARUTI ₹13k, etc.)

import logging
import os
import csv
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, Dict

import config as cfg

log = logging.getLogger("trade_manager")


@dataclass
class Position:
    symbol:       str
    entry_price:  float
    qty:          int
    sl:           float
    target:       float
    atr:          float
    prob:         float
    entry_time:   datetime
    mode:         str   = "INTRADAY"
    trailing_sl_active: bool  = False
    peak_price:         float = 0.0
    trailing_sl:        float = 0.0


class TradeManager:
    def __init__(self, dhan_client=None):
        self.dhan      = dhan_client
        self.positions: Dict[str, Position] = {}
        self.daily_pnl = 0.0
        os.makedirs(os.path.dirname(cfg.TRADE_LOG_PATH), exist_ok=True)
        # FIX I7: removed self._hdr = os.path.exists(...) — stale state bug
        # Header is now checked at write time via _needs_header()

    def reset_daily(self):
        """Call at session start to clear daily P&L counter."""
        self.daily_pnl = 0.0

    # ── Risk checks ──────────────────────────────────────────────────
    @property
    def daily_loss_breached(self) -> bool:
        """True when daily drawdown exceeds MAX_DAILY_LOSS fraction."""
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
        """Check position limits before entering."""
        if self.open_count >= cfg.MAX_OPEN_POSITIONS:
            return False
        from watchlist import SECTOR_MAP
        sec = SECTOR_MAP.get(symbol, "OTHER")
        if self.sector_count(sec) >= cfg.MAX_PER_SECTOR:
            return False
        return True

    # ── Sizing ───────────────────────────────────────────────────────
    def compute_qty(self, entry: float, sl: float) -> int:
        """
        Risk-based position sizing.
        FIX I8: Added 25% capital cap guard.

        Without the cap, high-price stocks can cause capital overflow:
          Example — MARUTI at ₹13,000, ATR=₹60:
            qty = int(500000 * 0.005 / 60) = 41
            position_value = 41 * 13000 = ₹533,000 > TOTAL_CAPITAL ₹500,000

        Fix: single position capped at TOTAL_CAPITAL * 0.25 = ₹125,000
          MARUTI: max_qty = int(125000 / 13000) = 9 shares → safe
        """
        risk_per_share = max(entry - sl, entry * cfg.MIN_SL_PCT)
        raw_qty        = int(cfg.TOTAL_CAPITAL * cfg.RISK_PER_TRADE / risk_per_share)

        # FIX I8: cap at 25% of total capital per position
        max_position_value = cfg.TOTAL_CAPITAL * 0.25
        max_qty_by_capital = int(max_position_value / max(entry, 1))

        qty = min(raw_qty, max_qty_by_capital)
        return max(qty, 1)

    # ── Entry ────────────────────────────────────────────────────────
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
            symbol=symbol,
            entry_price=entry,
            qty=qty,
            sl=sl,
            target=signal["target"],
            atr=signal["atr"],
            prob=signal["prob"],
            entry_time=datetime.now(),
            peak_price=entry,
            trailing_sl=sl,
        )

        if cfg.PAPER_MODE:
            log.info(
                f"[PAPER] BUY {symbol} qty={qty} entry={entry:.2f} "
                f"sl={sl:.2f} tp={signal['target']:.2f} prob={signal['prob']:.3f}"
            )
        else:
            try:
                self.dhan.place_order(
                    security_id=symbol,
                    exchange_segment="NSE_EQ",
                    transaction_type="BUY",
                    quantity=qty,
                    order_type="MARKET",
                    product_type="INTRADAY",
                    price=0,
                )
            except Exception as e:
                log.error(f"Order failed {symbol}: {e}")
                return None

        self.positions[symbol] = pos
        self._log(symbol, "ENTRY", qty, entry, sl, signal["target"], signal["prob"])
        return pos

    # ── Trailing SL update ───────────────────────────────────────────
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

    # ── Exit check on each candle ────────────────────────────────────
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

    # ── Force exit (EOD / manual) ────────────────────────────────────
    def force_exit(self, symbol: str, ltp: float, reason: str = "EOD"):
        self._exit_position(symbol, ltp, reason)

    # ── Internal exit ────────────────────────────────────────────────
    def _exit_position(self, symbol: str, exit_price: float, reason: str):
        pos = self.positions.pop(symbol, None)
        if not pos:
            return
        pnl = (exit_price - pos.entry_price) * pos.qty
        self.daily_pnl += pnl

        if not cfg.PAPER_MODE:
            try:
                self.dhan.place_order(
                    security_id=symbol,
                    exchange_segment="NSE_EQ",
                    transaction_type="SELL",
                    quantity=pos.qty,
                    order_type="MARKET",
                    product_type="INTRADAY",
                    price=0,
                )
            except Exception as e:
                log.error(f"Exit order failed {symbol}: {e}")

        log.info(
            f"EXIT {symbol} reason={reason} exit={exit_price:.2f} "
            f"pnl=\u20b9{pnl:.2f} daily=\u20b9{self.daily_pnl:.2f}"
        )
        self._log(
            symbol, "EXIT", pos.qty, exit_price,
            pos.sl, pos.target, pos.prob, pnl, reason,
        )

    # ── CSV header helper ────────────────────────────────────────────
    @staticmethod
    def _needs_header(path: str) -> bool:
        """
        FIX I7: Check file size at write time — not at init.
        Returns True if the file does not exist or is empty (header needed).
        The old pattern (self._hdr = os.path.exists(...) at __init__) had a
        stale-state bug: if the file was deleted mid-session, _hdr remained
        True and the next write would produce a headerless CSV row.
        """
        try:
            return not os.path.exists(path) or os.path.getsize(path) == 0
        except OSError:
            return True

    # ── CSV trade log ────────────────────────────────────────────────
    def _log(
        self,
        symbol, action, qty, price, sl, target, prob,
        pnl=None, reason=None,
    ):
        row = {
            "time":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "mode":        "PAPER" if cfg.PAPER_MODE else "LIVE",
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
            # FIX I7: check header at write time, not at init
            if self._needs_header(cfg.TRADE_LOG_PATH):
                w.writeheader()
            w.writerow(row)
