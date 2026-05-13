# ============================================================
# bot/live_bot.py — PhD-grade NSE intraday trading loop
#
# Research basis:
#   - Chordia et al. (2000): intraday liquidity patterns on NSE
#   - Anand & Chakravarty (2007): informed trading & order flow
#   - Berkman et al. (2012): opening/closing auction effects
#   - Kumar & Lee (2006): retail sentiment and reversal risk
#   - NSE circular: CNC margin framework, SEBI LODR compliance
#
# Modes:
#   BOT_MODE=test   → one scan cycle, print signals, exit
#   BOT_MODE=paper  → full loop, simulated fills, P&L tracked
#   BOT_MODE=live   → real bracket orders via Dhan API
# ============================================================

from __future__ import annotations

import csv
import logging
import os
import time
from datetime import datetime, time as dtime
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join("config", ".env"))

# ── Mode validation (fail-fast before any imports) ───────────
BOT_MODE = os.getenv("BOT_MODE", "paper").lower().strip()
if BOT_MODE not in ("test", "paper", "live"):
    raise SystemExit(
        f"[ERROR] BOT_MODE='{BOT_MODE}' invalid. Use: test | paper | live"
    )

# ── Project imports ──────────────────────────────────────────
from bot.dhan_api import DhanBroker
from bot.signal_engine import SignalEngine
from bot.risk_manager import RiskManager
from bot.telegram_alert import (
    alert_bot_started, alert_entry, alert_exit,
    alert_trail_update, alert_daily_summary,
    alert_circuit_breaker, _send,
)
from bot.trade_policy import BLOCKED_SYMBOLS
from config.config import (
    WATCHLIST, SECTOR_MAP, TRADE_MODE, MAX_OPEN_TRADES,
    MARKET_OPEN, MARKET_CLOSE, INTRADAY_CUTOFF,
    NO_NEW_TRADE_AFTER, NO_NEW_TRADE_BEFORE,
    TRADE_LOG, LOG_FILE, CAPITAL,
)

# ── Runtime parameters (env-configurable, no restart needed) ─
MAX_PER_SECTOR          = int(os.getenv("MAX_PER_SECTOR",           "2"))
NEW_TRADE_LOSS_PAUSE    = float(os.getenv("NEW_TRADE_LOSS_PAUSE",   "-0.03"))
NIFTY50_SECURITY_ID     = os.getenv("NIFTY50_SECURITY_ID",          "13")
MONITOR_INTERVAL        = int(os.getenv("MONITOR_INTERVAL",         "60"))
SCAN_INTERVAL           = int(os.getenv("SCAN_INTERVAL",            "300"))
AUTO_EXIT_TIME          = os.getenv("AUTO_EXIT_TIME",               "15:15")
AUTO_EXIT_THRESHOLD     = float(os.getenv("AUTO_EXIT_THRESHOLD",    "-0.01"))
EOD_RESET_TIME          = os.getenv("EOD_RESET_TIME",               "15:30")

# Execution quality gates
EXEC_MAX_SPREAD_PCT     = float(os.getenv("EXEC_MAX_SPREAD_PCT",    "0.0005"))
EXEC_MAX_DRIFT_PCT      = float(os.getenv("EXEC_MAX_DRIFT_PCT",     "0.0010"))
LIQUIDITY_MIN_VALUE     = float(os.getenv("LIQUIDITY_MIN_VALUE",    "5000000"))
LIQUIDITY_LOOKBACK_BARS = int(os.getenv("LIQUIDITY_LOOKBACK_BARS",  "3"))
ENTRY_REPRICE_SECS      = int(os.getenv("ENTRY_REPRICE_SECS",       "10"))

# Rotation parameters
ROTATION_MIN_PROFIT     = float(os.getenv("ROTATION_MIN_PROFIT",    "0.005"))
ROTATION_MIN_EDGE       = float(os.getenv("ROTATION_MIN_EDGE",      "0.05"))

# ── NSE session schedule (Chordia et al. 2000 — avoid first 15min noise) ─
# The opening 15 minutes on NSE exhibit the widest spreads and highest
# adverse-selection costs. NO_NEW_TRADE_BEFORE in config should be "09:30".
# Last 30 min also have elevated volatility from index rebalancing flows.

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level    = logging.INFO,
    format   = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    handlers = [
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("live_bot")
TRADE_ANALYSIS_LOG = Path("logs/trade_analysis.csv")
MODE_LABEL = {
    "test":  "TEST   — one scan cycle, no orders, exits after",
    "paper": "PAPER  — full loop, no real orders, simulated P&L",
    "live":  "LIVE   — real bracket orders on Dhan, real money",
}

# ── Candle period (minutes) — must match features.py / retrain ─
CANDLE_MINUTES = 5


# ─────────────────────────────────────────────────────────────
#  Trade dataclass
# ─────────────────────────────────────────────────────────────
class Trade:
    """
    Immutable identity fields set at entry.
    Mutable fields updated during position monitoring.
    """
    __slots__ = (
        "symbol", "security_id", "side", "qty",
        "entry", "stop_loss", "target", "order_id", "mode",
        "running_high", "open_time",
        "last_prob", "rr", "atr",
        "candles_held", "trail_count",
    )

    def __init__(
        self,
        symbol:      str,
        security_id,
        side:        str,
        qty:         int,
        entry:       float,
        stop_loss:   float,
        target:      float,
        order_id:    str,
        mode:        str,
        last_prob:   float = 0.5,
        rr:          float = 0.0,
        atr:         float = 0.0,
    ):
        self.symbol      = symbol
        self.security_id = security_id
        self.side        = side
        self.qty         = qty
        self.entry       = entry
        self.stop_loss   = stop_loss
        self.target      = target
        self.order_id    = order_id
        self.mode        = mode
        self.running_high = entry
        self.open_time   = datetime.now()
        self.last_prob   = last_prob
        self.rr          = rr
        self.atr         = atr
        self.candles_held  = 0     # incremented each monitor tick
        self.trail_count   = 0     # number of trailing SL updates

    def unrealised_pnl(self, ltp: float) -> float:
        if self.side == "LONG":
            return (ltp - self.entry) * self.qty
        return (self.entry - ltp) * self.qty

    def hold_minutes(self) -> float:
        return (datetime.now() - self.open_time).total_seconds() / 60.0

    def __repr__(self) -> str:
        return (
            f"Trade({self.symbol} {self.side} qty={self.qty} "
            f"entry={self.entry:.2f} SL={self.stop_loss:.2f} "
            f"TP={self.target:.2f} held={self.hold_minutes():.0f}m)"
        )


# ─────────────────────────────────────────────────────────────
#  Utility functions
# ─────────────────────────────────────────────────────────────
def _log_trade_csv(row: dict):
    """Append one trade row to TRADE_LOG CSV. Thread-safe for single process."""
    exists = os.path.exists(TRADE_LOG)
    with open(TRADE_LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if not exists:
            w.writeheader()
        w.writerow(row)


def _now_time() -> dtime:
    return datetime.now().time()


def _parse_time(s: str) -> dtime:
    h, m = map(int, s.split(":"))
    return dtime(h, m)


def _is_market_open() -> bool:
    t = _now_time()
    return _parse_time(MARKET_OPEN) <= t <= _parse_time(MARKET_CLOSE)


def _is_cutoff_passed() -> bool:
    """True after intraday CNC square-off deadline."""
    return _now_time() >= _parse_time(INTRADAY_CUTOFF)


def _no_new_trades() -> bool:
    return _now_time() >= _parse_time(NO_NEW_TRADE_AFTER)


def _is_auto_exit_time() -> bool:
    """2:45 PM safety — exit positions in weak market before close."""
    return _now_time() >= _parse_time(AUTO_EXIT_TIME)


def _is_candle_boundary() -> bool:
    """
    True at the close of each 5-min candle.
    Used to gate signal-flip checks and candle-low SL validation —
    avoid over-checking intra-candle noise (Berkman et al. 2012).
    """
    return datetime.now().minute % CANDLE_MINUTES == 0


# ─────────────────────────────────────────────────────────────
#  LiveBot
# ─────────────────────────────────────────────────────────────
class LiveBot:
    """
    Main trading loop. Responsibilities:
      - Nifty regime context refresh
      - Stock scan + confidence ranking
      - Execution quality gates (spread, drift, liquidity)
      - Entry: bracket order (live) or paper simulation
      - Monitor: trailing SL, target, signal-flip, candle-low SL
      - Exit: market order (live) or paper simulation
      - Daily P&L, circuit breaker, EOD reset
      - Portfolio rotation: exit weakest, enter strongest signal

    Research note:
      Chordia et al. (2000) show that NSE intraday liquidity follows
      a U-shaped pattern — widest spreads at open and close.
      This bot avoids first-15-min entries (NO_NEW_TRADE_BEFORE=09:30)
      and auto-exits weak positions before 2:45 PM close.
    """

    def __init__(self):
        self.broker  = DhanBroker()
        self.engine  = SignalEngine()
        self.risk    = RiskManager()
        self.trades: dict[str, Trade] = {}
        self.closed_trades: list[dict] = []
        self.paper_pnl        = 0.0
        self.sl_blacklist:    set[str] = set()
        self._last_scan_ts    = 0.0
        self._entry_quotes:   dict[str, dict] = {}
        self._nifty_refreshed = False

    # ──────────────────────────────────────────────────────────
    #  Nifty regime context
    # ──────────────────────────────────────────────────────────
    def _refresh_nifty(self):
        """
        Fetch latest Nifty 50 candles and push into SignalEngine.
        Nifty regime features (trend, RSI, ATR%) directly gate
        individual stock confidence scores — see features.py.
        If fetch fails, engine uses neutral (0.0) Nifty values.
        """
        try:
            nifty_df = self.broker.get_candles(
                security_id = NIFTY50_SECURITY_ID,
                symbol      = "NIFTY50",
                days_back   = 5,
            )
            if not nifty_df.empty:
                self.engine.update_nifty(nifty_df)
                self._nifty_refreshed = True
                log.info("Nifty refreshed: %d candles, last=%s",
                         len(nifty_df), nifty_df.index[-1])
            else:
                log.warning("Nifty fetch empty — using previous values.")
        except Exception as e:
            log.warning("Nifty refresh failed: %s — engine uses neutral values.", e)

    # ──────────────────────────────────────────────────────────
    #  Execution quality gate
    # ──────────────────────────────────────────────────────────
    def _recent_liquidity_value(self, df) -> float:
        """
        Turnover (₹) over last N bars.
        Anand & Chakravarty (2007): low-turnover stocks carry higher
        adverse-selection cost — minimum ₹50L turnover per 3 candles.
        """
        if df is None or df.empty:
            return 0.0
        tail = df.tail(LIQUIDITY_LOOKBACK_BARS)
        return float((tail["close"] * tail["volume"]).sum()) if not tail.empty else 0.0

    def _passes_entry_filters(
        self,
        symbol:       str,
        sec_id,
        entry_price:  float,
        reason_prefix: str = "",
    ) -> tuple[bool, dict]:
        """
        Three-gate execution filter:
          1. Bid-ask spread < EXEC_MAX_SPREAD_PCT (skips on scan, hard gate on entry)
          2. Recent turnover >= LIQUIDITY_MIN_VALUE
          3. Price drift since signal < EXEC_MAX_DRIFT_PCT

        Returns (passed: bool, quote_dict: dict)

        Research note (Chordia et al. 2000):
          Execution cost = half-spread + price impact.
          For a 0.10% edge model, spread alone must be < 0.05% to
          be profitable net of all costs.
        """
        is_entry_stage = "[ENTRY]" in reason_prefix or "[EXEC]" in reason_prefix
        try:
            ltp = self.broker.get_ltp(str(sec_id), symbol)
            if ltp <= 0:
                log.info("%s%s: LTP unavailable — skip.", reason_prefix, symbol)
                return False, {}

            bid = ask = spread_pct = spread_abs = None
            quote: dict = {}

            if hasattr(self.broker, "get_quote"):
                try:
                    raw = self.broker.get_quote(str(sec_id), symbol) or {}
                    meta = raw.get("meta", {})
                    bid, ask    = raw.get("bid"), raw.get("ask")
                    spread_abs  = raw.get("spread_abs")
                    spread_pct  = raw.get("spread_pct")

                    if meta:
                        log.info(
                            "%s%s: quote dhan_ok=%s partial=%s fallback=%s attempts=%s",
                            reason_prefix, symbol,
                            meta.get("dhan_success"),
                            meta.get("partial_success"),
                            meta.get("fallback_used"),
                            meta.get("attempts"),
                        )

                    dhan_ok = meta.get("dhan_success", True) if meta else True
                    if not dhan_ok:
                        if is_entry_stage:
                            log.info("%s%s: quote unavailable — skip entry.", reason_prefix, symbol)
                            return False, raw
                        # Scan stage: degrade gracefully — continue with LTP only
                        raw = {}

                    if bid is not None and ask is not None and spread_pct is None:
                        spread_abs = float(ask) - float(bid)
                        spread_pct = spread_abs / ltp if ltp > 0 else 1.0

                    # Hard spread gate at entry; advisory only at scan
                    if is_entry_stage and spread_pct is not None:
                        if spread_pct > EXEC_MAX_SPREAD_PCT:
                            log.info(
                                "%s%s: spread %.5f%% > max %.5f%% — skip.",
                                reason_prefix, symbol,
                                spread_pct * 100, EXEC_MAX_SPREAD_PCT * 100,
                            )
                            return False, raw

                    quote = raw
                except Exception as e:
                    log.warning("%s%s: quote error: %s", reason_prefix, symbol, e)
                    if is_entry_stage:
                        return False, {}

            # ── Liquidity gate ────────────────────────────────
            df = self.broker.get_candles(sec_id, symbol, days_back=1)
            if df.empty:
                log.info("%s%s: candles unavailable — skip.", reason_prefix, symbol)
                return False, quote

            liquidity = self._recent_liquidity_value(df)
            if liquidity < LIQUIDITY_MIN_VALUE:
                log.info(
                    "%s%s: liquidity ₹%.0f < min ₹%.0f — skip.",
                    reason_prefix, symbol, liquidity, LIQUIDITY_MIN_VALUE,
                )
                return False, quote

            # ── Price drift gate ──────────────────────────────
            drift = abs(ltp - entry_price) / entry_price if entry_price > 0 else 1.0
            if drift > EXEC_MAX_DRIFT_PCT:
                log.info(
                    "%s%s: drift %.4f%% > max %.4f%% (signal=%.2f ltp=%.2f) — skip.",
                    reason_prefix, symbol,
                    drift * 100, EXEC_MAX_DRIFT_PCT * 100,
                    entry_price, ltp,
                )
                return False, quote

            return True, {
                "ltp":        ltp,
                "bid":        bid,
                "ask":        ask,
                "spread_abs": spread_abs,
                "spread_pct": spread_pct,
                "meta":       quote.get("meta", {}),
            }

        except Exception as e:
            log.warning("%s%s: filter error: %s", reason_prefix, symbol, e)
            return False, {}

    # ──────────────────────────────────────────────────────────
    #  TEST MODE
    # ──────────────────────────────────────────────────────────
    def run_test(self):
        """
        Scans all watchlist stocks once, logs signals to Telegram.
        No orders placed. Useful for pre-market confidence check.
        """
        log.info("=" * 55)
        log.info("TEST MODE — scanning %d stocks, no orders", len(WATCHLIST))
        log.info("=" * 55)

        _send(
            f"🤖 <b>Bot TEST MODE</b>\n"
            f"Scanning {len(WATCHLIST)} stocks...\n"
            f"No orders placed.\n"
            f"Capital: ₹{CAPITAL:,} | Mode: {TRADE_MODE.upper()}"
        )

        self._refresh_nifty()
        results: list[str] = []

        for symbol, sec_id in WATCHLIST.items():
            try:
                if symbol.upper() in BLOCKED_SYMBOLS:
                    results.append(f"  {symbol:<14} BLOCKED by policy")
                    continue

                df = self.broker.get_candles(sec_id, symbol, days_back=10)
                if df.empty:
                    results.append(f"  {symbol:<14} no data")
                    continue

                r      = self.engine.score(df, symbol=symbol)
                signal = r["signal"]
                prob   = r["prob_up"]
                entry  = r["entry"]
                sl     = r["sl"]
                target = r["target"]
                rr     = r["rr"]

                if entry <= 0 or sl <= 0 or target <= 0 or rr <= 0:
                    results.append(f"  {symbol:<14} invalid signal")
                    continue

                qty    = self.risk.position_size(entry, sl)
                sector = SECTOR_MAP.get(symbol, "?")
                risk_amt = (entry - sl) * qty

                results.append(
                    f"  {symbol:<14} {signal:<5} ₹{entry:.1f}"
                    f"  conf={prob:.1%}  SL=₹{sl:.1f}"
                    f"  R:R={rr:.2f}x  qty={qty}"
                    f"  risk=₹{risk_amt:.0f}  [{sector}]"
                )
                log.info("  %s: %s prob=%.3f entry=%.2f R:R=%.2fx [%s]",
                         symbol, signal, prob, entry, rr, sector)

            except Exception as e:
                results.append(f"  {symbol:<14} error: {e}")
                log.error("  %s: error: %s", symbol, e)

        # Chunk Telegram messages (max 15 lines each)
        chunk_size = 15
        chunks = [results[i:i+chunk_size] for i in range(0, len(results), chunk_size)]
        for idx, chunk in enumerate(chunks):
            header = (
                f"📊 <b>TEST RESULTS ({idx+1}/{len(chunks)})</b> "
                f"— {datetime.now().strftime('%d %b %Y')}\n"
                f"{'─' * 30}\n"
            )
            footer = "\n\n✅ Bot OK. Set BOT_MODE=paper to start paper trading." \
                     if idx == len(chunks) - 1 else ""
            _send(header + "\n".join(chunk) + footer)

        log.info("Test scan complete.")

    # ──────────────────────────────────────────────────────────
    #  Portfolio rotation
    # ──────────────────────────────────────────────────────────
    def _should_rotate(self, new_prob: float) -> tuple[bool, Optional[str]]:
        """
        Rotation logic: replace the weakest open position with a
        significantly stronger new signal, but only if the weakest
        position is already in profit (to avoid realising losses).

        Kumar & Lee (2006) show retail herding on NSE amplifies
        momentum — rotating into higher-confidence setups at scale
        has measurable edge over static hold.

        Returns: (should_rotate: bool, symbol_to_exit: str | None)
        """
        if not self.trades:
            return False, None

        id_map = {str(t.security_id): sym for sym, t in self.trades.items()}
        prices = self.broker.get_ltp_batch(id_map)

        weakest_symbol: Optional[str] = None
        weakest_prob   = new_prob

        for symbol, trade in self.trades.items():
            ltp = prices.get(str(trade.security_id), 0.0)
            if ltp <= 0:
                continue

            profit_pct = (ltp - trade.entry) / trade.entry
            if profit_pct < ROTATION_MIN_PROFIT:
                log.info("%s: rotation skip — profit %.2f%% < min %.2f%%",
                         symbol, profit_pct * 100, ROTATION_MIN_PROFIT * 100)
                continue

            if new_prob >= trade.last_prob + ROTATION_MIN_EDGE:
                if weakest_symbol is None or trade.last_prob < weakest_prob:
                    weakest_symbol = symbol
                    weakest_prob   = trade.last_prob
                    log.info(
                        "Rotation candidate: exit %s (prob=%.3f) "
                        "→ new signal (prob=%.3f edge=%.3f)",
                        symbol, trade.last_prob, new_prob,
                        new_prob - trade.last_prob,
                    )

        return (weakest_symbol is not None), weakest_symbol

    # ──────────────────────────────────────────────────────────
    #  Dhan position sync (live mode only)
    # ──────────────────────────────────────────────────────────
    def _sync_with_dhan(self):
        """
        Cross-check bot's open trades against Dhan's live positions.
        If a trade disappeared from Dhan (manual close, rejected order,
        margin call) — sync the bot state to avoid ghost positions.
        """
        if BOT_MODE != "live" or not self.trades:
            return
        try:
            dhan_pos = self.broker.get_positions()
            if dhan_pos.empty:
                log.warning("sync_dhan: Dhan returned empty positions — API glitch? skipping.")
                return

            sym_col = next(
                (c for c in ["tradingSymbol", "trading_symbol", "symbol"]
                 if c in dhan_pos.columns), None
            )
            if sym_col is None:
                log.warning("sync_dhan: cannot find symbol column in Dhan positions.")
                return

            open_on_dhan = set(dhan_pos[sym_col].str.upper())
            log.debug("sync_dhan: Dhan open = %s", open_on_dhan)

            for symbol, trade in list(self.trades.items()):
                if symbol.upper() not in open_on_dhan:
                    log.warning("%s: missing from Dhan — manual close? syncing.", symbol)
                    ltp = self.broker.get_ltp(str(trade.security_id), symbol)
                    exit_price = ltp if ltp > 0 else trade.stop_loss
                    self._exit_trade(trade, exit_price, "CLOSED_ON_DHAN")
                    self.sl_blacklist.add(symbol)
        except Exception as e:
            log.error("sync_dhan: failed: %s", e)

    # ──────────────────────────────────────────────────────────
    #  Scan and enter
    # ──────────────────────────────────────────────────────────
    def _scan_and_enter(self):
        """
        Full watchlist scan → rank BUY signals by confidence →
        pick highest-quality setup → execute.

        Design principles:
          - Skip opening 15 min (NO_NEW_TRADE_BEFORE=09:30)
          - Max 2 positions per GICS sector (concentration risk)
          - Spread, liquidity, drift gates before execution
          - Confidence-ranked selection (best model edge wins)
          - Rotation only when new signal strictly dominates
        """
        # ── Session time gates ────────────────────────────────
        if _now_time() < _parse_time(NO_NEW_TRADE_BEFORE):
            log.info("Pre-trade window: waiting until %s", NO_NEW_TRADE_BEFORE)
            return
        if _no_new_trades():
            return

        # ── Daily loss pause ──────────────────────────────────
        daily_loss_pct = self.risk.daily_pnl / CAPITAL
        if daily_loss_pct <= NEW_TRADE_LOSS_PAUSE:
            log.warning(
                "Daily loss %.2f%% ≤ pause threshold %.2f%% — no new entries.",
                daily_loss_pct * 100, NEW_TRADE_LOSS_PAUSE * 100,
            )
            return

        max_slots_full = len(self.trades) >= MAX_OPEN_TRADES
        log.info("── Scan: ranking %d eligible stocks ──", len(WATCHLIST))

        candidates: list[tuple[str, str, dict, dict]] = []

        for symbol, sec_id in WATCHLIST.items():
            # ── Symbol-level gates ────────────────────────────
            if symbol.upper() in BLOCKED_SYMBOLS:
                continue
            if symbol in self.trades:
                continue
            if symbol in self.sl_blacklist:
                log.debug("%s: SL-blacklisted today — skip.", symbol)
                continue

            # ── Sector concentration gate ─────────────────────
            sector = SECTOR_MAP.get(symbol, "unknown")
            sector_count = sum(1 for s in self.trades
                               if SECTOR_MAP.get(s, "unknown") == sector)
            if sector_count >= MAX_PER_SECTOR:
                continue

            # ── Data fetch ────────────────────────────────────
            df = self.broker.get_candles(sec_id, symbol, days_back=10)
            if df.empty:
                continue

            # ── Signal engine score ───────────────────────────
            result = self.engine.score(df, symbol=symbol)
            if result["signal"] != "BUY":
                continue

            entry  = result["entry"]
            sl     = result["sl"]
            target = result["target"]
            rr     = result["rr"]

            # Sanity-check engine output
            if any(v <= 0 for v in [entry, sl, target, rr]):
                log.warning("%s: invalid engine output entry=%.2f sl=%.2f — skip",
                            symbol, entry, sl)
                continue

            # ── Scan-stage execution quality ──────────────────
            ok, quote = self._passes_entry_filters(
                symbol, sec_id, entry, reason_prefix="[SCAN] ")
            if not ok:
                continue

            candidates.append((symbol, sec_id, result, quote))
            log.info(
                "  Candidate %-14s prob=%.3f  entry=%.2f  SL=%.2f  "
                "TP=%.2f  R:R=%.2fx  spread=%.4f%%  [%s]",
                symbol, result["prob_up"], entry, sl, target, rr,
                (quote.get("spread_pct") or 0.0) * 100, sector,
            )

        if not candidates:
            log.info("No BUY signals this scan.")
            return

        # ── Select highest-confidence signal ──────────────────
        candidates.sort(key=lambda x: (x[2]["prob_up"] * 0.7+ x[2].get("atr_ratio", 1.0) * 0.3),reverse=True,)
        best_sym, best_sec_id, best_result, best_quote = candidates[0]
        log.info(
            "Best signal: %-14s prob=%.3f  (%d candidates ranked)",
            best_sym, best_result["prob_up"], len(candidates),
        )

        # ── Portfolio rotation check ──────────────────────────
        if max_slots_full:
            should_rotate, exit_sym = self._should_rotate(best_result["prob_up"])
            if should_rotate and exit_sym:
                trade_to_exit = self.trades[exit_sym]
                id_map  = {str(trade_to_exit.security_id): exit_sym}
                prices  = self.broker.get_ltp_batch(id_map)
                ltp     = prices.get(str(trade_to_exit.security_id), trade_to_exit.entry)
                log.info("ROTATION: exiting %s → entering %s", exit_sym, best_sym)
                self._exit_trade(trade_to_exit, ltp, "ROTATION_BETTER_SIGNAL")
            else:
                log.info("Max open trades (%d). No rotation opportunity.", MAX_OPEN_TRADES)
                return

        # ── Final validation ──────────────────────────────────
                # ─────────────────────────────────────────────
        # HARD RISK SAFETY GATES
        # ─────────────────────────────────────────────

        # Circuit breaker protection
        if self.risk.is_halted():
            log.critical(
                "RiskManager halted trading — skipping new entries."
            )
            return

        # Hard cap on concurrent positions
        if len(self.trades) >= MAX_OPEN_TRADES:
            log.warning(
                "Max open trades reached (%d/%d) — skip entry.",
                len(self.trades),
                MAX_OPEN_TRADES,
            )
            return

        # ── Final validation ──────────────────────────
        entry  = best_result["entry"]
        sl     = best_result["sl"]
        target = best_result["target"]
        rr     = best_result["rr"]

        if any(v <= 0 for v in [entry, sl, target, rr]):
            log.warning("%s: final proposal invalid — skip.", best_sym)
            return

        qty = self.risk.position_size(entry, sl)

        if qty <= 0:
            log.warning("%s: position size = 0 — skip.", best_sym)
            return

        log.info(
            "SIGNAL BUY %-14s  entry=%.2f  SL=%.2f  TP=%.2f"
            "  qty=%d  prob=%.3f  R:R=%.2fx  sector=%s  spread=%.4f%%",
            best_sym,
            entry,
            sl,
            target,
            qty,
            best_result["prob_up"],
            rr,
            SECTOR_MAP.get(best_sym, "?"),
            (best_quote.get("spread_pct") or 0.0) * 100,
        )

        if BOT_MODE == "live":

            ok, quote = self._passes_entry_filters(
                best_sym,
                best_sec_id,
                entry,
                reason_prefix="[ENTRY] ",
            )

            if not ok:
                log.warning(
                    "%s: entry execution filters failed.",
                    best_sym,
                )
                return

            self._entry_quotes[best_sym] = {
                "entry": entry,
                "ts": time.time(),
            }

            self._enter_live(
                best_sym,
                best_sec_id,
                qty,
                entry,
                sl,
                target,
                best_result,
            )

        else:

            self._enter_paper(
                best_sym,
                best_sec_id,
                qty,
                entry,
                sl,
                target,
                best_result,
            )

    # ──────────────────────────────────────────────────────────
    #  Entry: live
    # ──────────────────────────────────────────────────────────
    def _enter_live(
        self, symbol: str, sec_id, qty: int,
        entry: float, sl: float, target: float, result: dict,
    ):
        """
        Place bracket order on Dhan with entry re-pricing loop.
        If LTP drifts beyond EXEC_MAX_DRIFT_PCT before fill, abort.
        Re-price for up to ENTRY_REPRICE_SECS seconds.
        """
        start         = time.time()
        initial_entry = self._entry_quotes.get(symbol, {}).get("entry", entry)

        try:
            # ── Re-price loop ─────────────────────────────────
            while True:
                drift_ok, quote = self._passes_entry_filters(
                    symbol, sec_id, initial_entry, reason_prefix="[EXEC] ")
                if not drift_ok:
                    log.info("%s: execution aborted — drift/spread gate failed.", symbol)
                    return

                ltp = quote.get("ltp") or entry
                if abs(ltp - initial_entry) / initial_entry <= EXEC_MAX_DRIFT_PCT:
                    entry = ltp   # use latest LTP as execution price
                    break

                if time.time() - start > ENTRY_REPRICE_SECS:
                    log.info("%s: re-price window %ds expired — abort.", symbol, ENTRY_REPRICE_SECS)
                    return

                time.sleep(1)

            # ── Place bracket order ───────────────────────────
            resp = self.broker.place_bracket_order(
                symbol       = symbol,
                security_id  = sec_id,
                quantity     = qty,
                entry_price  = entry,
                stop_loss    = sl,
                target       = target,
                trade_type   = TRADE_MODE,
            )

            if resp.get("status") == "success":
                order_id = resp["data"]["orderId"]
                self.trades[symbol] = Trade(
                    symbol=symbol, security_id=sec_id, side="LONG",
                    qty=qty, entry=entry, stop_loss=sl, target=target,
                    order_id=order_id, mode=TRADE_MODE,
                    last_prob=result["prob_up"],
                    rr=result.get("rr", 0.0),
                    atr=result.get("atr", 0.0),
                )
                _log_trade_csv({
                    "time":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "mode":    "LIVE",
                    "symbol":  symbol,
                    "action":  "ENTRY",
                    "side":    "LONG",
                    "qty":     qty,
                    "price":   entry,
                    "sl":      sl,
                    "target":  target,
                    "prob_up": result["prob_up"],
                    "rr":      result.get("rr", 0.0),
                    "pnl":     "",
                })
                alert_entry(
                    symbol, entry, sl, target, qty,
                    result["prob_up"], TRADE_MODE, entry * qty,
                )
                log.info("LIVE ENTRY %s @ %.2f qty=%d order=%s", symbol, entry, qty, order_id)
            else:
                log.error("Order FAILED %s: %s", symbol, resp)
                _send(f"❌ <b>ORDER FAILED</b> {symbol}\n{resp}")

        except Exception as e:
            log.error("_enter_live %s: %s", symbol, e)
        finally:
            self._entry_quotes.pop(symbol, None)

    # ──────────────────────────────────────────────────────────
    #  Entry: paper
    # ──────────────────────────────────────────────────────────
    def _enter_paper(
        self, symbol: str, sec_id, qty: int,
        entry: float, sl: float, target: float, result: dict,
    ):
        self.trades[symbol] = Trade(
            symbol=symbol, security_id=sec_id, side="LONG",
            qty=qty, entry=entry, stop_loss=sl, target=target,
            order_id="PAPER-" + datetime.now().strftime("%H%M%S"),
            mode=TRADE_MODE,
            last_prob=result["prob_up"],
            rr=result.get("rr", 0.0),
            atr=result.get("atr", 0.0),
        )
        _log_trade_csv({
            "time":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "mode":    "PAPER",
            "symbol":  symbol,
            "action":  "ENTRY",
            "side":    "LONG",
            "qty":     qty,
            "price":   entry,
            "sl":      sl,
            "target":  target,
            "prob_up": result["prob_up"],
            "rr":      result.get("rr", 0.0),
            "pnl":     "",
        })
        log.info(
            "[PAPER] BUY %-14s qty=%d @ %.2f  SL=%.2f  TP=%.2f"
            "  R:R=%.2fx  prob=%.3f  [%s]",
            symbol, qty, entry, sl, target,
            result.get("rr", 0.0), result["prob_up"],
            SECTOR_MAP.get(symbol, "?"),
        )
        alert_entry(symbol, entry, sl, target, qty,
                    result["prob_up"], TRADE_MODE, entry * qty)

    # ──────────────────────────────────────────────────────────
    #  Monitor open positions
    # ──────────────────────────────────────────────────────────
    def _monitor_positions(self):
        """
        Per-tick (every MONITOR_INTERVAL seconds) position checks.
        Candle-boundary checks (signal flip, candle-low SL) run only
        at 5-min close to avoid acting on intra-candle noise.
        """

        if not self.trades:
            return

        id_map = {str(t.security_id): sym for sym, t in self.trades.items()}
        prices = self.broker.get_ltp_batch(id_map)

        for symbol, trade in list(self.trades.items()):

            ltp = prices.get(str(trade.security_id), 0.0)

            if ltp <= 0:
                log.warning("%s: LTP unavailable — skip tick.", symbol)
                continue

            # ── Update running high ──────────────────────────
            if ltp > trade.running_high:
                trade.running_high = ltp

            pnl        = trade.unrealised_pnl(ltp)
            profit_pct = (ltp - trade.entry) / trade.entry
            hold_mins  = trade.hold_minutes()

            # ── 1. Target hit ────────────────────────────────
            if ltp >= trade.target:

                log.info(
                    "%s: TARGET HIT ltp=%.2f TP=%.2f held=%.0fmin",
                    symbol,
                    ltp,
                    trade.target,
                    hold_mins,
                )

                self._exit_trade(
                    trade,
                    ltp,
                    "TARGET_HIT"
                )

                continue

            # ── 2. Adaptive trailing stop ───────────────────
            should_trail, new_sl = self.risk.should_trail(
                entry=trade.entry,
                current=ltp,
                running_high=trade.running_high,
            )

            if should_trail and new_sl > trade.stop_loss:

                old_sl = trade.stop_loss

                trade.stop_loss = round(new_sl, 2)

                trade.trail_count += 1

                locked_pct = (
                    (trade.stop_loss - trade.entry)
                    / trade.entry
                ) * 100

                log.info(
                    "%s: TRAIL #%d  SL %.2f → %.2f  "
                    "LTP=%.2f  Locked=%.2f%%  "
                    "P&L=₹%+.0f",
                    symbol,
                    trade.trail_count,
                    old_sl,
                    trade.stop_loss,
                    ltp,
                    locked_pct,
                    pnl,
                )

                alert_trail_update(
                    symbol,
                    trade.stop_loss,
                    ltp,
                    pnl,
                )

            # ── 3. LTP SL check ─────────────────────────────
            if ltp <= trade.stop_loss:

                log.warning(
                    "%s: SL HIT (LTP) ltp=%.2f SL=%.2f held=%.0fmin",
                    symbol,
                    ltp,
                    trade.stop_loss,
                    hold_mins,
                )

                self._exit_trade(
                    trade,
                    ltp,
                    "SL_HIT"
                )

                self.sl_blacklist.add(symbol)

                continue

            # ── Candle-boundary checks ──────────────────────
            if _is_candle_boundary():

                # ── New candle completed ────────────────────
                trade.candles_held += 1

                # ── Momentum failure exit ───────────────────
                # Good breakout trades should work quickly.
                # If still negative after 3 completed candles,
                # continuation probability drops sharply.

                if (
                    trade.candles_held >= 3
                    and pnl < 0
                ):

                    log.warning(
                        "%s: MOMENTUM FAILURE EXIT "
                        "candles=%d pnl=₹%.0f",
                        symbol,
                        trade.candles_held,
                        pnl,
                    )

                    self._exit_trade(
                        trade,
                        ltp,
                        "MOMENTUM_FAILURE"
                    )

                    continue

                # ── 4. Candle-low SL check ─────────────────
                # NSE Level-2 data shows wicks often breach SL intra-candle.
                # Checking candle low catches wick-hits accurately.

                try:

                    df_check = self.broker.get_candles(
                        trade.security_id,
                        symbol,
                        days_back=1,
                    )

                    if not df_check.empty:

                        last_low = float(df_check["low"].iloc[-1])

                        if last_low <= trade.stop_loss:

                            log.warning(
                                "%s: SL HIT (candle low) "
                                "low=%.2f ≤ SL=%.2f held=%.0fmin",
                                symbol,
                                last_low,
                                trade.stop_loss,
                                hold_mins,
                            )

                            self._exit_trade(
                                trade,
                                trade.stop_loss,
                                "SL_HIT_CANDLE_LOW"
                            )

                            self.sl_blacklist.add(symbol)

                            continue

                except Exception as e:

                    log.warning(
                        "%s: candle-low SL check error: %s",
                        symbol,
                        e,
                    )

                # ── 5. Auto-exit: 2:45 PM weak market ─────
                if (
                    _is_auto_exit_time()
                    and profit_pct <= AUTO_EXIT_THRESHOLD
                ):

                    log.warning(
                        "%s: AUTO-EXIT 2:45 PM "
                        "profit %.2f%% ≤ threshold %.2f%%",
                        symbol,
                        profit_pct * 100,
                        AUTO_EXIT_THRESHOLD * 100,
                    )

                    self._exit_trade(
                        trade,
                        ltp,
                        "AUTO_EXIT_EOD_WEAK"
                    )

                    continue

                # ── 6. Signal flip / model deterioration ──
                try:

                    df_live = self.broker.get_candles(
                        trade.security_id,
                        symbol,
                        days_back=5,
                    )

                    if not df_live.empty:

                        scored = self.engine.score(
                            df_live,
                            symbol=symbol,
                        )

                        trade.last_prob = scored["prob_up"]

                        if self.engine.should_exit(
                            df_live,
                            trade.side,
                            symbol=symbol,
                        ):

                            log.info(
                                "%s: SIGNAL FLIP "
                                "exit=%.2f prob=%.3f "
                                "P&L=₹%.0f held=%.0fmin "
                                "candles=%d",
                                symbol,
                                ltp,
                                scored["prob_up"],
                                pnl,
                                hold_mins,
                                trade.candles_held,
                            )

                            self._exit_trade(
                                trade,
                                ltp,
                                "SIGNAL_FLIP"
                            )

                            continue

                except Exception as e:

                    log.warning(
                        "%s: signal flip check error: %s",
                        symbol,
                        e,
                    )

            # ── Holding log ───────────────────────────────
            log.info(
                "HOLD %-14s LTP=%.2f  Entry=%.2f  "
                "SL=%.2f  TP=%.2f  ATR=%.2f  "
                "prob=%.3f  R:R=%.2fx  "
                "P&L=₹%+.0f  pct=%+.2f%%  "
                "held=%.0fm  [%s] held_candles=%d",

                symbol,
                ltp,
                trade.entry,
                trade.stop_loss,
                trade.target,
                trade.atr,
                trade.last_prob,
                trade.rr,
                pnl,
                profit_pct * 100,
                hold_mins,
                SECTOR_MAP.get(symbol, "?"),
                trade.candles_held,
            )
    # ──────────────────────────────────────────────────────────
    #  Force exit all (EOD / circuit breaker)
    # ──────────────────────────────────────────────────────────
    def _force_exit_all(self, reason: str = "CNC_EOD_CUTOFF"):
        if not self.trades:
            return
        log.warning("Force-exiting %d position(s) — reason: %s",
                    len(self.trades), reason)
        id_map = {str(t.security_id): sym for sym, t in self.trades.items()}
        prices = self.broker.get_ltp_batch(id_map)

        for symbol, trade in list(self.trades.items()):
            ltp = prices.get(str(trade.security_id), 0.0)
            if ltp <= 0:
                # Fallback: use running high (best realistic fill estimate)
                ltp = trade.running_high if trade.running_high > trade.entry else trade.entry
                log.warning("%s: LTP unavailable — fallback exit at %.2f", symbol, ltp)
            self._exit_trade(trade, ltp, reason=reason)

    # ──────────────────────────────────────────────────────────
    #  Exit trade
    # ──────────────────────────────────────────────────────────
    def _exit_trade(self, trade: Trade, exit_price: float, reason: str):
        pnl = trade.unrealised_pnl(exit_price)

        if BOT_MODE == "live":
            try:
                self.broker.place_market_sell(
                    trade.security_id, trade.qty, trade.mode)
            except Exception as e:
                log.error("EXIT ORDER FAILED %s: %s — recording P&L anyway", trade.symbol, e)
                _send(f"❌ <b>EXIT ORDER FAILED</b> {trade.symbol}\n{e}")
        else:
            self.paper_pnl += pnl
            log.info("[PAPER] SELL %s @ %.2f  P&L=₹%+.0f", trade.symbol, exit_price, pnl)

        self.risk.update_pnl(pnl)
        self.engine.reset_symbol(trade.symbol)

        _log_trade_csv({
            "time":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "mode":     BOT_MODE.upper(),
            "symbol":   trade.symbol,
            "action":   "EXIT",
            "side":     trade.side,
            "qty":      trade.qty,
            "price":    exit_price,
            "sl":       trade.stop_loss,
            "target":   trade.target,
            "prob_up":  "",
            "rr":       trade.rr,
            "pnl":      round(pnl, 2),
        })

        log.info(
            "EXIT %-14s reason=%-25s exit=%.2f  entry=%.2f"
            "  P&L=₹%+.0f  held=%.0fm  trails=%d",
            trade.symbol, reason, exit_price, trade.entry,
            pnl, trade.hold_minutes(), trade.trail_count,
        )

        alert_exit(
            symbol     = trade.symbol,
            buy_price  = trade.entry,
            sell_price = exit_price,
            quantity   = trade.qty,
            pnl        = pnl,
            reason     = reason,
            trade_mode = trade.mode,
        )

        self.closed_trades.append({
            "symbol": trade.symbol,
            "entry":  trade.entry,
            "exit":   exit_price,
            "qty":    trade.qty,
            "pnl":    round(pnl, 2),
            "reason": reason,
            "held_min": round(trade.hold_minutes(), 1),
        })
        # ── Quant analytics log ─────────────────────────────
        slippage = abs(exit_price - trade.entry)

        trade_duration = round(trade.hold_minutes(), 1)

        market_regime = (
            "BULL"
            if trade.last_prob >= 0.70
            else "SIDEWAYS"
            if trade.last_prob >= 0.55
            else "WEAK"
        )

        self.log_trade_analysis(
            symbol=trade.symbol,
            side=trade.side,
            probability=trade.last_prob,
            entry_price=trade.entry,
            exit_price=exit_price,
            qty=trade.qty,
            pnl=pnl,
            slippage=slippage,
            trade_duration=trade_duration,
            market_regime=market_regime,
            reason=reason,
        )
        del self.trades[trade.symbol]
    
    # ──────────────────────────────────────────────────────────
    #  New trade log
    # ──────────────────────────────────────────────────────────    
    def log_trade_analysis(
        self,
        symbol,
        side,
        probability,
        entry_price,
        exit_price,
        qty,
        pnl,
        slippage,
        trade_duration,
        market_regime,
        reason,
    ):
        """
        Quant-grade trade analytics logger
        """

        file_exists = TRADE_ANALYSIS_LOG.exists()

        with open(TRADE_ANALYSIS_LOG, "a", newline="") as f:

            writer = csv.writer(f)

            # Write header once
            if not file_exists:

                writer.writerow([
                    "datetime",
                    "symbol",
                    "side",
                    "probability",
                    "entry_price",
                    "exit_price",
                    "qty",
                    "pnl",
                    "slippage",
                    "trade_duration",
                    "market_regime",
                    "reason",
                ])

            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                symbol,
                side,
                round(probability, 4),
                round(entry_price, 2),
                round(exit_price, 2),
                qty,
                round(pnl, 2),
                round(slippage, 2),
                trade_duration,
                market_regime,
                reason,
            ])
    # ──────────────────────────────────────────────────────────
    #  Daily EOD reset
    # ──────────────────────────────────────────────────────────
    def _eod_reset(self):
        """
        Sends daily P&L summary, resets all intraday state.
        Called once after 15:30 on any trading day.
        """
        # Summarise today
        total_pnl = self.risk.daily_pnl
        n_trades  = len(self.closed_trades)
        winners   = [t for t in self.closed_trades if t["pnl"] > 0]
        losers    = [t for t in self.closed_trades if t["pnl"] <= 0]

        alert_daily_summary(
            total_pnl = total_pnl,
            trades    = self.closed_trades,
            capital   = CAPITAL + total_pnl,
        )

        if BOT_MODE == "paper":
            avg_win  = sum(t["pnl"] for t in winners) / len(winners)  if winners else 0
            avg_loss = sum(t["pnl"] for t in losers)  / len(losers)   if losers  else 0
            _send(
                f"📊 <b>PAPER DAY COMPLETE</b>\n"
                f"P&L       : ₹{self.paper_pnl:+,.0f}\n"
                f"Trades    : {n_trades}  "
                f"(W={len(winners)} L={len(losers)})\n"
                f"Avg win   : ₹{avg_win:+.0f}\n"
                f"Avg loss  : ₹{avg_loss:+.0f}\n"
                f"SL blacklist: {', '.join(self.sl_blacklist) or 'none'}"
            )

        # Reset state
        self.closed_trades   = []
        self.paper_pnl       = 0.0
        self.sl_blacklist.clear()
        self._last_scan_ts   = 0.0
        self._nifty_refreshed = False
        self.risk.reset_day()
        log.info("EOD reset complete.")

    # ──────────────────────────────────────────────────────────
    #  Main loop
    # ──────────────────────────────────────────────────────────
    def run(self):
        log.info("=" * 60)
        log.info("  %s", MODE_LABEL[BOT_MODE])
        log.info("  Capital: ₹%s | Type: %s | Max/sector: %d",
                 f"{CAPITAL:,}", TRADE_MODE.upper(), MAX_PER_SECTOR)
        log.info("  Monitor: %ds | Scan: %ds | Stocks: %d",
                 MONITOR_INTERVAL, SCAN_INTERVAL, len(WATCHLIST))
        log.info("=" * 60)

        self.risk.reset_day()
        self.sl_blacklist.clear()
        self._last_scan_ts = 0.0

        alert_bot_started(
            capital    = CAPITAL,
            trade_mode = f"{TRADE_MODE.upper()} [{BOT_MODE.upper()}]",
            watchlist  = list(WATCHLIST.keys()),
        )

        if BOT_MODE == "paper":
            _send(
                f"📋 <b>PAPER MODE STARTED</b>\n"
                f"Capital : ₹{CAPITAL:,}\n"
                f"Monitor : every {MONITOR_INTERVAL}s\n"
                f"Scan    : every {SCAN_INTERVAL//60} min\n"
                f"Sectors : max {MAX_PER_SECTOR} per sector\n"
                f"Stocks  : {len(WATCHLIST)}\n"
                f"Set BOT_MODE=live when paper results are consistent."
            )

        daily_summary_sent = False

        while True:
            try:
                # ── Market closed ─────────────────────────────
                if not _is_market_open():
                    if not daily_summary_sent and \
                            _now_time() >= _parse_time(EOD_RESET_TIME):
                        self._eod_reset()
                        daily_summary_sent = True

                    log.info("Market closed (%s). Sleeping 60s...",
                             datetime.now().strftime("%H:%M"))
                    time.sleep(60)
                    continue

                daily_summary_sent = False

                # ── Circuit breaker ───────────────────────────
                if self.risk.is_halted():
                    alert_circuit_breaker(
                        daily_loss = self.risk.daily_pnl,
                        capital    = CAPITAL + self.risk.daily_pnl,
                    )
                    log.critical("CIRCUIT BREAKER ACTIVE — halted for today.")
                    time.sleep(300)
                    continue

                # ── EOD CNC cutoff ────────────────────────────
                if _is_cutoff_passed():
                    self._force_exit_all(reason="CNC_EOD_CUTOFF")
                    log.info("Past CNC cutoff — all positions closed.")
                    time.sleep(60)
                    continue

                # ── Per-tick: monitor open positions ──────────
                self._monitor_positions()

                # ── Per-scan: find new entries ─────────────────
                now_ts = time.time()
                if now_ts - self._last_scan_ts >= SCAN_INTERVAL:
                    log.info("── Scan cycle [%s] %s ──",
                             BOT_MODE.upper(),
                             datetime.now().strftime("%H:%M:%S"))
                    self._refresh_nifty()
                    self._sync_with_dhan()
                    self._scan_and_enter()
                    self._last_scan_ts = now_ts

                time.sleep(MONITOR_INTERVAL)

            except KeyboardInterrupt:
                log.warning("KeyboardInterrupt — exiting gracefully.")
                if self.trades:
                    log.warning("Open trades at shutdown: %s",
                                list(self.trades.keys()))
                    _send(
                        f"⚠️ <b>BOT SHUTDOWN (keyboard)</b>\n"
                        f"Open trades: {', '.join(self.trades.keys())}\n"
                        f"Close them manually if needed."
                    )
                break

            except Exception as e:
                log.error("Main loop exception: %s", e, exc_info=True)
                _send(f"⚠️ <b>LOOP ERROR</b>\n{e}")
                time.sleep(30)   # brief pause before retry


# ─────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    bot = LiveBot()
    if BOT_MODE == "test":
        bot.run_test()
    else:
        bot.run()
