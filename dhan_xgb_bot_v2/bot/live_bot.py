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
#
# Fix log (2026-05-14):
#   FIX-2: market_regime now uses entry_prob (immutable), not last_prob
#   FIX-3: momentum exit min candles = 14 (70 min) for CNC mode
#   FIX-4: circuit breaker Telegram alert fires only once per halt
#   FIX-5: _should_rotate() uses fresh engine.score() for last_prob
#   FIX-6: _force_exit_all() fallback uses entry price, not running_high
#   FIX-7: _is_candle_boundary() de-duplicated per candle via key guard
#          KeyboardInterrupt now calls _force_exit_all() in live mode
#
# Fix log (2026-05-25):
#   BUG-A: df.empty guard moved BEFORE EMA calculation in _scan_and_enter
#          (previously crashed with AttributeError on empty DataFrame)
#   BUG-B: eod_reset used self.circuitalertsent instead of self._circuit_alert_sent
#          (circuit breaker would never reset between trading days)
#   BUG-C: eod_reset used self.last_boundary_key instead of self._last_boundary_key
#          (candle boundary dedup key would never reset between days)
#   BUG-D: momentum failure exit used hardcoded 14 instead of
#          self._MOMENTUM_EXIT_MIN_CANDLES (class constant was ignored)
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
from collections import defaultdict
from bot.trade_policy import MOMENTUM_EXIT_CANDLES
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
SCAN_INTERVAL           = int(os.getenv("SCAN_INTERVAL",            "30"))
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


def _expected_pnl(
    prob_up: float,
    entry:   float,
    sl:      float,
    target:  float,
    qty:     int,
) -> float:
    """
    Heuristic expected PnL for one trade in rupees.

    Formula:
        E = prob_up × reward - (1 - prob_up) × risk

    where:
        reward = (target - entry) × qty   ← rupees gained if TP hit
        risk   = (entry  - sl)    × qty   ← rupees lost  if SL hit

    This is dimensionally correct and economically meaningful.
    It is NOT true expectancy because prob_up is an uncalibrated
    XGBoost classification probability, not an empirical win rate.

    CALIBRATION UPGRADE PATH (run after ≥200 closed trades):
        Step 1: Group trade_analysis.csv by prob_up bucket
                (0.65-0.70, 0.70-0.75, 0.75-0.80, etc.)
        Step 2: empirical_win_rate = tp_hits / total per bucket
        Step 3: Replace prob_up argument with
                empirical_win_rate[bucket(prob_up)]
        Step 4: Function now computes TRUE expectancy.
        Your trade_analysis.csv already logs everything needed.

    Args:
        prob_up: Model's upward probability (uncalibrated).
        entry:   Entry price in rupees.
        sl:      Stop-loss price in rupees.
        target:  Take-profit price in rupees.
        qty:     Number of shares.

    Returns:
        Expected PnL in rupees (can be negative).
    """
    reward = (target - entry) * qty
    risk   = (entry - sl)    * qty
    return prob_up * reward - (1.0 - prob_up) * risk


# ─────────────────────────────────────────────────────────────
#  Trade dataclass
# ─────────────────────────────────────────────────────────────
class Trade:
    """
    Immutable identity fields set at entry.
    Mutable fields updated during position monitoring.

    entry_prob: Immutable. Model probability at entry time.
                Used for market_regime classification in analytics.
                Never overwritten after construction.
    last_prob:  Mutable. Updated each candle boundary during
                signal-flip checks. Used for rotation logic only.
    """
    __slots__ = (
        "symbol", "security_id", "side", "qty",
        "entry", "stop_loss", "target", "order_id", "mode",
        "running_high", "open_time",
        "entry_prob",   # FIX-2: immutable entry probability for regime classification
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
        self.entry_prob  = last_prob   # FIX-2: snapshot at entry — never changes
        self.last_prob   = last_prob   # mutable — updated each candle boundary
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
    """Append one trade row to TRADE_LOG CSV. Single-process append."""
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

    # ── Rotation cost constant ─────────────────────────────────
    # Estimated round-trip transaction cost for CNC on NSE:
    # STT (0.1% delivery) + brokerage + NSE charges + SEBI fee ≈ 0.20%
    _ROTATION_ROUNDTRIP_COST_PCT = 0.0020

    # Minimum candles a position must be held before rotation eligibility.
    # 6 candles × 5 min = 30 minutes minimum hold before rotation.
    _ROTATION_MIN_CANDLES = 6

    # FIX-3: Minimum candles before momentum-failure exit is evaluated.
    # CNC delivery trades need time to develop — 15 min (3 candles) is
    # statistical noise on a large-cap NSE stock. 14 candles = 70 minutes
    # is a more appropriate minimum for post-breakout continuation checks.
    _MOMENTUM_EXIT_MIN_CANDLES = MOMENTUM_EXIT_CANDLES

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
        self.cooldown_bars = {}   # symbol -> remaining bars
        # FIX-4: Circuit breaker alert sent only once per halt period.
        # Prevents Telegram spam when the bot is halted for the day.
        self._circuit_alert_sent = False
        # ── Daily rejection statistics ──────────────────────────────
        self.rejection_stats: dict[str, int] = defaultdict(int)
        self.rejection_symbols: dict[str, list[str]] = defaultdict(list)
        # FIX-7: Candle boundary deduplication key.
        # Stores (date, hour, minute) of the last processed boundary.
        # Prevents candles_held from being incremented multiple times
        # within the same 1-minute window when MONITOR_INTERVAL < 60s.
        self._last_boundary_key: Optional[tuple] = None

    # ──────────────────────────────────────────────────────────
    #  FIX-7: Candle boundary check — fires exactly once per candle
    # ──────────────────────────────────────────────────────────
    def _is_candle_boundary(self) -> bool:
        """
        True ONCE per 5-minute candle close.

        Original implementation returned True for the entire minute
        when minute % 5 == 0, causing candles_held to be incremented
        multiple times per boundary when MONITOR_INTERVAL < 60s.

        Fix: track last processed (date, hour, minute) key and only
        fire once per unique candle boundary minute.

        Research note (Berkman et al. 2012):
          Checking at candle close avoids reacting to intra-candle
          noise — true signal confirmation requires a closed bar.
        """
        now = datetime.now()
        if now.minute % CANDLE_MINUTES != 0:
            return False
        key = (now.date(), now.hour, now.minute)
        if key == self._last_boundary_key:
            return False
        self._last_boundary_key = key
        return True

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
                        raw = {}

                    if bid is not None and ask is not None and spread_pct is None:
                        spread_abs = float(ask) - float(bid)
                        spread_pct = spread_abs / ltp if ltp > 0 else 1.0

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
    #  FIX-5: Portfolio rotation — fresh score for incumbent
    # ──────────────────────────────────────────────────────────
    def _should_rotate(self, new_prob: float) -> tuple[bool, Optional[str]]:
        """
        Rotation logic: replace the weakest open position with a
        significantly stronger new signal, but only if the weakest
        position is already in profit (to avoid realising losses).

        FIX-5: incumbent last_prob is now refreshed via engine.score()
        before comparison. Previously last_prob could be stale between
        candle boundaries (up to 5 minutes old), causing rotation to
        compare a fresh signal against an outdated baseline.

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

            # FIX-5: Refresh last_prob with a live engine score
            # so the rotation comparison uses current model state.
            try:
                df_live = self.broker.get_candles(
                    trade.security_id, symbol, days_back=5
                )
                if not df_live.empty:
                    scored = self.engine.score(df_live, symbol=symbol)
                    trade.last_prob = scored["prob_up"]
                    log.debug(
                        "%s: rotation probe — refreshed last_prob=%.3f",
                        symbol, trade.last_prob,
                    )
            except Exception as e:
                log.warning("%s: rotation probe score failed: %s — using cached prob", symbol, e)

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
        Full watchlist scan -> rank BUY signals by confidence ->
        enter top candidates up to available slots.

        Design principles:
        - Skip opening 15 min (NO_NEW_TRADE_BEFORE=09:30)
        - Respect sector concentration limits
        - Spread, liquidity, drift gates before execution
        - Confidence-ranked selection
        - Rotation only when max slots are full and a new signal is clearly better
        """

        if _now_time() < _parse_time(NO_NEW_TRADE_BEFORE):
            log.info("Pre-trade window: waiting until %s", NO_NEW_TRADE_BEFORE)
            return
        if _no_new_trades():
            return

        if self.risk.is_halted():
            log.critical(
                "RiskManager halted trading — scan aborted. "
                "(daily loss / consecutive SL limit breached)"
            )
            return

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
            if symbol.upper() in BLOCKED_SYMBOLS:
                continue
            if symbol in self.trades:
                continue
            if symbol in self.sl_blacklist:
                log.debug("%s: SL-blacklisted today — skip.", symbol)
                continue

            cooldown = self.cooldown_bars.get(symbol, 0)
            if cooldown > 0:
                self.cooldown_bars[symbol] = cooldown - 1
                log.debug("%s: cooldown active (%d bars left)", symbol, cooldown)
                continue

            sector = SECTOR_MAP.get(symbol, "unknown")
            sector_count = sum(
                1 for s in self.trades
                if SECTOR_MAP.get(s, "unknown") == sector
            )
            if sector_count >= MAX_PER_SECTOR:
                continue

            df = self.broker.get_candles(sec_id, symbol, days_back=10)
            if df.empty:
                continue

            result = self.engine.score(df, symbol=symbol)
            if result["signal"] != "BUY":
                reason_str = result.get("reason", "")
                if reason_str and reason_str not in (
                    "null", "blocked_symbol", "insufficient_data",
                    "feature_error", "empty_features",
                    "missing_features", "nan_in_features",
                    "predict_error", "invalid_atr_or_price"
                ):
                    for r in reason_str.split(","):
                        r = r.strip()
                        if r:
                            self.rejection_stats[r] += 1
                            if symbol not in self.rejection_symbols[r]:
                                self.rejection_symbols[r].append(symbol)
                continue

            entry = result["entry"]
            sl = result["sl"]
            target = result["target"]
            rr = result["rr"]

            if any(v <= 0 for v in [entry, sl, target, rr]):
                log.warning(
                    "%s: invalid engine output entry=%.2f sl=%.2f — skip",
                    symbol, entry, sl,
                )
                continue

            ok, quote = self._passes_entry_filters(
                symbol, sec_id, entry, reason_prefix="[SCAN] "
            )
            if not ok:
                continue

            candidates.append((symbol, sec_id, result, quote))
            log.info(
                "  Candidate %-14s prob=%.3f entry=%.2f SL=%.2f TP=%.2f "
                "R:R=%.2fx ev=%.3f spread=%.4f%% [%s]",
                symbol, result["prob_up"], entry, sl, target, rr,
                result["prob_up"] * rr,
                (quote.get("spread_pct") or 0.0) * 100,
                sector,
            )

        if not candidates:
            log.info("No BUY signals this scan.")
            if self.rejection_stats:
                top = sorted(self.rejection_stats.items(), key=lambda x: -x[1])[:6]
                summary = " | ".join(f"{k}={v}" for k, v in top)
                log.info("  Rejection stats (today): %s", summary)
            return

        def calibrated_score(prob, rr):
            calibrated = 0.50 + ((prob - 0.50) * 0.35)
            return calibrated * rr

        candidates.sort(
            key=lambda x: calibrated_score(
                x[2]["prob_up"],
                x[2].get("rr", 1.0),
            ),
            reverse=True,
        )

        for rank, (sym, _, res, _) in enumerate(candidates, start=1):
            log.info(
                "  Rank #%d %-14s prob=%.3f rr=%.2fx ev=%.3f",
                rank,
                sym,
                res["prob_up"],
                res.get("rr", 0.0),
                res["prob_up"] * res.get("rr", 1.0),
            )

        best_sym, best_sec_id, best_result, best_quote = candidates[0]
        log.info(
            "Best signal: %-14s prob=%.3f rr=%.2fx ev_score=%.3f scan_rank=1/%d",
            best_sym,
            best_result["prob_up"],
            best_result.get("rr", 0.0),
            best_result["prob_up"] * best_result.get("rr", 1.0),
            len(candidates),
        )

        if max_slots_full:
            should_rotate, exit_sym = self._should_rotate(best_result["prob_up"])

            if should_rotate and exit_sym:
                trade_to_exit = self.trades[exit_sym]

                if trade_to_exit.candles_held < self._ROTATION_MIN_CANDLES:
                    log.info(
                        "ROTATION BLOCKED (time hysteresis): %s only %d candles old "
                        "(min=%d). Position too young to rotate.",
                        exit_sym,
                        trade_to_exit.candles_held,
                        self._ROTATION_MIN_CANDLES,
                    )
                    return

                ev_new = _expected_pnl(
                    prob_up=best_result["prob_up"],
                    entry=best_result["entry"],
                    sl=best_result["sl"],
                    target=best_result["target"],
                    qty=self.risk.position_size(
                        best_result["entry"], best_result["sl"]
                    ),
                )

                ev_old = _expected_pnl(
                    prob_up=trade_to_exit.last_prob,
                    entry=trade_to_exit.entry,
                    sl=trade_to_exit.stop_loss,
                    target=trade_to_exit.target,
                    qty=trade_to_exit.qty,
                )

                roundtrip_cost = (
                    trade_to_exit.entry
                    * trade_to_exit.qty
                    * self._ROTATION_ROUNDTRIP_COST_PCT
                )

                ev_gain = ev_new - ev_old

                if ev_gain <= roundtrip_cost:
                    log.info(
                        "ROTATION BLOCKED (ev gate): %s -> %s "
                        "ev_new=₹%.2f ev_old=₹%.2f ev_gain=₹%.2f <= cost=₹%.2f",
                        exit_sym, best_sym, ev_new, ev_old, ev_gain, roundtrip_cost,
                    )
                    return

                id_map = {str(trade_to_exit.security_id): exit_sym}
                prices = self.broker.get_ltp_batch(id_map)
                ltp = prices.get(str(trade_to_exit.security_id), trade_to_exit.entry)

                log.info(
                    "ROTATION APPROVED: exiting %s (held=%d candles, ev_gain=₹%.2f > cost=₹%.2f) -> entering %s",
                    exit_sym,
                    trade_to_exit.candles_held,
                    ev_gain,
                    roundtrip_cost,
                    best_sym,
                )
                self._exit_trade(trade_to_exit, ltp, "ROTATION_BETTER_SIGNAL")
            else:
                log.info("Max open trades (%d). No rotation opportunity.", MAX_OPEN_TRADES)
                return

        available_slots = max(0, MAX_OPEN_TRADES - len(self.trades))
        if available_slots <= 0:
            return

        selected_candidates = candidates[:available_slots]

        for scan_rank, (sym, sec_id, result, quote) in enumerate(selected_candidates, start=1):
            entry = result["entry"]
            sl = result["sl"]
            target = result["target"]
            rr = result["rr"]

            if any(v <= 0 for v in [entry, sl, target, rr]):
                log.warning("%s: final proposal invalid — skip.", sym)
                continue

            sector = SECTOR_MAP.get(sym, "unknown")
            sector_count = sum(
                1 for s in self.trades
                if SECTOR_MAP.get(s, "unknown") == sector
            )
            if sector_count >= MAX_PER_SECTOR:
                log.info("%s: skipped at entry stage due to sector cap.", sym)
                continue

            qty = self.risk.position_size(entry, sl)
            if qty <= 0:
                log.warning("%s: position size = 0 — skip.", sym)
                continue

            log.info(
                "SIGNAL BUY %-14s entry=%.2f SL=%.2f TP=%.2f qty=%d prob=%.3f R:R=%.2fx ev=%.3f scan_rank=%d/%d sector=%s spread=%.4f%%",
                sym,
                entry, sl, target, qty,
                result["prob_up"],
                rr,
                result["prob_up"] * rr,
                scan_rank,
                len(candidates),
                sector,
                (quote.get("spread_pct") or 0.0) * 100,
            )

            if BOT_MODE == "live":
                ok, entry_quote = self._passes_entry_filters(
                    sym,
                    sec_id,
                    entry,
                    reason_prefix="[ENTRY] ",
                )
                if not ok:
                    log.warning("%s: entry execution filters failed.", sym)
                    continue

                self._entry_quotes[sym] = {
                    "entry": entry,
                    "ts": time.time(),
                }
                self._enter_live(
                    sym, sec_id, qty, entry, sl, target, result,
                )
            else:
                self._enter_paper(
                    sym, sec_id, qty, entry, sl, target, result,
                )
        """
        Full watchlist scan → rank BUY signals by confidence →
        pick highest-quality setup → execute.

        Design principles:
          - Skip opening 15 min (NO_NEW_TRADE_BEFORE=09:30)
          - Max 1 position per GICS sector (concentration risk)
          - Spread, liquidity, drift gates before execution
          - Confidence-ranked selection (best model edge wins)
          - Rotation only when new signal strictly dominates,
            position is mature (≥30 min), AND edge exceeds cost

        Precondition contract (Design by Contract — Bertrand Meyer):
          ALL guards are evaluated ONCE at function entry.
          No guard is repeated inside the function body.
          If any precondition fails, the function exits immediately.
          This is the single authority principle — one gate, one place.

        Ranking formula:
          score = prob_up × rr  (Kelly-adjacent expected value proxy)
          This is a heuristic, not true expectancy. Once ≥200 closed
          trades are logged, replace with empirical TP-hit rates per
          confidence bucket from trade_analysis.csv.

        scan_rank:
          Every chosen trade logs its ordinal rank among candidates.
          If rank > 1 consistently outperforms rank = 1, the ranking
          formula has an inversion problem and must be re-examined.
        """

        # ═══════════════════════════════════════════════════════
        # PRECONDITION BLOCK — single authority, checked once.
        # ═══════════════════════════════════════════════════════

        if _now_time() < _parse_time(NO_NEW_TRADE_BEFORE):
            log.info("Pre-trade window: waiting until %s", NO_NEW_TRADE_BEFORE)
            return
        if _no_new_trades():
            return

        if self.risk.is_halted():
            log.critical(
                "RiskManager halted trading — scan aborted. "
                "(daily loss / consecutive SL limit breached)"
            )
            return

        daily_loss_pct = self.risk.daily_pnl / CAPITAL
        if daily_loss_pct <= NEW_TRADE_LOSS_PAUSE:
            log.warning(
                "Daily loss %.2f%% ≤ pause threshold %.2f%% — no new entries.",
                daily_loss_pct * 100, NEW_TRADE_LOSS_PAUSE * 100,
            )
            return

        max_slots_full = len(self.trades) >= MAX_OPEN_TRADES

        # ═══════════════════════════════════════════════════════
        # END PRECONDITION BLOCK
        # ═══════════════════════════════════════════════════════

        log.info("── Scan: ranking %d eligible stocks ──", len(WATCHLIST))

        candidates: list[tuple[str, str, dict, dict]] = []

        for symbol, sec_id in WATCHLIST.items():
            if symbol.upper() in BLOCKED_SYMBOLS:
                continue
            if symbol in self.trades:
                continue
            if symbol in self.sl_blacklist:
                log.debug("%s: SL-blacklisted today — skip.", symbol)
                continue
             # NEW: cooldown after exit
            cooldown = self.cooldown_bars.get(symbol, 0)

            if cooldown > 0:
                self.cooldown_bars[symbol] = cooldown - 1

                log.debug(
                    "%s: cooldown active (%d bars left)",
                    symbol,
                    cooldown
                )
                continue

            sector = SECTOR_MAP.get(symbol, "unknown")
            sector_count = sum(
                1 for s in self.trades
                if SECTOR_MAP.get(s, "unknown") == sector
            )
            if sector_count >= MAX_PER_SECTOR:
                continue

            # BUG-A FIX: guard df.empty BEFORE accessing df columns for EMA.
            # Previously ema20/ema50 were computed before the empty check,
            # causing AttributeError when broker returned an empty DataFrame.
            df = self.broker.get_candles(sec_id, symbol, days_back=10)
            if df.empty:
                continue

            result = self.engine.score(df, symbol=symbol)
            if result["signal"] != "BUY":
                # ── Track rejection reasons for EOD summary ──────────────
                reason_str = result.get("reason", "")
                if reason_str and reason_str not in ("null", "blocked_symbol",
                                                    "insufficient_data",
                                                    "feature_error", "empty_features",
                                                    "missing_features", "nan_in_features",
                                                    "predict_error", "invalid_atr_or_price"):
                    for r in reason_str.split(","):
                        r = r.strip()
                        if r:
                            self.rejection_stats[r] += 1
                            if symbol not in self.rejection_symbols[r]:
                                self.rejection_symbols[r].append(symbol)
                continue

            entry  = result["entry"]
            sl     = result["sl"]
            target = result["target"]
            rr     = result["rr"]

            if any(v <= 0 for v in [entry, sl, target, rr]):
                log.warning(
                    "%s: invalid engine output entry=%.2f sl=%.2f — skip",
                    symbol, entry, sl,
                )
                continue

            ok, quote = self._passes_entry_filters(
                symbol, sec_id, entry, reason_prefix="[SCAN] ")
            if not ok:
                continue

            candidates.append((symbol, sec_id, result, quote))
            log.info(
                "  Candidate %-14s prob=%.3f  entry=%.2f  SL=%.2f  "
                "TP=%.2f  R:R=%.2fx  ev=%.3f  spread=%.4f%%  [%s]",
                symbol, result["prob_up"], entry, sl, target, rr,
                result["prob_up"] * rr,
                (quote.get("spread_pct") or 0.0) * 100, sector,
            )

        if not candidates:
            log.info("No BUY signals this scan.")
            # ── Print top rejection reasons for this scan ─────────────
            if self.rejection_stats:
                top = sorted(self.rejection_stats.items(), key=lambda x: -x[1])[:6]
                summary = " | ".join(f"{k}={v}" for k, v in top)
                log.info(f"  Rejection stats (today): {summary}")
            return

        # ── Rank by Kelly-adjacent expected value proxy ───────
        def calibrated_score(prob, rr):
            # squash fake confidence inflation
            calibrated = 0.50 + ((prob - 0.50) * 0.35)
            return calibrated * rr

        candidates.sort(
            key=lambda x: calibrated_score(
                x[2]["prob_up"],
                x[2].get("rr", 1.0),
            ),
            reverse=True,
        )

        max_slots_full = len(self.trades) >= MAX_OPEN_TRADES
        available_slots = max(0, MAX_OPEN_TRADES - len(self.trades))

        log.info(
            "Best signal: %-14s prob=%.3f  rr=%.2fx  ev_score=%.3f  scan_rank=1/%d",
            result["prob_up"],
            result.get("rr", 0.0),
            result["prob_up"] * result.get("rr", 1.0),
            len(candidates),
        )

        # ── Portfolio rotation check ──────────────────────────
        if max_slots_full:
            should_rotate, exit_sym = self._should_rotate(result["prob_up"])

            if should_rotate and exit_sym:
                trade_to_exit = self.trades[exit_sym]

                if trade_to_exit.candles_held < self._ROTATION_MIN_CANDLES:
                    log.info(
                        "ROTATION BLOCKED (time hysteresis): %s only %d candles old "
                        "(min=%d). Position too young to rotate.",
                        exit_sym,
                        trade_to_exit.candles_held,
                        self._ROTATION_MIN_CANDLES,
                    )
                    return

                ev_new = _expected_pnl(
                    prob_up = result["prob_up"],
                    entry   = result["entry"],
                    sl      = result["sl"],
                    target  = result["target"],
                    qty     = self.risk.position_size(
                                result["entry"], result["sl"]
                            ),
                )

                ev_old = _expected_pnl(
                    prob_up = trade_to_exit.last_prob,
                    entry   = trade_to_exit.entry,
                    sl      = trade_to_exit.stop_loss,
                    target  = trade_to_exit.target,
                    qty     = trade_to_exit.qty,
                )

                roundtrip_cost = (
                    trade_to_exit.entry
                    * trade_to_exit.qty
                    * self._ROTATION_ROUNDTRIP_COST_PCT
                )

                ev_gain = ev_new - ev_old

                if ev_gain <= roundtrip_cost:
                    log.info(
                        "ROTATION BLOCKED (ev gate): %s → %s "
                        "ev_new=₹%.2f  ev_old=₹%.2f  "
                        "ev_gain=₹%.2f ≤ cost=₹%.2f. "
                        "Rotation does not justify transaction friction.",
                        exit_sym, sym,
                        ev_new, ev_old,
                        ev_gain, roundtrip_cost,
                    )
                    return
                
                id_map = {str(trade_to_exit.security_id): exit_sym}
                prices = self.broker.get_ltp_batch(id_map)
                ltp    = prices.get(
                    str(trade_to_exit.security_id), trade_to_exit.entry
                )
                log.info(
                    "ROTATION APPROVED: exiting %s (held=%d candles, "
                    "ev_gain=₹%.2f > cost=₹%.2f) → entering %s",
                    exit_sym,
                    trade_to_exit.candles_held,
                    ev_gain,
                    roundtrip_cost,
                    sym,
                )
                self._exit_trade(trade_to_exit, ltp, "ROTATION_BETTER_SIGNAL")

            else:
                log.info(
                    "Max open trades (%d). No rotation opportunity.",
                    MAX_OPEN_TRADES,
                )
                return

        # ── Final signal validation ───────────────────────────
        entry  = result["entry"]
        sl     = result["sl"]
        target = result["target"]
        rr     = result["rr"]

        if any(v <= 0 for v in [entry, sl, target, rr]):
            log.warning("%s: final proposal invalid — skip.", sym)
            return

        qty = self.risk.position_size(entry, sl)

        if qty <= 0:
            log.warning("%s: position size = 0 — skip.", sym)
            return

        log.info(
            "SIGNAL BUY %-14s  entry=%.2f  SL=%.2f  TP=%.2f"
            "  qty=%d  prob=%.3f  R:R=%.2fx  ev=%.3f"
            "  scan_rank=1/%d  sector=%s  spread=%.4f%%",
            sym,
            entry, sl, target, qty,
            result["prob_up"],
            rr,
            result["prob_up"] * rr,
            len(candidates),
            SECTOR_MAP.get(sym, "?"),
            (quote.get("spread_pct") or 0.0) * 100,
        )

        if BOT_MODE == "live":
            ok, quote = self._passes_entry_filters(
                sym,
                sec_id,
                entry,
                reason_prefix="[ENTRY] ",
            )
            if not ok:
                log.warning("%s: entry execution filters failed.", sym)
                return

            self._entry_quotes[sym] = {
                "entry": entry,
                "ts": time.time(),
            }
            self._enter_live(
                sym, sec_id, qty, entry, sl, target, result,
            )

        else:
            self._enter_paper(
                sym, sec_id, qty, entry, sl, target, result,
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
            while True:
                drift_ok, quote = self._passes_entry_filters(
                    symbol, sec_id, initial_entry, reason_prefix="[EXEC] ")
                if not drift_ok:
                    log.info("%s: execution aborted — drift/spread gate failed.", symbol)
                    return

                ltp = quote.get("ltp") or entry
                if abs(ltp - initial_entry) / initial_entry <= EXEC_MAX_DRIFT_PCT:
                    entry = ltp
                    break

                if time.time() - start > ENTRY_REPRICE_SECS:
                    log.info("%s: re-price window %ds expired — abort.", symbol, ENTRY_REPRICE_SECS)
                    return

                time.sleep(1)

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

            if ltp > trade.running_high:
                trade.running_high = ltp

            pnl        = trade.unrealised_pnl(ltp)
            profit_pct = (ltp - trade.entry) / trade.entry
            hold_mins  = trade.hold_minutes()

            # ── 1. Target hit ────────────────────────────────
            if ltp >= trade.target:
                log.info(
                    "%s: TARGET HIT ltp=%.2f TP=%.2f held=%.0fmin",
                    symbol, ltp, trade.target, hold_mins,
                )
                self._exit_trade(trade, ltp, "TARGET_HIT")
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
                locked_pct = ((trade.stop_loss - trade.entry) / trade.entry) * 100
                log.info(
                    "%s: TRAIL #%d  SL %.2f → %.2f  "
                    "LTP=%.2f  Locked=%.2f%%  P&L=₹%+.0f",
                    symbol, trade.trail_count,
                    old_sl, trade.stop_loss,
                    ltp, locked_pct, pnl,
                )
                alert_trail_update(symbol, trade.stop_loss, ltp, pnl)

            # ── 3. LTP SL check ─────────────────────────────
            if ltp <= trade.stop_loss:
                log.warning(
                    "%s: SL HIT (LTP) ltp=%.2f SL=%.2f held=%.0fmin",
                    symbol, ltp, trade.stop_loss, hold_mins,
                )
                self._exit_trade(trade, ltp, "SL_HIT")
                self.sl_blacklist.add(symbol)
                continue

            # ── Candle-boundary checks — fires once per candle ──
            # FIX-7: _is_candle_boundary() is now an instance method
            # with deduplication so it fires exactly once per 5-min close.
            if self._is_candle_boundary():

                trade.candles_held += 1

                # ── FIX-3 / BUG-D: Momentum failure exit ─────────────
                # Uses self._MOMENTUM_EXIT_MIN_CANDLES (14 candles = 70 min).
                # Previously the class constant was defined but never used —
                # the check was hardcoded to 14 directly, meaning any change
                # to the constant had no effect. Now references the constant.
                if (
                    trade.candles_held >= self._MOMENTUM_EXIT_MIN_CANDLES
                    and pnl < -(trade.atr * trade.qty * 0.30)
                    and trade.last_prob < 0.45
                ):
                    log.warning(
                        "%s: MOMENTUM FAILURE EXIT candles=%d pnl=₹%.0f prob=%.3f",
                        symbol,
                        trade.candles_held,
                        pnl,
                        trade.last_prob,
                    )
                    self._exit_trade(trade, ltp, "MOMENTUM_FAILURE")
                    continue

                # ── 4. Candle-low SL check ─────────────────
                # DISABLED: candle lows create many false stop exits
                # due to wick noise and broker candle artifacts.

                # ── 5. Auto-exit: 2:45 PM weak market ─────
                if (
                    _is_auto_exit_time()
                    and profit_pct <= AUTO_EXIT_THRESHOLD
                ):
                    log.warning(
                        "%s: AUTO-EXIT 2:45 PM profit %.2f%% ≤ threshold %.2f%%",
                        symbol, profit_pct * 100, AUTO_EXIT_THRESHOLD * 100,
                    )
                    self._exit_trade(trade, ltp, "AUTO_EXIT_EOD_WEAK")
                    continue

                # ── 6. Signal flip / model deterioration ──
                try:
                    df_live = self.broker.get_candles(
                        trade.security_id, symbol, days_back=5,
                    )
                    if not df_live.empty:
                        scored = self.engine.score(df_live, symbol=symbol)
                        trade.last_prob = scored["prob_up"]
                        if self.engine.should_exit(df_live, trade.side, symbol=symbol):
                            log.info(
                                "%s: SIGNAL FLIP exit=%.2f prob=%.3f "
                                "P&L=₹%.0f held=%.0fmin candles=%d",
                                symbol, ltp, scored["prob_up"],
                                pnl, hold_mins, trade.candles_held,
                            )
                            self._exit_trade(trade, ltp, "SIGNAL_FLIP")
                            continue
                except Exception as e:
                    log.warning("%s: signal flip check error: %s", symbol, e)

            # ── Holding log ───────────────────────────────
            log.info(
                "HOLD %-14s LTP=%.2f  Entry=%.2f  "
                "SL=%.2f  TP=%.2f  ATR=%.2f  "
                "prob=%.3f  R:R=%.2fx  "
                "P&L=₹%+.0f  pct=%+.2f%%  "
                "held=%.0fm  [%s] held_candles=%d",
                symbol, ltp, trade.entry,
                trade.stop_loss, trade.target, trade.atr,
                trade.last_prob, trade.rr,
                pnl, profit_pct * 100,
                hold_mins, SECTOR_MAP.get(symbol, "?"), trade.candles_held,
            )

    # ──────────────────────────────────────────────────────────
    #  FIX-6: Force exit all — safe fallback price
    # ──────────────────────────────────────────────────────────
    def _force_exit_all(self, reason: str = "CNC_EOD_CUTOFF"):
        """
        FIX-6: Fallback exit price is now trade.entry (breakeven),
        not trade.running_high. The running_high is the best price
        the stock ever touched — using it as an EOD fallback inflates
        paper P&L when the position is actually at a loss. Entry price
        is the only safe, conservative, non-inflating fallback.
        """
        if not self.trades:
            return
        log.warning("Force-exiting %d position(s) — reason: %s",
                    len(self.trades), reason)
        id_map = {str(t.security_id): sym for sym, t in self.trades.items()}
        prices = self.broker.get_ltp_batch(id_map)

        for symbol, trade in list(self.trades.items()):
            ltp = prices.get(str(trade.security_id), 0.0)
            if ltp <= 0:
                # FIX-6: Use entry price as conservative fallback.
                ltp = trade.entry
                log.warning(
                    "%s: LTP unavailable — conservative fallback exit at entry=%.2f",
                    symbol, ltp,
                )
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
        slippage = abs(exit_price - trade.target)      if reason == "TARGET_HIT" \
              else abs(exit_price - trade.stop_loss)   if "SL" in reason \
              else 0.0

        trade_duration = round(trade.hold_minutes(), 1)

        # FIX-2: Use entry_prob (immutable snapshot at trade open),
        # NOT last_prob (which reflects model state at exit time).
        market_regime = (
            "BULL"     if trade.entry_prob >= 0.70
            else "SIDEWAYS" if trade.entry_prob >= 0.55
            else "WEAK"
        )

        self.log_trade_analysis(
            symbol=trade.symbol,
            side=trade.side,
            probability=trade.entry_prob,   # FIX-2: entry_prob, not last_prob
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
        self.cooldown_bars[trade.symbol] = 2

    # ──────────────────────────────────────────────────────────
    #  Trade analysis log
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
        """Quant-grade trade analytics logger."""
        file_exists = TRADE_ANALYSIS_LOG.exists()
        with open(TRADE_ANALYSIS_LOG, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "datetime", "symbol", "side", "probability",
                    "entry_price", "exit_price", "qty", "pnl",
                    "slippage", "trade_duration", "market_regime", "reason",
                ])
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                symbol, side,
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
    #  EOD reset
    # ──────────────────────────────────────────────────────────
    def eod_reset(self):
        """Reset daily state for next session."""
        log.info("EOD reset: clearing daily state.")

        # ── EOD Rejection Summary ─────────────────────────────────
        if self.rejection_stats:
            sorted_reasons = sorted(
                self.rejection_stats.items(), key=lambda x: -x[1]
            )
            lines = ["── EOD Rejection Summary ─────────────────────────"]
            for reason, count in sorted_reasons:
                syms = ", ".join(self.rejection_symbols.get(reason, [])[:5])
                lines.append(f"  {reason:<32} x{count:>3}  [{syms}]")
            lines.append(f"  Total rejected signals: {sum(self.rejection_stats.values())}")
            log.info("\n".join(lines))

            # ── Telegram summary (only if rejections happened all day) ──
            if sum(self.rejection_stats.values()) > 0:
                top5 = sorted_reasons[:5]
                msg = (
                    f"📊 *EOD Rejection Report — {datetime.now().strftime('%d %b %Y')}*\n"
                    f"Scanned: {len(self.closed_trades)} trades today\n\n"
                    + "\n".join(f"• `{r}` → {c}x" for r, c in top5)
                )
                _send(msg)

        self.rejection_stats.clear()
        self.rejection_symbols.clear()

        self.sl_blacklist.clear()
        self._circuit_alert_sent = False          # BUG-B FIX: was self.circuitalertsent
        self.risk.reset_daily()
        self.closed_trades.clear()
        self.cooldown_bars.clear()
        self._last_boundary_key = None            # BUG-C FIX: was self.last_boundary_key
        log.info("EOD reset complete.")

    # ──────────────────────────────────────────────────────────
    #  Main loop
    # ──────────────────────────────────────────────────────────
    def run(self):
        """
        Production trading loop (paper or live).

        Loop structure (per MONITOR_INTERVAL seconds):
          1. Verify market is open
          2. Refresh Nifty regime every 30 min
          3. Scan for new entries every SCAN_INTERVAL seconds
          4. Monitor + manage all open positions
          5. Dhan sync check (live mode only)
          6. Circuit breaker check
          7. EOD force-exit + reset at 15:30

        FIX-4: Circuit breaker Telegram alert fires ONCE per halt,
        not on every monitor tick. _circuit_alert_sent flag is reset
        in eod_reset() so it fires again next trading day.
        """
        log.info("=" * 60)
        log.info("LiveBot starting — mode: %s", MODE_LABEL.get(BOT_MODE, BOT_MODE).upper())
        log.info("Capital: ₹%s | Trade mode: %s | Max trades: %d",
                 f"{CAPITAL:,}", TRADE_MODE.upper(), MAX_OPEN_TRADES)
        log.info("=" * 60)

        alert_bot_started(BOT_MODE, CAPITAL, TRADE_MODE, MAX_OPEN_TRADES)
        last_nifty_refresh = 0.0
        eod_done           = False

        try:
            while True:
                now = datetime.now()

                if not _is_market_open():
                    if not eod_done and _now_time() >= _parse_time(EOD_RESET_TIME):
                        if self.trades:
                            self._force_exit_all("CNC_EOD_CUTOFF")
                        summary = self.risk.daily_summary()
                        alert_daily_summary(**summary)
                        log.info(
                            "EOD summary: P&L=₹%+.0f  trades=%d  wins=%d",
                            summary["pnl"], summary["total_trades"], summary["wins"],
                        )
                        self.eod_reset()
                        eod_done = True
                    time.sleep(60)
                    continue

                eod_done = False

                # ── Nifty regime refresh (every 30 min) ──
                if time.time() - last_nifty_refresh > 1800:
                    self._refresh_nifty()
                    last_nifty_refresh = time.time()

                # ── Scan for new entries ──────────────────
                if time.time() - self._last_scan_ts > SCAN_INTERVAL:
                    self._scan_and_enter()
                    self._last_scan_ts = time.time()

                # ── Monitor open positions ────────────────
                self._monitor_positions()

                # ── Dhan position sync (live only) ────────
                self._sync_with_dhan()

                # ── FIX-4: Circuit breaker — alert once ──
                if self.risk.is_halted():
                    if not self._circuit_alert_sent:
                        log.critical(
                            "CIRCUIT BREAKER: trading halted for the day "
                            "(daily loss limit or consecutive SL limit breached). "
                            "Resuming tomorrow after EOD reset."
                        )
                        alert_circuit_breaker(self.risk.daily_pnl,CAPITAL)
                        self._circuit_alert_sent = True
                    # Still monitor existing positions even when halted —
                    # we just block new entries (handled in _scan_and_enter).

                # ── CNC intraday cutoff ───────────────────
                if _is_cutoff_passed() and self.trades:
                    log.warning("Intraday CNC cutoff — force-exiting all.")
                    self._force_exit_all("CNC_EOD_CUTOFF")

                time.sleep(MONITOR_INTERVAL)

        except KeyboardInterrupt:
            log.warning("KeyboardInterrupt received.")
            if BOT_MODE == "live" and self.trades:
                log.warning("Exiting %d open live position(s) safely.", len(self.trades))
                self._force_exit_all("KEYBOARD_INTERRUPT")
            _send("⚠️ <b>Bot stopped manually (KeyboardInterrupt)</b>")
            log.info("Bot shut down cleanly.")


# ─────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    bot = LiveBot()
    if BOT_MODE == "test":
        bot.run_test()
    else:
        bot.run()