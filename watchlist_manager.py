# watchlist_manager.py — dhan_xgb_bot_v2
# =============================================================
# OODA Self-Evolving Watchlist Pipeline
# -------------------------------------------------------------
# OBSERVE  : Scan broader NSE universe via Dhan API every 5 min
# ORIENT   : Score each stock with live XGBoost probability
# DECIDE   : Apply add / prune rules against configurable gates
# ACT      : Atomically rewrite watchlist.json + notify Telegram
# =============================================================
# Architecture decisions (documented for maintainability):
#
#  1. Single-model scoring — we use the existing global XGBoost
#     model to score candidate stocks.  No per-stock inference;
#     the model generalises well across the 35-40 stock universe.
#
#  2. Atomic JSON writes — we write to a temp file and os.replace()
#     so bot.py never reads a half-written file (POSIX atomic).
#     portalocker adds a cross-platform advisory lock for Windows.
#
#  3. Cooldown counters — each stock that was pruned sits in a
#     PRUNE_COOLDOWN_BARS cooldown before it can re-enter. This
#     prevents thrashing in choppy markets.
#
#  4. Risk gates — a stock must pass ALL of:
#        a. prob_up  >= ADD_THRESHOLD (default 0.60)
#        b. avg_vol  >= MIN_DAILY_VOL_CR (200 Cr intraday)
#        c. atr_pct  between ATR_MIN_PCT and ATR_MAX_PCT
#        d. sector   count < MAX_PER_SECTOR in current watchlist
#        e. NOT in BLOCKED_SYMBOLS (permanent blocklist)
#        f. NOT currently in cooldown
#
#  5. Prune gates — a stock is removed if ANY of:
#        a. rolling 5-scan avg prob_up < PRUNE_THRESHOLD (0.45)
#        b. consecutive_losses >= MAX_CONSEC_LOSSES
#        c. atr_pct > ATR_MAX_PCT  (vol spike / circuit risk)
#        d. daily volume < MIN_DAILY_VOL_CR / 2  (liquidity drop)
#
#  6. Universe scan is done ONLY pre-market (9:05-9:14 AM) and
#     every 30 min during market hours to keep API call budget
#     low. Intra-candle scoring is done only for current watchlist.
# =============================================================

import json, logging, os, pickle, tempfile, time, threading
from collections import defaultdict, deque
from datetime import datetime, time as dtime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import portalocker
    _HAS_LOCK = True
except ImportError:
    _HAS_LOCK = False

try:
    import schedule
    _HAS_SCHEDULE = True
except ImportError:
    _HAS_SCHEDULE = False

try:
    import requests as _req
except ImportError:
    _req = None

import config as cfg
from features import build_features, FEATURE_COLS

log = logging.getLogger("watchlist_manager")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# ── constants ──────────────────────────────────────────────────
WATCHLIST_PATH      = Path(getattr(cfg, "WATCHLIST_JSON_PATH", "watchlist.json"))
ADD_THRESHOLD       = float(getattr(cfg, "WM_ADD_THRESHOLD",   0.60))
PRUNE_THRESHOLD     = float(getattr(cfg, "WM_PRUNE_THRESHOLD", 0.45))
SCAN_INTERVAL_MIN   = int(getattr(cfg, "WM_SCAN_INTERVAL_MIN", 5))
UNIVERSE_RESCAN_MIN = int(getattr(cfg, "WM_UNIVERSE_RESCAN_MIN", 30))
MIN_DAILY_VOL_CR    = float(getattr(cfg, "WM_MIN_DAILY_VOL_CR", 200.0))   # Cr
ATR_MIN_PCT         = float(getattr(cfg, "WM_ATR_MIN_PCT",      0.005))
ATR_MAX_PCT         = float(getattr(cfg, "WM_ATR_MAX_PCT",      0.060))
MAX_WATCHLIST_SIZE  = int(getattr(cfg, "WM_MAX_WATCHLIST_SIZE", 40))
MAX_PER_SECTOR      = int(getattr(cfg, "MAX_PER_SECTOR",        6))        # watchlist slots/sector
PRUNE_SCORE_WINDOW  = int(getattr(cfg, "WM_PRUNE_SCORE_WINDOW", 5))
PRUNE_COOLDOWN_BARS = int(getattr(cfg, "WM_PRUNE_COOLDOWN_BARS", 24))     # 24×5min = 2hrs
MAX_CONSEC_LOSSES   = int(getattr(cfg, "WM_MAX_CONSEC_LOSSES",  4))

MARKET_OPEN  = dtime(9, 15)
MARKET_CLOSE = dtime(15, 30)
UNIV_WINDOW_START = dtime(9, 5)
UNIV_WINDOW_END   = dtime(15, 20)

# ── broad NSE universe to scan (Nifty500 liquid subset) ───────
# Curated to 80 names: passes min 200Cr/day volume screen.
# The add/prune rules will dynamically narrow to 35-40 slots.
BROAD_UNIVERSE = [
    # BANKING / FINANCE
    "HDFCBANK","ICICIBANK","AXISBANK","SBIN","KOTAKBANK",
    "INDUSINDBK","BAJFINANCE","BAJAJFINSV","CHOLAFIN","HDFCLIFE",
    "SBILIFE","MUTHOOTFIN","MANAPPURAM","AUBANK","FEDERALBNK",
    # IT / TECH
    "TCS","INFY","HCLTECH","WIPRO","TECHM",
    "LTIM","PERSISTENT","COFORGE","MPHASIS","KPITTECH",
    # AUTO
    "MARUTI","TATAMOTORS","M&M","BAJAJ-AUTO","EICHERMOT",
    "HEROHEROMOTOCO","BOSCHLTD","MOTHERSON","BALKRISIND","TIINDIA",
    # PHARMA / HEALTH
    "SUNPHARMA","DRREDDY","CIPLA","DIVISLAB","APOLLOHOSP",
    "MAXHEALTH","FORTIS","ALKEM","LUPIN","BIOCON",
    # INFRA / CAPITAL GOODS
    "LT","HAL","BEL","BHEL_EXCL","CGPOWER",
    "SIEMENS","ABB","THERMAX","CUMMINSIND","GRINDWELL",
    # ENERGY
    "RELIANCE","NTPC","POWERGRID","TATAPOWER","ADANIGREEN",
    "BPCL","IOC","HINDPETRO","GAIL","PETRONET",
    # METALS
    "JSWSTEEL","TATASTEEL","HINDALCO","VEDL","COALINDIA",
    "NMDC","NATIONALUM","APLAPOLLO","JINDALSAW","RATNAMANI",
    # FMCG / CONSUMER
    "HINDUNILVR","ITC","NESTLEIND","BRITANNIA","DABUR",
    "MARICO","COLPAL","GODREJCP","EMAMILTD","TATACONSUM",
    # TELECOM / MEDIA
    "BHARTIARTL","VBL",
    # REALTY / RETAIL
    "DLF","GODREJPROP","TRENT","ETERNAL","IRCTC",
]

# Remove permanently blocked names from universe at import time
try:
    import watchlist as _wl
    _BLOCKED = set(getattr(_wl, "BLOCKED_SYMBOLS", []))
except Exception:
    _BLOCKED = set()

BROAD_UNIVERSE = [s for s in BROAD_UNIVERSE if s not in _BLOCKED]


# ══════════════════════════════════════════════════════════════
class WatchlistManager:
    """
    Autonomous OODA pipeline that keeps watchlist.json healthy.

    Thread-safe: uses a threading.Lock for in-memory state +
    atomic os.replace() for file writes.
    """

    def __init__(self, dhan_client, model=None, scaler=None, feature_cols=None):
        """
        Parameters
        ----------
        dhan_client : dhanhq.dhanhq  — authenticated Dhan API client
        model       : loaded XGBoost model (optional; loaded from disk if None)
        scaler      : fitted StandardScaler  (optional)
        feature_cols: list[str]             (optional)
        """
        self.dhan   = dhan_client
        self.model  = model
        self.scaler = scaler
        self.features = feature_cols or FEATURE_COLS
        self._lock  = threading.Lock()

        # rolling prob scores per symbol: deque(maxlen=PRUNE_SCORE_WINDOW)
        self._prob_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=PRUNE_SCORE_WINDOW))
        # prune cooldown counter per symbol
        self._cooldown: dict[str, int] = {}
        # consecutive losses counter (incremented by trade_manager feedback)
        self._consec_losses: dict[str, int] = defaultdict(int)

        self._last_universe_scan = datetime.min
        self._candidate_scores: dict[str, float] = {}  # scores from last universe scan

        if self.model is None:
            self._load_model()

        log.info("WatchlistManager initialised. watchlist=%s", WATCHLIST_PATH)

    # ── model helpers ─────────────────────────────────────────
    def _load_model(self):
        try:
            with open(cfg.MODEL_PATH,  "rb") as f: self.model   = pickle.load(f)
            with open(cfg.SCALER_PATH, "rb") as f: self.scaler  = pickle.load(f)
            with open(cfg.FEATURE_PATH,"rb") as f: self.features= pickle.load(f)
            log.info("Model loaded from disk.")
        except FileNotFoundError as e:
            log.warning("Model not found (%s) — scoring disabled until model exists.", e)

    def reload_model(self):
        """Called by auto_retrain.py after a successful retrain."""
        with self._lock:
            self._load_model()

    # ── watchlist.json I/O ────────────────────────────────────
    @staticmethod
    def _read_watchlist() -> dict:
        """Read watchlist.json; return empty structure on any error."""
        try:
            if WATCHLIST_PATH.exists():
                with open(WATCHLIST_PATH, "r") as f:
                    data = json.load(f)
                # Normalise legacy format (plain list) to new dict format
                if isinstance(data, list):
                    data = {"tier_a": data, "tier_b": [], "metadata": {}}
                return data
        except (json.JSONDecodeError, OSError) as e:
            log.error("watchlist.json read error: %s — using empty list.", e)
        return {"tier_a": [], "tier_b": [], "metadata": {}}

    @staticmethod
    def _write_watchlist(data: dict):
        """Atomically write watchlist.json via temp-file + os.replace()."""
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=WATCHLIST_PATH.parent, suffix=".json.tmp"
        )
        try:
            with os.fdopen(tmp_fd, "w") as f:
                if _HAS_LOCK:
                    portalocker.lock(f, portalocker.LOCK_EX)
                json.dump(data, f, indent=2)
                if _HAS_LOCK:
                    portalocker.unlock(f)
            os.replace(tmp_path, WATCHLIST_PATH)
        except Exception as e:
            log.error("watchlist.json write failed: %s", e)
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def _all_symbols(self) -> list[str]:
        data = self._read_watchlist()
        return list(dict.fromkeys(data.get("tier_a", []) + data.get("tier_b", [])))

    def _sector_counts(self, symbols: list[str]) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        try:
            import watchlist as wl
            sm = wl.SECTOR_MAP
        except Exception:
            sm = {}
        for s in symbols:
            counts[sm.get(s, "OTHER")] += 1
        return counts

    # ── Dhan API data helpers ─────────────────────────────────
    def _fetch_candles(self, symbol: str, n: int = 250) -> Optional[pd.DataFrame]:
        """
        Fetch last `n` 5-min candles for `symbol` via Dhan intraday API.
        Returns a DataFrame with columns [open, high, low, close, volume]
        and a DatetimeIndex, or None on failure.
        """
        try:
            # dhanhq intraday_minute_data: interval in minutes
            resp = self.dhan.intraday_minute_data(
                security_id=self._symbol_to_id(symbol),
                exchange_segment="NSE_EQ",
                instrument_type="EQUITY",
            )
            if not resp or "data" not in resp:
                return None
            rows = resp["data"]
            if not rows:
                return None
            df = pd.DataFrame(rows)
            # normalise column names from Dhan response
            col_map = {
                "open": "open", "high": "high", "low": "low",
                "close": "close", "volume": "volume",
                "startTime": "datetime", "timestamp": "datetime",
            }
            df.rename(columns={k: v for k, v in col_map.items() if k in df.columns},
                      inplace=True)
            if "datetime" in df.columns:
                df["datetime"] = pd.to_datetime(df["datetime"])
                df.set_index("datetime", inplace=True)
            df = df[["open", "high", "low", "close", "volume"]].astype(float)
            df.sort_index(inplace=True)
            return df.tail(n)
        except Exception as e:
            log.debug("Candle fetch failed %s: %s", symbol, e)
            return None

    def _symbol_to_id(self, symbol: str) -> str:
        """
        Resolve NSE symbol → Dhan security_id.
        Uses dhanhq search or a cached lookup table.
        Falls back to symbol string itself (works for index instruments).
        """
        try:
            result = self.dhan.search_scrip(exchange_segment="NSE_EQ", searchtext=symbol)
            if result and isinstance(result, list):
                for item in result:
                    if item.get("tradingSymbol", "") == symbol:
                        return str(item.get("securityId", symbol))
        except Exception:
            pass
        return symbol  # fallback — dhanhq also accepts ticker strings in some endpoints

    def _get_daily_vol_cr(self, symbol: str) -> float:
        """Estimate today's traded value in Cr from intraday candles."""
        try:
            df = self._fetch_candles(symbol, n=80)
            if df is None or df.empty:
                return 0.0
            # total traded value = sum(close × volume) / 1e7  (Cr)
            return float((df["close"] * df["volume"]).sum() / 1e7)
        except Exception:
            return 0.0

    # ── XGBoost scoring ───────────────────────────────────────
    def _score_symbol(self, symbol: str, df: Optional[pd.DataFrame] = None) -> float:
        """
        Return P(BUY=1) for `symbol` using latest candle features.
        Returns 0.0 if model unavailable or data insufficient.
        """
        if self.model is None or self.scaler is None:
            return 0.0
        try:
            if df is None:
                df = self._fetch_candles(symbol, n=250)
            if df is None or len(df) < 50:
                return 0.0
            feat = build_features(df)
            X = feat.iloc[-1][self.features].values.reshape(1, -1)
            X = np.where(np.isfinite(X), X, 0.0)
            X_sc = self.scaler.transform(X)
            prob = float(self.model.predict_proba(X_sc)[0, 1])
            return prob
        except Exception as e:
            log.debug("Score failed %s: %s", symbol, e)
            return 0.0

    # ── core OODA tick ────────────────────────────────────────
    def tick(self):
        """
        One OODA cycle.  Called every SCAN_INTERVAL_MIN minutes.
        Steps:
          1. Check market hours — skip if outside.
          2. Score all current watchlist symbols → update prob history.
          3. PRUNE: remove symbols that fail prune gates.
          4. ADD: if watchlist has room, score universe candidates,
             add those that pass add gates.
          5. Write updated watchlist.json atomically.
          6. Send Telegram summary if any changes occurred.
        """
        now = datetime.now()
        now_t = now.time()

        if not (MARKET_OPEN <= now_t <= MARKET_CLOSE):
            return  # outside market hours — do nothing

        with self._lock:
            self._tick_inner(now)

    def _tick_inner(self, now: datetime):
        data      = self._read_watchlist()
        tier_a    = list(data.get("tier_a", []))
        tier_b    = list(data.get("tier_b", []))
        metadata  = dict(data.get("metadata", {}))
        all_syms  = list(dict.fromkeys(tier_a + tier_b))

        changes_added   = []
        changes_pruned  = []

        # ── STEP 1: score current watchlist ──────────────────
        for sym in all_syms:
            prob = self._score_symbol(sym)
            self._prob_history[sym].append(prob)
            log.debug("SCORE %s → %.3f", sym, prob)

        # ── STEP 2: PRUNE ────────────────────────────────────
        to_remove = set()
        for sym in all_syms:
            hist = list(self._prob_history[sym])
            avg_prob = np.mean(hist) if hist else 0.0
            atr_df = self._fetch_candles(sym, 30)
            atr_pct = 0.01  # safe default
            if atr_df is not None and len(atr_df) > 14:
                atr_val = atr_df["close"].rolling(14).std().iloc[-1]
                atr_pct = atr_val / atr_df["close"].iloc[-1]

            prune_reason = None
            if len(hist) >= PRUNE_SCORE_WINDOW and avg_prob < PRUNE_THRESHOLD:
                prune_reason = f"low_avg_prob({avg_prob:.3f})"
            elif atr_pct > ATR_MAX_PCT:
                prune_reason = f"vol_spike(atr_pct={atr_pct:.3f})"
            elif self._consec_losses.get(sym, 0) >= MAX_CONSEC_LOSSES:
                prune_reason = f"consec_losses({self._consec_losses[sym]})"

            if prune_reason:
                to_remove.add(sym)
                self._cooldown[sym] = PRUNE_COOLDOWN_BARS
                metadata.setdefault("prune_log", {})[sym] = {
                    "reason": prune_reason,
                    "pruned_at": now.isoformat(),
                    "avg_prob": round(avg_prob, 4),
                }
                changes_pruned.append((sym, prune_reason))
                log.info("PRUNE %s — %s", sym, prune_reason)

        # Decrement cooldown counters
        expired = [s for s, v in self._cooldown.items() if v <= 1]
        for s in expired:
            del self._cooldown[s]
        for s in set(self._cooldown) - set(expired):
            self._cooldown[s] -= 1

        tier_a = [s for s in tier_a if s not in to_remove]
        tier_b = [s for s in tier_b if s not in to_remove]
        all_syms = list(dict.fromkeys(tier_a + tier_b))

        # ── STEP 3: ADD from universe ─────────────────────────
        now_t = now.time()
        do_universe_scan = (
            (now - self._last_universe_scan).total_seconds() >= UNIVERSE_RESCAN_MIN * 60
            and UNIV_WINDOW_START <= now_t <= UNIV_WINDOW_END
        )

        if do_universe_scan and len(all_syms) < MAX_WATCHLIST_SIZE:
            self._last_universe_scan = now
            sector_counts = self._sector_counts(all_syms)
            try:
                import watchlist as wl
                sm = wl.SECTOR_MAP
            except Exception:
                sm = {}

            # Score candidates not already in watchlist
            candidates = [
                s for s in BROAD_UNIVERSE
                if s not in all_syms
                and s not in to_remove
                and s not in self._cooldown
                and s not in _BLOCKED
            ]

            scored = []
            for sym in candidates:
                if len(all_syms) + len(changes_added) >= MAX_WATCHLIST_SIZE:
                    break
                prob = self._score_symbol(sym)
                if prob < ADD_THRESHOLD:
                    continue
                sector = sm.get(sym, "OTHER")
                if sector_counts.get(sector, 0) >= MAX_PER_SECTOR:
                    log.debug("SKIP %s — sector cap (%s=%d)", sym, sector, sector_counts[sector])
                    continue
                # Liquidity gate
                vol_cr = self._get_daily_vol_cr(sym)
                if vol_cr < MIN_DAILY_VOL_CR:
                    log.debug("SKIP %s — vol %.1f Cr < %.0f", sym, vol_cr, MIN_DAILY_VOL_CR)
                    continue
                # ATR gate
                df_tmp = self._fetch_candles(sym, 30)
                if df_tmp is not None and len(df_tmp) > 14:
                    atr_v = df_tmp["close"].rolling(14).std().iloc[-1]
                    atp   = atr_v / df_tmp["close"].iloc[-1]
                    if not (ATR_MIN_PCT <= atp <= ATR_MAX_PCT):
                        log.debug("SKIP %s — atr_pct=%.4f out of range", sym, atp)
                        continue

                scored.append((sym, prob))
                sector_counts[sector] = sector_counts.get(sector, 0) + 1

            # Sort by prob desc, add highest-scoring first
            scored.sort(key=lambda x: x[1], reverse=True)
            for sym, prob in scored:
                if len(tier_b) + len(changes_added) < 20:  # tier_b cap
                    tier_b.append(sym)
                    changes_added.append((sym, prob))
                    metadata.setdefault("add_log", {})[sym] = {
                        "prob": round(prob, 4),
                        "added_at": now.isoformat(),
                    }
                    log.info("ADD %s — prob=%.3f", sym, prob)

        # ── STEP 4: write if anything changed ────────────────
        if changes_added or changes_pruned:
            metadata["last_updated"] = now.isoformat()
            metadata["size"] = len(tier_a) + len(tier_b)
            new_data = {"tier_a": tier_a, "tier_b": tier_b, "metadata": metadata}
            self._write_watchlist(new_data)
            log.info(
                "watchlist.json updated — added=%d pruned=%d total=%d",
                len(changes_added), len(changes_pruned), metadata["size"],
            )
            self._telegram_notify(changes_added, changes_pruned, metadata["size"])

    # ── trade_manager feedback ────────────────────────────────
    def record_trade_result(self, symbol: str, pnl: float):
        """
        Called by trade_manager after each trade closes.
        Tracks consecutive losses for prune gate.
        """
        with self._lock:
            if pnl < 0:
                self._consec_losses[symbol] += 1
            else:
                self._consec_losses[symbol] = 0

    # ── Telegram notification ─────────────────────────────────
    def _telegram_notify(self, added: list, pruned: list, total: int):
        token = getattr(cfg, "TELEGRAM_TOKEN", None)
        chat  = getattr(cfg, "TELEGRAM_CHAT_ID", None)
        if not token or not chat or _req is None:
            return
        lines = ["📊 *Watchlist Update*"]
        if added:
            lines.append("\n✅ *Added:*")
            for sym, prob in added:
                lines.append(f"  • {sym}  (prob={prob:.3f})")
        if pruned:
            lines.append("\n❌ *Pruned:*")
            for sym, reason in pruned:
                lines.append(f"  • {sym}  ({reason})")
        lines.append(f"\n🔢 Universe size: *{total}*")
        msg = "\n".join(lines)
        try:
            _req.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat, "text": msg, "parse_mode": "Markdown"},
                timeout=5,
            )
        except Exception as e:
            log.debug("Telegram notify failed: %s", e)

    # ── blocking run loop ─────────────────────────────────────
    def run_forever(self):
        """
        Standalone blocking loop.  Use this when running
        watchlist_manager.py directly as a process.

        For integration with bot.py, prefer run_scheduled()
        which registers with the `schedule` library.
        """
        log.info("Starting OODA loop every %d min.", SCAN_INTERVAL_MIN)
        while True:
            try:
                self.tick()
            except Exception as e:
                log.error("Tick error: %s", e, exc_info=True)
            time.sleep(SCAN_INTERVAL_MIN * 60)

    def run_scheduled(self):
        """
        Register with the `schedule` library (used in bot.py).
        Call schedule.run_pending() in the bot's main loop.
        """
        if not _HAS_SCHEDULE:
            raise ImportError("pip install schedule")
        schedule.every(SCAN_INTERVAL_MIN).minutes.do(self.tick)
        log.info("Registered OODA tick with schedule every %d min.", SCAN_INTERVAL_MIN)


# ══════════════════════════════════════════════════════════════
# CLI entry-point — run standalone if needed
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import dhanhq  # pip install dhanhq
    client = dhanhq.dhanhq(
        client_id=cfg.DHAN_CLIENT_ID,
        access_token=cfg.DHAN_ACCESS_TOKEN,
    )
    wm = WatchlistManager(dhan_client=client)
    wm.run_forever()
