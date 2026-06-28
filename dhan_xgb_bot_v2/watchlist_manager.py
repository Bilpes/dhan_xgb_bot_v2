# watchlist_manager.py — dhan_xgb_bot_v2
# =============================================================
# OODA dynamic watchlist management.
#
# ISSUE-20 FIX: add get_sector(symbol) public method.
#   trade_manager._get_sector() calls wm.get_sector(symbol) but
#   the method didn't exist — every enter_trade() raised AttributeError
#   → no trades were ever entered even when a valid BUY signal fired.
#
# Purpose
# -------
# The static watchlist.json defines a 38-stock starting universe.
# WatchlistManager runs every WM_SCAN_INTERVAL_MIN minutes to:
#
#   OBSERVE  — score every stock in a broader candidate universe
#              using the live XGBoost model
#   ORIENT   — filter by ATR%, daily volume, liquidity
#   DECIDE   — decide to ADD (high-score candidates not yet in
#              watchlist) or PRUNE (chronic underperformers)
#   ACT      — atomically update watchlist.json and call
#              watchlist._refresh_static() so bot.py picks up
#              the change in the same scan cycle (no restart)
#
# Why OODA?
# ---------
# A fixed watchlist trained once every 7 days cannot adapt to
# intraday regime shifts. BANKNIFTY rotation, sector momentum,
# and post-result stock moves all create short-lived alpha that
# a static universe misses.  The OODA loop adds stocks during
# high-momentum periods and removes them when they stop working,
# keeping the live scanning universe tightly focused.
#
# Integration
# -----------
#   bot.py.__init__  :  wm = WatchlistManager(dhan, model, scaler, features)
#                       tm.set_watchlist_manager(wm)
#   bot.py.run()     :  wm.run_scheduled()     # registers wm.tick()
#   bot.py.reload()  :  wm.reload_model(model, scaler)  # after auto_retrain
#   trade_manager.py :  wm.record_trade_result(symbol, pnl)
#                       wm.get_sector(symbol)            # ISSUE-20 new
#
# Thread safety
# -------------
# bot.py is single-threaded (schedule loop).  All wm.tick() calls
# are made from the same thread as scan().  No locks are needed.
# =============================================================

import json
import logging
import os
import tempfile
from collections import defaultdict, deque
from datetime import datetime, time as dtime
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import pandas as pd
import schedule

import config as cfg
from features import build_features, FEATURE_COLS

log = logging.getLogger("watchlist_manager")

_WL_PATH = Path(__file__).parent / "watchlist.json"

# -- Broader candidate universe scanned by WatchlistManager --
# These are NSE stocks that are NOT in the base 38-stock watchlist
# but are candidates for dynamic addition if the model scores them
# above WM_ADD_THRESHOLD.  They are NEVER added if in BLOCKED_SYMBOLS.
#
# Criteria for being in this list:
#   - Nifty 500 constituent
#   - Daily average volume >= 200 Cr
#   - No corporate governance red flags (as of 2026-Q2)
#   - Clean intraday OHLCV available from Dhan API
CANDIDATE_UNIVERSE: List[str] = [
    # Banking / NBFC extras
    "FEDERALBNK", "IDFCFIRSTB", "BANDHANBNK",
    # IT extras
    "MPHASIS", "COFORGE",
    # Auto extras
    "BAJAJ-AUTO", "EICHERMOT", "HEROMOTOCO",
    # FMCG / Consumer
    "HINDUNILVR", "ITC", "NESTLEIND", "DABUR",
    # Pharma extras
    "CIPLA", "LUPIN", "ALKEM",
    # Energy / Infra extras
    "ONGC", "BPCL", "IOC", "GAIL", "SIEMENS",
    # Metals / Mining
    "TATASTEEL", "HINDALCO", "NATIONALUM",
    # Realty
    "OBEROIRLTY", "PRESTIGE",
    # Financials / Insurance
    "ICICIPRULI", "HDFCLIFE", "SBILIFE",
    # Diversified
    "GODREJCP",
    # Note: PIDILITIND removed — in BLOCKED_SYMBOLS (illiquid for algo)
]


class WatchlistManager:
    """
    OODA dynamic watchlist manager.

    Parameters
    ----------
    dhan_client   : dhanhq client (can be None in paper mode if CSV files exist)
    model         : loaded XGBClassifier from SignalEngine
    scaler        : loaded StandardScaler from SignalEngine
    feature_cols  : list of feature column names (from FEATURE_COLS)
    """

    def __init__(self, dhan_client, model, scaler, feature_cols=None):
        self.dhan         = dhan_client
        self.model        = model
        self.scaler       = scaler
        self.feature_cols = feature_cols or FEATURE_COLS

        # -- Per-symbol state --
        # prob_history  : rolling window of recent XGBoost probs (for prune check)
        # consec_losses : consecutive loss counter reset on win or on prune cooldown
        # prune_cooldown: bars remaining before a pruned stock can be re-added
        self._prob_history:   Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=cfg.WM_PRUNE_SCORE_WINDOW)
        )
        self._consec_losses:  Dict[str, int]   = defaultdict(int)
        self._prune_cooldown: Dict[str, int]   = defaultdict(int)
        self._last_tick: Optional[datetime]    = None

        log.info(
            "WatchlistManager init | "
            f"add_thr={cfg.WM_ADD_THRESHOLD} prune_thr={cfg.WM_PRUNE_THRESHOLD} "
            f"max_size={cfg.WM_MAX_WATCHLIST_SIZE} scan={cfg.WM_SCAN_INTERVAL_MIN}m"
        )

    # -- ISSUE-20 FIX: get_sector() public method ----------------
    def get_sector(self, symbol: str) -> str:
        """
        Return the sector for a symbol from watchlist.json SECTOR_MAP.
        Called by trade_manager._get_sector() for MAX_PER_SECTOR enforcement.

        Fallback chain:
          1. watchlist.json SECTOR_MAP (live, reflects WM updates)
          2. watchlist.SECTOR_MAP module-level dict (import-time snapshot)
          3. 'UNKNOWN'
        """
        data = self._read_json()
        sector_map = data.get("SECTOR_MAP", {})
        sector = sector_map.get(symbol)
        if sector:
            return sector.upper()

        # Fallback: module-level SECTOR_MAP (import-time snapshot)
        try:
            from watchlist import SECTOR_MAP
            return SECTOR_MAP.get(symbol, "UNKNOWN").upper()
        except Exception:
            return "UNKNOWN"

    # -- Model hot-reload (called by bot.reload() after retrain) --
    def reload_model(self, model=None, scaler=None):
        """
        Sync model/scaler with the freshly retrained version.
        Called from bot.reload() after auto_retrain completes.
        ISSUE-21: bot.py now passes engine.model, engine.scaler explicitly
        so we don't need to re-read from disk (saves ~40ms pickle load).
        If model/scaler not provided, falls back to reading from disk.
        """
        if model is not None and scaler is not None:
            self.model  = model
            self.scaler = scaler
            log.info("WatchlistManager: model/scaler reloaded from engine")
        else:
            try:
                import pickle
                with open(cfg.MODEL_PATH,  "rb") as f: self.model  = pickle.load(f)
                with open(cfg.SCALER_PATH, "rb") as f: self.scaler = pickle.load(f)
                log.info("WatchlistManager: model/scaler reloaded from disk")
            except Exception as e:
                log.warning(f"WatchlistManager: reload failed — {e}")

    # -- Trade result feedback (called by trade_manager.exit_trade) --
    def record_trade_result(self, symbol: str, pnl: float):
        """
        Called on every exit (SL_HIT, TARGET_HIT, EOD, force_exit).
        Updates the consecutive-loss counter for prune decisions.
        A win resets the streak; a loss increments it.
        """
        if pnl >= 0:
            self._consec_losses[symbol] = 0
            log.debug(f"[WM] {symbol} WIN pnl=Rs{pnl:.2f} -> streak reset")
        else:
            self._consec_losses[symbol] += 1
            streak = self._consec_losses[symbol]
            log.info(f"[WM] {symbol} LOSS pnl=Rs{pnl:.2f} -> consec_losses={streak}")
            if streak >= cfg.WM_MAX_CONSEC_LOSSES:
                log.warning(
                    f"[WM] {symbol} hit {streak} consecutive losses — "
                    "scheduling prune on next tick"
                )

    # -- Scheduled tick (registered by run_scheduled) -------------
    def tick(self):
        """
        Single OODA cycle.  Called every WM_SCAN_INTERVAL_MIN by schedule.
        Skips outside market hours to avoid wasting API quota.
        """
        now = datetime.now().time()
        if now < dtime(9, 15) or now > dtime(15, 25):
            return

        self._last_tick = datetime.now()
        data = self._read_json()

        # -- 1. PRUNE pass — check existing watchlist -------------
        tier_a = data.get("tier_a", [])
        tier_b = data.get("tier_b", [])
        blocked = set(data.get("BLOCKED_SYMBOLS", []))
        modified = False

        pruned = []
        for sym in list(tier_b):  # only prune tier_b; tier_a is anchored
            if self._should_prune(sym, blocked):
                tier_b.remove(sym)
                self._prune_cooldown[sym] = cfg.WM_PRUNE_COOLDOWN_BARS
                self._consec_losses[sym]  = 0
                self._prob_history[sym].clear()
                pruned.append(sym)
                modified = True
                log.info(f"[WM] PRUNED {sym} from tier_b")

        # -- 2. ADD pass — scan candidate universe ----------------
        current_size = len(tier_a) + len(tier_b)
        if current_size < cfg.WM_MAX_WATCHLIST_SIZE:
            existing = set(tier_a + tier_b + list(blocked))
            for sym in CANDIDATE_UNIVERSE:
                if current_size >= cfg.WM_MAX_WATCHLIST_SIZE:
                    break
                if sym in existing:
                    continue
                if self._prune_cooldown.get(sym, 0) > 0:
                    self._prune_cooldown[sym] -= 1
                    continue

                score = self._score_symbol(sym)
                if score is not None:
                    self._prob_history[sym].append(score)

                if score is not None and score >= cfg.WM_ADD_THRESHOLD:
                    if self._passes_liquidity_gate(sym):
                        tier_b.append(sym)
                        existing.add(sym)
                        current_size += 1
                        modified = True
                        log.info(
                            f"[WM] ADDED {sym} to tier_b | score={score:.3f}"
                        )

        # -- 3. Tick down prune cooldowns -------------------------
        for sym in list(self._prune_cooldown):
            if self._prune_cooldown[sym] > 0:
                self._prune_cooldown[sym] -= 1

        # -- 4. Persist if changed --------------------------------
        if modified:
            data["tier_b"] = tier_b
            self._write_json(data)
            log.info(
                f"[WM] Watchlist updated | "
                f"pruned={pruned} | tier_b={len(tier_b)} | "
                f"total={len(tier_a) + len(tier_b)}"
            )

    # -- Score a single symbol with the live model ----------------
    def _score_symbol(self, symbol: str) -> Optional[float]:
        """
        Fetch latest OHLCV, build features, return model probability.
        Returns None if data unavailable or model not loaded.
        """
        if self.model is None or self.scaler is None:
            return None
        df = self._fetch(symbol)
        if df is None or len(df) < 50:
            return None
        try:
            feat = build_features(df)
            row  = feat.iloc[[-1]][self.feature_cols]
            X    = row.replace([np.inf, -np.inf], np.nan).fillna(0).values
            X_sc = self.scaler.transform(X)
            prob = float(self.model.predict_proba(X_sc)[0, 1])
            return prob
        except Exception as e:
            log.debug(f"[WM] score_symbol {symbol} error: {e}")
            return None

    # -- Prune decision logic -------------------------------------
    def _should_prune(self, symbol: str, blocked: set) -> bool:
        """
        True if any of the following prune conditions are met:
          1. In BLOCKED_SYMBOLS
          2. Consecutive losses >= WM_MAX_CONSEC_LOSSES
          3. Rolling avg prob < WM_PRUNE_THRESHOLD (sustained underperformance)
        """
        if symbol in blocked:
            return True

        # Consecutive loss gate
        if self._consec_losses.get(symbol, 0) >= cfg.WM_MAX_CONSEC_LOSSES:
            log.debug(f"[WM] {symbol}: prune — consec_losses={self._consec_losses[symbol]}")
            return True

        # Rolling prob gate
        hist = list(self._prob_history.get(symbol, []))
        if len(hist) >= cfg.WM_PRUNE_SCORE_WINDOW:
            avg_prob = np.mean(hist)
            if avg_prob < cfg.WM_PRUNE_THRESHOLD:
                log.debug(f"[WM] {symbol}: prune — avg_prob={avg_prob:.3f}")
                return True

        return False

    # -- Liquidity / ATR gate ------------------------------------
    def _passes_liquidity_gate(self, symbol: str) -> bool:
        """
        Quick liquidity and volatility sanity check.
        Returns True if the symbol is suitable for algo trading.
        """
        df = self._fetch(symbol, n=20)
        if df is None or len(df) < 10:
            return False
        try:
            avg_price  = df["close"].mean()
            avg_volume = df["volume"].mean()
            # Rough daily Cr estimate: avg_price * avg_vol_per_candle * 75 candles
            daily_vol_cr = (avg_price * avg_volume * 75) / 1e7

            if daily_vol_cr < cfg.WM_MIN_DAILY_VOL_CR:
                log.debug(
                    f"[WM] {symbol}: liquidity gate FAIL "
                    f"daily_vol~{daily_vol_cr:.0f}Cr < {cfg.WM_MIN_DAILY_VOL_CR}Cr"
                )
                return False

            # ATR% check
            from ta.volatility import average_true_range
            atr     = average_true_range(df["high"], df["low"], df["close"], 14).iloc[-1]
            atr_pct = atr / avg_price

            if atr_pct < cfg.WM_ATR_MIN_PCT:
                log.debug(f"[WM] {symbol}: ATR% too flat ({atr_pct:.4f} < {cfg.WM_ATR_MIN_PCT})")
                return False
            if atr_pct > cfg.WM_ATR_MAX_PCT:
                log.debug(f"[WM] {symbol}: ATR% too wild ({atr_pct:.4f} > {cfg.WM_ATR_MAX_PCT})")
                return False

            return True

        except Exception as e:
            log.debug(f"[WM] {symbol}: liquidity gate error: {e}")
            return False

    # -- Data fetch -----------------------------------------------
    def _fetch(self, symbol: str, n: int = 250) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data for a symbol.
        Priority: local CSV cache -> Dhan API -> None.
        """
        csv_path = f"data/{symbol}_5min.csv"
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(
                    csv_path, parse_dates=["datetime"], index_col="datetime"
                )
                df.columns = [c.lower() for c in df.columns]
                df = df.sort_index().tail(n)
                if len(df) >= 30:
                    return df
            except Exception:
                pass

        if self.dhan is None:
            return None

        try:
            r = self.dhan.intraday_minute_data(
                security_id      = symbol,
                exchange_segment = "NSE_EQ",
                instrument_type  = "EQUITY",
            )
            raw = r.get("data", [])
            if not raw:
                return None
            df = pd.DataFrame(raw)
            df["datetime"] = pd.to_datetime(df["start_Time"])
            df = df.set_index("datetime").sort_index()
            df = df[["open", "high", "low", "close", "volume"]]
            os.makedirs("data", exist_ok=True)
            df.to_csv(csv_path)
            return df.tail(n)
        except Exception as e:
            log.debug(f"[WM] fetch {symbol} API error: {e}")
            return None

    # -- JSON I/O -------------------------------------------------
    @staticmethod
    def _read_json() -> dict:
        try:
            with open(_WL_PATH) as f:
                data = json.load(f)
            if isinstance(data, list):
                return {
                    "tier_a": data, "tier_b": [],
                    "SECURITY_IDS": {}, "SECTOR_MAP": {},
                    "BLOCKED_SYMBOLS": [], "ALT_USED": {},
                }
            return data
        except Exception:
            return {
                "tier_a": [], "tier_b": [],
                "SECURITY_IDS": {}, "SECTOR_MAP": {},
                "BLOCKED_SYMBOLS": [], "ALT_USED": {},
            }

    @staticmethod
    def _write_json(data: dict):
        """
        Atomic write: write to a temp file, then os.replace().
        Ensures watchlist.json is never partially-written (torn write).
        Also calls watchlist._refresh_static() so bot.py's in-process
        module-level ALL_SYMBOLS / SECTOR_MAP reflects the change
        without a restart.
        """
        try:
            dir_  = _WL_PATH.parent
            fd, tmp = tempfile.mkstemp(dir=dir_, suffix=".tmp")
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, _WL_PATH)
            log.debug("[WM] watchlist.json written atomically")

            try:
                from watchlist import _refresh_static
                _refresh_static()
            except Exception as e:
                log.debug(f"[WM] _refresh_static error: {e}")

        except Exception as e:
            log.error(f"[WM] JSON write failed: {e}")

    # -- Register with schedule (called from bot.run) -------------
    def run_scheduled(self):
        """
        Register wm.tick() with the `schedule` library at
        WM_SCAN_INTERVAL_MIN frequency.  bot.run()'s main while-loop
        calls schedule.run_pending() every 10 seconds, which drives
        both scan() and wm.tick() from the same thread.
        No extra thread or asyncio event loop needed.
        """
        interval = cfg.WM_SCAN_INTERVAL_MIN
        schedule.every(interval).minutes.do(self.tick)
        log.info(
            f"[WM] OODA tick registered every {interval}m — "
            f"driven by schedule.run_pending() in bot.run()"
        )
