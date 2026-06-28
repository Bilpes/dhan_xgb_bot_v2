# ============================================================
# signal_engine.py — dhan_xgb_bot_v2
# Audit-patched 2026-06-28
# Fix I7: CSV header now checked at write time via _needs_header()
#         Prevents headerless scan-log rows if file deleted mid-session
# BUY signal generator with Redis feature/prediction caching,
# cooldown enforcement, circuit-breaker, and duplicate-order guard
# ============================================================

import pickle, logging, os, json, hashlib
import numpy as np
import pandas as pd
from datetime import datetime

from features import build_features, FEATURE_COLS
import config as cfg

log = logging.getLogger("signal_engine")

# ── Redis helper (lazy import — never crashes if Redis is down) ──
def _get_redis():
    try:
        import redis as _redis
        pool = _redis.ConnectionPool(
            host=cfg.REDIS_HOST, port=cfg.REDIS_PORT, db=cfg.REDIS_DB,
            password=cfg.REDIS_PASSWORD, max_connections=cfg.REDIS_MAX_CONNECTIONS,
            socket_timeout=cfg.REDIS_SOCKET_TIMEOUT,
            retry_on_timeout=cfg.REDIS_RETRY_ON_TIMEOUT,
            decode_responses=True,
        )
        r = _redis.Redis(connection_pool=pool)
        r.ping()
        return r
    except Exception:
        return None

_R = None  # module-level Redis client (initialised once)

def _r():
    global _R
    if not cfg.REDIS_ENABLED:
        return None
    if _R is None:
        _R = _get_redis()
        if _R is None:
            log.warning("[Redis] Unavailable — running without cache")
    return _R

def _safe_get(key):
    try:
        r = _r()
        return r.get(key) if r else None
    except Exception:
        return None

def _safe_set(key, value, ex):
    try:
        r = _r()
        if r:
            r.setex(key, ex, value)
    except Exception:
        pass

def _safe_exists(key):
    try:
        r = _r()
        return bool(r.exists(key)) if r else False
    except Exception:
        return False

def _safe_setex_nx(key, ex, value):
    """SET key value EX ex NX — atomic, only if not exists."""
    try:
        r = _r()
        if r:
            return r.set(key, value, ex=ex, nx=True)
    except Exception:
        pass
    return None


class SignalEngine:
    def __init__(self):
        self.model    = None
        self.scaler   = None
        self.features = FEATURE_COLS
        self._load_model()
        os.makedirs(os.path.dirname(cfg.SIGNAL_LOG_PATH), exist_ok=True)
        # FIX I7: removed self._hdr = os.path.exists(...) — stale state bug
        # Header is now checked at write time via _needs_header()

    # ── CSV header helper ────────────────────────────────────────────
    @staticmethod
    def _needs_header(path: str) -> bool:
        """
        FIX I7: Check file size at write time — not at init.
        Returns True if the file does not exist or is empty (header needed).
        """
        try:
            return not os.path.exists(path) or os.path.getsize(path) == 0
        except OSError:
            return True

    # ── Model I/O ───────────────────────────────────────────────
    def _load_model(self):
        try:
            with open(cfg.MODEL_PATH,   "rb") as f: self.model    = pickle.load(f)
            with open(cfg.SCALER_PATH,  "rb") as f: self.scaler   = pickle.load(f)
            with open(cfg.FEATURE_PATH, "rb") as f: self.features = pickle.load(f)
            log.info("Model loaded")
        except FileNotFoundError:
            log.warning("No model found — run train.py first")

    def reload_model(self):
        self._load_model()

    # ── Main signal method ──────────────────────────────────────
    def get_signal(self, symbol, df, current_positions,
                   nifty_regime="BULL", nifty_ret_5c=0.0):

        result = {"action": "HOLD", "prob": 0.0, "entry": 0.0,
                  "sl": 0.0, "target": 0.0, "reject_reason": None, "atr": 0.0}
        now = datetime.now().time()

        def reject(reason):
            result["reject_reason"] = reason
            self._log(symbol, result)
            return result

        # ── Time gate ──────────────────────────────────────────
        if now < cfg.NO_NEW_TRADE_BEFORE or now >= cfg.NO_NEW_TRADE_AFTER:
            return reject("OUTSIDE_HOURS")

        # ── Model gate ─────────────────────────────────────────
        if self.model is None:
            return reject("NO_MODEL")

        # ── Data quality ───────────────────────────────────────
        if len(df) < 210:
            return reject(f"INSUFFICIENT_DATA({len(df)})")
        if df["close"].iloc[-1] < cfg.MIN_STOCK_PRICE:
            return reject(f"PRICE_LOW({df['close'].iloc[-1]:.0f})")

        # ── Position limits ────────────────────────────────────
        if symbol in current_positions:
            return reject("ALREADY_OPEN")

        import watchlist as wl
        sector = wl.SECTOR_MAP.get(symbol, "OTHER")
        sector_count = sum(
            1 for s in current_positions
            if wl.SECTOR_MAP.get(s, "X") == sector
        )
        if sector_count >= cfg.MAX_PER_SECTOR:
            return reject(f"SECTOR_LIMIT({sector})")
        if len(current_positions) >= cfg.MAX_OPEN_POSITIONS:
            return reject("MAX_POSITIONS")

        # ── Cooldown check (Redis) ─────────────────────────────
        cooldown_key = f"bot:cooldown:{symbol}"
        if _safe_exists(cooldown_key):
            return reject("COOLDOWN")

        # ── Circuit breaker (Redis) ────────────────────────────
        cb_key = "bot:circuit_breaker"
        if _safe_exists(cb_key):
            return reject("CIRCUIT_BREAKER_OPEN")

        # ── Duplicate order guard (Redis) ──────────────────────
        dedup_key = f"bot:dedup:{symbol}:{datetime.now().strftime('%Y%m%d%H%M')}"
        if _safe_exists(dedup_key):
            return reject("DUPLICATE_SIGNAL")

        # ── Feature vector (Redis cache) ───────────────────────
        feat_key  = f"bot:feat:{symbol}"
        feat_hash = hashlib.md5(str(df["close"].iloc[-1]).encode()).hexdigest()[:8]
        cached_feat_raw = _safe_get(feat_key + ":val")
        cached_feat_hsh = _safe_get(feat_key + ":hash")
        feat = None

        if cached_feat_raw and cached_feat_hsh == feat_hash:
            try:
                feat = pd.DataFrame([json.loads(cached_feat_raw)])
                log.debug(f"[Redis] Feature cache HIT {symbol}")
            except Exception:
                feat = None

        if feat is None:
            try:
                df2 = df.copy()
                df2["nifty_ret_5c"]      = nifty_ret_5c
                df2["nifty_above_ema20"] = 1 if nifty_regime in ("BULL", "NEUTRAL") else 0
                feat_full = build_features(df2)
                feat = feat_full.iloc[[-1]]
                row_dict = feat.iloc[0].to_dict()
                row_json = json.dumps({
                    k: (v if not (isinstance(v, float) and not np.isfinite(v)) else 0.0)
                    for k, v in row_dict.items()
                })
                _safe_set(feat_key + ":val",  row_json,  cfg.TTL_FEATURE)
                _safe_set(feat_key + ":hash", feat_hash, cfg.TTL_FEATURE)
            except Exception as e:
                return reject(f"FEAT_ERR({e})")

        # ── XGBoost prediction (Redis cache) ───────────────────
        pred_key = f"bot:pred:{symbol}:{feat_hash}"
        cached_prob = _safe_get(pred_key)
        if cached_prob is not None:
            prob = float(cached_prob)
            log.debug(f"[Redis] Prediction cache HIT {symbol} prob={prob:.4f}")
        else:
            try:
                X = feat.iloc[0][self.features].values.reshape(1, -1)
                X = np.where(np.isfinite(X), X, 0.0)
                X_sc = self.scaler.transform(X)
                prob = float(self.model.predict_proba(X_sc)[0, 1])
                _safe_set(pred_key, str(prob), cfg.TTL_PREDICTION)
            except Exception as e:
                return reject(f"PREDICT_ERR({e})")

        result["prob"] = prob

        # ── Threshold gate ─────────────────────────────────────
        thr = cfg.BUY_THRESHOLD_WEAK if nifty_regime == "WEAK" else cfg.BUY_THRESHOLD_DEFAULT
        if prob < thr:
            return reject(f"LOW_PROB({prob:.3f})")

        # ── Volume filter ──────────────────────────────────────
        vol_ratio = feat.iloc[0].get("vol_ratio", 1.0)
        if vol_ratio < cfg.MIN_VOLUME_RATIO:
            return reject(f"LOW_VOL({vol_ratio:.2f})")

        # ── ATR SL/TP (Redis ATR cache) ────────────────────────
        price = df["close"].iloc[-1]
        atr_key = f"bot:atr:{symbol}"
        cached_atr = _safe_get(atr_key)
        if cached_atr is not None:
            atr = float(cached_atr)
        else:
            atr = float(feat.iloc[0].get("atr14", price * 0.005))
            _safe_set(atr_key, str(atr), cfg.TTL_ATR)

        atr    = max(atr, price * 0.005)
        sl     = max(price - cfg.ATR_SL_MULT * atr, price * (1 - cfg.MAX_SL_PCT))
        sl     = min(sl, price * (1 - cfg.MIN_SL_PCT))
        target = price + cfg.ATR_TP_MULT * atr
        rr     = ((target - price) / (price - sl)) if price > sl else 0

        if rr < cfg.MIN_RR_RATIO:
            return reject(f"LOW_RR({rr:.2f})")

        # ── Mark duplicate guard (atomic NX) ───────────────────
        _safe_setex_nx(dedup_key, cfg.TTL_DEDUP_ORDER, "1")

        result.update({
            "action":        "BUY",
            "entry":         round(price,  2),
            "sl":            round(sl,     2),
            "target":        round(target, 2),
            "atr":           round(atr,    2),
            "rr_ratio":      round(rr,     2),
            "reject_reason": None,
        })
        self._log(symbol, result)
        log.info(
            f"BUY {symbol} entry={price:.2f} sl={sl:.2f} tp={target:.2f} "
            f"prob={prob:.4f} rr={rr:.2f}"
        )
        return result

    # ── Cooldown setter (called by trade_manager on SL hit) ─────
    @staticmethod
    def set_cooldown(symbol: str, seconds: int = None):
        ttl = seconds or cfg.TTL_COOLDOWN
        _safe_setex_nx(f"bot:cooldown:{symbol}", ttl, "1")
        log.info(f"[Cooldown] {symbol} → {ttl}s")

    # ── Circuit breaker ─────────────────────────────────────────
    @staticmethod
    def trip_circuit_breaker(seconds: int = None):
        ttl = seconds or cfg.TTL_CIRCUIT_BREAKER
        _safe_setex_nx("bot:circuit_breaker", ttl, "1")
        log.warning(f"[CircuitBreaker] Tripped for {ttl}s")

    @staticmethod
    def reset_circuit_breaker():
        try:
            r = _r()
            if r:
                r.delete("bot:circuit_breaker")
        except Exception:
            pass

    # ── CSV scan log ────────────────────────────────────────────
    def _log(self, symbol, result):
        row = {
            "time":          datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "symbol":        symbol,
            "action":        result["action"],
            "prob":          round(result["prob"], 4),
            "entry":         result["entry"],
            "sl":            result["sl"],
            "target":        result["target"],
            "reject_reason": result.get("reject_reason", ""),
        }
        # FIX I7: check header at write time, not at init
        write_header = self._needs_header(cfg.SIGNAL_LOG_PATH)
        pd.DataFrame([row]).to_csv(
            cfg.SIGNAL_LOG_PATH, mode="a",
            header=write_header, index=False
        )


# ── Nifty regime (cached in Redis) ──────────────────────────────
def get_nifty_regime(nifty_df) -> tuple:
    cached = _safe_get("bot:nifty:regime")
    if cached:
        try:
            d = json.loads(cached)
            return d["regime"], float(d["ret5"])
        except Exception:
            pass

    if len(nifty_df) < 50:
        return "NEUTRAL", 0.0

    c   = nifty_df["close"]
    e20 = c.ewm(span=20).mean()
    e50 = c.ewm(span=50).mean()
    slp = (e20.iloc[-1] - e20.iloc[-5]) / e20.iloc[-5]
    ret5 = (c.iloc[-1] - c.iloc[-6]) / c.iloc[-6]

    if   c.iloc[-1] > e20.iloc[-1] > e50.iloc[-1] and slp > 0: regime = "BULL"
    elif c.iloc[-1] < e20.iloc[-1] < e50.iloc[-1] and slp < 0: regime = "WEAK"
    else:                                                        regime = "NEUTRAL"

    _safe_set(
        "bot:nifty:regime",
        json.dumps({"regime": regime, "ret5": ret5}),
        cfg.TTL_NIFTY_REGIME,
    )
    return regime, ret5
