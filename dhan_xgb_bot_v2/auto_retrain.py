# ============================================================
# auto_retrain.py — dhan_xgb_bot_v2
# Weekly walk-forward retrain with:
#   - Redis distributed lock (prevents duplicate retrains)
#   - 14-day embargo enforced in train.py
#   - Fallback to file-based marker if Redis unavailable
#
# PATCH 2026-06-28:
#   - Import changed to: from watchlist import ALL_SYMBOLS
#     (was missing — caused ImportError on startup)
#   - cfg.RETRAIN_EVERY_DAYS used (was RETRAIN_INTERVAL_DAYS)
# ============================================================

import logging, os, json
from datetime import datetime
import config as cfg
from train import train_and_save
from watchlist import ALL_SYMBOLS, get_watchlist

log = logging.getLogger("auto_retrain")

MARKER     = "models/last_retrain.txt"
REDIS_LOCK = "bot:retrain:lock"
REDIS_META = "bot:retrain:meta"


# ── Redis helper ─────────────────────────────────────────────
def _r():
    if not cfg.REDIS_ENABLED:
        return None
    try:
        import redis
        client = redis.Redis(
            host=cfg.REDIS_HOST, port=cfg.REDIS_PORT, db=cfg.REDIS_DB,
            password=cfg.REDIS_PASSWORD, socket_timeout=cfg.REDIS_SOCKET_TIMEOUT,
            decode_responses=True,
        )
        client.ping()
        return client
    except Exception:
        return None


def _last_retrain_from_redis():
    try:
        r = _r()
        if r:
            meta = r.get(REDIS_META)
            if meta:
                d = json.loads(meta)
                return datetime.fromisoformat(d["last_retrain"])
    except Exception:
        pass
    return None


def _last_retrain_from_file():
    if not os.path.exists(MARKER):
        return None
    try:
        with open(MARKER) as f:
            return datetime.fromisoformat(f.read().strip())
    except Exception:
        return None


def should_retrain() -> bool:
    last = _last_retrain_from_redis() or _last_retrain_from_file()
    if last is None:
        return True
    elapsed_days = (datetime.now() - last).days
    log.info(f"Last retrain: {last.date()} ({elapsed_days} days ago)")
    return elapsed_days >= cfg.RETRAIN_EVERY_DAYS   # FIX: was RETRAIN_INTERVAL_DAYS


def _acquire_lock(ttl_seconds=7200) -> bool:
    """Atomic NX lock — prevents parallel retrain runs."""
    try:
        r = _r()
        if r:
            acquired = r.set(REDIS_LOCK, datetime.now().isoformat(),
                             ex=ttl_seconds, nx=True)
            if not acquired:
                log.warning("[Retrain] Another retrain is already running (Redis lock held)")
                return False
    except Exception:
        pass
    return True


def _release_lock():
    try:
        r = _r()
        if r:
            r.delete(REDIS_LOCK)
    except Exception:
        pass


def _record_success(symbols: list):
    now_iso = datetime.now().isoformat()
    try:
        r = _r()
        if r:
            r.set(REDIS_META, json.dumps({
                "last_retrain": now_iso,
                "symbols":      len(symbols),
                "horizon":      cfg.HORIZON,
                "embargo_days": cfg.EMBARGO_DAYS,
            }))
    except Exception:
        pass
    os.makedirs("models", exist_ok=True)
    with open(MARKER, "w") as f:
        f.write(now_iso)
    log.info(f"[Retrain] Marker updated: {now_iso}")


def run_retrain(force: bool = False) -> bool:
    # Always use live watchlist in case WatchlistManager updated it
    symbols = get_watchlist()
    if not symbols:
        symbols = ALL_SYMBOLS   # fallback to import-time snapshot

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    log.info("=" * 58)
    log.info(f"  AUTO-RETRAIN  {datetime.now().strftime('%Y-%m-%d %H:%M')} IST")
    log.info(f"  Window: 90 days | Stocks: {len(symbols)}")
    log.info("=" * 58)

    if not force and not should_retrain():
        log.info("[Retrain] Not due yet — skipping")
        return False

    if not _acquire_lock():
        return False

    log.info(f"[Retrain] Starting — symbols={len(symbols)} "
             f"embargo={cfg.EMBARGO_DAYS}d folds={cfg.WALK_FORWARD_FOLDS}")
    try:
        train_and_save(symbols)
        _record_success(symbols)

        # Invalidate stale prediction/feature cache
        try:
            r = _r()
            if r:
                keys = r.keys("bot:pred:*") + r.keys("bot:feat:*")
                if keys:
                    r.delete(*keys)
                    log.info(f"[Retrain] Cleared {len(keys)} stale cache keys")
        except Exception:
            pass

        log.info("[Retrain] Complete ✓")
        return True

    except Exception as e:
        log.error(f"[Retrain] Failed: {e}")
        return False

    finally:
        _release_lock()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s")
    run_retrain(force=True)
