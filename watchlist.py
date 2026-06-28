# watchlist.py — dhan_xgb_bot_v2 / v3
# =============================================================
# Single source of truth for symbol universe.
# Schema (watchlist.json):
#   tier_a         : list[str]  — scanned from 09:20 AM
#   tier_b         : list[str]  — scanned from 10:00 AM
#   SECURITY_IDS   : dict       — symbol → Dhan security_id string
#   SECTOR_MAP     : dict       — symbol → sector (UPPERCASE)
#   BLOCKED_SYMBOLS: list[str]  — never traded, never added by WM
#   ALT_USED       : dict       — symbol → API alt name override
#
# PATCH 2026-06-28:
#   - ALL_SYMBOLS added as module-level list alias for train.py / auto_retrain.py
#   - get_watchlist() remains the live OODA-safe reader (re-reads JSON every call)
#   - ALL_SYMBOLS is a snapshot at import time — safe for training loops
# =============================================================

import json
from pathlib import Path
from typing import Optional

_WL_PATH = Path(__file__).parent / "watchlist.json"


# ── JSON loader ────────────────────────────────────────────
def _load_json() -> dict:
    """
    Re-reads watchlist.json on every call.
    WatchlistManager atomic-writes this file; callers always
    get the latest version on each scan tick without restarting.
    """
    try:
        with open(_WL_PATH) as f:
            data = json.load(f)
        # Legacy support: old schema was a flat list
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


# ── universe helpers ───────────────────────────────────────
def get_watchlist() -> list:
    """
    Combined tier_a + tier_b, deduped, in scan priority order.
    Called on every scan tick — always reflects the live JSON.
    Safe for bot.py OODA loop (re-reads watchlist.json on every call).
    """
    data = _load_json()
    combined = data.get("tier_a", []) + data.get("tier_b", [])
    return list(dict.fromkeys(combined))  # preserve order, dedup


def get_tier_a() -> list:
    """Tier A stocks — scanned from 09:20 AM."""
    return list(_load_json().get("tier_a", []))


def get_tier_b() -> list:
    """Tier B stocks — scanned from 10:00 AM."""
    return list(_load_json().get("tier_b", []))


def get_security_id(symbol: str) -> Optional[str]:
    """
    Return Dhan security_id string for a symbol, or None.
    Used by signal_engine and trade_manager instead of
    any hardcoded ID dict elsewhere in the codebase.
    """
    return _load_json().get("SECURITY_IDS", {}).get(symbol)


def is_tradeable(symbol: str) -> bool:
    """True if symbol is in active watchlist and not blocked."""
    blocked = set(_load_json().get("BLOCKED_SYMBOLS", []))
    return symbol in get_watchlist() and symbol not in blocked


# ── module-level static copies (fast in-process lookups) ──
# Source of truth is watchlist.json.  Call _refresh_static()
# after any JSON write to sync these without restarting.
def _build_sector_map() -> dict:
    return {k: v.upper() for k, v in _load_json().get("SECTOR_MAP", {}).items()}


def _build_blocked() -> list:
    return _load_json().get("BLOCKED_SYMBOLS", [])


SECTOR_MAP:      dict = _build_sector_map()
BLOCKED_SYMBOLS: list = _build_blocked()

# ── ALL_SYMBOLS — snapshot alias used by train.py / auto_retrain.py ──
# This is a list captured at import time — stable for training loops.
# bot.py uses get_watchlist() (live) instead of this.
# If watchlist.json changes mid-session, re-import or call get_watchlist().
ALL_SYMBOLS: list = get_watchlist()


def _refresh_static() -> None:
    """
    Hot-refresh SECTOR_MAP + BLOCKED_SYMBOLS + ALL_SYMBOLS after
    WatchlistManager._write_watchlist() atomically updates JSON.
    Call this at the end of every WM write so the live bot
    sees sector/universe changes without a restart.

    Usage in watchlist_manager.py:
        from watchlist import _refresh_static
        ...
        self._write_watchlist(new_data)
        _refresh_static()
    """
    global SECTOR_MAP, BLOCKED_SYMBOLS, ALL_SYMBOLS
    SECTOR_MAP      = _build_sector_map()
    BLOCKED_SYMBOLS = _build_blocked()
    ALL_SYMBOLS     = get_watchlist()
