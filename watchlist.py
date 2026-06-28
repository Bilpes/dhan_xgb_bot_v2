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
# OODA change 2026-06-28:
#   - SECTOR_MAP and BLOCKED_SYMBOLS now loaded FROM JSON
#     so watchlist.json is the single source of truth.
#   - get_security_id() added for Dhan API calls.
#   - _refresh_static() lets WatchlistManager hot-refresh
#     module-level vars after atomic JSON write.
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
def get_watchlist() -> list[str]:
    """
    Combined tier_a + tier_b, deduped, in scan priority order.
    Called on every scan tick — always reflects the live JSON.
    """
    data = _load_json()
    combined = data.get("tier_a", []) + data.get("tier_b", [])
    return list(dict.fromkeys(combined))  # preserve order, dedup


def get_tier_a() -> list[str]:
    """Tier A stocks — scanned from 09:20 AM."""
    return list(_load_json().get("tier_a", []))


def get_tier_b() -> list[str]:
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
def _build_sector_map() -> dict[str, str]:
    return {k: v.upper() for k, v in _load_json().get("SECTOR_MAP", {}).items()}


def _build_blocked() -> list[str]:
    return _load_json().get("BLOCKED_SYMBOLS", [])


SECTOR_MAP: dict[str, str] = _build_sector_map()
BLOCKED_SYMBOLS: list[str] = _build_blocked()


def _refresh_static() -> None:
    """
    Hot-refresh SECTOR_MAP + BLOCKED_SYMBOLS after
    WatchlistManager._write_watchlist() atomically updates JSON.
    Call this at the end of every WM write so the live bot
    sees sector changes without a restart.

    Usage in watchlist_manager.py:
        from watchlist import _refresh_static
        ...
        self._write_watchlist(new_data)
        _refresh_static()
    """
    global SECTOR_MAP, BLOCKED_SYMBOLS
    SECTOR_MAP      = _build_sector_map()
    BLOCKED_SYMBOLS = _build_blocked()