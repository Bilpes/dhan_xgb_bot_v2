# watchlist.py — dhan_xgb_bot_v2 / v3
# Single source of truth for symbol universe + sector mapping.
# WatchlistManager reads watchlist.json dynamically; this file
# provides the static sector map and helper functions.

import json
from pathlib import Path

_WL_PATH = Path(__file__).parent / "watchlist.json"

# ── dynamic loader ─────────────────────────────────────────
def _load_json() -> dict:
    try:
        with open(_WL_PATH) as f:
            data = json.load(f)
        if isinstance(data, list):
            return {"tier_a": data, "tier_b": [], "metadata": {}}
        return data
    except Exception:
        return {"tier_a": [], "tier_b": [], "metadata": {}}

def get_watchlist() -> list[str]:
    """Return combined tier_a + tier_b, deduped, respecting current JSON."""
    data = _load_json()
    combined = data.get("tier_a", []) + data.get("tier_b", [])
    return list(dict.fromkeys(combined))

def get_tier_a() -> list[str]:
    return list(_load_json().get("tier_a", []))

def get_tier_b() -> list[str]:
    return list(_load_json().get("tier_b", []))

def is_tradeable(symbol: str) -> bool:
    return symbol in get_watchlist() and symbol not in BLOCKED_SYMBOLS

# ── sector map ─────────────────────────────────────────────
# Static sector classification used by signal_engine.py for
# per-sector position limits.  WatchlistManager also uses this
# to prevent sector concentration when adding new stocks.
SECTOR_MAP: dict[str, str] = {
    # BANKING
    "HDFCBANK":    "BANKING",
    "ICICIBANK":   "BANKING",
    "AXISBANK":    "BANKING",
    "SBIN":        "BANKING",
    "KOTAKBANK":   "BANKING",
    "INDUSINDBK":  "BANKING",
    "FEDERALBNK":  "BANKING",
    "AUBANK":      "BANKING",
    # FINANCE
    "BAJFINANCE":  "FINANCE",
    "BAJAJFINSV":  "FINANCE",
    "CHOLAFIN":    "FINANCE",
    "HDFCLIFE":    "FINANCE",
    "SBILIFE":     "FINANCE",
    "MUTHOOTFIN":  "FINANCE",
    "MANAPPURAM":  "FINANCE",
    # IT
    "TCS":         "IT",
    "INFY":        "IT",
    "HCLTECH":     "IT",
    "WIPRO":       "IT",
    "TECHM":       "IT",
    "LTIM":        "IT",
    "PERSISTENT":  "IT",
    "COFORGE":     "IT",
    "MPHASIS":     "IT",
    "KPITTECH":    "IT",
    # AUTO
    "TATAMOTORS":  "AUTO",
    "MARUTI":      "AUTO",
    "M&M":         "AUTO",
    "BAJAJ-AUTO":  "AUTO",
    "EICHERMOT":   "AUTO",
    "MOTHERSON":   "AUTO",
    "BALKRISIND":  "AUTO",
    "TIINDIA":     "AUTO",
    # PHARMA
    "SUNPHARMA":   "PHARMA",
    "DRREDDY":     "PHARMA",
    "CIPLA":       "PHARMA",
    "DIVISLAB":    "PHARMA",
    "APOLLOHOSP":  "PHARMA",
    "MAXHEALTH":   "PHARMA",
    "LUPIN":       "PHARMA",
    # ENERGY / INFRA
    "RELIANCE":    "ENERGY",
    "NTPC":        "ENERGY",
    "POWERGRID":   "ENERGY",
    "TATAPOWER":   "ENERGY",
    "BPCL":        "ENERGY",
    "IOC":         "ENERGY",
    "GAIL":        "ENERGY",
    "LT":          "INFRA",
    "HAL":         "INFRA",
    "BEL":         "INFRA",
    "CGPOWER":     "INFRA",
    "SIEMENS":     "INFRA",
    "ABB":         "INFRA",
    # TELECOM
    "BHARTIARTL":  "TELECOM",
    # CONSUMER
    "ETERNAL":     "CONSUMER",
    "TRENT":       "CONSUMER",
    "TITAN":       "CONSUMER",
    "IRCTC":       "CONSUMER",
    "HAVELLS":     "CONSUMER",
    "HINDUNILVR":  "CONSUMER",
    "ITC":         "CONSUMER",
    "NESTLEIND":   "CONSUMER",
    # METALS / REALTY
    "JSWSTEEL":    "METALS",
    "TATASTEEL":   "METALS",
    "HINDALCO":    "METALS",
    "VEDL":        "METALS",
    "COALINDIA":   "METALS",
    "DLF":         "REALTY",
    "GODREJPROP":  "REALTY",
    "ADANIPORTS":  "PORTS",
}

# ── permanent blocklist ────────────────────────────────────
# These symbols are NEVER added by WatchlistManager regardless
# of their XGBoost score.  Reasons: regulatory risk, Adani
# group news-event driven moves, extreme illiquidity, or
# historically consistent false-positive signals.
BLOCKED_SYMBOLS: list[str] = [
    # Adani group — news event driven, not technical
    "ADANIENT", "ADANITRANS", "ADANIPOWER", "ADANIGREEN",
    "ADANIWILMAR", "ADANIENSOL",
    # Extreme illiquidity / low float
    "YESBANK", "IDEA", "RBLBANK", "PAYTM",
    # Duplicate / removed from universe
    "HINDCOPPER",    # previously in TIER_B AND blocked — resolved
    "PIDILITIND",    # <250Cr daily vol
    "BANKBARODA",
    "IPCALAB",
    "KALYANKJIL",
    "DELHIVERY",
    "IRFC",          # government bond proxy — low alpha
    "RVNL",          # erratic order flow
    "BHEL_EXCL",     # placeholder entry removed
]
