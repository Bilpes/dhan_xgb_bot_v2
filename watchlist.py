# ============================================================
# watchlist.py — dhan_xgb_bot_v3
# 38-stock curated list: price >₹200, MCap >₹30k Cr, algo-friendly
# Blocked: all penny/PSU/low-price noisy stocks (confirmed losers)
# ============================================================

TIER_A = [
    # Banking — high liquidity, strong momentum, best for algo
    "ICICIBANK", "HDFCBANK", "AXISBANK", "SBIN", "BAJFINANCE",
    # IT — trend well intraday, low spread
    "TCS", "INFY", "HCLTECH", "WIPRO",
    # Diversified large-cap
    "RELIANCE", "LT", "BHARTIARTL",
    # Auto
    "TATAMOTORS", "MARUTI",
    # Pharma
    "SUNPHARMA",
    # Infra/PSU quality
    "ADANIPORTS", "NTPC", "POWERGRID", "HAL", "BEL",
]

TIER_B = [
    # IT mid-large
    "LTIM", "PERSISTENT",
    # NBFC
    "CHOLAFIN", "MUTHOOTFIN",
    # Capital goods
    "CGPOWER", "HAVELLS",
    # Consumer / retail
    "TRENT", "TITAN",
    # Pharma
    "DRREDDY", "APOLLOHOSP", "PIDILITIND",
    # Metals (selective — only large MCap)
    "JSWSTEEL", "TATASTEEL",
    # Others
    "DLF", "TATACONSUM", "ULTRACEMCO", "ADANIENT", "HINDCOPPER",
]

ALL_SYMBOLS = TIER_A + TIER_B

SECTOR_MAP = {
    "ICICIBANK":  "BANKING",  "HDFCBANK":   "BANKING",  "AXISBANK":   "BANKING",
    "SBIN":       "BANKING",  "BAJFINANCE": "BANKING",  "CHOLAFIN":   "BANKING",
    "MUTHOOTFIN": "BANKING",
    "TCS":        "IT",       "INFY":        "IT",       "HCLTECH":    "IT",
    "WIPRO":      "IT",       "LTIM":        "IT",       "PERSISTENT": "IT",
    "RELIANCE":   "OIL_GAS",
    "LT":         "INFRA",    "NTPC":        "INFRA",    "POWERGRID":  "INFRA",
    "ADANIPORTS": "INFRA",    "HAL":         "INFRA",    "BEL":        "INFRA",
    "CGPOWER":    "INFRA",    "DLF":         "INFRA",    "ULTRACEMCO": "INFRA",
    "BHARTIARTL": "TELECOM",
    "TATAMOTORS": "AUTO",     "MARUTI":      "AUTO",
    "SUNPHARMA":  "PHARMA",   "DRREDDY":     "PHARMA",   "APOLLOHOSP": "PHARMA",
    "TRENT":      "RETAIL",   "TITAN":       "RETAIL",
    "HAVELLS":    "ELECTRICALS",
    "JSWSTEEL":   "METALS",   "TATASTEEL":   "METALS",   "HINDCOPPER": "METALS",
    "ADANIENT":   "CONGLOMERATE",
    "PIDILITIND": "CHEMICALS",
    "TATACONSUM": "FMCG",
}

# ── Explicitly blocked ────────────────────────────────────────
# Reason: price <₹200 → SL hit on tick noise; or confirmed losers in paper logs
BLOCKED_SYMBOLS = [
    # Confirmed losers from paper trade log (Jun 2026)
    "YESBANK", "IDEA", "NHPC", "SJVN", "IRFC", "RVNL", "IRCON",
    "IDFCFIRSTB", "CANBK", "BANKBARODA", "SAIL", "BHEL", "PNB",
    "MOIL", "JIOFINANCE", "MOTHERSON", "NYKAA", "PAYTM",
    "ASHOKLEY", "IOC", "BPCL", "ONGC", "COALINDIA", "GAIL",
    "ADANIPOWER", "RBLBANK", "FEDERALBNK", "INDIANB",
    # PSU/penny general
    "RECLTD", "PFC", "HUDCO", "IREDA", "NBCC", "RITES", "RAILTEL",
    "GMRAIRPORT", "SUZLON", "ZOMATO", "SWIGGY",
]

def is_tradeable(symbol: str) -> bool:
    """Quick check: symbol in ALL_SYMBOLS and not in BLOCKED_SYMBOLS."""
    return symbol in ALL_SYMBOLS and symbol not in BLOCKED_SYMBOLS
