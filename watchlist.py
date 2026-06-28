# ============================================================
# watchlist.py — dhan_xgb_bot_v2
# Audit-patched 2026-06-28
# Fixes: I4a (HINDCOPPER), I4b (ADANIENT), I4c (PIDILITIND), I4d (MUTHOOTFIN)
# 38-stock curated list: price >₹200, MCap >₹30k Cr, algo-friendly
# ============================================================

TIER_A = [
    # Banking — high liquidity, strong momentum, best for algo
    "ICICIBANK", "HDFCBANK", "AXISBANK", "SBIN", "BAJFINANCE", "KOTAKBANK",
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
    # NBFC — FIX I4d: removed MUTHOOTFIN (gold-price driven, mean-reverting)
    #                 added BAJAJFINSV (₹800Cr ADV, trend persistence 0.68)
    "CHOLAFIN", "BAJAJFINSV",
    # Capital goods
    "CGPOWER", "HAVELLS",
    # Consumer / retail
    "TRENT", "TITAN",
    # Pharma
    "DRREDDY", "APOLLOHOSP",
    # FIX I4c: removed PIDILITIND — price ₹2900+ causes compute_qty capital overflow
    #           at ATR≈₹25: qty=int(500000*0.005/25)=100, position=100*2900=₹290k (58% capital)
    # Metals — FIX I4a: removed HINDCOPPER (₹180Cr ADV, LME copper gap risk)
    #           TATASTEEL retained: ₹900Cr ADV, domestic demand driven
    "JSWSTEEL", "TATASTEEL",
    # Others — FIX I4b: removed ADANIENT (SEBI/court event-driven 2026, non-technical)
    #           added M_M (₹1100Cr ADV, clean intraday trend structure)
    #           added ETERNAL (fmr Zomato ticker change; ₹600Cr ADV, strong momentum 2026)
    "DLF", "TATACONSUM", "ULTRACEMCO",
    "M_M", "ETERNAL",
]

ALL_SYMBOLS = list(dict.fromkeys(TIER_A + TIER_B))  # dedup, preserve order

SECTOR_MAP = {
    "ICICIBANK":  "BANKING",  "HDFCBANK":   "BANKING",  "AXISBANK":   "BANKING",
    "SBIN":       "BANKING",  "BAJFINANCE": "BANKING",  "CHOLAFIN":   "BANKING",
    "KOTAKBANK":  "BANKING",  "BAJAJFINSV": "BANKING",
    "TCS":        "IT",       "INFY":        "IT",       "HCLTECH":    "IT",
    "WIPRO":      "IT",       "LTIM":        "IT",       "PERSISTENT": "IT",
    "RELIANCE":   "OIL_GAS",
    "LT":         "INFRA",    "NTPC":        "INFRA",    "POWERGRID":  "INFRA",
    "ADANIPORTS": "INFRA",    "HAL":         "INFRA",    "BEL":        "INFRA",
    "CGPOWER":    "INFRA",    "DLF":         "INFRA",    "ULTRACEMCO": "INFRA",
    "BHARTIARTL": "TELECOM",
    "TATAMOTORS": "AUTO",     "MARUTI":      "AUTO",     "M_M":        "AUTO",
    "SUNPHARMA":  "PHARMA",   "DRREDDY":     "PHARMA",   "APOLLOHOSP": "PHARMA",
    "TRENT":      "RETAIL",   "TITAN":       "RETAIL",   "ETERNAL":    "RETAIL",
    "HAVELLS":    "ELECTRICALS",
    "JSWSTEEL":   "METALS",   "TATASTEEL":   "METALS",
    "TATACONSUM": "FMCG",
}

# ── Explicitly blocked ────────────────────────────────────────
# Reason: price <₹200 → SL hit on tick noise; commodity-driven gaps;
#         regulatory/news-event stocks; confirmed paper-mode losers
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
    # FIX I4a: commodity-driven gap risk (LME copper)
    "HINDCOPPER",
    # FIX I4b: SEBI/court regulatory event-driven — technical model invalid 2026
    "ADANIENT", "ADANIGREEN", "ADANITRANS",
    # FIX I4c: high price → compute_qty capital overflow
    "PIDILITIND",
    # FIX I4d: gold NBFC — mean-reverting, not momentum
    "MUTHOOTFIN",
]


def is_tradeable(symbol: str) -> bool:
    """Quick check: symbol in ALL_SYMBOLS and not in BLOCKED_SYMBOLS."""
    return symbol in ALL_SYMBOLS and symbol not in BLOCKED_SYMBOLS
