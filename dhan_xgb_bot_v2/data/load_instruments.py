# ============================================================
#  data/load_instruments.py
#  Auto-fetches Security IDs for 150 stocks from Dhan master CSV
#  Source: https://dhanhq.co/docs/v2/instruments/
#  Universe: Nifty 50 + Nifty Next 50 + 50 High-Quality Intraday Stocks
#  Last updated: 2026-05-22  |  Total: ~150 stocks
# ============================================================
"""
Run once before going live. Auto-runs every Sunday via Task Scheduler.

    python data/load_instruments.py

What it does:
  1. Downloads Dhan master CSV (~32MB, 256k instruments)
  2. Filters to NSE equity EQ series only
  3. Matches each symbol using SEM_TRADING_SYMBOL
  4. Tries alternate spellings for tricky symbols (M&M, BAJAJ-AUTO etc)
  5. Saves config/watchlist.json
  6. Sends Telegram summary -- green if all found, red if any missing

Watchlist universe (~150 stocks):
  [N50]  = Nifty 50 constituent
  [NN50] = Nifty Next 50 constituent
  [HQ]   = High-quality intraday pick (high ATR%, strong volume, momentum)
"""


import os, sys, json, requests, logging
import pandas as pd
from io import StringIO
from collections import Counter


sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


log = logging.getLogger("load_instruments")
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)s  %(message)s",
    handlers= [
        logging.StreamHandler(),
        logging.FileHandler("logs/instruments.log", mode="a"),
    ]
)


MASTER_CSV_URL = "https://images.dhan.co/api-data/api-scrip-master.csv"
WATCHLIST_JSON = os.path.join(
    os.path.dirname(__file__), "..", "config", "watchlist.json"
)


# ── NIFTY 50 (57 symbols incl. recent additions) ────────────────────────────
# Key   = exact NSE trading symbol (SEM_TRADING_SYMBOL in Dhan CSV)
# Value = sector (used to avoid 2 stocks from same sector simultaneously)

NIFTY50_SYMBOLS = {
    # BANKING (7)
    "HDFCBANK":    "banking",   # [N50]  Largest private bank, Nifty anchor
    "ICICIBANK":   "banking",   # [N50]  High beta, consistent momentum
    "SBIN":        "banking",   # [N50]  Highest volume PSU bank
    "KOTAKBANK":   "banking",   # [N50]  Low beta, clean intraday structure
    "AXISBANK":    "banking",   # [N50]  Volatile, good breakout patterns
    "INDUSINDBK":  "banking",   # [N50]  High ATR, strong intraday moves
    "BANKBARODA":  "banking",   # [N50]  PSU with sharp intraday spikes

    # IT (5)
    "TCS":         "it",        # [N50]  IT anchor stock, clean trends
    "INFY":        "it",        # [N50]  High volume, reactive to global cues
    "WIPRO":       "it",        # [N50]  Consistent trending
    "HCLTECH":     "it",        # [N50]  Good ATR for IT sector
    "TECHM":       "it",        # [N50]  High beta IT, good breakout patterns

    # ENERGY (7)
    "RELIANCE":    "energy",    # [N50]  Highest weight in N50, anchor
    "ONGC":        "energy",    # [N50]  Crude-linked, highly reactive
    "NTPC":        "energy",    # [N50]  Power sector anchor
    "POWERGRID":   "energy",    # [N50]  Defensive, consistent trending
    "COALINDIA":   "energy",    # [N50]  High dividend yield, good volume
    "BPCL":        "energy",    # [N50]  Crude proxy, high ATR on oil move days
    "IOC":         "energy",    # [N50]  High volume OMC

    # FMCG (5)
    "HINDUNILVR":  "fmcg",      # [N50]  Defensive FMCG anchor
    "ITC":         "fmcg",      # [N50]  High volume, cigarette + FMCG
    "NESTLEIND":   "fmcg",      # [N50]  Premium FMCG, low ATR but liquid
    "BRITANNIA":   "fmcg",      # [N50]  Bakery leader, clean trending patterns
    "TATACONSUM":  "fmcg",      # [N50]  Tea + food, Tata group

    # AUTO (6)
    "MARUTI":      "auto",      # [N50]  Bellwether auto, most liquid
    "M&M":         "auto",      # [N50]  EV pivot story, strong uptrend
    "BAJAJ-AUTO":  "auto",      # [N50]  Export-linked, strong momentum
    "EICHERMOT":   "auto",      # [N50]  RE parent, clean breakout trends
    "HEROMOTOCO":  "auto",      # [N50]  High volume 2-wheeler
    "TATAMOTORS":  "auto",      # [N50]  JLR-linked, high beta plays

    # PHARMA (4)
    "SUNPHARMA":   "pharma",    # [N50]  Pharma sector anchor
    "DRREDDY":     "pharma",    # [N50]  Export-linked, ADR correlation
    "CIPLA":       "pharma",    # [N50]  Domestic + export blend
    "DIVISLAB":    "pharma",    # [N50]  API leader, clean trending

    # INFRA (3)
    "LT":          "infra",     # [N50]  Engineering conglomerate anchor
    "ADANIPORTS":  "infra",     # [N50]  Port + logistics, high volume
    "ADANIENT":    "infra",     # [N50]  Adani flagship, high beta

    # CEMENT (2)
    "ULTRACEMCO":  "cement",    # [N50]  Cement sector anchor
    "GRASIM":      "cement",    # [N50]  Aditya Birla + Ultratech parent

    # FINANCE / NBFC (3)
    "BAJFINANCE":  "finance",   # [N50]  Highest-beta NBFC
    "BAJAJFINSV":  "finance",   # [N50]  Bajaj group financial holding
    "SHRIRAMFIN":  "finance",   # [N50]  CV finance, good ATR

    # INSURANCE (2)
    "HDFCLIFE":    "insurance",  # [N50] Life insurance anchor
    "SBILIFE":     "insurance",  # [N50] PSU life insurance giant

    # METALS (3)
    "JSWSTEEL":    "metals",    # [N50]  Steel anchor, global cues linked
    "TATASTEEL":   "metals",    # [N50]  High volume steel player
    "HINDALCO":    "metals",    # [N50]  Aluminium, global macro linked

    # DEFENCE (2)
    "HAL":         "defence",   # [N50]  HAL Aerospace, high ATR
    "BEL":         "defence",   # [N50]  Electronics defence, consistent

    # TELECOM (1)
    "BHARTIARTL":  "telecom",   # [N50]  Telecom anchor, consistent trend

    # CONSUMER (3)
    "TITAN":       "consumer",  # [N50]  Jewellery + watches, consistent
    "ASIANPAINT":  "consumer",  # [N50]  Paint anchor, quality trend stock
    "ETERNAL":     "consumer",  # [N50]  Food delivery (Zomato), high ATR

    # RETAIL (1)
    "TRENT":       "retail",    # [N50]  Retail breakout -- Zara, Westside

    # TRAVEL (1)
    "INDIGO":      "travel",    # [N50]  Aviation, high ATR, oil-linked

    # HEALTHCARE (1)
    "MAXHEALTH":   "healthcare", # [N50]  Max Hospital, breakout stock

    # FINTECH (1)
    "JIOFINANCE":  "fintech",   # [N50]  Jio Financial -- new entrant, reactive
}


# ── NIFTY NEXT 50 ────────────────────────────────────────────────────────────
NIFTY_NEXT50_SYMBOLS = {
    # BANKING (4)
    "FEDERALBNK":  "banking",   # [NN50] Mid-size private, responsive
    "IDFCFIRSTB":  "banking",   # [NN50] High volume penny bank momentum
    "CANBK":       "banking",   # [NN50] PSU, high volume
    "PNB":         "banking",   # [NN50] High volume PSU, news-reactive

    # IT (4)
    "LTIM":        "it",        # [NN50] LTIMindtree -- strong momentum
    "PERSISTENT":  "it",        # [NN50] High-growth IT, large ATR%
    "COFORGE":     "it",        # [NN50] Mid-cap IT, high ATR%
    "TATAELXSI":   "it",        # [NN50] EV/design IT, strong trends

    # AUTO (2)
    "TVSMOTOR":    "auto",      # [NN50] Strong EV story, high ATR%
    "ASHOKLEY":    "auto",      # [NN50] CV cycle, reactive to IIP data

    # ENERGY (5)
    "ADANIPOWER":  "energy",    # [NN50] High beta power, news-reactive
    "ADANIENERGY": "energy",    # [NN50] Power transmission, infra play
    "JSWENERGY":   "energy",    # [NN50] Renewable pivot, momentum
    "NHPC":        "energy",    # [NN50] Hydro PSU, budget-reactive
    "SJVN":        "energy",    # [NN50] Hydro + renewable PSU

    # INFRA / CAPITAL GOODS (6)
    "SIEMENS":     "infra",     # [NN50] Capital goods, strong trends
    "ABB":         "infra",     # [NN50] Power infra automation
    "CGPOWER":     "infra",     # [NN50] CG Power -- transformer boom stock
    "HAVELLS":     "infra",     # [NN50] Consumer electrical, consistent
    "BHEL":        "infra",     # [NN50] PSU capex play, high volume
    "THERMAX":     "infra",     # [NN50] Industrial energy, good ATR

    # PHARMA / HEALTH (5)
    "TORNTPHARM":  "pharma",    # [NN50] Strong domestic franchise
    "AUROPHARMA":  "pharma",    # [NN50] US-generic exposure, high ATR
    "APOLLOHOSP":  "healthcare", # [NN50] Hospital chain, consumer health
    "ZYDUSLIFE":   "pharma",    # [NN50] Domestic pharma + CDMO growth
    "LUPIN":       "pharma",    # [NN50] US market exposure

    # FINANCE (4)
    "CHOLAFIN":    "finance",   # [NN50] Chola Finance -- NBFC rising star
    "MUTHOOTFIN":  "finance",   # [NN50] Gold loan, reactive to gold prices
    "IRFC":        "finance",   # [NN50] Rail financing PSU, high volume
    "ABCAPITAL":   "finance",   # [NN50] Aditya Birla financial holding

    # INSURANCE (1)
    "ICICIGI":     "insurance",  # [NN50] General insurance, strong franchise

    # FMCG (3)
    "GODREJCP":    "fmcg",      # [NN50] Godrej Consumer, domestic FMCG
    "DABUR":       "fmcg",      # [NN50] Ayurvedic FMCG, consistent
    "MARICO":      "fmcg",      # [NN50] Hair + food brands

    # METALS (3)
    "VEDL":        "metals",    # [NN50] Diversified metals, high ATR
    "HINDZINC":    "metals",    # [NN50] Zinc monopoly, high dividend
    "SAIL":        "metals",    # [NN50] PSU steel, high volume

    # CEMENT (1)
    "AMBUJACEM":   "cement",    # [NN50] Adani group cement

    # CHEMICALS (2)
    "PIDILITIND":  "chemicals", # [NN50] Fevicol monopoly, consistent
    "SOLARINDS":   "chemicals", # [NN50] Solar Explosives -- high ATR%

    # REALTY (3)
    "DLF":         "realty",    # [NN50] Real estate anchor stock
    "GODREJPROP":  "realty",    # [NN50] Godrej Properties -- premium
    "PRESTIGE":    "realty",    # [NN50] South India realty, good ATR

    # RETAIL / CONSUMER (3)
    "DMART":       "retail",    # [NN50] Supermarket chain, value pick
    "NYKAA":       "consumer",  # [NN50] Beauty retail, volatile
    "VOLTAS":      "consumer",  # [NN50] AC + cooling, seasonal momentum

    # FINTECH (1)
    "PAYTM":       "fintech",   # [NN50] One97 -- high ATR, news-reactive

    # DEFENCE (1)
    "MAZAGONDOCK": "defence",   # [NN50] Warship builder, high ATR%

    # HOSPITALITY (1)
    "INDHOTEL":    "hospitality", # [NN50] Indian Hotels (Taj)
}


# ── 50 HIGH-QUALITY INTRADAY PICKS ───────────────────────────────────────────
# Selection: High ATR%, good volume, strong momentum characteristics
HQ_SYMBOLS = {
    # BANKING (2)
    "RBLBANK":     "banking",   # [HQ]   High ATR%, volatile, breakout-prone
    "YESBANK":     "banking",   # [HQ]   Extreme volume, high beta moves

    # IT (4)
    "MPHASIS":     "it",        # [HQ]   DXC-backed, clean technical patterns
    "LTTS":        "it",        # [HQ]   L&T Tech -- engineering IT momentum
    "KPITTECH":    "it",        # [HQ]   EV software, very high ATR%
    "OFSS":        "it",        # [HQ]   Oracle Fin -- breakout, large-cap IT

    # AUTO (5)
    "MOTHERSON":   "auto",      # [HQ]   Global auto component, volatile
    "BHARATFORG":  "auto",      # [HQ]   Defence + auto dual catalyst
    "BALKRISIND":  "auto",      # [HQ]   Tyre, export-linked, low correlation
    "APOLLOTYRE":  "auto",      # [HQ]   Tyre sector momentum play
    "EXIDEIND":    "auto",      # [HQ]   Battery/EV play, good ATR

    # ENERGY / GAS (3)
    "GAIL":        "energy",    # [HQ]   Gas utility, budget-reactive
    "PETRONET":    "energy",    # [HQ]   LNG terminal, defensive energy
    "IGL":         "energy",    # [HQ]   City gas distribution, steady

    # INFRA (4)
    "CUMMINSIND":  "infra",     # [HQ]   Industrial engines, steady trends
    "RVNL":        "infra",     # [HQ]   Rail PSU -- high ATR%, budget play
    "IRCON":       "infra",     # [HQ]   Rail infra PSU, reactive to news
    "RAILVIKAS":   "infra",     # [HQ]   RVNL peer, infra momentum

    # PHARMA (4)
    "MANKIND":     "pharma",    # [HQ]   Consumer health, IPO momentum
    "GLENMARK":    "pharma",    # [HQ]   Specialty pharma, high ATR
    "IPCALAB":     "pharma",    # [HQ]   IPCA Labs -- solid domestic
    "ALKEM":       "pharma",    # [HQ]   Domestic branded pharma

    # FINANCE (4)
    "HDFCAMC":     "finance",   # [HQ]   AMC sector, MF flow linked
    "ANGELONE":    "finance",   # [HQ]   Broking, reactive to market volumes
    "MOTILALOFS":  "finance",   # [HQ]   Wealth + broking, MF AUM linked
    "ICICPRULI":   "insurance",  # [HQ]  ICICI Pru Life -- growing momentum

    # FMCG (3)
    "COLPAL":      "fmcg",      # [HQ]   Colgate -- defensive, low noise
    "EMAMILTD":    "fmcg",      # [HQ]   High ATR for FMCG sector
    "VBL":         "fmcg",      # [HQ]   Varun Beverages -- PepsiCo bottler

    # METALS (3)
    "NATIONALUM":  "metals",    # [HQ]   Aluminium PSU, breakout stock
    "MOIL":        "metals",    # [HQ]   Manganese -- niche commodity play
    "HINDCOPPER":  "metals",    # [HQ]   Copper proxy, high ATR%

    # DEFENCE (2)
    "GRSE":        "defence",   # [HQ]   Garden Reach Shipbuilders, volatile
    "COCHINSHIP":  "defence",   # [HQ]   Cochin Shipyard, defence + commercial

    # CEMENT (2)
    "SHREECEM":    "cement",    # [HQ]   Premium cement, quality trend stock
    "JKCEMENT":    "cement",    # [HQ]   Mid-cap cement, strong regional

    # CHEMICALS (3)
    "PIIND":       "chemicals", # [HQ]   PI Industries -- agrochem + CDMO
    "DEEPAKNTR":   "chemicals", # [HQ]   Deepak Nitrite -- volatile chemical
    "AARTI":       "chemicals", # [HQ]   Aarti Industries -- specialty chem

    # REALTY (2)
    "OBEROIRLTY":  "realty",    # [HQ]   Premium Mumbai real estate
    "PHOENIXLTD":  "realty",    # [HQ]   Mall + residential, consistent

    # CONSUMER / RETAIL (4)
    "IRCTC":       "consumer",  # [HQ]   Rail ticketing, high ATR, news-reactive
    "JUBLFOOD":    "consumer",  # [HQ]   Jubilant -- Domino's India
    "INDIAMART":   "consumer",  # [HQ]   B2B marketplace, volatile
    "KALYANKJIL":  "consumer",  # [HQ]   Kalyan Jewellers -- retail momentum

    # LOGISTICS (1)
    "DELHIVERY":   "logistics", # [HQ]   Express logistics, ecomm linked

    # TELECOM (1)
    "IDEA":        "telecom",   # [HQ]   Vi -- extreme volume, high ATR%
}


# ── Merge all symbols into a single NIFTY50_SYMBOLS dict ────────────────────
# This keeps backward compatibility with your existing code
_all_sources = {}
_all_sources.update(NIFTY50_SYMBOLS)
_all_sources.update(NIFTY_NEXT50_SYMBOLS)
_all_sources.update(HQ_SYMBOLS)

# This is the master dict your existing code references
NIFTY50_SYMBOLS = _all_sources


# ── Alternate spellings Dhan sometimes uses ──────────────────────────────────
SYMBOL_ALTERNATES = {
    "M&M":          ["MM", "M AND M", "MAHINDRA"],
    "BAJAJ-AUTO":   ["BAJAJAUTO", "BAJAJ AUTO"],
    "DRREDDY":      ["DRREDDY", "DR REDDY"],
    "NESTLEIND":    ["NESTLE", "NESTLEIND"],
    "SHRIRAMFIN":   ["SHRIRAMFIN", "SHRIRAM FIN", "SHRIRAMCIT"],
    "ADANIENT":     ["ADANIENT", "ADANI ENT"],
    "HEROMOTOCO":   ["HEROMOTOCO", "HERO MOTO"],
    "TATAMOTORS":   ["TATAMTRDVR", "TATA MOTORS", "TMCV"],
    "ETERNAL":      ["ETERNAL", "ZOMATO"],
    "LTIM":         ["LTIM#", "LTM", "LTIMINDTREE", "LTI MINDTREE"],
    "INDIGO":       ["INDIGO", "INTERGLOBEAVIATION", "INTERGLOBE"],
    "ZYDUSLIFE":    ["ZYDUSLIFE", "CADILAHC", "ZYDUS"],
    "ASHOKLEY":     ["ASHOKLEY", "ASHOKLEI", "ASHOK LEYLAND"],
    "IDFCFIRSTB":   ["IDFCFIRSTB", "IDFCFIRST", "IDFC FIRST"],
    "ADANIENERGY":  ["ADANIENSOL", "ADANI ENERGY", "ADANITRANSMISSION"],
    "MAZAGONDOCK":  ["MAZDOCK", "MAZAGON DOCK", "MDL"],
    "RAILVIKAS":    ["RAILVIKAS", "RVNL", "RAIL VIKAS NIGAM"],
    "JIOFINANCE":   ["JIOFINANCE", "JIO FINANCIAL", "JIOFIN"],
    "PHOENIXLTD":   ["PHOENIXLTD", "PHOENIX LTD", "PHOENIXMILL"],
    "OBEROIRLTY":   ["OBEROIRLTY", "OBEROI REALTY"],
    "COCHINSHIP":   ["COCHINSHIP", "CSL", "COCHIN SHIPYARD"],
    "NATIONALUM":   ["NATIONALUM", "NALCO", "NATIONAL ALUMINIUM"],
    "HINDCOPPER":   ["HINDCOPPER", "HIND COPPER", "HCL"],
    "IPCALAB":      ["IPCALAB", "IPCA LAB", "IPCA"],
    "ICICPRULI":    ["ICICIPRULI", "ICICI PRU LIFE"],
    "MOTILALOFS":   ["MOTILALOFS", "MOFSL", "MOTILAL OSWAL"],
    "ANGELONE":     ["ANGELONE", "ANGEL ONE", "ANGELBROKING"],
    "ABCAPITAL":    ["ABCAPITAL", "ADITYA BIRLA CAPITAL", "ABCAP"],
    "DELHIVERY":    ["DELHIVERY"],
    "KALYANKJIL":   ["KALYANKJIL", "KALYAN JEWELLERS", "KALYAN"],
    "INDIAMART":    ["INDIAMART", "INDIA MART", "INDMRT"],
    "AARTI":        ["AARTIIND", "AARTI INDUSTRIES"],
}


# ── Step 1: Download master CSV ──────────────────────────────────────────────
def download_master_csv() -> pd.DataFrame:
    log.info("Downloading Dhan instrument master CSV...")
    log.info("URL: %s", MASTER_CSV_URL)
    try:
        resp = requests.get(MASTER_CSV_URL, timeout=30)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        log.error("Download failed: %s", e)
        raise
    log.info("Downloaded %.1f KB", len(resp.content) / 1024)
    df = pd.read_csv(StringIO(resp.text), low_memory=False)
    log.info("Total instruments in master: %d", len(df))
    return df


# ── Step 2: Filter to NSE equity ─────────────────────────────────────────────
def filter_nse_equity(df: pd.DataFrame) -> pd.DataFrame:
    filtered = df[
        (df["SEM_EXM_EXCH_ID"] == "NSE") &
        (df["SEM_SEGMENT"]     == "E")   &
        (df["SEM_SERIES"]      == "EQ")
    ].copy()

    log.info("NSE EQ instruments found: %d", len(filtered))

    filtered["SECURITY_ID"] = (
        filtered["SEM_SMST_SECURITY_ID"]
        .astype(str).str.strip()
    )
    filtered["SYMBOL"] = (
        filtered["SEM_TRADING_SYMBOL"]
        .astype(str).str.strip().str.upper()
    )

    sample = filtered[["SECURITY_ID", "SYMBOL"]].head(8)
    log.info("Sample from filtered CSV:\n%s", sample.to_string())

    return filtered[["SECURITY_ID", "SYMBOL"]]


# ── Step 3: Match each symbol → Security ID ──────────────────────────────────
def build_watchlist(nse_df: pd.DataFrame) -> dict:
    symbol_to_id = dict(zip(nse_df["SYMBOL"], nse_df["SECURITY_ID"]))

    watchlist  = {}
    sector_map = {}
    not_found  = []
    used_alt   = {}

    for symbol, sector in NIFTY50_SYMBOLS.items():
        sec_id = symbol_to_id.get(symbol.upper())

        if not sec_id and symbol in SYMBOL_ALTERNATES:
            for alt in SYMBOL_ALTERNATES[symbol]:
                sec_id = symbol_to_id.get(alt.upper())
                if sec_id:
                    used_alt[symbol] = alt
                    break

        if sec_id:
            watchlist[symbol]  = sec_id
            sector_map[symbol] = sector
            alt_note = f" (via alternate: {used_alt[symbol]})" if symbol in used_alt else ""
            log.info("  %-15s -> ID: %-8s  sector: %s%s",
                     symbol, sec_id, sector, alt_note)
        else:
            not_found.append(symbol)
            log.warning("  %-15s -> NOT FOUND in Dhan CSV", symbol)

    return {
        "WATCHLIST":  watchlist,
        "SECTOR_MAP": sector_map,
        "NOT_FOUND":  not_found,
        "ALT_USED":   used_alt,
    }


# ── Step 4: Save to JSON ──────────────────────────────────────────────────────
def save_watchlist(data: dict):
    os.makedirs(os.path.dirname(WATCHLIST_JSON), exist_ok=True)
    with open(WATCHLIST_JSON, "w") as f:
        json.dump(data, f, indent=2)
    log.info("Saved -> %s", WATCHLIST_JSON)


# ── Step 5: Print summary to terminal ────────────────────────────────────────
def print_summary(data: dict):
    print("\n" + "=" * 60)
    print(f"  Loaded  : {len(data['WATCHLIST'])} stocks")
    print(f"  Missing : {len(data['NOT_FOUND'])} stocks")

    if data.get("ALT_USED"):
        print(f"\n  Alternate spellings used:")
        for sym, alt in data["ALT_USED"].items():
            print(f"    {sym} -> matched as {alt}")

    if data["NOT_FOUND"]:
        print(f"\n  NOT FOUND -- these will be SKIPPED by the bot:")
        for s in data["NOT_FOUND"]:
            print(f"    x {s}")
        print(f"\n  Fix: add correct symbol to SYMBOL_ALTERNATES")
    else:
        print("\n  All symbols found successfully.")

    counts = Counter(data["SECTOR_MAP"].values())
    print("\n  Sectors loaded:")
    for s, c in sorted(counts.items()):
        bar = "X" * c
        print(f"    {s:<15} {bar}  ({c})")
    print("=" * 60)


# ── Step 6: Telegram alert ────────────────────────────────────────────────────
def send_telegram_summary(data: dict):
    try:
        from bot.telegram_alert import _send
        from datetime import datetime

        loaded   = len(data["WATCHLIST"])
        missing  = data["NOT_FOUND"]
        alt_used = data.get("ALT_USED", {})
        counts   = Counter(data["SECTOR_MAP"].values())
        date_str = datetime.now().strftime("%d %b %Y, %I:%M %p")

        sector_lines = "\n".join(
            f"  {s:<14} {c} stocks"
            for s, c in sorted(counts.items())
        )

        alt_lines = ""
        if alt_used:
            alt_lines = "\n\n<b>Alternate spellings used:</b>\n" + "\n".join(
                f"  {sym} -> matched as {alt}"
                for sym, alt in alt_used.items()
            )

        if missing:
            missing_str = "\n".join(f"  x {s}" for s in missing)
            msg = (
                f"Warning <b>INSTRUMENT REFRESH -- ACTION NEEDED</b>\n"
                f"{date_str}\n"
                f"{'Loaded':-<28}\n"
                f"Loaded  : <b>{loaded} stocks</b>\n"
                f"Missing : <b>{len(missing)} stocks</b>\n\n"
                f"<b>NOT FOUND -- bot will skip these:</b>\n"
                f"<code>{missing_str}</code>\n\n"
                f"Fix: add correct trading symbol to\n"
                f"<code>SYMBOL_ALTERNATES</code> in\n"
                f"<code>data/load_instruments.py</code>"
                f"{alt_lines}\n\n"
                f"<b>Sectors loaded:</b>\n<code>{sector_lines}</code>"
            )
        else:
            msg = (
                f"OK <b>INSTRUMENT REFRESH -- ALL OK</b>\n"
                f"{date_str}\n"
                f"{'Loaded':-<28}\n"
                f"Loaded : <b>{loaded} stocks</b>\n"
                f"{alt_lines}\n\n"
                f"<b>Sectors:</b>\n<code>{sector_lines}</code>\n\n"
                f"Weekly retrain starts in 30 minutes."
            )

        _send(msg)
        log.info("Telegram summary sent.")

    except Exception as e:
        log.warning("Telegram not sent (bot still works): %s", e)


# ── Main ──────────────────────────────────────────────────────────────────────
def run():
    os.makedirs("logs", exist_ok=True)

    log.info("=" * 60)
    log.info("Instrument Loader -- Dhan API | Universe: ~150 stocks")
    log.info("=" * 60)

    master_df = download_master_csv()
    nse_df    = filter_nse_equity(master_df)
    data      = build_watchlist(nse_df)
    save_watchlist(data)
    print_summary(data)
    send_telegram_summary(data)

    if data["NOT_FOUND"]:
        print(
            f"\nBot will run with {len(data['WATCHLIST'])} stocks. "
            f"{len(data['NOT_FOUND'])} skipped."
        )
        print("Add missing symbols to SYMBOL_ALTERNATES and rerun.")
    else:
        print("\nAll done. Run next: python data/download_data.py")


if __name__ == "__main__":
    run()
