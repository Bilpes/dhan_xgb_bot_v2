# ============================================================
#  data/load_instruments.py
#  Auto-fetches Security IDs for Nifty 50 from Dhan master CSV
#  Source: https://dhanhq.co/docs/v2/instruments/
# ============================================================
"""
Run once before going live. Auto-runs every Sunday via Task Scheduler.

    python data/load_instruments.py

What it does:
  1. Downloads Dhan master CSV (~32MB, 256k instruments)
  2. Filters to NSE equity EQ series only
  3. Matches each Nifty 50 symbol using SEM_TRADING_SYMBOL
  4. Tries alternate spellings for tricky symbols (M&M, BAJAJ-AUTO etc)
  5. Saves config/watchlist.json
  6. Sends Telegram summary — green if all found, red if any missing
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

# ── Nifty 50 symbols with sector mapping ────────────────────
# Key   = exact NSE trading symbol (SEM_TRADING_SYMBOL in Dhan CSV)
# Value = sector (used to avoid 2 stocks from same sector simultaneously)
NIFTY50_SYMBOLS = {
    # Banking
    "HDFCBANK":    "banking",
    "ICICIBANK":   "banking",
    "SBIN":        "banking",
    "KOTAKBANK":   "banking",
    "AXISBANK":    "banking",
    "INDUSINDBK":  "banking",
    "BANKBARODA":  "banking",
    # IT
    "TCS":         "it",
    "INFY":        "it",
    "WIPRO":       "it",
    "HCLTECH":     "it",
    "TECHM":       "it",
    # Energy / Power
    "RELIANCE":    "energy",
    "ONGC":        "energy",
    "NTPC":        "energy",
    "POWERGRID":   "energy",
    "COALINDIA":   "energy",
    "BPCL":        "energy",
    "IOC":         "energy",
    # FMCG
    "HINDUNILVR":  "fmcg",
    "ITC":         "fmcg",
    "NESTLEIND":   "fmcg",
    "BRITANNIA":   "fmcg",
    "TATACONSUM":  "fmcg",
    # Auto
    "MARUTI":      "auto",
    "TATAMOTORS":  "auto",
    "M&M":         "auto",
    "BAJAJ-AUTO":  "auto",
    "EICHERMOT":   "auto",
    "HEROMOTOCO":  "auto",
    # Pharma
    "SUNPHARMA":   "pharma",
    "DRREDDY":     "pharma",
    "CIPLA":       "pharma",
    "DIVISLAB":    "pharma",
    # Infra / Cement
    "LT":          "infra",
    "ULTRACEMCO":  "infra",
    "ADANIPORTS":  "infra",
    "ADANIENT":    "infra",
    "SHREECEM":    "infra",
    # Finance / NBFC
    "BAJFINANCE":  "finance",
    "BAJAJFINSV":  "finance",
    "SHRIRAMFIN":  "finance",
    # Defence
    "HAL":         "defence",
    "BEL":         "defence",
    # Telecom / Consumer
    "BHARTIARTL":  "telecom",
    "TITAN":       "consumer",
    "ASIANPAINT":  "consumer",
    # Metals
    "JSWSTEEL":    "metals",
    "TATASTEEL":   "metals",
    "HINDALCO":    "metals",
    "VEDL":        "metals",
}

# ── Alternate spellings Dhan sometimes uses ─────────────────
# If primary symbol not found, bot tries these automatically
SYMBOL_ALTERNATES = {
    "M&M":        ["MM", "M AND M", "MAHINDRA"],
    "BAJAJ-AUTO": ["BAJAJAUTO", "BAJAJ AUTO"],
    "DRREDDY":    ["DRREDDY", "DR REDDY"],
    "NESTLEIND":  ["NESTLE", "NESTLEIND"],
    "SHRIRAMFIN": ["SHRIRAMFIN", "SHRIRAM FIN", "SHRIRAMCIT"],
    "ADANIENT":   ["ADANIENT", "ADANI ENT"],
    "HEROMOTOCO": ["HEROMOTOCO", "HERO MOTO"],
}


# ── Step 1: Download master CSV ──────────────────────────────
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


# ── Step 2: Filter to NSE equity ────────────────────────────
def filter_nse_equity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keeps only:
      SEM_EXM_EXCH_ID  = NSE
      SEM_SEGMENT      = E  (Equity)
      SEM_SERIES       = EQ (regular equity, not BE/IL/SM etc)

    Uses SEM_TRADING_SYMBOL — the actual NSE trading symbol
    e.g. HDFCBANK, TCS, RELIANCE (not the full company name)
    """
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

    # Show a sample so you can confirm symbols look right
    sample = filtered[["SECURITY_ID", "SYMBOL"]].head(8)
    log.info("Sample from filtered CSV:\n%s", sample.to_string())

    return filtered[["SECURITY_ID", "SYMBOL"]]


# ── Step 3: Match each symbol → Security ID ─────────────────
def build_watchlist(nse_df: pd.DataFrame) -> dict:
    """
    For each symbol in NIFTY50_SYMBOLS:
      1. Try exact match in SEM_TRADING_SYMBOL
      2. If not found, try alternate spellings from SYMBOL_ALTERNATES
      3. If still not found, log warning — bot will skip that stock
    """
    symbol_to_id = dict(zip(nse_df["SYMBOL"], nse_df["SECURITY_ID"]))

    watchlist  = {}
    sector_map = {}
    not_found  = []
    used_alt   = {}   # tracks which alternate was used

    for symbol, sector in NIFTY50_SYMBOLS.items():
        # Try primary spelling first
        sec_id = symbol_to_id.get(symbol.upper())

        # Try alternates if primary failed
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
            log.info("  %-15s → ID: %-8s  sector: %s%s",
                     symbol, sec_id, sector, alt_note)
        else:
            not_found.append(symbol)
            log.warning("  %-15s → NOT FOUND in Dhan CSV", symbol)

    return {
        "WATCHLIST":  watchlist,
        "SECTOR_MAP": sector_map,
        "NOT_FOUND":  not_found,
        "ALT_USED":   used_alt,
    }


# ── Step 4: Save to JSON ─────────────────────────────────────
def save_watchlist(data: dict):
    os.makedirs(os.path.dirname(WATCHLIST_JSON), exist_ok=True)
    with open(WATCHLIST_JSON, "w") as f:
        json.dump(data, f, indent=2)
    log.info("Saved → %s", WATCHLIST_JSON)


# ── Step 5: Print summary to terminal ───────────────────────
def print_summary(data: dict):
    print("\n" + "=" * 55)
    print(f"  Loaded  : {len(data['WATCHLIST'])} stocks")
    print(f"  Missing : {len(data['NOT_FOUND'])} stocks")

    if data.get("ALT_USED"):
        print(f"\n  Alternate spellings used:")
        for sym, alt in data["ALT_USED"].items():
            print(f"    {sym} → matched as {alt}")

    if data["NOT_FOUND"]:
        print(f"\n  NOT FOUND — these will be SKIPPED by the bot:")
        for s in data["NOT_FOUND"]:
            print(f"    ✗ {s}")
        print(f"\n  To fix: add the correct Dhan trading symbol")
        print(f"  to SYMBOL_ALTERNATES in data/load_instruments.py")
    else:
        print("\n  ✓ All symbols found successfully.")

    counts = Counter(data["SECTOR_MAP"].values())
    print("\n  Sectors loaded:")
    for s, c in sorted(counts.items()):
        bar = "█" * c
        print(f"    {s:<15} {bar}  ({c})")
    print("=" * 55)


# ── Step 6: Telegram alert ───────────────────────────────────
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
                f"  {sym} → matched as {alt}"
                for sym, alt in alt_used.items()
            )

        if missing:
            missing_str = "\n".join(f"  ✗ {s}" for s in missing)
            msg = (
                f"⚠️ <b>INSTRUMENT REFRESH — ACTION NEEDED</b>\n"
                f"{date_str}\n"
                f"{'─' * 28}\n"
                f"✅ Loaded  : <b>{loaded} stocks</b>\n"
                f"❌ Missing : <b>{len(missing)} stocks</b>\n\n"
                f"<b>NOT FOUND — bot will skip these:</b>\n"
                f"<code>{missing_str}</code>\n\n"
                f"Fix: add correct trading symbol to\n"
                f"<code>SYMBOL_ALTERNATES</code> in\n"
                f"<code>data/load_instruments.py</code>"
                f"{alt_lines}\n\n"
                f"<b>Sectors loaded:</b>\n<code>{sector_lines}</code>"
            )
        else:
            msg = (
                f"✅ <b>INSTRUMENT REFRESH — ALL OK</b>\n"
                f"{date_str}\n"
                f"{'─' * 28}\n"
                f"Loaded : <b>{loaded} stocks</b>\n"
                f"{alt_lines}\n\n"
                f"<b>Sectors:</b>\n<code>{sector_lines}</code>\n\n"
                f"⏰ Weekly retrain starts in 30 minutes."
            )

        _send(msg)
        log.info("Telegram summary sent.")

    except Exception as e:
        log.warning("Telegram not sent (bot still works): %s", e)


# ── Main ─────────────────────────────────────────────────────
def run():
    os.makedirs("logs", exist_ok=True)

    log.info("=" * 55)
    log.info("Nifty 50 Instrument Loader — Dhan API")
    log.info("=" * 55)

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