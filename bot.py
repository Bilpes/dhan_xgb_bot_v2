# bot.py — dhan_xgb_bot_v2  (OODA-wired 2026-06-28)
# =============================================================
# Changes from previous version:
#   1. WatchlistManager imported and instantiated in __init__
#   2. Symbol universe loaded DYNAMICALLY from watchlist.json
#      via get_watchlist() — no more static TIER_A / TIER_B
#   3. wm.run_scheduled() registers OODA tick with `schedule`
#   4. TradeManager receives wm reference so _exit_position
#      calls wm.record_trade_result(symbol, pnl) on every close
#   5. reload() hot-reloads both engine and wm after retrain
# =============================================================

import time
import logging
import schedule
from datetime import datetime, time as dtime

import config as cfg

# ── dynamic watchlist ─────────────────────────────────────────
# OODA: universe driven by watchlist.json, not a static list.
# get_watchlist() re-reads the JSON on every call so live
# adds/prunes from WatchlistManager appear in the same scan.
from watchlist import get_watchlist, get_tier_a, get_tier_b, BLOCKED_SYMBOLS

# ── core engine / manager imports ────────────────────────────
from signal_engine import SignalEngine, get_nifty_regime
from trade_manager import TradeManager
from watchlist_manager import WatchlistManager

try:
    from telegram import Bot as TelegramBot
    _TG = bool(getattr(cfg, "TELEGRAM_BOT_TOKEN", ""))
except ImportError:
    _TG = False

import os
os.makedirs("logs",   exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("data",   exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/bot.log"),
    ],
)
log = logging.getLogger("bot")


class DhanXGBBot:
    def __init__(self):
        self.dhan   = self._connect()
        self.engine = SignalEngine()
        self.tm     = TradeManager(self.dhan)
        self._tg    = TelegramBot(cfg.TELEGRAM_BOT_TOKEN) if _TG else None
        self.regime = "NEUTRAL"
        self.nifty_r5c = 0.0

        # ── OODA WatchlistManager ─────────────────────────────
        # Pass the already-loaded model/scaler from engine so we
        # don't deserialise the pickle twice (~40ms saved on init).
        self.wm = WatchlistManager(
            dhan_client  = self.dhan,
            model        = self.engine.model,
            scaler       = self.engine.scaler,
            feature_cols = self.engine.features,
        )
        # Give TradeManager a reference so every exit automatically
        # feeds the consecutive-loss counter in WatchlistManager.
        self.tm.set_watchlist_manager(self.wm)

        log.info("DhanXGBBot v3 — OODA watchlist pipeline active")

    # ── Dhan connection ───────────────────────────────────────
    def _connect(self):
        if cfg.PAPER_TRADE:
            log.info("PAPER_TRADE=True — dhan client is None")
            return None
        from dhanhq import dhanhq
        client = dhanhq(cfg.DHAN_CLIENT_ID, cfg.DHAN_ACCESS_TOKEN)
        log.info("Dhan API connected")
        return client

    # ── Telegram helper ───────────────────────────────────────
    def notify(self, msg: str):
        log.info(f"[MSG] {msg}")
        if self._tg and getattr(cfg, "TELEGRAM_CHAT_ID", ""):
            try:
                self._tg.send_message(chat_id=cfg.TELEGRAM_CHAT_ID, text=msg)
            except Exception:
                pass

    # ── data fetch ────────────────────────────────────────────
    def fetch(self, symbol: str, n: int = 250):
        import pandas as pd
        if cfg.PAPER_TRADE:
            try:
                df = pd.read_csv(
                    f"data/{symbol}_5min.csv",
                    parse_dates=["datetime"],
                    index_col="datetime",
                )
                df.columns = [c.lower() for c in df.columns]
                return df.sort_index().tail(n)
            except Exception:
                return None
        else:
            try:
                r = self.dhan.intraday_minute_data(
                    security_id=symbol,
                    exchange_segment="NSE_EQ",
                    instrument_type="EQUITY",
                )
                df = pd.DataFrame(r["data"])
                df["datetime"] = pd.to_datetime(df["start_Time"])
                df = df.set_index("datetime").sort_index()
                return df[["open", "high", "low", "close", "volume"]].tail(n)
            except Exception as e:
                log.warning(f"Fetch failed {symbol}: {e}")
                return None

    # ── Nifty regime ──────────────────────────────────────────
    def update_regime(self):
        df = self.fetch("NIFTY50")
        if df is not None:
            self.regime, self.nifty_r5c = get_nifty_regime(df)

    # ── model reload (called by auto_retrain.py) ──────────────
    def reload(self):
        """Hot-reload model after auto_retrain completes."""
        self.engine.reload_model()
        self.wm.reload_model()   # keep wm in sync with engine
        log.info("Model reloaded in engine + wm")

    # ── main scan loop ────────────────────────────────────────
    def scan(self):
        now = datetime.now().time()

        # ── EOD force-exit window ─────────────────────────────
        if now >= cfg.AUTO_EXIT_TIME:
            for sym in list(self.tm.positions):
                df = self.fetch(sym, 3)
                ltp = (
                    df["close"].iloc[-1]
                    if df is not None
                    else self.tm.positions[sym].entry_price
                )
                self.tm.force_exit(sym, ltp, "EOD_CUTOFF")
            if not self.tm.positions:
                self.notify(f"EOD done | Daily PnL ₹{self.tm.daily_pnl:.2f}")
            return

        # ── Pre-market gate ───────────────────────────────────
        if now < cfg.NO_NEW_TRADE_BEFORE:
            return

        # ── Daily loss circuit breaker ────────────────────────
        if self.tm.daily_loss_breached:
            log.warning("Daily loss limit hit — scan paused")
            return

        self.update_regime()

        # ── OODA: universe is live from watchlist.json ────────
        # get_watchlist() re-reads watchlist.json on every scan.
        # Stocks added/pruned by WatchlistManager appear/disappear
        # in the SAME scan cycle without restarting the bot.
        all_syms = get_watchlist()
        tier_a   = set(get_tier_a())
        syms = [
            s for s in all_syms
            if s not in BLOCKED_SYMBOLS
            and (s in tier_a or now >= dtime(10, 0))
        ]

        for sym in syms:
            # ── Manage existing position ──────────────────────
            if sym in self.tm.positions:
                df = self.fetch(sym, 5)
                if df is None:
                    continue
                ltp = df["close"].iloc[-1]
                self.tm.update_trailing_sl(sym, ltp)
                reason = self.tm.check_exits(sym, df.iloc[-1].to_dict())
                if reason:
                    self.notify(
                        f"EXIT {sym} [{reason}] | Daily PnL ₹{self.tm.daily_pnl:.2f}"
                    )
                continue

            # ── Seek new entry ────────────────────────────────
            df = self.fetch(sym)
            if df is None or len(df) < 50:
                continue

            sig = self.engine.get_signal(
                sym, df, self.tm.positions, self.regime, self.nifty_r5c
            )

            if sig["action"] == "BUY":
                pos = self.tm.enter(sym, sig)
                if pos:
                    self.notify(
                        f"🟢 BUY {sym} ₹{sig['entry']:.0f} "
                        f"SL={sig['sl']:.0f} TP={sig['target']:.0f} "
                        f"p={sig['prob']:.2f} rr={sig.get('rr_ratio',0):.1f} "
                        f"regime={self.regime}"
                    )

        log.info(
            f"Scan done | universe={len(syms)} | open={list(self.tm.positions)} | "
            f"Daily PnL=₹{self.tm.daily_pnl:.2f}"
        )

    # ── scheduler / run ───────────────────────────────────────
    def run(self):
        self.tm.reset_daily()

        # Register every 5-min slot 09:15 → 15:25
        for h in range(9, 16):
            for m in range(0, 60, 5):
                t = f"{h:02d}:{m:02d}"
                schedule.every().day.at(t).do(self.scan)

        # ── OODA: register WatchlistManager tick ──────────────
        # wm.run_scheduled() registers wm.tick() with `schedule`
        # at WM_SCAN_INTERVAL_MIN frequency (default 5 min).
        # The existing schedule.run_pending() below drives it —
        # no second thread or loop needed.
        self.wm.run_scheduled()

        self.notify(
            "🤖 DhanXGBBot v3 started | "
            f"PAPER={cfg.PAPER_TRADE} | OODA watchlist active"
        )
        log.info("Scheduler running — OODA + scan registered")

        while True:
            schedule.run_pending()   # drives both scan() + wm.tick()
            time.sleep(10)


if __name__ == "__main__":
    DhanXGBBot().run()
