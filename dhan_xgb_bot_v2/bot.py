# bot.py — dhan_xgb_bot_v2
# =============================================================
# PATCH 2026-06-29 (x10 audit pass — ISSUE-21 fix):
#   ISSUE-21: bot.reload() now passes engine.model and engine.scaler
#             explicitly to wm.reload_model() — avoids redundant pickle
#             re-read (~40ms saved) and keeps wm in sync with engine.
#
# Prior patches (2026-06-28):
#   ISSUE-9:  fetch() PAPER_TRADE fallback hits Dhan API when CSV absent
#   ISSUE-11: update_regime() uses fetch_index() (NSE_INDICES segment)
#   ISSUE-12: schedule loop starts at 09:15, not 09:00
#   Original: WatchlistManager OODA pipeline wired in
# =============================================================

import time
import logging
import schedule
from datetime import datetime, time as dtime

import config as cfg

# -- dynamic watchlist ----------------------------------------
from watchlist import get_watchlist, get_tier_a, get_tier_b, BLOCKED_SYMBOLS

# -- core engine / manager imports ----------------------------
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
        self.regime    = "NEUTRAL"
        self.nifty_r5c = 0.0

        # -- OODA WatchlistManager --------------------------------
        # Pass the already-loaded model/scaler from engine so we
        # don't deserialise the pickle twice (~40ms saved on init).
        self.wm = WatchlistManager(
            dhan_client  = self.dhan,
            model        = self.engine.model,
            scaler       = self.engine.scaler,
            feature_cols = self.engine.features,
        )

        # Wire WatchlistManager into TradeManager:
        #   - every exit calls wm.record_trade_result(symbol, pnl)
        #   - enter_trade() calls wm.get_sector() for sector limits
        # ISSUE-13 FIX: set_watchlist_manager() now exists on TradeManager
        self.tm.set_watchlist_manager(self.wm)

        log.info("DhanXGBBot v3 — OODA watchlist pipeline active")

    # -- Dhan connection ------------------------------------------
    def _connect(self):
        if cfg.PAPER_TRADE:
            log.info("PAPER_TRADE=True — dhan client is None")
            return None
        from dhanhq import dhanhq
        client = dhanhq(cfg.DHAN_CLIENT_ID, cfg.DHAN_ACCESS_TOKEN)
        log.info("Dhan API connected")
        return client

    # -- Telegram helper ------------------------------------------
    def notify(self, msg: str):
        log.info(f"[MSG] {msg}")
        if self._tg and getattr(cfg, "TELEGRAM_CHAT_ID", ""):
            try:
                self._tg.send_message(chat_id=cfg.TELEGRAM_CHAT_ID, text=msg)
            except Exception:
                pass

    # -- ISSUE-9: paper mode fetch with API fallback --------------
    def fetch(self, symbol: str, n: int = 250):
        """
        Fetch 5-min OHLCV data for a symbol.

        PAPER_TRADE mode:
          1. Try local CSV  data/{symbol}_5min.csv  (fast, cached)
          2. Fall back to Dhan API if CSV absent or too short

        LIVE mode:
          Always hits Dhan intraday_minute_data API.
        """
        import pandas as pd

        if cfg.PAPER_TRADE:
            csv_path = f"data/{symbol}_5min.csv"
            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(
                        csv_path, parse_dates=["datetime"], index_col="datetime",
                    )
                    df.columns = [c.lower() for c in df.columns]
                    df = df.sort_index().tail(n)
                    if len(df) >= 50:
                        return df
                except Exception as e:
                    log.debug(f"CSV read failed {symbol}: {e}")

            if not cfg.DHAN_CLIENT_ID or not cfg.DHAN_ACCESS_TOKEN:
                log.warning(
                    f"{symbol}: no CSV and DHAN credentials not set — "
                    "set DHAN_CLIENT_ID + DHAN_ACCESS_TOKEN or place CSV in data/"
                )
                return None
            try:
                from dhanhq import dhanhq
                dhan = dhanhq(cfg.DHAN_CLIENT_ID, cfg.DHAN_ACCESS_TOKEN)
                r = dhan.intraday_minute_data(
                    security_id=symbol,
                    exchange_segment="NSE_EQ",
                    instrument_type="EQUITY",
                )
                df = pd.DataFrame(r["data"])
                df["datetime"] = pd.to_datetime(df["start_Time"])
                df = df.set_index("datetime").sort_index()
                df = df[["open", "high", "low", "close", "volume"]]
                df.to_csv(csv_path)
                log.debug(f"Fetched and cached {symbol} ({len(df)} rows)")
                return df.tail(n)
            except Exception as e:
                log.warning(f"API fetch failed {symbol}: {e}")
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

    # -- ISSUE-11: separate index fetch (NSE_INDICES segment) -----
    def fetch_index(self, symbol: str, n: int = 250):
        """
        Fetch 5-min data for NSE index symbols (NIFTY50, BANKNIFTY).
        Uses exchange_segment=NSE_INDICES, NOT NSE_EQ.
        """
        import pandas as pd

        if cfg.PAPER_TRADE and not cfg.DHAN_CLIENT_ID:
            csv_path = f"data/{symbol}_5min.csv"
            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(
                        csv_path, parse_dates=["datetime"], index_col="datetime"
                    )
                    df.columns = [c.lower() for c in df.columns]
                    return df.sort_index().tail(n)
                except Exception:
                    pass
            return None

        try:
            dhan_client = self.dhan
            if dhan_client is None:
                from dhanhq import dhanhq
                dhan_client = dhanhq(cfg.DHAN_CLIENT_ID, cfg.DHAN_ACCESS_TOKEN)

            r = dhan_client.intraday_minute_data(
                security_id=symbol,
                exchange_segment="NSE_INDICES",
                instrument_type="INDEX",
            )
            df = pd.DataFrame(r["data"])
            df["datetime"] = pd.to_datetime(df["start_Time"])
            df = df.set_index("datetime").sort_index()
            return df[["open", "high", "low", "close", "volume"]].tail(n)
        except Exception as e:
            log.warning(f"Index fetch failed {symbol}: {e}")
            return None

    # -- ISSUE-11: use fetch_index() for regime -------------------
    def update_regime(self):
        df = self.fetch_index("NIFTY50")
        if df is not None:
            self.regime, self.nifty_r5c = get_nifty_regime(df)
        else:
            log.debug("Nifty fetch failed — keeping last regime: %s", self.regime)

    # -- ISSUE-21 FIX: pass model/scaler explicitly on reload -----
    def reload(self):
        """Hot-reload model after auto_retrain completes."""
        self.engine.reload_model()
        # Pass freshly-loaded model and scaler from engine so wm
        # doesn't need a second disk read (saves ~40ms, avoids
        # a brief window where engine and wm are on different versions).
        self.wm.reload_model(
            model  = self.engine.model,
            scaler = self.engine.scaler,
        )
        log.info("Model reloaded in engine + wm (same instance)")

    # -- main scan loop -------------------------------------------
    def scan(self):
        now = datetime.now().time()

        # -- EOD force-exit window --------------------------------
        if now >= cfg.AUTO_EXIT_TIME:
            for sym in list(self.tm.positions):
                df = self.fetch(sym, 3)
                # ISSUE-19: pos.entry_price property now exists
                ltp = (
                    df["close"].iloc[-1]
                    if df is not None and len(df) > 0
                    else self.tm.positions[sym].entry_price
                )
                # ISSUE-14: force_exit() now exists on TradeManager
                self.tm.force_exit(sym, ltp, "EOD_CUTOFF")
            if not self.tm.positions:
                self.notify(f"EOD done | Daily PnL Rs{self.tm.daily_pnl:.2f}")
            return

        # -- Pre-market gate --------------------------------------
        if now < cfg.NO_NEW_TRADE_BEFORE:
            return

        # -- ISSUE-15: daily_loss_breached property now exists ----
        if self.tm.daily_loss_breached:
            log.warning("Daily loss limit hit — scan paused")
            return

        self.update_regime()

        # -- OODA: universe is live from watchlist.json -----------
        all_syms = get_watchlist()
        tier_a   = set(get_tier_a())
        syms = [
            s for s in all_syms
            if s not in BLOCKED_SYMBOLS
            and (s in tier_a or now >= dtime(10, 0))
        ]

        for sym in syms:
            # -- Manage existing position -------------------------
            if sym in self.tm.positions:
                df = self.fetch(sym, 5)
                if df is None:
                    continue
                ltp = df["close"].iloc[-1]
                self.tm.update_trailing_sl(sym, ltp)
                # ISSUE-16: check_exits(sym, candle_dict) now exists
                reason = self.tm.check_exits(sym, df.iloc[-1].to_dict())
                if reason:
                    self.notify(
                        f"EXIT {sym} [{reason}] | Daily PnL Rs{self.tm.daily_pnl:.2f}"
                    )
                continue

            # -- Seek new entry -----------------------------------
            df = self.fetch(sym)
            if df is None or len(df) < 50:
                continue

            sig = self.engine.get_signal(
                sym, df, self.tm.positions, self.regime, self.nifty_r5c
            )

            if sig["action"] == "BUY":
                # ISSUE-18: enter(sym, sig_dict) now exists on TradeManager
                pos = self.tm.enter(sym, sig)
                if pos:
                    self.notify(
                        f"BUY {sym} Rs{sig['entry']:.0f} "
                        f"SL={sig['sl']:.0f} TP={sig['target']:.0f} "
                        f"p={sig['prob']:.2f} rr={sig.get('rr_ratio',0):.1f} "
                        f"regime={self.regime}"
                    )

        log.info(
            f"Scan done | universe={len(syms)} | open={list(self.tm.positions)} | "
            f"Daily PnL=Rs{self.tm.daily_pnl:.2f} | regime={self.regime}"
        )

    # -- scheduler / run ------------------------------------------
    def run(self):
        # ISSUE-17: reset_daily() now exists on TradeManager
        self.tm.reset_daily()

        # ISSUE-12: Start at 09:15, not 09:00.
        for h in range(9, 16):
            for m in range(0, 60, 5):
                t_obj = dtime(h, m)
                if h == 9 and m < 15:
                    continue
                if h == 15 and m >= 30:
                    continue
                schedule.every().day.at(f"{h:02d}:{m:02d}").do(self.scan)

        # -- OODA: register WatchlistManager tick -----------------
        self.wm.run_scheduled()

        self.notify(
            "DhanXGBBot v3 started | "
            f"PAPER={cfg.PAPER_TRADE} | OODA watchlist active"
        )
        log.info("Scheduler running — OODA + scan registered")

        while True:
            schedule.run_pending()
            time.sleep(10)


if __name__ == "__main__":
    DhanXGBBot().run()
