# bot.py — dhan_xgb_bot_v3 — Main event loop

import time
import logging
import schedule
from datetime import datetime, time as dtime

import config as cfg
from watchlist import TIER_A, TIER_B, BLOCKED_SYMBOLS
from signal_engine import SignalEngine, get_nifty_regime
from trade_manager import TradeManager

try:
    from telegram import Bot as TelegramBot
    _TG = bool(cfg.TELEGRAM_BOT_TOKEN)
except ImportError:
    _TG = False

import os
os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

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
        self.dhan = self._connect()
        self.engine = SignalEngine()
        self.tm = TradeManager(self.dhan)
        self._tg = TelegramBot(cfg.TELEGRAM_BOT_TOKEN) if _TG else None
        self.regime = "NEUTRAL"
        self.nifty_r5c = 0.0
        log.info("DhanXGBBot v3 ready")

    def _connect(self):
        if cfg.PAPER_MODE:
            return None
        from dhanhq import dhanhq
        return dhanhq(cfg.DHAN_CLIENT_ID, cfg.DHAN_ACCESS_TOKEN)

    def notify(self, msg: str):
        log.info(f"[MSG] {msg}")
        if self._tg and cfg.TELEGRAM_CHAT_ID:
            try:
                self._tg.send_message(chat_id=cfg.TELEGRAM_CHAT_ID, text=msg)
            except Exception:
                pass

    def fetch(self, symbol: str, n: int = 250):
        import pandas as pd
        if cfg.PAPER_MODE:
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

    def update_regime(self):
        df = self.fetch("NIFTY50")
        if df is not None:
            self.regime, self.nifty_r5c = get_nifty_regime(df)

    def scan(self):
        now = datetime.now().time()

        # ── EOD force-exit window ─────────────────────────────────────
        if now >= cfg.INTRADAY_EXIT_TIME:
            for sym in list(self.tm.positions):
                df = self.fetch(sym, 3)
                ltp = (
                    df["close"].iloc[-1]
                    if df is not None
                    else self.tm.positions[sym].entry_price
                )
                self.tm.force_exit(sym, ltp, "EOD_CUTOFF")
            if not self.tm.positions:
                self.notify(f"EOD done | Daily PnL \u20b9{self.tm.daily_pnl:.2f}")
            return

        # ── Pre-market gate ───────────────────────────────────────────
        if now < cfg.NO_NEW_TRADE_BEFORE:
            return

        # ── Daily loss circuit breaker ────────────────────────────────
        if self.tm.daily_loss_breached:
            log.warning("Daily loss limit hit — scan paused")
            return

        self.update_regime()

        # Tier B only after 10:00
        syms = TIER_A + (TIER_B if now >= dtime(10, 0) else [])
        syms = [s for s in syms if s not in BLOCKED_SYMBOLS]

        for sym in syms:
            # ── Manage existing position ──────────────────────────────
            if sym in self.tm.positions:
                df = self.fetch(sym, 5)
                if df is None:
                    continue
                ltp = df["close"].iloc[-1]
                self.tm.update_trailing_sl(sym, ltp)
                reason = self.tm.check_exits(sym, df.iloc[-1].to_dict())
                if reason:
                    self.notify(
                        f"EXIT {sym} [{reason}] | Daily PnL \u20b9{self.tm.daily_pnl:.2f}"
                    )
                continue

            # ── Seek new entry ────────────────────────────────────────
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
                        f"\U0001f7e2 BUY {sym} \u20b9{sig['entry']:.0f} "
                        f"SL={sig['sl']:.0f} TP={sig['target']:.0f} "
                        f"p={sig['prob']:.2f} rr={sig.get('rr_ratio', 0):.1f} "
                        f"regime={self.regime}"
                    )

        log.info(
            f"Scan done | open={list(self.tm.positions)} | "
            f"Daily PnL=\u20b9{self.tm.daily_pnl:.2f}"
        )

    def run(self):
        self.tm.reset_daily()
        # Schedule every 5 min from 09:15 to 15:25
        for h in range(9, 16):
            for m in range(0, 60, 5):
                t = f"{h:02d}:{m:02d}"
                schedule.every().day.at(t).do(self.scan)
        self.notify("\U0001f916 DhanXGBBot v3 started | PAPER=" + str(cfg.PAPER_MODE))
        log.info("Scheduler running — waiting for market open")
        while True:
            schedule.run_pending()
            time.sleep(10)


if __name__ == "__main__":
    DhanXGBBot().run()
