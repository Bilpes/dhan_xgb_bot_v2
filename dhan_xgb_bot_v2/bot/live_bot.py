# ============================================================
#  bot/live_bot.py  —  Main trading loop
# ============================================================

import time
import logging
import csv
import os
from datetime import datetime, time as dtime
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join("config", ".env"))

BOT_MODE = os.getenv("BOT_MODE", "paper").lower().strip()

if BOT_MODE not in ("test", "paper", "live"):
    print(f"[ERROR] BOT_MODE='{BOT_MODE}' is invalid. Use: test, paper, live")
    raise SystemExit(1)

from bot.dhan_api       import DhanBroker
from bot.signal_engine  import SignalEngine
from bot.risk_manager   import RiskManager
from bot.telegram_alert import (
    alert_bot_started, alert_entry, alert_exit,
    alert_trail_update, alert_daily_summary,
    alert_circuit_breaker, _send,
)
from config.config import (
    WATCHLIST, SECTOR_MAP, TRADE_MODE, MAX_OPEN_TRADES,
    MARKET_OPEN, MARKET_CLOSE, INTRADAY_CUTOFF,
    NO_NEW_TRADE_AFTER,NO_NEW_TRADE_BEFORE, TRADE_LOG, LOG_FILE, CAPITAL,
)

MAX_PER_SECTOR       = int(os.getenv("MAX_PER_SECTOR", "2"))
NEW_TRADE_LOSS_PAUSE = float(os.getenv("NEW_TRADE_LOSS_PAUSE", "-0.03"))

# ── Timing constants ──────────────────────────────────────────
MONITOR_INTERVAL  = 60    # check open positions every 60 seconds
SCAN_INTERVAL     = 300   # scan for new entries every 5 minutes (one candle)

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    handlers= [
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("live_bot")

MODE_LABEL = {
    "test":  "TEST   -- one scan cycle, no orders, then exits",
    "paper": "PAPER  -- full loop, no real orders, simulated P&L",
    "live":  "LIVE   -- real orders on Dhan with real money",
}


# ── Trade state ───────────────────────────────────────────────
class Trade:
    def __init__(self, symbol, security_id, side, qty,
                 entry, stop_loss, target, order_id, mode):
        self.symbol       = symbol
        self.security_id  = security_id
        self.side         = side
        self.qty          = qty
        self.entry        = entry
        self.stop_loss    = stop_loss
        self.target       = target
        self.order_id     = order_id
        self.mode         = mode
        self.running_high = entry
        self.open_time    = datetime.now()

    def unrealised_pnl(self, ltp: float) -> float:
        if self.side == "LONG":
            return (ltp - self.entry) * self.qty
        return (self.entry - ltp) * self.qty


# ── Trade CSV logger ──────────────────────────────────────────
def log_trade(row: dict):
    exists = os.path.exists(TRADE_LOG)
    with open(TRADE_LOG, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if not exists:
            w.writeheader()
        w.writerow(row)


# ── Time helpers ──────────────────────────────────────────────
def now_time() -> dtime:
    return datetime.now().time()

def time_from_str(s: str) -> dtime:
    h, m = map(int, s.split(":")); return dtime(h, m)

def is_market_open() -> bool:
    t = now_time()
    return time_from_str(MARKET_OPEN) <= t <= time_from_str(MARKET_CLOSE)

def is_cutoff_passed() -> bool:
    return now_time() >= time_from_str(INTRADAY_CUTOFF)

def no_new_trades() -> bool:
    return now_time() >= time_from_str(NO_NEW_TRADE_AFTER)

def is_auto_exit_time() -> bool:
    return now_time() >= time_from_str(os.getenv("AUTO_EXIT_TIME", "14:45"))


# ── Main bot ──────────────────────────────────────────────────
class LiveBot:

    def __init__(self):
        self.broker        = DhanBroker()
        self.engine        = SignalEngine()
        self.risk          = RiskManager()
        self.trades        = {}
        self.closed_trades = []
        self.paper_pnl     = 0.0
        self.sl_blacklist  = set()

        # Track when we last ran a full candle scan
        self._last_scan_time = 0.0

    # =========================================================
    # TEST MODE
    # =========================================================
    def run_test(self):
        log.info("=" * 55)
        log.info("TEST MODE -- scanning all %d watchlist stocks", len(WATCHLIST))
        log.info("No orders will be placed. Bot exits after scan.")
        log.info("=" * 55)

        _send(
            f"Bot TEST MODE\n"
            f"Scanning {len(WATCHLIST)} stocks...\n"
            f"No orders placed.\n"
            f"Capital: Rs.{CAPITAL:,} | Mode: {TRADE_MODE.upper()}"
        )

        results = []
        for symbol, sec_id in WATCHLIST.items():
            log.info("Scanning %s ...", symbol)
            try:
                df = self.broker.get_candles(sec_id, symbol, days_back=10)
                if df.empty:
                    results.append(f"  {symbol:<14} no data")
                    continue
                r      = self.engine.score(df)
                signal = r["signal"]
                prob   = r["prob_up"]
                price  = r["entry"]
                atr    = r["atr"]
                sl     = self.risk.calc_stop_loss(price, atr, TRADE_MODE)
                qty    = self.risk.position_size(price, sl)
                sector = SECTOR_MAP.get(symbol, "?")
                results.append(
                    f"  {symbol:<14} {signal:<5} "
                    f"Rs.{price:.1f}  conf={prob:.1%}  "
                    f"SL=Rs.{sl:.1f}  qty={qty}  [{sector}]"
                )
                log.info("  %s: %s prob=%.3f price=%.2f [%s]",
                         symbol, signal, prob, price, sector)
            except Exception as e:
                results.append(f"  {symbol:<14} error: {e}")
                log.error("  %s error: %s", symbol, e)

        chunk_size = 15
        chunks = [results[i:i+chunk_size] for i in range(0, len(results), chunk_size)]
        for idx, chunk in enumerate(chunks):
            msg = (
                f"TEST RESULTS ({idx+1}/{len(chunks)})"
                f" -- {datetime.now().strftime('%d %b %Y')}\n"
                f"{'─' * 28}\n"
                + "\n".join(chunk)
            )
            if idx == len(chunks) - 1:
                msg += "\n\nBot OK. Set BOT_MODE=paper to start."
            _send(msg)

        log.info("Test complete.")

    # =========================================================
    # SCAN AND ENTER — ranked by confidence (best signal first)
    # =========================================================
    def scan_and_enter(self):

        # Wait for market settle time
        if now_time() < time_from_str(NO_NEW_TRADE_BEFORE):
            log.info("Waiting for market to settle until %s ...",
                     NO_NEW_TRADE_BEFORE)
            return
                    
        if no_new_trades():
            return
        if len(self.trades) >= MAX_OPEN_TRADES:
            return

        # Loss pause — stop new entries at -3% daily loss
        if self.risk.daily_pnl / CAPITAL <= NEW_TRADE_LOSS_PAUSE:
            log.warning("Daily loss %.1f%% -- pausing new entries.",
                        self.risk.daily_pnl / CAPITAL * 100)
            return

        log.info("-- Full scan: ranking all eligible stocks --")

        candidates = []

        for symbol, sec_id in WATCHLIST.items():
            if symbol in self.trades:
                continue
            if symbol in self.sl_blacklist:
                log.info("%s: Skipping -- SL blacklisted today.", symbol)
                continue

            stock_sector = SECTOR_MAP.get(symbol, "unknown")
            sector_count = sum(
                1 for sym in self.trades
                if SECTOR_MAP.get(sym, "unknown") == stock_sector
            )
            if sector_count >= MAX_PER_SECTOR:
                continue

            df = self.broker.get_candles(sec_id, symbol, days_back=10)
            if df.empty:
                continue

            result = self.engine.score(df)

            if result["signal"] == "BUY":
                candidates.append((symbol, sec_id, result))
                log.info("  Candidate: %s prob=%.3f [%s]",
                         symbol, result["prob_up"], stock_sector)

        if not candidates:
            log.info("No BUY signals this scan.")
            return

        # Sort by confidence — highest prob_up first
        candidates.sort(key=lambda x: x[2]["prob_up"], reverse=True)

        log.info("Top signal: %s prob=%.3f (from %d candidates)",
                 candidates[0][0], candidates[0][2]["prob_up"], len(candidates))

        # Pick best signal
        best_symbol, best_sec_id, best_result = candidates[0]
        stock_sector = SECTOR_MAP.get(best_symbol, "?")

        entry  = best_result["entry"]
        atr    = best_result["atr"]
        sl     = self.risk.calc_stop_loss(entry, atr, TRADE_MODE)
        target = self.risk.calc_target(entry, sl)
        qty    = self.risk.position_size(entry, sl)

        if qty <= 0:
            log.warning("%s: qty=0 -- skipping.", best_symbol)
            return

        log.info("SIGNAL BUY | %s [%s] | entry=%.2f | SL=%.2f | target=%.0f | qty=%d | prob=%.3f",
                 best_symbol, stock_sector, entry, sl, target, qty, best_result["prob_up"])

        if BOT_MODE == "live":
            self._enter_live(best_symbol, best_sec_id, qty, entry, sl, target, best_result)
        else:
            self._enter_paper(best_symbol, best_sec_id, qty, entry, sl, target, best_result)

    def _enter_live(self, symbol, sec_id, qty, entry, sl, target, result):
        resp = self.broker.place_bracket_order(
            symbol=symbol, security_id=sec_id,
            quantity=qty, entry_price=entry,
            stop_loss=sl, target=target, trade_type=TRADE_MODE,
        )
        if resp.get("status") == "success":
            order_id = resp["data"]["orderId"]
            self.trades[symbol] = Trade(
                symbol=symbol, security_id=sec_id, side="LONG",
                qty=qty, entry=entry, stop_loss=sl, target=target,
                order_id=order_id, mode=TRADE_MODE,
            )
            log_trade({
                "time": datetime.now(), "mode": "LIVE", "symbol": symbol,
                "action": "ENTRY", "side": "LONG", "qty": qty,
                "price": entry, "sl": sl, "target": target,
                "prob_up": result["prob_up"], "pnl": "",
            })
            alert_entry(symbol, entry, sl, target, qty,
                        result["prob_up"], TRADE_MODE, entry * qty)
        else:
            log.error("Order FAILED for %s: %s", symbol, resp)

    def _enter_paper(self, symbol, sec_id, qty, entry, sl, target, result):
        self.trades[symbol] = Trade(
            symbol=symbol, security_id=sec_id, side="LONG",
            qty=qty, entry=entry, stop_loss=sl, target=target,
            order_id="PAPER-" + datetime.now().strftime("%H%M%S"),
            mode=TRADE_MODE,
        )
        log_trade({
            "time": datetime.now(), "mode": "PAPER", "symbol": symbol,
            "action": "ENTRY", "side": "LONG", "qty": qty,
            "price": entry, "sl": sl, "target": target,
            "prob_up": result["prob_up"], "pnl": "",
        })
        log.info("[PAPER] Simulated BUY %s [%s] | qty=%d @ %.2f | SL=%.2f | target=%.0f",
                 symbol, SECTOR_MAP.get(symbol, "?"), qty, entry, sl, target)
        alert_entry(symbol, entry, sl, target, qty,
                    result["prob_up"], TRADE_MODE, entry * qty)

    # =========================================================
    # MONITOR POSITIONS — runs every 60 seconds
    # Catches fast SL hits and trailing stops promptly
    # =========================================================
    def monitor_positions(self):
        if not self.trades:
            return

        # ONE batch LTP call for all open positions
        id_symbol_map = {
            str(trade.security_id): symbol
            for symbol, trade in self.trades.items()
        }
        prices = self.broker.get_ltp_batch(id_symbol_map)

        for symbol, trade in list(self.trades.items()):
            ltp = prices.get(str(trade.security_id), 0.0)

            if ltp <= 0:
                log.warning("%s: LTP unavailable -- skipping.", symbol)
                continue

            if ltp > trade.running_high:
                trade.running_high = ltp

            pnl = trade.unrealised_pnl(ltp)

            # Trailing stop
            should_trail, new_sl = self.risk.should_trail(
                trade.entry, ltp, trade.running_high)
            if should_trail and new_sl > trade.stop_loss:
                trade.stop_loss = new_sl
                log.info("%s: Trail SL -> %.2f | LTP=%.2f | P&L=%.0f",
                         symbol, new_sl, ltp, pnl)
                alert_trail_update(symbol, new_sl, ltp, pnl)

            # Stop-loss breach
            if ltp <= trade.stop_loss:
                log.warning("%s: SL hit at %.2f", symbol, ltp)
                self._exit_trade(trade, ltp, "SL_HIT")
                self.sl_blacklist.add(symbol)
                log.info("%s: Added to SL blacklist for today.", symbol)
                continue

            # CNC safety -- auto exit at 2:45 PM if in loss
            if is_auto_exit_time():
                auto_thr = float(os.getenv("AUTO_EXIT_THRESHOLD", "-0.01"))
                if (ltp - trade.entry) / trade.entry <= auto_thr:
                    log.warning("%s: Auto-exit 2:45 PM CNC safety", symbol)
                    self._exit_trade(trade, ltp, "AUTO_EXIT_WEAK_MARKET")
                    continue

            # Signal flip — only check on candle boundaries (every 5 min)
            # Skip during fast 60s monitor cycles to avoid noise
            if self._is_candle_boundary():
                df = self.broker.get_candles(trade.security_id, symbol, days_back=5)
                if not df.empty and self.engine.should_exit(df, trade.side):
                    log.info("%s: Signal flip -> exit at %.2f | P&L=%.0f",
                             symbol, ltp, pnl)
                    self._exit_trade(trade, ltp, "SIGNAL_FLIP")
                    continue

            log.info("%s [%s]: Holding | LTP=%.2f | SL=%.2f | P&L=%.0f",
                     symbol, SECTOR_MAP.get(symbol, "?"), ltp, trade.stop_loss, pnl)

    def _is_candle_boundary(self) -> bool:
        """True when current time is on or near a 5-minute candle boundary."""
        return datetime.now().minute % 5 == 0

    # =========================================================
    # FORCE EXIT
    # =========================================================
    def force_exit_all(self, reason="CNC_EOD_CUTOFF"):
        if not self.trades:
            return

        log.warning("Force-exiting %d position(s) -- reason: %s",
                    len(self.trades), reason)

        id_symbol_map = {
            str(trade.security_id): symbol
            for symbol, trade in self.trades.items()
        }
        prices = self.broker.get_ltp_batch(id_symbol_map)

        for symbol, trade in list(self.trades.items()):
            ltp = prices.get(str(trade.security_id), 0.0)
            if ltp <= 0:
                ltp = trade.running_high if trade.running_high > trade.entry \
                      else trade.entry
                log.warning("%s: LTP unavailable -- fallback price %.2f",
                            symbol, ltp)
            log.warning("%s: Force exit at %.2f | reason=%s", symbol, ltp, reason)
            self._exit_trade(trade, ltp, reason=reason)

    # =========================================================
    # EXIT TRADE
    # =========================================================
    def _exit_trade(self, trade: Trade, exit_price: float, reason: str):
        pnl = trade.unrealised_pnl(exit_price)

        if BOT_MODE == "live":
            self.broker.place_market_sell(trade.security_id, trade.qty, trade.mode)
        else:
            self.paper_pnl += pnl
            log.info("[PAPER] Simulated SELL %s @ %.2f | P&L=%.0f",
                     trade.symbol, exit_price, pnl)

        self.risk.update_pnl(pnl)

        log_trade({
            "time": datetime.now(), "mode": BOT_MODE.upper(),
            "symbol": trade.symbol, "action": "EXIT", "side": trade.side,
            "qty": trade.qty, "price": exit_price,
            "sl": trade.stop_loss, "target": trade.target,
            "prob_up": "", "pnl": round(pnl, 2),
        })

        log.info("EXIT %s | reason=%s | exit=%.2f | P&L=%.2f",
                 trade.symbol, reason, exit_price, pnl)

        alert_exit(
            symbol=trade.symbol, buy_price=trade.entry,
            sell_price=exit_price, quantity=trade.qty,
            pnl=pnl, reason=reason, trade_mode=trade.mode,
        )

        self.closed_trades.append({
            "symbol": trade.symbol, "entry": trade.entry,
            "exit": exit_price, "qty": trade.qty, "pnl": round(pnl, 2),
        })
        del self.trades[trade.symbol]

    # =========================================================
    # MAIN LOOP
    # =========================================================
    def run(self):
        log.info("=" * 55)
        log.info("%s", MODE_LABEL[BOT_MODE])
        log.info("Capital: Rs.%s | Trade type: %s | Max/sector: %d",
                 f"{CAPITAL:,}", TRADE_MODE.upper(), MAX_PER_SECTOR)
        log.info("Monitor: every %ds | Scan: every %ds",
                 MONITOR_INTERVAL, SCAN_INTERVAL)
        log.info("Watching %d stocks", len(WATCHLIST))
        log.info("=" * 55)

        self.risk.reset_day()
        self.sl_blacklist.clear()
        self._last_scan_time = 0.0

        alert_bot_started(
            capital    = CAPITAL,
            trade_mode = f"{TRADE_MODE.upper()} [{BOT_MODE.upper()}]",
            watchlist  = list(WATCHLIST.keys()),
        )

        if BOT_MODE == "paper":
            _send(
                "PAPER TRADE MODE\n"
                "Monitor: every 60s | Scan: every 5 min\n"
                "Best confidence signal picked each candle.\n"
                f"Max {MAX_PER_SECTOR} stocks per sector.\n"
                "Set BOT_MODE=live when ready."
            )

        daily_summary_sent = False

        while True:

            # Outside market hours
            if not is_market_open():
                if not daily_summary_sent and now_time() >= time_from_str("15:30"):
                    alert_daily_summary(
                        total_pnl = self.risk.daily_pnl,
                        trades    = self.closed_trades,
                        capital   = CAPITAL + self.risk.daily_pnl,
                    )
                    if BOT_MODE == "paper":
                        _send(
                            f"Paper trade day complete\n"
                            f"Simulated P&L: Rs.{self.paper_pnl:+,.0f}\n"
                            f"Trades: {len(self.closed_trades)}\n"
                            f"SL blacklist: {', '.join(self.sl_blacklist) or 'none'}"
                        )
                    daily_summary_sent = True
                    self.closed_trades = []
                    self.paper_pnl     = 0.0
                    self.sl_blacklist.clear()
                    self._last_scan_time = 0.0

                log.info("Market closed. Sleeping 60s...")
                time.sleep(60)
                continue

            daily_summary_sent = False

            # Circuit breaker
            if self.risk.is_halted():
                alert_circuit_breaker(
                    daily_loss = self.risk.daily_pnl,
                    capital    = CAPITAL + self.risk.daily_pnl,
                )
                log.critical("CIRCUIT BREAKER -- halted for today.")
                time.sleep(300)
                continue

            # EOD cutoff
            if is_cutoff_passed():
                self.force_exit_all(reason="CNC_EOD_CUTOFF")
                log.info("Past cutoff. All positions closed.")
                time.sleep(60)
                continue

            now_ts = time.time()

            # ── Monitor positions every 60 seconds ───────────
            # Catches SL hits and trailing stops quickly
            self.monitor_positions()

            # ── Scan for new entries every 5 minutes ─────────
            # Only on candle boundaries — avoids noisy mid-candle signals
            if now_ts - self._last_scan_time >= SCAN_INTERVAL:
                log.info("== Candle scan [%s] %s ==",
                         BOT_MODE.upper(), datetime.now().strftime("%H:%M"))
                self.scan_and_enter()
                self._last_scan_time = now_ts

            # Sleep 60 seconds then monitor again
            time.sleep(MONITOR_INTERVAL)


# ── Entry point ───────────────────────────────────────────────
if __name__ == "__main__":
    bot = LiveBot()
    if BOT_MODE == "test":
        bot.run_test()
    else:
        bot.run()