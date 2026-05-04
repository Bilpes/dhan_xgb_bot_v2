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
    NO_NEW_TRADE_AFTER, NO_NEW_TRADE_BEFORE, TRADE_LOG, LOG_FILE, CAPITAL,
)

MAX_PER_SECTOR       = int(os.getenv("MAX_PER_SECTOR", "2"))
NEW_TRADE_LOSS_PAUSE = float(os.getenv("NEW_TRADE_LOSS_PAUSE", "-0.03"))

MONITOR_INTERVAL = 60    # check positions every 60 seconds
SCAN_INTERVAL    = 300   # scan for new entries every 5 minutes

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
    # ROTATION — exit weak position to enter stronger signal
    # =========================================================
    def _should_rotate(self, new_prob: float) -> tuple:
        """
        Returns (should_rotate: bool, symbol_to_exit: str|None)
        Rotates only if:
          1. New signal confidence > existing + ROTATION_MIN_EDGE (5%)
          2. Existing position profit > ROTATION_MIN_PROFIT (0.5%)
        """
        if not self.trades:
            return False, None

        rotation_min_profit = float(os.getenv("ROTATION_MIN_PROFIT", "0.005"))
        rotation_min_edge   = float(os.getenv("ROTATION_MIN_EDGE", "0.05"))

        weakest_symbol = None
        weakest_prob   = new_prob

        id_symbol_map = {
            str(t.security_id): sym
            for sym, t in self.trades.items()
        }
        prices = self.broker.get_ltp_batch(id_symbol_map)

        for symbol, trade in self.trades.items():
            ltp = prices.get(str(trade.security_id), 0.0)
            if ltp <= 0:
                continue

            profit_pct = (ltp - trade.entry) / trade.entry
            if profit_pct < rotation_min_profit:
                log.info("%s: Rotation skip -- profit %.2f%% below min",
                         symbol, profit_pct * 100)
                continue

            df = self.broker.get_candles(trade.security_id, symbol, days_back=5)
            if df.empty:
                continue
            existing_prob = self.engine.score(df)["prob_up"]

            if new_prob >= existing_prob + rotation_min_edge:
                if weakest_symbol is None or existing_prob < weakest_prob:
                    weakest_symbol = symbol
                    weakest_prob   = existing_prob
                    log.info("Rotation candidate: exit %s (%.3f) for new (%.3f)",
                             symbol, existing_prob, new_prob)

        return (weakest_symbol is not None), weakest_symbol

    # =========================================================
    # SYNC WITH DHAN — live mode only
    # Detects positions closed by Dhan bracket order (SL/target hit)
    # and syncs bot memory accordingly
    # =========================================================
    def sync_with_dhan(self):
        """
        Live mode only — syncs bot memory with Dhan actual positions.
        Since we use simple LIMIT orders (not bracket), positions only
        disappear from Dhan if manually sold or bot's market sell fired.
        """
        if BOT_MODE != "live":
            return
        if not self.trades:
            return

        try:
            dhan_positions = self.broker.get_positions()

            # SAFETY GUARD: if API returns empty, could be a glitch
            # Only act if we have multiple confirmation attempts
            if dhan_positions.empty:
                log.warning("sync_with_dhan: Dhan returned no positions "
                            "-- could be API glitch, skipping auto-close")
                # Do NOT auto-close here — bot's 60s monitor will catch
                # real SL hits via LTP check. Only log the warning.
                return

            # Build set of symbols open on Dhan
            col = None
            for c in ["tradingSymbol", "trading_symbol", "symbol"]:
                if c in dhan_positions.columns:
                    col = c
                    break

            if col is None:
                log.warning("sync_with_dhan: cannot find symbol column")
                return

            open_on_dhan = set(dhan_positions[col].str.upper().tolist())
            log.debug("sync_with_dhan: open on Dhan = %s", open_on_dhan)

            for symbol, trade in list(self.trades.items()):
                if symbol.upper() not in open_on_dhan:
                    log.warning("%s: Not found in Dhan positions -- manually "
                                "sold or order rejected?", symbol)
                    ltp = self.broker.get_ltp(str(trade.security_id), symbol)
                    exit_price = ltp if ltp > 0 else trade.stop_loss
                    self._exit_trade(trade, exit_price, "CLOSED_ON_DHAN")
                    self.sl_blacklist.add(symbol)

        except Exception as e:
            log.error("sync_with_dhan failed: %s", e)

    # =========================================================
    # SCAN AND ENTER — ranked by confidence, with rotation
    # =========================================================
    def scan_and_enter(self):
        if now_time() < time_from_str(NO_NEW_TRADE_BEFORE):
            log.info("Waiting for market to settle until %s ...", NO_NEW_TRADE_BEFORE)
            return

        if no_new_trades():
            return

        if self.risk.daily_pnl / CAPITAL <= NEW_TRADE_LOSS_PAUSE:
            log.warning("Daily loss %.1f%% -- pausing new entries.",
                        self.risk.daily_pnl / CAPITAL * 100)
            return

        max_reached = len(self.trades) >= MAX_OPEN_TRADES

        log.info("-- Full scan: ranking all eligible stocks --")

        # Step 1: Collect ALL BUY candidates
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
                entry = result["entry"]
                atr   = result["atr"]
                sl    = self.risk.calc_stop_loss(entry, atr, TRADE_MODE)
                target= self.risk.calc_target(entry, sl)
                log.info("  Candidate: %s prob=%.3f | CP=%.2f | SL=%.2f | TP=%.0f [%s]",
                          symbol, result["prob_up"], entry, sl, target, stock_sector)
               
        # Step 2: Pick best signal
        if not candidates:
            log.info("No BUY signals this scan.")
            return

        candidates.sort(key=lambda x: x[2]["prob_up"], reverse=True)
        best_symbol, best_sec_id, best_result = candidates[0]
        stock_sector = SECTOR_MAP.get(best_symbol, "?")

        log.info("Top signal: %s prob=%.3f (from %d candidates)",
                 best_symbol, best_result["prob_up"], len(candidates))

        # Step 3: Handle max trades via rotation
        if max_reached:
            should_rotate, exit_symbol = self._should_rotate(best_result["prob_up"])
            if should_rotate:
                trade_to_exit = self.trades[exit_symbol]
                id_map = {str(trade_to_exit.security_id): exit_symbol}
                prices = self.broker.get_ltp_batch(id_map)
                ltp = prices.get(str(trade_to_exit.security_id), trade_to_exit.entry)
                log.info("ROTATION: exiting %s -> entering %s", exit_symbol, best_symbol)
                self._exit_trade(trade_to_exit, ltp, "ROTATION_BETTER_SIGNAL")
            else:
                log.info("Max trades reached, no rotation opportunity.")
                return

        # Step 4: Calculate and place entry
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
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "mode": "LIVE", "symbol": symbol,
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
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "mode": "PAPER", "symbol": symbol,
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
    # =========================================================
    def monitor_positions(self):
        if not self.trades:
            return

        # ONE batch call for all open positions
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

            # FIX: Candle LOW check — catches SL hits that recover before
            # the 60s monitor cycle runs. Checks actual candle low price
            # which is more reliable than LTP snapshot.
            if self._is_candle_boundary():
                try:
                    df_check = self.broker.get_candles(
                        trade.security_id, symbol, days_back=1)
                    if not df_check.empty:
                        last_low = float(df_check["low"].iloc[-1])
                        if last_low <= trade.stop_loss:
                            log.warning(
                                "%s: SL hit via candle low=%.2f <= SL=%.2f",
                                symbol, last_low, trade.stop_loss)
                            self._exit_trade(trade, trade.stop_loss, "SL_HIT")
                            self.sl_blacklist.add(symbol)
                            log.info("%s: Added to SL blacklist.", symbol)
                            continue
                except Exception as e:
                    log.warning("%s: Candle low SL check failed: %s", symbol, e)

            # LTP stop-loss breach (fast check every 60s)
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

            # Signal flip — only on candle boundary to avoid mid-candle noise
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
                log.warning("%s: LTP unavailable -- fallback %.2f", symbol, ltp)
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
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "mode": BOT_MODE.upper(),
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

            if self.risk.is_halted():
                alert_circuit_breaker(
                    daily_loss = self.risk.daily_pnl,
                    capital    = CAPITAL + self.risk.daily_pnl,
                )
                log.critical("CIRCUIT BREAKER -- halted for today.")
                time.sleep(300)
                continue

            if is_cutoff_passed():
                self.force_exit_all(reason="CNC_EOD_CUTOFF")
                log.info("Past cutoff. All positions closed.")
                time.sleep(60)
                continue

            now_ts = time.time()

            # Monitor every 60s — catches SL and trailing fast
            self.monitor_positions()

            # Every 5 min — sync with Dhan + scan for new entries
            if now_ts - self._last_scan_time >= SCAN_INTERVAL:
                log.info("== Candle scan [%s] %s ==",
                         BOT_MODE.upper(), datetime.now().strftime("%H:%M"))
                self.sync_with_dhan()      # live mode: detect bracket SL/target hits
                self.scan_and_enter()
                self._last_scan_time = now_ts

            time.sleep(MONITOR_INTERVAL)


# ── Entry point ───────────────────────────────────────────────
if __name__ == "__main__":
    bot = LiveBot()
    if BOT_MODE == "test":
        bot.run_test()
    else:
        bot.run()