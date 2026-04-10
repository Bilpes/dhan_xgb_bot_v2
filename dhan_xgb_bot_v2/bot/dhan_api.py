# ============================================================
#  bot/dhan_api.py  —  Dhan API wrapper (orders + market data)
# ============================================================

import requests
import logging
import pandas as pd
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path=os.path.join("config", ".env"))

from dhanhq import dhanhq
from config.config import (
    DHAN_CLIENT_ID,
    DHAN_ACCESS_TOKEN,
    CANDLE_INTERVAL,
)

log = logging.getLogger("dhan_api")
BASE_URL = "https://api.dhan.co/v2"


class DhanBroker:

    def __init__(self):
        self.client_id  = DHAN_CLIENT_ID
        self.token      = DHAN_ACCESS_TOKEN
        self.dhan       = dhanhq(self.client_id, self.token)
        self.headers    = {
            "access-token": self.token,
            "client-id":    self.client_id,
            "Content-Type": "application/json",
            "Accept":       "application/json",
        }
        self._ltp_cache = {}    # security_id -> (price, timestamp)
        self._cache_ttl = 4     # seconds
        log.info("Dhan connected — client %s", self.client_id)

    # =========================================================
    #  BATCH LTP — all positions in ONE call
    # =========================================================

    def get_ltp_batch(self, id_symbol_map: dict) -> dict:
        """
        Fetch LTP for multiple stocks in ONE API call.

        Args:
            id_symbol_map: dict of {security_id: symbol_name}
                           e.g. {"1333": "HDFCBANK", "4963": "ICICIBANK"}

        Returns:
            dict: {security_id: price}  e.g. {"1333": 754.7}

        This is called ONCE per candle cycle for ALL open positions.
        Dhan allows 1000 instruments per call at 1 req/sec.
        """
        if not id_symbol_map:
            return {}

        now      = time.time()
        result   = {}
        to_fetch = {}   # security_id -> symbol for uncached

        # Serve fresh cache hits first
        for sid, sym in id_symbol_map.items():
            cached = self._ltp_cache.get(str(sid))
            if cached and (now - cached[1]) < self._cache_ttl:
                result[str(sid)] = cached[0]
            else:
                to_fetch[str(sid)] = sym

        if not to_fetch:
            return result

        # Try Dhan batch endpoint
        dhan_success = False
        try:
            payload = {"NSE_EQ": [int(sid) for sid in to_fetch]}
            resp    = requests.post(
                f"{BASE_URL}/marketfeed/ltp",
                json=payload, headers=self.headers, timeout=10,
            )

            if resp.status_code == 429:
                log.warning("Batch LTP: rate limited — waiting 2s then retry")
                time.sleep(2)
                resp = requests.post(
                    f"{BASE_URL}/marketfeed/ltp",
                    json=payload, headers=self.headers, timeout=10,
                )

            if resp.status_code == 200:
                data = resp.json().get("data", {}).get("NSE_EQ", {})
                for sid_str, val in data.items():
                    price = float(val.get("last_price", 0))
                    if price > 0:
                        result[sid_str]          = price
                        self._ltp_cache[sid_str] = (price, now)
                        to_fetch.pop(sid_str, None)   # remove from fallback list
                log.debug("Batch LTP: got %d prices from Dhan", len(data))
                dhan_success = True
            else:
                log.warning("Batch LTP: HTTP %d — using yfinance fallback",
                            resp.status_code)

        except Exception as e:
            log.warning("Batch LTP exception: %s — using yfinance fallback", e)

        # yfinance fallback for any that Dhan didn't return
        if to_fetch:
            for sid, sym in to_fetch.items():
                price = self._get_ltp_yfinance(sym)
                if price > 0:
                    result[sid]          = price
                    self._ltp_cache[sid] = (price, now)
                    log.info("yfinance LTP %s: %.2f", sym, price)
                else:
                    log.warning("LTP unavailable for %s — will skip this cycle", sym)

        return result

    def get_ltp(self, security_id: str, symbol: str = "") -> float:
        """Single LTP — uses batch internally."""
        prices = self.get_ltp_batch({str(security_id): symbol})
        price  = prices.get(str(security_id), 0.0)
        return price

    def _get_ltp_yfinance(self, symbol: str) -> float:
        """yfinance LTP fallback."""
        try:
            import yfinance as yf
            hist = yf.Ticker(f"{symbol}.NS").history(period="1d", interval="1m")
            if not hist.empty:
                return float(hist["Close"].iloc[-1])
        except Exception as e:
            log.error("yfinance LTP %s: %s", symbol, e)
        return 0.0

    # =========================================================
    #  CANDLE DATA
    # =========================================================

    def get_candles(self, security_id: str, symbol: str,
                    days_back: int = 10) -> pd.DataFrame:
        to_dt    = datetime.now()
        from_dt  = to_dt - timedelta(days=days_back)
        payload  = {
            "securityId":      str(security_id),
            "exchangeSegment": "NSE_EQ",
            "instrument":      "EQUITY",
            "interval":        CANDLE_INTERVAL,
            "oi":              False,
            "fromDate":        from_dt.strftime("%Y-%m-%d 09:15:00"),
            "toDate":          to_dt.strftime("%Y-%m-%d 15:30:00"),
        }
        try:
            resp = requests.post(
                f"{BASE_URL}/charts/intraday",
                json=payload, headers=self.headers, timeout=15,
            )
            if resp.status_code == 200:
                data = resp.json()
                if all(k in data for k in
                       ["open","high","low","close","volume","timestamp"]
                       ) and data["timestamp"]:
                    df = pd.DataFrame({
                        "datetime": pd.to_datetime(
                            data["timestamp"], unit="s", utc=True
                        ).tz_convert("Asia/Kolkata"),
                        "open":   data["open"],
                        "high":   data["high"],
                        "low":    data["low"],
                        "close":  data["close"],
                        "volume": data["volume"],
                    }).set_index("datetime").sort_index()
                    df.index = df.index.tz_localize(None)
                    return df
            log.warning("Dhan candles %s: HTTP %d — yfinance fallback",
                        symbol, resp.status_code)
        except Exception as e:
            log.warning("Dhan candles %s: %s — yfinance fallback", symbol, e)

        return self._get_candles_yfinance(symbol, days_back)

    def _get_candles_yfinance(self, symbol: str,
                               days_back: int = 10) -> pd.DataFrame:
        try:
            import yfinance as yf
            df = yf.download(
                f"{symbol}.NS",
                period=f"{days_back}d", interval=f"{CANDLE_INTERVAL}m",
                auto_adjust=True, progress=False,
            )
            if df.empty:
                return pd.DataFrame()
            df.index.name = "datetime"
            df.columns    = [c.lower() for c in df.columns]
            return df[["open","high","low","close","volume"]].dropna()
        except Exception as e:
            log.error("yfinance candles %s: %s", symbol, e)
            return pd.DataFrame()

    # =========================================================
    #  ORDER PLACEMENT
    # =========================================================

    def place_bracket_order(self, symbol, security_id, quantity,
                             entry_price, stop_loss, target,
                             trade_type="cnc") -> dict:
        product = dhanhq.INTRA if trade_type == "intraday" else dhanhq.CNC
        try:
            resp = self.dhan.place_order(
                security_id=security_id, exchange_segment=dhanhq.NSE,
                transaction_type=dhanhq.BUY, quantity=quantity,
                order_type=dhanhq.LIMIT, product_type=product,
                price=round(entry_price, 2), trigger_price=0,
                bo_profit_value=round(target - entry_price, 2),
                bo_stop_loss_value=round(entry_price - stop_loss, 2),
            )
            log.info("ORDER PLACED | %s | qty=%d | entry=%.2f | SL=%.2f | target=%.2f",
                     symbol, quantity, entry_price, stop_loss, target)
            return resp
        except Exception as e:
            log.error("place_bracket_order %s: %s", symbol, e)
            return {"status": "error", "message": str(e)}

    def place_market_sell(self, security_id, quantity,
                           trade_type="cnc") -> dict:
        product = dhanhq.INTRA if trade_type == "intraday" else dhanhq.CNC
        try:
            resp = self.dhan.place_order(
                security_id=security_id, exchange_segment=dhanhq.NSE,
                transaction_type=dhanhq.SELL, quantity=quantity,
                order_type=dhanhq.MARKET, product_type=product,
                price=0, trigger_price=0,
            )
            log.info("MARKET SELL | security=%s | qty=%d", security_id, quantity)
            return resp
        except Exception as e:
            log.error("place_market_sell %s: %s", security_id, e)
            return {"status": "error", "message": str(e)}

    def get_positions(self) -> pd.DataFrame:
        try:
            resp = self.dhan.get_positions()
            if not resp or resp.get("status") != "success":
                return pd.DataFrame()
            return pd.DataFrame(resp.get("data", []))
        except Exception as e:
            log.error("get_positions: %s", e)
            return pd.DataFrame()