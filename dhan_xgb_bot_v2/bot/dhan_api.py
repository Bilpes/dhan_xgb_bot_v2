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
_DHAN_ORDER_URL = "https://api.dhan.co/v2/orders"


class DhanBroker:

    def __init__(self):
        self.client_id = DHAN_CLIENT_ID
        self.token = DHAN_ACCESS_TOKEN
        self.dhan = dhanhq(self.client_id, self.token)
        self.headers = {
            "access-token": self.token,
            "client-id": self.client_id,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        self._ltp_cache = {}
        self._cache_ttl = 4
        self._quote_cache = {}
        self._quote_cache_ttl = 2
        self._max_retries = int(os.getenv("DHAN_MAX_RETRIES", "2"))
        self._retry_sleep = float(os.getenv("DHAN_RETRY_SLEEP", "0.8"))
        self._retry_timeout = float(os.getenv("DHAN_RETRY_TIMEOUT", "6"))
        log.info("Dhan connected — client %s", self.client_id)

    def _post_with_retries(self, path: str, payload: dict, timeout: float = None):
        timeout = timeout or self._retry_timeout
        last_resp = None
        for attempt in range(1, self._max_retries + 1):
            try:
                resp = requests.post(f"{BASE_URL}{path}", json=payload, headers=self.headers, timeout=timeout)
                last_resp = resp
                if resp.status_code == 429 and attempt < self._max_retries:
                    log.warning("%s: rate limited — retry %d/%d", path, attempt, self._max_retries)
                    time.sleep(self._retry_sleep * attempt)
                    continue
                return resp, True, attempt
            except Exception as e:
                log.warning("%s: attempt %d/%d failed: %s", path, attempt, self._max_retries, e)
                if attempt < self._max_retries:
                    time.sleep(self._retry_sleep * attempt)
                else:
                    return None, False, attempt
        return last_resp, False, self._max_retries

    def get_ltp_batch(self, id_symbol_map: dict) -> dict:
        if not id_symbol_map:
            return {}
        now = time.time()
        result = {}
        to_fetch = {}
        for sid, sym in id_symbol_map.items():
            sid = str(sid)
            cached = self._ltp_cache.get(sid)
            if cached and (now - cached[1]) < self._cache_ttl:
                result[sid] = cached[0]
            else:
                to_fetch[sid] = sym
        if not to_fetch:
            return result
        payload = {"NSE_EQ": [int(sid) for sid in to_fetch]}
        resp, dhan_success, attempts = self._post_with_retries("/marketfeed/ltp", payload, timeout=8)
        partial_success = False
        if resp is not None and resp.status_code == 200:
            try:
                data = resp.json().get("data", {}).get("NSE_EQ", {})
                for sid_str, val in data.items():
                    price = float(val.get("last_price", 0) or 0)
                    if price > 0:
                        result[sid_str] = price
                        self._ltp_cache[sid_str] = (price, now)
                        to_fetch.pop(sid_str, None)
                partial_success = len(to_fetch) > 0
                log.debug("Batch LTP: got %d prices from Dhan", len(data))
            except Exception as e:
                log.warning("Batch LTP parse error: %s", e)
                partial_success = True
                dhan_success = False
        else:
            dhan_success = False
            partial_success = bool(to_fetch)
            if resp is not None:
                log.warning("Batch LTP: HTTP %d — Dhan fetch failed", resp.status_code)
            else:
                log.warning("Batch LTP: request failed after %d attempts", attempts)
        if to_fetch:
            log.warning("Batch LTP incomplete for %d symbol(s) — Dhan success=%s partial=%s", len(to_fetch), dhan_success, partial_success)
        result["__meta__"] = {
            "dhan_success": dhan_success,
            "partial_success": partial_success,
            "fallback_used": False,
            "attempts": attempts,
            "missing": list(to_fetch.keys()),
        }
        return result

    def get_ltp(self, security_id: str, symbol: str = "") -> float:
        prices = self.get_ltp_batch({str(security_id): symbol})
        return prices.get(str(security_id), 0.0)

    def get_quote(self, security_id: str, symbol: str = "") -> dict:
        sid = str(security_id)
        now = time.time()
        cached = self._quote_cache.get(sid)
        if cached and (now - cached[1]) < self._quote_cache_ttl:
            return cached[0]

        payload = {"NSE_EQ": [int(sid)]}
        resp, dhan_success, attempts = self._post_with_retries("/marketfeed/quote", payload, timeout=6)
        meta = {"dhan_success": False, "partial_success": False, "fallback_used": False, "attempts": attempts}

        if resp is None:
            log.warning("get_quote %s: failed after %d attempts", symbol or sid, attempts)
            return {"meta": meta}

        if resp.status_code != 200:
            log.warning("get_quote %s: HTTP %d", symbol or sid, resp.status_code)
            return {"meta": meta}

        try:
            data = resp.json() if resp.content else {}
            node = (data.get("data") or {}).get("NSE_EQ", {})

            q = None
            if isinstance(node, dict):
                q = node.get(sid)
                if q is None and node:
                    q = next(iter(node.values()))
            elif isinstance(node, list) and node:
                q = node[0]

            if not isinstance(q, dict):
                log.warning("get_quote %s: unexpected quote structure", symbol or sid)
                return {"meta": meta}

            depth = q.get("depth") or {}
            buy_depth = depth.get("buy") or []
            sell_depth = depth.get("sell") or []

            bid = (
                q.get("best_bid_price")
                or q.get("bid_price")
                or q.get("bid")
                or q.get("bestBid")
                or q.get("bidPrice")
            )

            ask = (
                q.get("best_ask_price")
                or q.get("ask_price")
                or q.get("ask")
                or q.get("bestAsk")
                or q.get("askPrice")
            )

            if bid in (None, "", 0, 0.0) and buy_depth and isinstance(buy_depth[0], dict):
                bid = buy_depth[0].get("price")

            if ask in (None, "", 0, 0.0) and sell_depth and isinstance(sell_depth[0], dict):
                ask = sell_depth[0].get("price")

            ltp = (
                q.get("last_price")
                or q.get("ltp")
                or q.get("last_traded_price")
                or (q.get("ohlc") or {}).get("close")
                or q.get("close")
                or 0
            )

            vol = q.get("volume") or q.get("vol") or q.get("traded_quantity") or 0

            bid_f = float(bid) if bid not in (None, "", 0, 0.0) else None
            ask_f = float(ask) if ask not in (None, "", 0, 0.0) else None
            ltp_f = float(ltp) if ltp not in (None, "") else 0.0
            vol_f = float(vol) if vol not in (None, "") else 0.0

            if ltp_f <= 0:
                ltp_f = self.get_ltp(sid, symbol)

            spread_abs = None
            spread_pct = None
            if bid_f is not None and ask_f is not None and ltp_f > 0:
                spread_abs = ask_f - bid_f
                spread_pct = spread_abs / ltp_f

            out = {
                "bid": bid_f,
                "ask": ask_f,
                "ltp": ltp_f,
                "volume": vol_f,
                "spread_abs": spread_abs,
                "spread_pct": spread_pct,
                "meta": {"dhan_success": True, "partial_success": False, "fallback_used": False, "attempts": attempts},
                "raw": q,
            }

            self._quote_cache[sid] = (out, now)
            return out

        except Exception as e:
            log.warning("get_quote %s failed: %s", symbol or sid, e)
            return {"meta": meta}
    
    def _get_ltp_yfinance(self, symbol: str) -> float:
        return 0.0

    def get_candles(self, security_id: str, symbol: str, days_back: int = 10) -> pd.DataFrame:
        to_dt = datetime.now()
        from_dt = to_dt - timedelta(days=days_back)
        payload = {
            "securityId": str(security_id),
            "exchangeSegment": "NSE_EQ",
            "instrument": "EQUITY",
            "interval": CANDLE_INTERVAL,
            "oi": False,
            "fromDate": from_dt.strftime("%Y-%m-%d 09:15:00"),
            "toDate": to_dt.strftime("%Y-%m-%d 15:30:00"),
        }
        try:
            resp = requests.post(f"{BASE_URL}/charts/intraday", json=payload, headers=self.headers, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                if all(k in data for k in ["open", "high", "low", "close", "volume", "timestamp"]) and data["timestamp"]:
                    df = pd.DataFrame({
                        "datetime": pd.to_datetime(data["timestamp"], unit="s", utc=True).tz_convert("Asia/Kolkata"),
                        "open": data["open"],
                        "high": data["high"],
                        "low": data["low"],
                        "close": data["close"],
                        "volume": data["volume"],
                    }).set_index("datetime").sort_index()
                    df.index = df.index.tz_localize(None)
                    return df
            log.warning("Dhan candles %s: HTTP %d — no fallback in production live path", symbol, resp.status_code)
        except Exception as e:
            log.warning("Dhan candles %s: %s — no fallback in production live path", symbol, e)
        return pd.DataFrame()

    def place_bracket_order(self, symbol, security_id, quantity, entry_price, stop_loss, target, trade_type="cnc") -> dict:
        product = dhanhq.INTRA if trade_type == "intraday" else dhanhq.CNC
        try:
            resp = self.dhan.place_order(
                security_id=security_id,
                exchange_segment=dhanhq.NSE,
                transaction_type=dhanhq.BUY,
                quantity=quantity,
                order_type=dhanhq.LIMIT,
                product_type=product,
                price=round(entry_price, 2),
                trigger_price=0,
                validity="DAY",
                tag=f"BOT-{symbol[:6]}",
            )
            log.info("BUY ORDER PLACED | %s | qty=%d | entry=%.2f | SL=%.2f | target=%.2f", symbol, quantity, entry_price, stop_loss, target)
            return resp
        except Exception as e:
            log.error("place_bracket_order %s: %s", symbol, e)
            return {"status": "error", "message": str(e)}

    def place_market_sell(self, security_id, quantity, trade_type="cnc") -> dict:
        product = dhanhq.INTRA if trade_type == "intraday" else dhanhq.CNC
        try:
            resp = self.dhan.place_order(
                security_id=security_id,
                exchange_segment=dhanhq.NSE,
                transaction_type=dhanhq.SELL,
                quantity=quantity,
                order_type=dhanhq.MARKET,
                product_type=product,
                price=0,
                trigger_price=0,
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
