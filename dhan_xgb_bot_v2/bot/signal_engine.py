# ============================================================
# bot/signal_engine.py — Load model, score live candles
# ============================================================

import pickle
import logging
import pandas as pd
import numpy as np

from data.features import build_features, FEATURE_COLS
from config.config import MODEL_PATH, SCALER_PATH, BUY_THRESHOLD, SELL_THRESHOLD

log = logging.getLogger("signal")

EXIT_LONG_THRESHOLD  = 0.42
EXIT_SHORT_THRESHOLD = 0.42
WEAK_THRESHOLD       = 0.48
WEAK_CANDLES_MAX     = 2


class SignalEngine:

    def __init__(self):
        with open(MODEL_PATH, "rb") as f:
            self.model = pickle.load(f)
        with open(SCALER_PATH, "rb") as f:
            self.scaler = pickle.load(f)
        self._weak_candles: dict = {}
        self._nifty_df = None 
        # ── NEW: Nifty50 candles stored here, updated every cycle ──
        # Starts as None — build_features() handles None gracefully
        # (fills nifty features with 0.0 = neutral until first update)
        self._nifty_df: pd.DataFrame = None

        log.info("XGBoost model loaded.")

    # ── NEW: Call this from live_bot.py every 5-min scan cycle ────
    def update_nifty(self, nifty_df: pd.DataFrame):
        """
        Store fresh Nifty50 candles so score() uses them automatically.
        Call this BEFORE scoring any stocks in each scan cycle:

            engine.update_nifty(nifty_candles)
            for symbol, df in watchlist.items():
                result = engine.score(df)
        """
        self._nifty_df = nifty_df
        log.debug("Nifty candles updated: %d rows, last=%s",
                  len(nifty_df), nifty_df.index[-1])

    def score(self, df: pd.DataFrame) -> dict:
        """
        Input: raw OHLCV DataFrame (last N candles)
        Output: {
            "signal": "BUY" | "SELL" | "HOLD",
            "prob_up": float,
            "atr":     float,
            "entry":   float,
        }
        """
        if len(df) < 55:
            return {"signal": "HOLD", "prob_up": 0.5, "atr": 0, "entry": 0}

        try:
            # ── CHANGED: pass self._nifty_df (None is safe) ───────
            feat_df = build_features(df, nifty_df=self._nifty_df)

            if feat_df.empty:
                return {"signal": "HOLD", "prob_up": 0.5, "atr": 0, "entry": 0}

            X_last   = feat_df[FEATURE_COLS].iloc[[-1]]
            X_scaled = self.scaler.transform(X_last)
            prob_up  = float(self.model.predict_proba(X_scaled)[0][1])

            entry = float(df["close"].iloc[-1])
            atr   = float(feat_df["atr_14"].iloc[-1])

            if prob_up >= BUY_THRESHOLD:
                signal = "BUY"
            elif prob_up <= SELL_THRESHOLD:
                signal = "SELL"
            else:
                signal = "HOLD"

            log.debug("Signal: %s | prob_up=%.3f | entry=%.2f | ATR=%.2f",
                      signal, prob_up, entry, atr)

            return {"signal": signal, "prob_up": prob_up,
                    "atr": atr, "entry": entry}

        except Exception as e:
            log.error("Signal scoring error: %s", e)
            return {"signal": "HOLD", "prob_up": 0.5, "atr": 0, "entry": 0}

    def should_exit(self, df: pd.DataFrame, position_side: str,
                    symbol: str = "__default__") -> bool:
        """
        Re-scores current candle and decides whether to exit.
        Nifty context is automatically included via self._nifty_df
        since score() uses it internally — no change needed here.
        """
        result  = self.score(df)   # ← nifty already included via score()
        prob_up = result["prob_up"]
        signal  = result["signal"]

        if position_side == "LONG":

            if signal == "SELL":
                log.info("Signal flip SELL | prob_up=%.3f | exiting LONG [%s]",
                         prob_up, symbol)
                self._weak_candles[symbol] = 0
                return True

            if prob_up < EXIT_LONG_THRESHOLD:
                log.info("Momentum fade | prob_up=%.3f < %.2f | exiting LONG [%s]",
                         prob_up, EXIT_LONG_THRESHOLD, symbol)
                self._weak_candles[symbol] = 0
                return True

            if prob_up < WEAK_THRESHOLD:
                self._weak_candles[symbol] = self._weak_candles.get(symbol, 0) + 1
                log.info("Weak candle %d/%d | prob_up=%.3f [%s]",
                         self._weak_candles[symbol], WEAK_CANDLES_MAX,
                         prob_up, symbol)
                if self._weak_candles[symbol] >= WEAK_CANDLES_MAX:
                    log.info("Consecutive weakness | exiting LONG [%s]", symbol)
                    self._weak_candles[symbol] = 0
                    return True
            else:
                self._weak_candles[symbol] = 0

        elif position_side == "SHORT":

            if signal == "BUY":
                log.info("Signal flip BUY | prob_up=%.3f | exiting SHORT [%s]",
                         prob_up, symbol)
                self._weak_candles[symbol] = 0
                return True

            if prob_up > EXIT_SHORT_THRESHOLD:
                log.info("Momentum fade up | prob_up=%.3f | exiting SHORT [%s]",
                         prob_up, symbol)
                self._weak_candles[symbol] = 0
                return True

        return False

    def reset_symbol(self, symbol: str):
        """Call when a position is closed — clears that symbol's weak counter."""
        self._weak_candles.pop(symbol, None)

    def score_batch(self, stocks: dict) -> dict:
        """
        Score multiple stocks at once.
        Nifty context is shared across all stocks via self._nifty_df —
        call update_nifty() once before score_batch(), not inside it.

        Args:
            stocks: {symbol: df}
        Returns:
            {symbol: score_dict}
        """
        results = {}
        for symbol, df in stocks.items():
            try:
                results[symbol] = self.score(df)
            except Exception as e:
                log.error("score_batch error for %s: %s", symbol, e)
                results[symbol] = {
                    "signal": "HOLD", "prob_up": 0.5,
                    "atr": 0, "entry": 0
                }
        return results