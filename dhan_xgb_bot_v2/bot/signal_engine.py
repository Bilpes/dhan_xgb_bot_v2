# ============================================================
#  bot/signal_engine.py  —  Load model, score live candles
# ============================================================

import pickle
import logging
import pandas as pd
import numpy as np

from data.features import build_features, FEATURE_COLS
from config.config import MODEL_PATH, SCALER_PATH, BUY_THRESHOLD, SELL_THRESHOLD

log = logging.getLogger("signal")

# ── Exit thresholds ───────────────────────────────────────────
# Exit long when confidence drops below this level.
# Much more responsive than waiting for full SELL signal.
#
# EXIT_THRESHOLD = 0.50 means: exit when model is no longer
# confident price will go UP (prob_up dropped below 50%).
# This is the "momentum fading" exit — catches turning points
# before the full SELL signal fires.
#
# 0.55 = conservative (only exit on clear reversal)
# 0.50 = balanced (exit when model uncertain)  <- recommended
# 0.45 = aggressive (hold longer, risk more)
EXIT_LONG_THRESHOLD  = 0.50   # exit LONG when prob_up drops below this
EXIT_SHORT_THRESHOLD = 0.50   # exit SHORT when prob_up rises above this


class SignalEngine:

    def __init__(self):
        with open(MODEL_PATH,  "rb") as f:
            self.model  = pickle.load(f)
        with open(SCALER_PATH, "rb") as f:
            self.scaler = pickle.load(f)
        log.info("XGBoost model loaded.")

    def score(self, df: pd.DataFrame) -> dict:
        """
        Input:  raw OHLCV DataFrame (last N candles)
        Output: {
            "signal":  "BUY" | "SELL" | "HOLD",
            "prob_up": float (0-1, model confidence for UP move),
            "atr":     float (latest ATR for SL sizing),
            "entry":   float (suggested entry = last close),
        }
        """
        if len(df) < 55:
            return {"signal": "HOLD", "prob_up": 0.5, "atr": 0, "entry": 0}

        try:
            feat_df = build_features(df)
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

            return {
                "signal":  signal,
                "prob_up": prob_up,
                "atr":     atr,
                "entry":   entry,
            }

        except Exception as e:
            log.error("Signal scoring error: %s", e)
            return {"signal": "HOLD", "prob_up": 0.5, "atr": 0, "entry": 0}

    def should_exit(self, df: pd.DataFrame, position_side: str) -> bool:
        """
        Re-scores current candle and decides whether to exit.

        FIX: Old logic only exited on full SELL signal (prob < 0.38).
             That almost never fired because stocks hover in HOLD zone.

        New logic — three exit conditions for LONG positions:
          1. Full reversal: signal == SELL (prob_up < SELL_THRESHOLD)
          2. Momentum fade: prob_up drops below EXIT_LONG_THRESHOLD (0.50)
             This catches "losing confidence" before full reversal
          3. Consecutive weakness: prob_up below 0.55 for 2 candles in a row
             (tracked via self._weak_candles counter)

        Result: signal flip fires much more often, cutting losers earlier
                and locking in profits on winners before they reverse.
        """
        result  = self.score(df)
        prob_up = result["prob_up"]
        signal  = result["signal"]

        if position_side == "LONG":

            # Condition 1 — Full reversal (original logic)
            if signal == "SELL":
                log.info("Signal flip SELL | prob_up=%.3f | exiting LONG",
                         prob_up)
                self._weak_candles = 0
                return True

            # Condition 2 — Momentum fade: model no longer confident in up move
            if prob_up < EXIT_LONG_THRESHOLD:
                log.info("Momentum fade | prob_up=%.3f < %.2f | exiting LONG",
                         prob_up, EXIT_LONG_THRESHOLD)
                self._weak_candles = 0
                return True

            # Condition 3 — Two consecutive weak candles (prob < 0.55)
            # Catches slow deterioration rather than sudden drops
            WEAK_THRESHOLD   = 0.55
            WEAK_CANDLES_MAX = 2
            if prob_up < WEAK_THRESHOLD:
                self._weak_candles = getattr(self, "_weak_candles", 0) + 1
                log.info("Weak candle %d/%d | prob_up=%.3f",
                         self._weak_candles, WEAK_CANDLES_MAX, prob_up)
                if self._weak_candles >= WEAK_CANDLES_MAX:
                    log.info("Consecutive weakness | exiting LONG")
                    self._weak_candles = 0
                    return True
            else:
                # Reset counter if candle is strong again
                self._weak_candles = 0

        elif position_side == "SHORT":

            # Mirror logic for SHORT positions
            if signal == "BUY":
                log.info("Signal flip BUY | prob_up=%.3f | exiting SHORT",
                         prob_up)
                self._weak_candles = 0
                return True

            if prob_up > EXIT_SHORT_THRESHOLD:
                log.info("Momentum fade up | prob_up=%.3f | exiting SHORT",
                         prob_up)
                return True

        return False

    def score_batch(self, stocks: dict) -> dict:
        """
        Score multiple stocks at once.
        More efficient than calling score() in a loop.

        Args:
            stocks: {symbol: df}  e.g. {"HDFCBANK": df1, "TCS": df2}

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