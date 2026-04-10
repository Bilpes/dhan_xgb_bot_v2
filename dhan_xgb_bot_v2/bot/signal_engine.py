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
            "signal":    "BUY" | "SELL" | "HOLD",
            "prob_up":   float  (0–1, model confidence for UP move),
            "atr":       float  (latest ATR for SL sizing),
            "entry":     float  (suggested entry = last close),
        }
        """
        if len(df) < 55:
            return {"signal": "HOLD", "prob_up": 0.5, "atr": 0, "entry": 0}

        try:
            feat_df  = build_features(df)
            if feat_df.empty:
                return {"signal": "HOLD", "prob_up": 0.5, "atr": 0, "entry": 0}

            X_last   = feat_df[FEATURE_COLS].iloc[[-1]]   # only latest row
            X_scaled = self.scaler.transform(X_last)
            prob_up  = float(self.model.predict_proba(X_scaled)[0][1])

            entry    = float(df["close"].iloc[-1])
            atr      = float(feat_df["atr_14"].iloc[-1])

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
        Re-score every candle while in a trade.
        Exit long if signal flips to SELL (model confidence reversal).
        Exit short if signal flips to BUY.
        """
        result = self.score(df)
        if position_side == "LONG"  and result["signal"] == "SELL":
            log.info("Signal flip → exit LONG (prob_up=%.3f)", result["prob_up"])
            return True
        if position_side == "SHORT" and result["signal"] == "BUY":
            log.info("Signal flip → exit SHORT (prob_up=%.3f)", result["prob_up"])
            return True
        return False
