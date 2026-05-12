# ============================================================
# bot/signal_engine.py — XGBoost signal scoring for live bot
#
# Fully aligned with:
#   trade_policy.py  — ATR_SL_MULT, ATR_TP_MULT (live execution only)
#                      TP_PCT, SL_PCT (what the model was trained to predict)
#                      BUY_THRESHOLD_DEFAULT, MIN_RR_RATIO,
#                      EXIT_LONG_THRESHOLD, WEAK_THRESHOLD,
#                      WEAK_CANDLES_MAX, BLOCKED_SYMBOLS
#   data/features.py — build_features(), FEATURE_COLS
#   models/train.py  — percentage-based labels (TP_PCT / SL_PCT)
#
# IMPORTANT DESIGN:
#   The model predicts: "will price rise TP_PCT% before falling SL_PCT%?"
#   Live SL/TP are placed using ATR (dynamic, adapts to volatility).
#   These two are intentionally different — train label = what to predict,
#   live SL/TP = how to execute it. This is correct and expected.
# ============================================================

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ── Trade quality filters ─────────────────────────
MIN_VOLUME_RATIO = 1.2     # avoid dead candles
MIN_ATR_PCT      = 0.003   # avoid low volatility chop
MODEL_THRESHOLD  = 0.60    # confidence threshold


from data.features import build_features, FEATURE_COLS
from bot.trade_policy import (
    ATR_PERIOD,
    ATR_SL_MULT,
    ATR_TP_MULT,
    BUY_THRESHOLD_DEFAULT,
    MIN_RR_RATIO,
    EXIT_LONG_THRESHOLD,
    EXIT_SHORT_THRESHOLD,
    WEAK_THRESHOLD,
    WEAK_CANDLES_MAX,
    BLOCKED_SYMBOLS,
)
from config.config import (
    MODEL_PATH,
    SCALER_PATH,
    BACKUP_MODEL_PATH,
    BACKUP_SCALER_PATH,
    STOP_LOSS_PCT,
    MIN_VOLUME_RATIO_CONFIRM,
    MIN_CANDLE_BODY_PCT,
    MAX_DISTANCE_FROM_EMA20,
)

log = logging.getLogger("signal_engine")

# ── Null result returned on any failure ──────────────────────
_NULL_RESULT = {
    "signal":  "HOLD",
    "prob_up": 0.0,
    "entry":   0.0,
    "sl":      0.0,
    "target":  0.0,
    "rr":      0.0,
    "atr":     0.0,
    "reason":  "null",
}


def _load_model_pair(model_path: str, scaler_path: str) -> tuple:
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    # Support both old and new model formats
    if isinstance(model_data, dict):
        model = model_data["model"]
    else:
        model = model_data

    return model, scaler


class SignalEngine:
    """
    Wraps the XGBoost model for live scoring.

    Public API consumed by live_bot.py:
      .score(df, symbol)               -> signal dict
      .should_exit(df, side, symbol)   -> bool
      .update_nifty(nifty_df)          -> None
      .reset_symbol(symbol)            -> None
      .reload()                        -> None  (called after auto_retrain)
    """

    def __init__(self, buy_threshold: float = BUY_THRESHOLD_DEFAULT):
        self.buy_threshold  = buy_threshold
        self.model          = None
        self.scaler         = None
        self._nifty_df: Optional[pd.DataFrame] = None
        # Per-symbol weak-candle counter for signal-deterioration exit
        self._weak_counts: dict[str, int] = {}
        self._load()

    # ── Model loading (with backup fallback) ──────────────────
    def _load(self):
        for mp, sp in [
            (MODEL_PATH,        SCALER_PATH),
            (BACKUP_MODEL_PATH, BACKUP_SCALER_PATH),
        ]:
            if Path(mp).exists() and Path(sp).exists():
                try:
                    self.model, self.scaler = _load_model_pair(mp, sp)
                    log.info("Model loaded: %s", mp)
                    return
                except Exception as e:
                    log.warning("Failed to load %s: %s", mp, e)
        raise FileNotFoundError(
            f"No valid model found at {MODEL_PATH} or {BACKUP_MODEL_PATH}.\n"
            "Run:  python models/train.py"
        )

    def reload(self):
        """Hot-reload model after auto_retrain deploys a new one."""
        log.info("Reloading model from disk...")
        self._load()

    # ── Nifty context injection ───────────────────────────────
    def update_nifty(self, nifty_df: pd.DataFrame):
        self._nifty_df = nifty_df.copy()

    # ── Per-symbol state reset (called after exit) ────────────
    def reset_symbol(self, symbol: str):
        self._weak_counts.pop(symbol, None)

    # ── Core scoring ──────────────────────────────────────────
    def score(self, df: pd.DataFrame, symbol: str = "STOCK") -> dict:
        """
        Score the latest candle and return a signal dict.

        Entry / SL / Target are computed with ATR-based volatility logic.

        Returns:
            {
                "signal":  "BUY" | "HOLD",
                "prob_up": float,
                "entry":   float,
                "sl":      float,
                "target":  float,
                "rr":      float,
                "atr":     float,
                "reason":  str,
            }
        """

        # ─────────────────────────────────────────
        # Basic validation
        # ─────────────────────────────────────────
        if symbol.upper() in BLOCKED_SYMBOLS:
            return {**_NULL_RESULT, "reason": "blocked_symbol"}

        if df is None or df.empty or len(df) < 50:
            return {**_NULL_RESULT, "reason": "insufficient_data"}

        # ─────────────────────────────────────────
        # Feature generation
        # ─────────────────────────────────────────
        try:
            feat = build_features(
                df.copy(),
                nifty_df=self._nifty_df,
                symbol=symbol
            )

        except Exception as e:
            log.warning("%s: build_features failed: %s", symbol, e)

            return {
                **_NULL_RESULT,
                "reason": f"feature_error:{e}"
            }

        if feat.empty:
            return {**_NULL_RESULT, "reason": "empty_features"}

        row = feat.iloc[-1]

        missing = [c for c in FEATURE_COLS if c not in feat.columns]

        if missing:
            log.warning("%s: missing features: %s", symbol, missing[:5])

            return {
                **_NULL_RESULT,
                "reason": "missing_features"
            }

        # ─────────────────────────────────────────
        # Model prediction
        # ─────────────────────────────────────────
        X_raw = row[FEATURE_COLS].values.reshape(1, -1).astype(np.float32)

        if np.isnan(X_raw).any():
            return {
                **_NULL_RESULT,
                "reason": "nan_in_features"
            }

        try:
            X_scaled = self.scaler.transform(X_raw)

            prob_up = float(
                self.model.predict_proba(X_scaled)[0, 1]
            )

        except Exception as e:
            log.warning("%s: model predict failed: %s", symbol, e)

            return {
                **_NULL_RESULT,
                "reason": f"predict_error:{e}"
            }

        # ─────────────────────────────────────────
        # Trade quality filters
        # ─────────────────────────────────────────
        vol_ratio = float(row.get("vol_ratio", 0.0))
        atr_pct   = float(row.get("atr_pct", 0.0))

        # Volume filter
        if vol_ratio < MIN_VOLUME_RATIO:

            return {
                **_NULL_RESULT,
                "prob_up": round(prob_up, 4),
                "reason": f"low_volume:{vol_ratio:.2f}",
            }

        # Volatility filter
        if atr_pct < MIN_ATR_PCT:

            return {
                **_NULL_RESULT,
                "prob_up": round(prob_up, 4),
                "reason": f"low_atr:{atr_pct:.4f}",
            }

        # ML confidence filter
        if prob_up < MODEL_THRESHOLD:

            return {
                **_NULL_RESULT,
                "prob_up": round(prob_up, 4),
                "reason": f"low_confidence:{prob_up:.3f}",
            }

        # ─────────────────────────────────────────
        # Price + ATR
        # ─────────────────────────────────────────
        entry = float(row.get("close", 0.0))
        atr   = float(row.get("atr_14", 0.0))

        if entry <= 0 or atr <= 0 or np.isnan(atr):

            return {
                **_NULL_RESULT,
                "prob_up": prob_up,
                "reason": "invalid_atr_or_price"
            }

        # ─────────────────────────────────────────
        # ATR-based SL / Target
        # ─────────────────────────────────────────
        sl_atr = entry - ATR_SL_MULT * atr
        tp_atr = entry + ATR_TP_MULT * atr

        # Hard SL cap
        sl_cap = entry * (1 - STOP_LOSS_PCT)

        sl     = round(max(sl_atr, sl_cap), 2)
        target = round(tp_atr, 2)

        risk   = entry - sl
        reward = target - entry

        rr = round(reward / risk, 3) if risk > 0 else 0.0

        # ─────────────────────────────────────────
        # Entry quality filters
        # ─────────────────────────────────────────
        current_open  = float(df["open"].iloc[-1])
        current_close = float(df["close"].iloc[-1])

        prev_high = float(df["high"].iloc[-2])

        # Candle body %
        body_pct = abs(current_close - current_open) / current_open

        # Breakout confirmation
        breakout_ok = (
            current_close > prev_high
            and vol_ratio >= MIN_VOLUME_RATIO_CONFIRM
        )

        # Candle strength
        body_ok = (
            body_pct >= MIN_CANDLE_BODY_PCT
        )

        # ─────────────────────────────────────────
        # Trend structure
        # ─────────────────────────────────────────
        ema20 = float(row["ema_20"])
        ema50 = float(row["ema_50"])
        vwap  = float(row["vwap"])

        trend_ok = (
            ema20 > ema50
            and current_close > vwap
        )

        # ─────────────────────────────────────────
        # Market regime filter
        # ─────────────────────────────────────────
        market_ok = (
            row["nifty_above_ema20"] == 1
            and row["nifty_trend"] == 1
        )

        # ─────────────────────────────────────────
        # Volatility expansion
        # ─────────────────────────────────────────
        volatility_expansion = (
            float(row["atr_ratio"]) > 1.1
        )

        # ─────────────────────────────────────────
        # Avoid extended entries
        # ─────────────────────────────────────────
        distance_from_ema20 = float(
            row["dist_from_ema20"]
        )

        not_extended = (
            distance_from_ema20 <= MAX_DISTANCE_FROM_EMA20
        )

        # ─────────────────────────────────────────
        # Avoid lunch-time chop
        # ─────────────────────────────────────────
        avoid_lunch = (
            12 <= int(row["hour"]) <= 13
        )

        # ─────────────────────────────────────────
        # Dynamic confidence threshold
        # ─────────────────────────────────────────
        dynamic_threshold = self.buy_threshold

        market_strong = (
            row["nifty_above_ema20"] == 1
            and row["nifty_trend"] == 1
            and float(row["atr_ratio"]) > 1.1
        )

        market_weak = (
            row["nifty_above_ema20"] == 0
            or row["nifty_trend"] == 0
            or float(row["atr_ratio"]) < 1.0
        )

        # Aggressive during strong trends
        if market_strong:
            dynamic_threshold = 0.58

        # Defensive during weak/choppy markets
        elif market_weak:
            dynamic_threshold = 0.70

        # ─────────────────────────────────────────
        # Signal gate
        # ─────────────────────────────────────────
        signal = "HOLD"
        reason = ""

        if prob_up < dynamic_threshold:

            reason = (
                f"prob_up={prob_up:.3f} "
                f"< threshold={dynamic_threshold:.2f}"
            )

        elif rr < MIN_RR_RATIO:

            reason = (
                f"R:R={rr:.2f} < min={MIN_RR_RATIO}"
            )

        elif sl <= 0:

            reason = "sl<=0"

        elif target <= entry:

            reason = "target<=entry"

        elif not breakout_ok:

            reason = "breakout_not_confirmed"

        elif not body_ok:

            reason = (
                f"weak_candle_body={body_pct:.4f}"
            )

        elif not trend_ok:

            reason = "trend_filter_failed"

        elif not market_ok:

            reason = "market_regime_bad"

        elif not volatility_expansion:

            reason = "low_volatility_expansion"

        elif not not_extended:

            reason = (
                f"too_far_from_ema20="
                f"{distance_from_ema20:.4f}"
            )

        elif avoid_lunch:

            reason = "lunch_hour_chop"

        else:

            signal = "BUY"

            self._weak_counts.pop(symbol, None)

        # ─────────────────────────────────────────
        # Debug logging
        # ─────────────────────────────────────────
        log.debug(
            "%s prob=%.3f signal=%s reason=%s "
            "body=%.4f vol=%.2f trend=%s "
            "market=%s vol_exp=%s "
            "entry=%.2f SL=%.2f TP=%.2f "
            "R:R=%.2f ATR=%.4f",

            symbol,
            prob_up,
            signal,
            reason,
            body_pct,
            vol_ratio,
            trend_ok,
            market_ok,
            volatility_expansion,
            entry,
            sl,
            target,
            rr,
            atr,
        )

        # ─────────────────────────────────────────
        # Final response
        # ─────────────────────────────────────────
        return {
            "signal":  signal,
            "prob_up": round(prob_up, 4),
            "entry":   round(entry, 2),
            "sl":      sl,
            "target":  target,
            "rr":      rr,
            "atr":     round(atr, 4),
            "atr_ratio": round(float(row["atr_ratio"]), 3),
            "reason":  reason,          
        }

    
    # ── Exit signal (signal flip / deterioration) ─────────────
    def should_exit(
        self, df: pd.DataFrame, side: str, symbol: str = "STOCK"
    ) -> bool:
        """
        True if the model recommends closing the open position.

        LONG exit triggers:
          1. Hard:  prob_up < EXIT_LONG_THRESHOLD  (immediate exit)
          2. Soft:  N consecutive candles where prob_up < WEAK_THRESHOLD

        Weak candle counter resets to 0 whenever prob_up >= WEAK_THRESHOLD.
        Counter is cleared entirely on trade exit via reset_symbol().
        """
        result = self.score(df, symbol=symbol)
        prob   = result["prob_up"]

        if side == "LONG":
            if prob < EXIT_LONG_THRESHOLD:
                log.info("%s: hard exit — prob_up=%.3f < EXIT_LONG=%.3f",
                         symbol, prob, EXIT_LONG_THRESHOLD)
                return True

            if prob < WEAK_THRESHOLD:
                count = self._weak_counts.get(symbol, 0) + 1
                self._weak_counts[symbol] = count
                log.info("%s: weak candle %d/%d  prob=%.3f",
                         symbol, count, WEAK_CANDLES_MAX, prob)
                if count >= WEAK_CANDLES_MAX:
                    log.info("%s: soft exit — %d consecutive weak candles",
                             symbol, count)
                    return True
            else:
                self._weak_counts[symbol] = 0

        elif side == "SHORT":
            if prob > EXIT_SHORT_THRESHOLD:
                log.info("%s: short exit — prob_up=%.3f > EXIT_SHORT=%.3f",
                         symbol, prob, EXIT_SHORT_THRESHOLD)
                return True

        return False
