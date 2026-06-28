# ============================================================
# bot/signal_engine.py — XGBoost signal scoring for live bot
#
# Fully aligned with:
#   trade_policy.py  — ATR_SL_MULT, ATR_TP_MULT (live execution only)
#                      BUY_THRESHOLD_DEFAULT, BUY_THRESHOLD_WEAK,
#                      MIN_RR_RATIO, EXIT_LONG_THRESHOLD,
#                      EXIT_SHORT_THRESHOLD, WEAK_THRESHOLD,
#                      WEAK_CANDLES_MAX, BLOCKED_SYMBOLS
#   data/features.py — build_features(), FEATURE_COLS
#   config/config.py — paths, filters (VWAP/volume/ATR), STOP_LOSS_PCT
#
# IMPORTANT DESIGN:
#   The model predicts: "will price rise by the trained target before
#   falling by the trained stop?"
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

from data.features import build_features, FEATURE_COLS
from bot.trade_policy import (
    ATR_SL_MULT,
    ATR_TP_MULT,
    BUY_THRESHOLD_DEFAULT,
    BUY_THRESHOLD_WEAK,
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
    MIN_VOLUME_RATIO,
    MIN_ATR_PCT,
    REQUIRE_BREAKOUT_CONFIRMATION,
    REQUIRE_VWAP_CONFIRM,
    TREND_STRENGTH_ENABLED,
    MAX_DISTANCE_FROM_VWAP,
    AVOID_LUNCH_HOURS,
)


log = logging.getLogger("signal_engine")


PROBABILITY_FLOOR = 0.0
PROBABILITY_CAP   = 0.95
ATR_FLOOR_PCT     = 0.003


_NULL_RESULT = {
    "signal":    "HOLD",
    "prob_up":   0.0,
    "entry":     0.0,
    "sl":        0.0,
    "target":    0.0,
    "rr":        0.0,
    "atr":       0.0,
    "atr_ratio": 0.0,
    "reason":    "null",
}


def _load_model_pair(model_path: str, scaler_path: str) -> tuple:
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    model = model_data["model"] if isinstance(model_data, dict) else model_data
    return model, scaler


class SignalEngine:
    """
    Wraps the XGBoost model for live scoring.

    Public API consumed by live_bot.py:
      .score(df, symbol)               -> signal dict
      .should_exit(df, side, symbol)   -> bool
      .update_nifty(nifty_df)          -> None
      .reset_symbol(symbol)            -> None
      .reload()                        -> None
    """

    def __init__(self, buy_threshold: float = BUY_THRESHOLD_DEFAULT):
        self.buy_threshold = buy_threshold
        self.model  = None
        self.scaler = None
        self._nifty_df: Optional[pd.DataFrame] = None
        self._weak_counts: dict[str, int] = {}
        self._load()

    def _load(self):
        for model_path, scaler_path in [
            (MODEL_PATH, SCALER_PATH),
            (BACKUP_MODEL_PATH, BACKUP_SCALER_PATH),
        ]:
            if Path(model_path).exists() and Path(scaler_path).exists():
                try:
                    self.model, self.scaler = _load_model_pair(model_path, scaler_path)
                    log.info("Model loaded: %s", model_path)
                    return
                except Exception as e:
                    log.warning("Failed to load %s: %s", model_path, e)

        raise FileNotFoundError(
            f"No valid model found at {MODEL_PATH} or {BACKUP_MODEL_PATH}.\n"
            "Run: python models/train.py"
        )

    def reload(self):
        log.info("Reloading model from disk...")
        self._load()

    def update_nifty(self, nifty_df: pd.DataFrame):
        self._nifty_df = nifty_df.copy()

    def reset_symbol(self, symbol: str):
        self._weak_counts.pop(symbol, None)

    def score(
        self,
        df: pd.DataFrame,
        symbol: str = "STOCK",
        for_exit: bool = False,
    ) -> dict:
        if symbol.upper() in BLOCKED_SYMBOLS:
            return {**_NULL_RESULT, "reason": "blocked_symbol"}

        if df is None or df.empty or len(df) < 50:
            return {**_NULL_RESULT, "reason": "insufficient_data"}

        try:
            feat = build_features(df.copy(), nifty_df=self._nifty_df, symbol=symbol)
        except Exception as e:
            log.warning("%s: build_features failed: %s", symbol, e)
            return {**_NULL_RESULT, "reason": f"feature_error:{e}"}

        if feat.empty:
            return {**_NULL_RESULT, "reason": "empty_features"}

        missing = [col for col in FEATURE_COLS if col not in feat.columns]
        if missing:
            log.warning("%s: missing features: %s", symbol, missing[:5])
            return {**_NULL_RESULT, "reason": "missing_features"}

        row   = feat.iloc[-1]
        x_raw = row[FEATURE_COLS].values.reshape(1, -1).astype(np.float32)

        if np.isnan(x_raw).any():
            return {**_NULL_RESULT, "reason": "nan_in_features"}

        try:
            x_scaled = self.scaler.transform(x_raw)
            raw_prob  = float(self.model.predict_proba(x_scaled)[0, 1])
            prob_up   = max(PROBABILITY_FLOOR, min(raw_prob, PROBABILITY_CAP))
        except Exception as e:
            log.warning("%s: model predict failed: %s", symbol, e)
            return {**_NULL_RESULT, "reason": f"predict_error:{e}"}

        vol_ratio = float(row.get("vol_ratio", 0.0))
        atr_pct   = float(row.get("atr_pct",   0.0))

        if not for_exit:
            if vol_ratio < MIN_VOLUME_RATIO:
                return {
                    **_NULL_RESULT,
                    "prob_up":     round(prob_up, 4),
                    "raw_prob_up": round(raw_prob, 4),
                    "reason":      f"low_volume:{vol_ratio:.2f}",
                }
            if atr_pct < MIN_ATR_PCT:
                return {
                    **_NULL_RESULT,
                    "prob_up":     round(prob_up, 4),
                    "raw_prob_up": round(raw_prob, 4),
                    "reason":      f"low_atr:{atr_pct:.4f}",
                }

        entry = float(row.get("close",  0.0))
        atr   = float(row.get("atr_14", 0.0))

        if entry <= 0 or atr <= 0 or np.isnan(atr):
            return {
                **_NULL_RESULT,
                "prob_up":     round(prob_up, 4),
                "raw_prob_up": round(raw_prob, 4),
                "reason":      "invalid_atr_or_price",
            }

        effective_atr = max(atr, entry * ATR_FLOOR_PCT)
        sl_atr  = entry - (ATR_SL_MULT * effective_atr)
        tp_atr  = entry + (ATR_TP_MULT * effective_atr)
        sl_cap  = entry * (1 - STOP_LOSS_PCT)

        sl     = round(max(sl_atr, sl_cap), 2)
        target = round(tp_atr, 2)

        risk   = entry - sl
        reward = target - entry
        rr     = round(reward / risk, 3) if risk > 0 else 0.0

        current_open  = float(df["open"].iloc[-1])
        current_close = float(df["close"].iloc[-1])
        prev_high     = float(df["high"].iloc[-2])

        body_pct    = abs(current_close - current_open) / current_open if current_open > 0 else 0.0
        breakout_ok = current_close > prev_high and vol_ratio >= MIN_VOLUME_RATIO_CONFIRM

        vwap           = float(row.get("vwap",           0.0))
        dist_from_vwap = float(row.get("dist_from_vwap", 999.0))
        avoid_lunch    = AVOID_LUNCH_HOURS and 12 <= int(row.get("hour", 0)) <= 13

        atr_ratio       = float(row.get("atr_ratio",       0.0))
        range_expansion = float(row.get("range_expansion", 0.0))

        # ── Dynamic threshold — regime adjustment ────────────
        # FIX: strong-market floor lowered from 0.56 → 0.50 so the
        # -0.03 bonus actually takes effect when buy_threshold=0.55.
        # FIX: weak-market delta raised 0.02 → 0.05 and cap lowered
        # 0.64 → 0.62 to properly reflect BUY_THRESHOLD_WEAK=0.60.
        dynamic_threshold = self.buy_threshold

        market_strong = (
            row.get("nifty_above_ema20", 0) == 1
            and float(row.get("nifty_trend", 0.0)) > 0
            and atr_ratio > 1.1
        )
        market_weak = (
            row.get("nifty_above_ema20", 0) == 0
            or float(row.get("nifty_trend", 0.0)) <= 0
            or atr_ratio < 1.0
        )

        if market_strong and range_expansion >= 1.1:
            # Lower bar on confirmed strong days — floor at 0.50 (was 0.56)
            dynamic_threshold = max(0.50, self.buy_threshold - 0.03)
        elif market_weak:
            # Raise bar on weak days — converges toward BUY_THRESHOLD_WEAK
            dynamic_threshold = min(BUY_THRESHOLD_WEAK, self.buy_threshold + 0.05)

        reject_reason: list[str] = []

        if not for_exit:
            if prob_up < dynamic_threshold:
                reject_reason.append("low_prob")

            if rr < MIN_RR_RATIO:
                reject_reason.append("low_rr")

            if vol_ratio < MIN_VOLUME_RATIO_CONFIRM:
                reject_reason.append("low_volume")

            if REQUIRE_BREAKOUT_CONFIRMATION and not breakout_ok:
                reject_reason.append("no_breakout")

            if REQUIRE_VWAP_CONFIRM and current_close < vwap:
                reject_reason.append("below_vwap")

            if dist_from_vwap > MAX_DISTANCE_FROM_VWAP * 1.35:
                reject_reason.append("too_far_from_vwap")

            if avoid_lunch:
                reject_reason.append("lunch_chop")

        signal = "HOLD"
        reason = "passed"

        if reject_reason:
            reason = ",".join(reject_reason)
            log.info(
                "%s rejected: %s | prob=%.3f rr=%.2f",
                symbol, reason, prob_up, rr,
            )
        else:
            signal = "BUY"
            self._weak_counts.pop(symbol, None)

        log.debug(
            "%s prob=%.3f raw=%.3f signal=%s reason=%s body=%.4f vol=%.2f "
            "range_exp=%.2f atr_ratio=%.2f dist_vwap=%.4f entry=%.2f SL=%.2f TP=%.2f R:R=%.2f ATR=%.4f",
            symbol, prob_up, raw_prob, signal, reason, body_pct, vol_ratio,
            range_expansion, atr_ratio, dist_from_vwap, entry, sl, target, rr, atr,
        )

        return {
            "signal":      signal,
            "prob_up":     round(prob_up, 4),
            "raw_prob_up": round(raw_prob, 4),
            "entry":       round(entry, 2),
            "sl":          sl,
            "target":      target,
            "rr":          rr,
            "atr":         round(atr, 4),
            "atr_ratio":   round(atr_ratio, 3),
            "reason":      reason,
        }

    def should_exit(
        self,
        df: pd.DataFrame,
        side: str,
        symbol: str = "STOCK",
    ) -> bool:
        """
        Exit logic depends only on model probability.
        Entry filters (volume, VWAP, breakout, lunch) are intentionally
        skipped here — they are irrelevant after a position is open.
        """
        result = self.score(df, symbol=symbol, for_exit=True)
        prob   = result.get("raw_prob_up", result["prob_up"])

        if side == "LONG":
            # Hard exit
            if prob < EXIT_LONG_THRESHOLD:
                log.info(
                    "%s: hard exit — prob_up=%.3f < EXIT_LONG=%.3f",
                    symbol, prob, EXIT_LONG_THRESHOLD,
                )
                self._weak_counts[symbol] = 0
                return True

            # Soft / weakening exit
            if prob < WEAK_THRESHOLD:
                count = self._weak_counts.get(symbol, 0) + 1
                self._weak_counts[symbol] = count
                log.info(
                    "%s: weak candle %d/%d prob=%.3f",
                    symbol, count, WEAK_CANDLES_MAX, prob,
                )
                if count >= WEAK_CANDLES_MAX:
                    log.info(
                        "%s: soft exit — %d consecutive weak candles",
                        symbol, count,
                    )
                    self._weak_counts[symbol] = 0
                    return True
            else:
                self._weak_counts[symbol] = 0

        elif side == "SHORT":
            if prob > EXIT_SHORT_THRESHOLD:
                log.info(
                    "%s: short exit — prob_up=%.3f > EXIT_SHORT=%.3f",
                    symbol, prob, EXIT_SHORT_THRESHOLD,
                )
                return True

        return False
