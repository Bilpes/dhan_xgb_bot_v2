# ============================================================
#  data/features.py  —  Build XGBoost features from OHLCV
# ============================================================
import pandas as pd
import numpy as np


# ── Helper: rolling calculations ────────────────────────────

def _ema(series, n):
    return series.ewm(span=n, adjust=False).mean()

def _rsi(close, n=14):
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(n).mean()
    loss  = (-delta.clip(upper=0)).rolling(n).mean()
    rs    = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def _atr(high, low, close, n=14):
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def _vwap(high, low, close, volume):
    tp  = (high + low + close) / 3
    cum = (tp * volume).cumsum() / volume.cumsum()
    return cum

def _macd(close, fast=12, slow=26, signal=9):
    macd_line   = _ema(close, fast) - _ema(close, slow)
    signal_line = _ema(macd_line, signal)
    return macd_line, signal_line

def _bollinger(close, n=20, k=2):
    mid = close.rolling(n).mean()
    std = close.rolling(n).std()
    return mid + k * std, mid - k * std   # upper, lower

def _stoch(high, low, close, k=14, d=3):
    lowest  = low.rolling(k).min()
    highest = high.rolling(k).max()
    pct_k   = 100 * (close - lowest) / (highest - lowest + 1e-9)
    pct_d   = pct_k.rolling(d).mean()
    return pct_k, pct_d


# ── Main feature builder ─────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input:  df with columns [open, high, low, close, volume]
            index = datetime, sorted ascending
    Output: df with 30+ features, NaN rows dropped
    """
    df = df.copy()
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]

    # ── Trend ────────────────────────────────────────────────
    df["ema_9"]        = _ema(c, 9)
    df["ema_21"]       = _ema(c, 21)
    df["ema_50"]       = _ema(c, 50)
    df["ema_cross"]    = df["ema_9"] - df["ema_21"]           # +ve = bullish
    df["price_vs_ema9"]= (c - df["ema_9"]) / df["ema_9"]

    # ── Momentum ─────────────────────────────────────────────
    df["rsi_14"]       = _rsi(c, 14)
    df["rsi_7"]        = _rsi(c, 7)
    macd, sig          = _macd(c)
    df["macd"]         = macd
    df["macd_signal"]  = sig
    df["macd_hist"]    = macd - sig
    df["stoch_k"], df["stoch_d"] = _stoch(h, l, c)
    df["roc_5"]        = c.pct_change(5)                      # 5-bar rate of change
    df["roc_10"]       = c.pct_change(10)

    # ── Volatility ───────────────────────────────────────────
    df["atr_14"]       = _atr(h, l, c, 14)
    df["atr_pct"]      = df["atr_14"] / c                     # normalised ATR
    bb_up, bb_lo       = _bollinger(c)
    df["bb_width"]     = (bb_up - bb_lo) / c
    df["bb_position"]  = (c - bb_lo) / (bb_up - bb_lo + 1e-9)

    # ── Volume ───────────────────────────────────────────────
    df["vol_ma20"]     = v.rolling(20).mean()
    df["vol_ratio"]    = v / (df["vol_ma20"] + 1e-9)          # surge = >1.5
    df["vwap"]         = _vwap(h, l, c, v)
    df["price_vs_vwap"]= (c - df["vwap"]) / df["vwap"]

    # ── Price structure ──────────────────────────────────────
    df["hl_range"]     = (h - l) / c                          # candle range
    df["body"]         = (c - o) / (h - l + 1e-9)            # body ratio (-1 to 1)
    df["upper_wick"]   = (h - df[["open","close"]].max(axis=1)) / (h - l + 1e-9)
    df["lower_wick"]   = (df[["open","close"]].min(axis=1) - l) / (h - l + 1e-9)
    df["gap"]          = (o - c.shift(1)) / c.shift(1)        # gap from prev close

    # ── Lagged returns (target leak-proof) ───────────────────
    for lag in [1, 2, 3, 5]:
        df[f"ret_lag{lag}"] = c.pct_change(lag)

    # ── Rolling high/low breakout ────────────────────────────
    df["high_20"]      = h.rolling(20).max()
    df["low_20"]       = l.rolling(20).min()
    df["near_high20"]  = (c - df["high_20"]) / df["high_20"]  # -ve = below high
    df["near_low20"]   = (c - df["low_20"])  / df["low_20"]

    # ── Target variable (for training) ───────────────────────
    # 1 = next candle close is higher than current close by > 0.3%
    # 0 = otherwise (flat or down)
    future_ret         = c.shift(-1) / c - 1
    df["target"]       = (future_ret > 0.003).astype(int)

    df.dropna(inplace=True)
    return df


FEATURE_COLS = [
    "ema_cross","price_vs_ema9",
    "rsi_14","rsi_7",
    "macd","macd_signal","macd_hist",
    "stoch_k","stoch_d",
    "roc_5","roc_10",
    "atr_pct","bb_width","bb_position",
    "vol_ratio","price_vs_vwap",
    "hl_range","body","upper_wick","lower_wick","gap",
    "ret_lag1","ret_lag2","ret_lag3","ret_lag5",
    "near_high20","near_low20",
]
