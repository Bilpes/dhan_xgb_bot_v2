# ============================================================
# data/features.py — Build XGBoost features from OHLCV
# ============================================================
import pandas as pd
import numpy as np

# ── Helper: rolling calculations ────────────────────────────

def _ema(series, n):
    return series.ewm(span=n, adjust=False).mean()

def _rsi(close, n=14):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def _atr(high, low, close, n=14):
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def _vwap(high, low, close, volume):
    tp = (high + low + close) / 3
    # FIX: reset VWAP every trading day (original used cumsum over entire
    # dataset which drifts far from real intraday VWAP after day 1)
    dates = tp.index.normalize()
    result = pd.Series(index=tp.index, dtype=float)
    for d in dates.unique():
        mask = dates == d
        tp_d  = tp[mask]
        vol_d = volume[mask]
        result[mask] = (tp_d * vol_d).cumsum() / vol_d.cumsum()
    return result

def _macd(close, fast=12, slow=26, signal=9):
    macd_line   = _ema(close, fast) - _ema(close, slow)
    signal_line = _ema(macd_line, signal)
    return macd_line, signal_line

def _bollinger(close, n=20, k=2):
    mid = close.rolling(n).mean()
    std = close.rolling(n).std()
    return mid + k * std, mid - k * std  # upper, lower

def _stoch(high, low, close, k=14, d=3):
    lowest  = low.rolling(k).min()
    highest = high.rolling(k).max()
    pct_k   = 100 * (close - lowest) / (highest - lowest + 1e-9)
    pct_d   = pct_k.rolling(d).mean()
    return pct_k, pct_d

# ── Main feature builder ─────────────────────────────────────

def build_features(df: pd.DataFrame, nifty_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Input:  df with columns [open, high, low, close, volume]
            index = datetime, sorted ascending
    Output: df with 30+ features, NaN rows dropped
    """
    df = df.copy()
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]

    # ── Trend ────────────────────────────────────────────────
    df["ema_9"]         = _ema(c, 9)
    df["ema_21"]        = _ema(c, 21)
    df["ema_50"]        = _ema(c, 50)
    df["ema_cross"]     = df["ema_9"] - df["ema_21"]   # +ve = bullish
    df["price_vs_ema9"] = (c - df["ema_9"]) / df["ema_9"]

    # ── Momentum ─────────────────────────────────────────────
    df["rsi_14"]          = _rsi(c, 14)
    df["rsi_7"]           = _rsi(c, 7)
    macd, sig             = _macd(c)
    df["macd"]            = macd
    df["macd_signal"]     = sig
    df["macd_hist"]       = macd - sig
    df["stoch_k"], df["stoch_d"] = _stoch(h, l, c)
    df["roc_5"]           = c.pct_change(5)   # 5-bar rate of change
    df["roc_10"]          = c.pct_change(10)

    # ── Volatility ───────────────────────────────────────────
    df["atr_14"]      = _atr(h, l, c, 14)
    df["atr_pct"]     = df["atr_14"] / c        # normalised ATR
    bb_up, bb_lo      = _bollinger(c)
    df["bb_width"]    = (bb_up - bb_lo) / c
    df["bb_position"] = (c - bb_lo) / (bb_up - bb_lo + 1e-9)

    # ── Volume ───────────────────────────────────────────────
    df["vol_ma20"]       = v.rolling(20).mean()
    df["vol_ratio"]      = v / (df["vol_ma20"] + 1e-9)   # surge = >1.5
    df["vwap"]           = _vwap(h, l, c, v)
    df["price_vs_vwap"]  = (c - df["vwap"]) / df["vwap"]

    # ── Price structure ──────────────────────────────────────
    df["hl_range"]   = (h - l) / c              # candle range
    df["body"]       = (c - o) / (h - l + 1e-9) # body ratio (-1 to 1)
    df["upper_wick"] = (h - df[["open","close"]].max(axis=1)) / (h - l + 1e-9)
    df["lower_wick"] = (df[["open","close"]].min(axis=1) - l) / (h - l + 1e-9)
    df["gap"]        = (o - c.shift(1)) / c.shift(1)  # gap from prev close

    # ── Lagged returns (target leak-proof) ───────────────────
    for lag in [1, 2, 3, 5]:
        df[f"ret_lag{lag}"] = c.pct_change(lag)

    # ── Rolling high/low breakout ────────────────────────────
    df["high_20"]    = h.rolling(20).max()
    df["low_20"]     = l.rolling(20).min()
    df["near_high20"]= (c - df["high_20"]) / df["high_20"]  # -ve = below high
    df["near_low20"] = (c - df["low_20"])  / df["low_20"]
    
    # ════════════════════════════════════════════════════════
    # NEW FEATURE #3 — Time of day
    # XGBoost learns that 9:15 open candles behave differently
    # from 13:30 afternoon drift. Encoded as integers so the
    # model can split on them (e.g. hour < 10 = opening range).
    # ════════════════════════════════════════════════════════
    
    df["hour"]           = df.index.hour                        # 9, 10, 11 ... 15
    df["minute"]         = df.index.minute                      # 0, 5, 10 ... 55
    df["mins_since_open"] = (df["hour"] - 9) * 60 + df["minute"] - 15
    # mins_since_open: 0 at 9:15, 75 at 10:30, 255 at 13:30, etc.
    # Negative values before 9:15 — dropna removes pre-market rows anyway.
    
    # ════════════════════════════════════════════════════════
    # NEW FEATURE #2 — Nifty50 Relative Strength
    # Compares stock's 5-bar return to Nifty's 5-bar return.
    # +ve = stock outperforming index (strong buy context)
    # -ve = stock lagging index (weak, avoid or skip entry)
    # Also adds nifty_roc5 so model knows index direction alone.
    # ════════════════════════════════════════════════════════
    if nifty_df is not None:
        # Align Nifty to same index as stock (forward-fill any gaps)
        nifty_close = (
            nifty_df["close"].reindex(df.index, method="ffill")
        )
        nifty_roc5         = nifty_close.pct_change(5)
        df["nifty_roc5"]   = nifty_roc5                       # raw index momentum
        df["rs_vs_nifty"]  = df["roc_5"] - nifty_roc5          # stock vs index  
        df["nifty_trend"]  = (_ema(nifty_close, 9) - _ema(nifty_close, 21))  # nifty ema cross
    else:
        # If Nifty data not passed (e.g. first candle of day), fill with 0
        # so feature set stays consistent — model treats 0 as neutral
        df["nifty_roc5"]  = 0.0
        df["rs_vs_nifty"] = 0.0
        df["nifty_trend"] = 0.0
        # ── Target variable (training only) ──────────────────────
        future_ret   = c.shift(-1) / c - 1
        df["target"] = (future_ret > 0.003).astype(int)
        df.dropna(inplace=True)
        return df

    # ── Target variable (for training) ───────────────────────
    # 1 = next candle close is higher than current close by > 0.3%
    # 0 = otherwise (flat or down)
    future_ret   = c.shift(-1) / c - 1
    df["target"] = (future_ret > 0.003).astype(int)
    
    # ── Nifty context features (safe — only added if nifty_df available) ──
    if nifty_df is not None and not nifty_df.empty:
        try:
            nifty_close = nifty_df["close"].reindex(df.index, method="ffill")
            df["nifty_ret_1"]  = nifty_close.pct_change(1)
            df["nifty_ret_5"]  = nifty_close.pct_change(5)
            df["nifty_above_ema20"] = (
                nifty_close > nifty_close.ewm(span=20).mean()
            ).astype(int)
        except Exception:
            df["nifty_ret_1"]       = 0.0
            df["nifty_ret_5"]       = 0.0
            df["nifty_above_ema20"] = 0
    else:
        # Nifty not available — fill neutral values so model still runs
        df["nifty_ret_1"]       = 0.0
        df["nifty_ret_5"]       = 0.0
        df["nifty_above_ema20"] = 0

    df.dropna(inplace=True)
    return df

# ════════════════════════════════════════════════════════════
# FEATURE_COLS — updated with ema_50, time, and Nifty features
# ════════════════════════════════════════════════════════════
FEATURE_COLS = [
    # Trend
    "ema_cross", "price_vs_ema9", "ema_50",          # ← ema_50 added (was missing)

    # Momentum
    "rsi_14", "rsi_7",
    "macd", "macd_signal", "macd_hist",
    "stoch_k", "stoch_d",
    "roc_5", "roc_10",

    # Volatility
    "atr_pct", "bb_width", "bb_position",

    # Volume
    "vol_ratio", "price_vs_vwap",

    # Price structure
    "hl_range", "body", "upper_wick", "lower_wick", "gap",

    # Lagged returns
    "ret_lag1", "ret_lag2", "ret_lag3", "ret_lag5",

    # Breakout
    "near_high20", "near_low20",

    # ── NEW: Time of day ──────────────────────────────────
    "hour", "minute", "mins_since_open",

    # ── NEW: Nifty50 relative strength ───────────────────
    "nifty_roc5", "rs_vs_nifty", "nifty_trend",
]