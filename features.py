# features.py — dhan_xgb_bot_v3
# LEAKAGE-FREE feature engineering + label construction
# KEY FIX: labels use open[t+1] as entry price, NOT close[t]

import numpy as np
import pandas as pd
import ta


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all technical indicators and derived features.
    VWAP is included as a numeric feature — NOT used as a hard filter.
    """
    df = df.copy().sort_index()
    c, h, l, o, v = df["close"], df["high"], df["low"], df["open"], df["volume"]

    # ── EMA trend ────────────────────────────────────────────────────
    df["ema9"]  = ta.trend.ema_indicator(c, 9)
    df["ema21"] = ta.trend.ema_indicator(c, 21)
    df["ema50"] = ta.trend.ema_indicator(c, 50)
    df["ema200"] = ta.trend.ema_indicator(c, 200)
    df["ema9_21_cross"]   = (df["ema9"]  > df["ema21"]).astype(int)
    df["ema21_50_cross"]  = (df["ema21"] > df["ema50"]).astype(int)
    df["price_above_ema50"]  = (c > df["ema50"]).astype(int)
    df["price_above_ema200"] = (c > df["ema200"]).astype(int)
    df["ema9_slope"]  = df["ema9"].diff(3)  / df["ema9"].shift(3)
    df["ema21_slope"] = df["ema21"].diff(3) / df["ema21"].shift(3)

    # ── RSI ──────────────────────────────────────────────────────────
    df["rsi14"]     = ta.momentum.rsi(c, 14)
    df["rsi7"]      = ta.momentum.rsi(c, 7)
    df["rsi_slope"] = df["rsi14"].diff(3)

    # ── MACD ─────────────────────────────────────────────────────────
    macd = ta.trend.MACD(c, 26, 12, 9)
    df["macd"]           = macd.macd()
    df["macd_signal"]    = macd.macd_signal()
    df["macd_hist"]      = macd.macd_diff()
    df["macd_hist_slope"] = df["macd_hist"].diff(2)

    # ── Stochastic ───────────────────────────────────────────────────
    stoch = ta.momentum.StochasticOscillator(h, l, c, 14, 3)
    df["stoch_k"]    = stoch.stoch()
    df["stoch_d"]    = stoch.stoch_signal()
    df["stoch_cross"] = (df["stoch_k"] > df["stoch_d"]).astype(int)

    # ── Rate of Change ───────────────────────────────────────────────
    df["roc5"]  = ta.momentum.roc(c, 5)
    df["roc10"] = ta.momentum.roc(c, 10)

    # ── ATR + Bollinger Bands ────────────────────────────────────────
    df["atr14"]   = ta.volatility.average_true_range(h, l, c, 14)
    df["atr_pct"] = df["atr14"] / c

    bb = ta.volatility.BollingerBands(c, 20, 2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / bb.bollinger_mavg()
    df["bb_pct"]   = bb.bollinger_pband()

    # ── Volume ───────────────────────────────────────────────────────
    df["vol_ma20"]   = v.rolling(20).mean()
    df["vol_ratio"]  = v / df["vol_ma20"]
    df["vol_ratio_5"] = v.rolling(5).mean() / df["vol_ma20"]
    df["obv"]       = ta.volume.on_balance_volume(c, v)
    df["obv_slope"] = df["obv"].diff(5) / df["obv"].shift(5).abs()

    # ── VWAP — feature only, NOT a hard filter ───────────────────────
    cum_vol = v.groupby(df.index.date).cumsum()
    cum_tp  = (c * v).groupby(df.index.date).cumsum()
    df["vwap"]          = cum_tp / cum_vol
    df["price_vs_vwap"] = (c - df["vwap"]) / df["vwap"]
    df["above_vwap"]    = (c > df["vwap"]).astype(int)

    # ── Candle structure ─────────────────────────────────────────────
    df["candle_body"]      = (c - o) / o
    df["candle_wick_up"]   = (h - c.clip(lower=o))  / (h - l + 1e-6)
    df["candle_wick_down"] = (c.clip(upper=o) - l)  / (h - l + 1e-6)
    df["is_green"]   = (c > o).astype(int)
    df["prev_return"] = c.pct_change()
    df["gap_up"]      = (o - c.shift(1)) / c.shift(1)
    df["high_break"]  = (c > h.shift(1)).astype(int)
    df["ret_3c"]      = c.pct_change(3)
    df["ret_5c"]      = c.pct_change(5)
    df["ret_10c"]     = c.pct_change(10)

    # ── Session time features ────────────────────────────────────────
    df["hour"]        = df.index.hour
    df["session_min"] = (df.index.hour - 9) * 60 + df.index.minute - 15

    # ── Nifty context (populated by bot before signal call) ──────────
    if "nifty_ret_5c" not in df.columns:
        df["nifty_ret_5c"] = 0.0
    if "nifty_above_ema20" not in df.columns:
        df["nifty_above_ema20"] = 1

    return df


def build_labels(
    df,
    horizon: int = 8,
    atr_tp_mult: float = 2.0,
    atr_sl_mult: float = 1.2,
    label_entry_shift: int = 1,
) -> pd.DataFrame:
    """
    LEAKAGE-FREE label construction.
    entry = open[t + label_entry_shift]  (default: next candle open)
    Simulates realistic execution — signal fires at close[t],
    order filled at open[t+1].
    """
    df = df.copy()
    atr = df["atr14"]
    labels = []

    for i in range(len(df)):
        ei = i + label_entry_shift
        if ei >= len(df):
            labels.append(np.nan)
            continue

        entry = df["open"].iloc[ei]
        tp    = entry + atr_tp_mult * atr.iloc[i]
        sl    = entry - atr_sl_mult * atr.iloc[i]

        end = min(ei + horizon, len(df))
        fut = df.iloc[ei:end]

        if fut.empty:
            labels.append(np.nan)
            continue

        tp_idx = fut[fut["high"] >= tp].index
        sl_idx = fut[fut["low"]  <= sl].index

        if len(tp_idx) == 0 and len(sl_idx) == 0:
            labels.append(0)   # neither hit within horizon
        elif len(tp_idx) == 0:
            labels.append(0)   # only SL hit
        elif len(sl_idx) == 0:
            labels.append(1)   # only TP hit
        else:
            labels.append(1 if tp_idx[0] <= sl_idx[0] else 0)

    df["label"] = labels
    return df


# Canonical feature column list — must match exactly between train and signal
FEATURE_COLS = [
    "ema9_21_cross", "ema21_50_cross", "price_above_ema50", "price_above_ema200",
    "ema9_slope", "ema21_slope",
    "rsi14", "rsi7", "rsi_slope",
    "macd", "macd_signal", "macd_hist", "macd_hist_slope",
    "stoch_k", "stoch_d", "stoch_cross",
    "roc5", "roc10",
    "atr_pct", "bb_width", "bb_pct",
    "vol_ratio", "vol_ratio_5", "obv_slope",
    "price_vs_vwap", "above_vwap",
    "candle_body", "candle_wick_up", "candle_wick_down", "is_green",
    "prev_return", "gap_up", "high_break",
    "ret_3c", "ret_5c", "ret_10c",
    "hour", "session_min",
    "nifty_ret_5c", "nifty_above_ema20",
]
