# ============================================================
# data/features.py — Build XGBoost features from OHLCV
# PhD-grade feature engineering for NSE 5-min time series
# Zero lookahead leakage. Works for both single-stock and
# multi-stock DataFrames.
# ============================================================
import pandas as pd
import numpy as np


def _ema(series, n):
    return series.ewm(span=n, adjust=False).mean()


def _rsi(close, n=14):
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(n).mean().shift(1)
    loss  = (-delta.clip(upper=0)).rolling(n).mean().shift(1)
    rs    = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def _atr(high, low, close, n=14):
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean().shift(1)


def _vwap_intraday(group):
    tp  = (group["high"] + group["low"] + group["close"]) / 3.0
    vol = group["volume"].astype(float)
    return (tp * vol).cumsum() / (vol.cumsum() + 1e-9)


def _macd(close, fast=12, slow=26, signal=9):
    macd_line = (
        _ema(close, fast).shift(1)
        - _ema(close, slow).shift(1)
    )

    signal_line = _ema(macd_line, signal).shift(1)

    return macd_line, signal_line


def _bollinger(close, n=20, k=2):
    mid = close.rolling(n).mean().shift(1)
    std = close.rolling(n).std().shift(1)
    return mid + k * std, mid - k * std


def _stoch(high, low, close, k=14, d=3):
    lowest  = low.rolling(k).min().shift(1)
    highest = high.rolling(k).max().shift(1)
    pct_k   = 100 * (close - lowest) / (highest - lowest + 1e-9)
    pct_d   = pct_k.rolling(d).mean().shift(1)
    return pct_k, pct_d


def _cci(high, low, close, n=20):
    tp  = (high + low + close) / 3.0
    ma  = tp.rolling(n).mean().shift(1)
    mad = (tp.rolling(n).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True).shift(1))
    return (tp - ma) / (0.015 * mad + 1e-9)


def _williams_r(high, low, close, n=14):
    hh = high.rolling(n).max().shift(1)
    ll = low.rolling(n).min().shift(1)
    return -100 * (hh - close) / (hh - ll + 1e-9)


def _keltner_channels(high, low, close, ema_n=20, atr_n=10, mult=1.5):
    mid = _ema(close, ema_n)
    atr = _atr(high, low, close, atr_n)
    return mid + mult * atr, mid - mult * atr


def _build_symbol_features(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy().sort_values("datetime").reset_index(drop=True)

    if "symbol" not in g.columns:
        g["symbol"] = "STOCK"

    o = g["open"].astype(float)
    h = g["high"].astype(float)
    l = g["low"].astype(float)
    c = g["close"].astype(float)
    v = g["volume"].astype(float)

    # ── Trend EMAs ───────────────────────────────────────────
    g["ema_9"]   = _ema(c, 9)
    g["ema_20"]  = _ema(c, 20)
    g["ema_21"]  = _ema(c, 21)
    g["ema_50"]  = _ema(c, 50)
    g["ema_200"] = _ema(c, 200)

    g["ema_cross"]    = g["ema_9"] - g["ema_21"]
    g["ema_cross_50"] = g["ema_21"] - g["ema_50"]

    g["price_vs_ema9"]  = (c - g["ema_9"]) / (g["ema_9"] + 1e-9)
    g["price_vs_ema21"] = (c - g["ema_21"]) / (g["ema_21"] + 1e-9)
    g["price_vs_ema50"] = (c - g["ema_50"]) / (g["ema_50"] + 1e-9)

    # ── Momentum ─────────────────────────────────────────────
    g["rsi_14"] = _rsi(c, 14)
    g["rsi_7"]  = _rsi(c, 7)
    g["rsi_21"] = _rsi(c, 21)

    g["rsi_slope"] = g["rsi_14"].diff(3)

    macd, sig = _macd(c)

    g["macd"]            = macd
    g["macd_signal"]     = sig
    g["macd_hist"]       = macd - sig
    g["macd_hist_slope"] = g["macd_hist"].diff(2)

    g["stoch_k"], g["stoch_d"] = _stoch(h, l, c)
    g["stoch_cross"] = g["stoch_k"] - g["stoch_d"]

    g["roc_3"]  = c.pct_change(3)
    g["roc_5"]  = c.pct_change(5)
    g["roc_10"] = c.pct_change(10)
    g["roc_20"] = c.pct_change(20)

    g["cci_20"]   = _cci(h, l, c, 20)
    g["willr_14"] = _williams_r(h, l, c, 14)

    # ── Volatility ───────────────────────────────────────────
    g["atr_14"] = _atr(h, l, c, 14)

    g["atr_pct"] = (
        g["atr_14"] / (c + 1e-9)
    )

    g["atr_ratio"] = (
        g["atr_14"]
        / (g["atr_14"].rolling(20).mean() + 1e-9)
    )

    bb_up, bb_lo = _bollinger(c)

    g["bb_width"] = (
        (bb_up - bb_lo) / (c + 1e-9)
    )

    g["bb_position"] = (
        (c - bb_lo)
        / (bb_up - bb_lo + 1e-9)
    )

    g["bb_squeeze"] = (
        g["bb_width"]
        < g["bb_width"]
            .rolling(20)
            .quantile(0.20)
            .shift(1)
    ).astype(int)

    kc_up, kc_lo = _keltner_channels(h, l, c)

    g["kc_position"] = (
        (c - kc_lo)
        / (kc_up - kc_lo + 1e-9)
    )

    g["hvol_20"] = (
        c.pct_change()
         .rolling(20)
         .std()
         * np.sqrt(75 * 252)
    )

    # ── Volume / VWAP ────────────────────────────────────────
    g["vol_ma20"] = v.rolling(20).mean()

    g["vol_ratio"] = (
        v / (g["vol_ma20"] + 1e-9)
    )

    g["vol_spike"] = (
        g["vol_ratio"] > 2.0
    ).astype(int)

    g["trade_date"] = g["datetime"].dt.normalize()

    g["vwap"] = (
        g.groupby("trade_date", group_keys=False)
         .apply(_vwap_intraday)
         .reset_index(level=0, drop=True)
    )

    g["price_vs_vwap"] = (
        (c - g["vwap"])
        / (g["vwap"] + 1e-9)
    )

    g["vwap_slope"] = (
        g["vwap"].diff(5)
        / (g["vwap"].shift(5) + 1e-9)
    )

    g["vwm_5"] = (
        (c.pct_change(1) * v).rolling(5).sum()
        / (v.rolling(5).sum() + 1e-9)
    )

    g["vwm_10"] = (
        (c.pct_change(1) * v).rolling(10).sum()
        / (v.rolling(10).sum() + 1e-9)
    )

    # ── Price structure / candle anatomy ────────────────────
    g["hl_range"] = (
        (h - l) / (c + 1e-9)
    )

    g["body"] = (
        (c - o) / (h - l + 1e-9)
    )

    g["upper_wick"] = (
        (h - pd.concat([o, c], axis=1).max(axis=1))
        / (h - l + 1e-9)
    )

    g["lower_wick"] = (
        (pd.concat([o, c], axis=1).min(axis=1) - l)
        / (h - l + 1e-9)
    )

    g["gap"] = (
        (o - c.shift(1))
        / (c.shift(1) + 1e-9)
    )

    g["doji"] = (
        g["body"].abs() < 0.1
    ).astype(int)

    g["hammer"] = (
        (g["lower_wick"] > 0.6)
        & (g["body"] > 0)
    ).astype(int)

    g["shooting_star"] = (
        (g["upper_wick"] > 0.6)
        & (g["body"] < 0)
    ).astype(int)

    # ── Candle quality ───────────────────────────────────────
    g["candle_body_pct"] = (
        (c - o).abs()
        / (c + 1e-9)
    )

    g["strong_bull_candle"] = (
        (c > o)
        & (g["body"] > 0.5)
        & (g["candle_body_pct"] > 0.003)
    ).astype(int)

    # ── Lagged returns ───────────────────────────────────────
    for lag in [1, 2, 3, 5, 8, 13]:
        g[f"ret_lag{lag}"] = c.pct_change(lag)

    # ── Autocorrelation regime ───────────────────────────────
    g["autocorr_5"] = (
        c.pct_change()
         .rolling(10)
         .apply(
             lambda x: pd.Series(x).autocorr(lag=5)
             if len(x) >= 6 else 0.0,
             raw=False
         )
         .fillna(0.0)
    )

    # ── Breakout / support-resistance ────────────────────────
    g["high_20"] = (
        h.rolling(20).max().shift(1)
    )

    g["low_20"] = (
        l.rolling(20).min().shift(1)
    )

    g["high_50"] = (
        h.rolling(50).max().shift(1)
    )

    g["low_50"] = (
        l.rolling(50).min().shift(1)
    )

    g["near_high20"] = (
        (c - g["high_20"])
        / (g["high_20"] + 1e-9)
    )

    g["near_low20"] = (
        (c - g["low_20"])
        / (g["low_20"] + 1e-9)
    )

    g["near_high50"] = (
        (c - g["high_50"])
        / (g["high_50"] + 1e-9)
    )

    g["range_pct_20"] = (
        (g["high_20"] - g["low_20"])
        / (c + 1e-9)
    )

    # ── Breakout confirmation ────────────────────────────────
    g["prev_5_high"] = (
        h.rolling(5)
         .max()
         .shift(1)
    )

    g["breakout_confirm"] = (
        c > (g["prev_5_high"] * 1.003)
    ).astype(int)

    # ── Extension filters ────────────────────────────────────
    g["dist_from_ema20"] = (
        (c - g["ema_20"]).abs()
        / (g["ema_20"] + 1e-9)
    )

    g["dist_from_vwap"] = (
        (c - g["vwap"]).abs()
        / (g["vwap"] + 1e-9)
    )

    # ── Time-of-day effects ──────────────────────────────────
    g["hour"] = g["datetime"].dt.hour
    g["minute"] = g["datetime"].dt.minute

    g["mins_since_open"] = (
        (g["hour"] - 9) * 60
        + g["minute"] - 15
    )

    g["is_first_30min"] = (
        g["mins_since_open"] <= 30
    ).astype(int)

    g["is_last_30min"] = (
        g["mins_since_open"] >= 330
    ).astype(int)

    g["session_frac"] = (
        g["mins_since_open"] / 375.0
    ).clip(0, 1)

    # ── Institutional trend structure ────────────────────────
    g["trend_strength"] = (
        (g["ema_20"] > g["ema_50"])
        & (c > g["vwap"])
    ).astype(int)

    trend_score = (
        (g["ema_20"] - g["ema_50"]).abs()
        / (c + 1e-9)
    )

    trend_mean = (
        trend_score
        .rolling(20)
        .mean()
        .shift(1)
    )

    g["is_trending"] = (
        trend_score > trend_mean
    ).astype(int)

    # ── Day-of-week ──────────────────────────────────────────
    g["day_of_week"] = (
        g["datetime"].dt.dayofweek
    )

    g["is_monday"] = (
        g["day_of_week"] == 0
    ).astype(int)

    g["is_friday"] = (
        g["day_of_week"] == 4
    ).astype(int)

    # ── VWAP alignment ───────────────────────────────────────
    g["above_vwap"] = (
        c > g["vwap"]
    ).astype(int)

    return g.drop(
        columns=["trade_date"],
        errors="ignore"
    )


def build_features(
    df: pd.DataFrame,
    nifty_df: pd.DataFrame = None,
    symbol: str = None,           # ← NEW: pass symbol name from train.py
) -> pd.DataFrame:
    """
    Build all features for a DataFrame.

    Accepts EITHER:
      (a) Multi-stock DataFrame with a 'symbol' column  (auto_retrain.py)
      (b) Single-stock DataFrame without 'symbol'       (models/train.py)
          → pass symbol='AXISBANK' or it defaults to 'STOCK'

    In both cases the output always has a 'symbol' column.
    """
    df = df.copy()

    # ── Normalise datetime column ─────────────────────────────
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
    else:
        df = df.reset_index().rename(columns={"index": "datetime"})
        df["datetime"] = pd.to_datetime(df["datetime"])

    # ── Ensure symbol column exists ───────────────────────────
    # Priority: existing column > caller-supplied name > filename stem
    if "symbol" not in df.columns:
        df["symbol"] = symbol if symbol else "STOCK"

    df   = df.sort_values(["symbol", "datetime"]).reset_index(drop=True)
    feat = (
        df.groupby("symbol", group_keys=False)
          .apply(_build_symbol_features)
          .reset_index(drop=True)
    )
    if "symbol" not in feat.columns:
        base_cols = [c for c in ["datetime", "symbol"] if c in df.columns]
        feat = feat.merge(df[base_cols], on="datetime", how="left")
        if "symbol" not in feat.columns:
            feat["symbol"] = symbol if symbol else "STOCK"

    # ── Nifty relative-strength features ─────────────────────
    if nifty_df is not None and not nifty_df.empty:
        try:
            nifty = nifty_df.copy()
            if "datetime" in nifty.columns:
                nifty["datetime"] = pd.to_datetime(nifty["datetime"])
                nifty = nifty.sort_values("datetime").set_index("datetime")
            else:
                nifty.index = pd.to_datetime(nifty.index)
                nifty = nifty.sort_index()

            nc = nifty["close"].reindex(feat["datetime"], method="ffill")
            feat["nifty_roc5"]         = nc.pct_change(5).values
            feat["rs_vs_nifty"]        = feat["roc_5"] - feat["nifty_roc5"]
            feat["nifty_trend"]        = (_ema(nc, 9) - _ema(nc, 21)).values
            feat["nifty_ret_1"]        = nc.pct_change(1).values
            feat["nifty_ret_5"]        = nc.pct_change(5).values
            feat["nifty_above_ema20"]  = (nc > nc.ewm(span=20).mean().shift(1)).astype(int).values
            feat["nifty_rsi"]          = _rsi(nc, 14).values
            feat["nifty_atr_pct"]      = (
                _atr(
                    nifty["high"].reindex(feat["datetime"], method="ffill"),
                    nifty["low"].reindex(feat["datetime"],  method="ffill"),
                    nc, 14,
                ) / (nc + 1e-9)
            ).values
        except Exception:
            for col in _NIFTY_COLS:
                feat[col] = 0.0
    else:
        for col in _NIFTY_COLS:
            feat[col] = 0.0

    # ── Forward-return label (zero-leakage) ───────────────────
    # Only created if 'target' column is NOT already present.
    # train.py / auto_retrain.py create their own ATR-based labels AFTER
    # calling build_features(); this default is a simple 1-bar return proxy.
    if "target" not in feat.columns:
        future_ret     = feat.groupby("symbol")["close"].shift(-1) / feat["close"] - 1
        feat["target"] = (future_ret > 0.003).astype(int)

    # ── Force all features to use ONLY past candles ───────────
    feat[FEATURE_COLS] = (
        feat.groupby("symbol")[FEATURE_COLS]
        .shift(1)
    )

    # ── Drop rows with NaN in any feature or target ───────────
    drop_cols = [c for c in FEATURE_COLS + ["target"] if c in feat.columns]

    feat = (
        feat.dropna(subset=drop_cols)
        .reset_index(drop=True)
    )

    return feat 

# ── Nifty column list (used in two places above) ─────────────
_NIFTY_COLS = [
    "nifty_roc5", "rs_vs_nifty", "nifty_trend",
    "nifty_ret_1", "nifty_ret_5", "nifty_above_ema20",
    "nifty_rsi", "nifty_atr_pct",
]

FEATURE_COLS = [

    # =========================================================
    # TREND STRUCTURE
    # Institutional trend alignment & directional bias
    # =========================================================
    "ema_cross",
    "ema_cross_50",

    "price_vs_ema9",
    "price_vs_ema21",
    "price_vs_ema50",

    #"ema_50",

    "trend_strength",
    "is_trending",

    "dist_from_ema20",
    "dist_from_vwap",

    "above_vwap",

    # =========================================================
    # MOMENTUM
    # Short-term impulse + continuation strength
    # =========================================================
    "rsi_14",
    "rsi_7",
    "rsi_21",
    "rsi_slope",

    "macd",
    "macd_signal",
    "macd_hist",
    "macd_hist_slope",

    "stoch_k",
    "stoch_d",
    "stoch_cross",

    "roc_3",
    "roc_5",
    "roc_10",
    "roc_20",

    "cci_20",
    "willr_14",

    # =========================================================
    # VOLATILITY REGIME
    # Detect expansion / contraction phases
    # =========================================================
    "atr_pct",
    "atr_ratio",

    "bb_width",
    "bb_position",
    "bb_squeeze",

    "kc_position",

    "hvol_20",

    # =========================================================
    # VOLUME + VWAP FLOW
    # Smart-money participation detection
    # =========================================================
    "vol_ratio",
    "vol_spike",

    "price_vs_vwap",
    "vwap_slope",

    "vwm_5",
    "vwm_10",

    # =========================================================
    # CANDLE STRUCTURE
    # Order-flow footprint inside candle anatomy
    # =========================================================
    "hl_range",

    "body",
    "upper_wick",
    "lower_wick",

    "gap",

    "doji",
    "hammer",
    "shooting_star",

    "candle_body_pct",
    "strong_bull_candle",

    # =========================================================
    # RETURN MEMORY
    # Multi-horizon momentum persistence
    # =========================================================
    "ret_lag1",
    "ret_lag2",
    "ret_lag3",
    "ret_lag5",
    "ret_lag8",
    "ret_lag13",

    "autocorr_5",

    # =========================================================
    # BREAKOUT + MARKET STRUCTURE
    # Expansion from compression / resistance escape
    # =========================================================
    "near_high20",
    "near_low20",
    "near_high50",

    "range_pct_20",

    "breakout_confirm",

    # =========================================================
    # SESSION CONTEXT
    # Intraday behavioural regime
    # =========================================================
    "hour",
    "minute",

    "mins_since_open",

    "is_first_30min",
    "is_last_30min",

    "session_frac",

    # =========================================================
    # WEEKDAY EFFECTS
    # Statistical behaviour by session type
    # =========================================================
    "day_of_week",
    "is_monday",
    "is_friday",

    # =========================================================
    # INDEX RELATIVE STRENGTH
    # Stock performance vs market regime
    # =========================================================
    "nifty_roc5",
    "rs_vs_nifty",

    "nifty_trend",

    "nifty_ret_1",
    "nifty_ret_5",

    "nifty_above_ema20",

    "nifty_rsi",
    "nifty_atr_pct",
]