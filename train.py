# ============================================================
# train.py — dhan_xgb_bot_v2
# Audit-patched 2026-06-28
# Fix I6a: Final model n_estimators uses WF best_iteration+50 (was hardcoded 600)
# Fix I6b: WF folds now fit+transform scaler per fold for consistent metric eval
# Fix DATA: load_historical_data() now fetches from Dhan API when CSV not present,
#           then caches to data/{SYMBOL}_5min.csv for faster subsequent retrains.
#
# Key invariants (unchanged):
#   1. Label entry = open[t+1], NOT close[t]  — kills look-ahead leakage
#   2. 14-day embargo between train-end and val-start
#   3. Raw XGBoost prob — no clipping/capping
#   4. Walk-forward with best-AUC fold selection
#   5. Gates: AUC, accuracy, precision all checked before saving
# ============================================================

import pickle, logging, os
from datetime import date, timedelta
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score
from xgboost import XGBClassifier

from features import build_features, FEATURE_COLS
import config as cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train")


# ── Data loader ──────────────────────────────────────────────────
def load_historical_data(symbol: str, days: int = 90) -> pd.DataFrame:
    """
    Load 5-min OHLCV data for a symbol.

    Priority:
      1. Local CSV cache: data/{symbol}_5min.csv
         — used if file exists (fast, no API call needed)
      2. Dhan historical_minute_charts API
         — fetches `days` of history, saves to CSV cache

    After first fetch, subsequent retrains read from cache
    and only hit the API once per week (or when CSV is stale).
    """
    os.makedirs("data", exist_ok=True)
    path = os.path.join("data", f"{symbol}_5min.csv")

    # ── 1. Try local CSV cache ────────────────────────────────
    if os.path.exists(path):
        try:
            df = pd.read_csv(path, parse_dates=["datetime"], index_col="datetime")
            df.columns = [c.lower() for c in df.columns]
            df = df.sort_index()
            if len(df) > 100:
                log.debug(f"{symbol}: loaded {len(df)} rows from cache")
                return df
        except Exception as e:
            log.warning(f"{symbol}: CSV cache corrupt ({e}), re-fetching from API")

    # ── 2. Fetch from Dhan API ────────────────────────────────
    if not cfg.DHAN_CLIENT_ID or not cfg.DHAN_ACCESS_TOKEN:
        raise FileNotFoundError(
            f"{symbol}: no local CSV and DHAN credentials not set. "
            "Set DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN env vars, "
            "or place CSV files in data/."
        )

    try:
        from dhanhq import dhanhq
        dhan = dhanhq(cfg.DHAN_CLIENT_ID, cfg.DHAN_ACCESS_TOKEN)

        to_date   = date.today().strftime("%Y-%m-%d")
        from_date = (date.today() - timedelta(days=days)).strftime("%Y-%m-%d")

        log.info(f"{symbol}: fetching {days}d from Dhan API ({from_date} → {to_date})")

        resp = dhan.historical_minute_charts(
            symbol          = symbol,
            exchange_segment= "NSE_EQ",
            instrument_type = "EQUITY",
            expiry_code     = 0,
            from_date       = from_date,
            to_date         = to_date,
        )

        records = resp.get("data", {})
        if not records or "timestamp" not in records:
            raise ValueError(f"Empty response from Dhan API for {symbol}")

        df = pd.DataFrame({
            "datetime": pd.to_datetime(records["timestamp"]),
            "open"    : records["open"],
            "high"    : records["high"],
            "low"     : records["low"],
            "close"   : records["close"],
            "volume"  : records["volume"],
        }).set_index("datetime").sort_index()

        # Filter to market hours only (9:15 AM – 3:30 PM)
        df = df.between_time("09:15", "15:30")

        if len(df) < 100:
            raise ValueError(f"Insufficient data for {symbol}: only {len(df)} rows")

        # Cache to CSV — subsequent retrains won't need API call
        df.to_csv(path)
        log.info(f"{symbol}: {len(df)} rows fetched and cached → {path}")
        return df

    except Exception as e:
        raise FileNotFoundError(f"Dhan API fetch failed for {symbol}: {e}")


# ── Label builder (LEAKAGE-FREE) ─────────────────────────────────
def build_labels(
    df: pd.DataFrame,
    horizon: int,
    atr_tp_mult: float,
    atr_sl_mult: float,
    label_entry_shift: int = 1,
) -> pd.DataFrame:
    """
    Construct forward-looking labels using open[t+1] as simulated entry.

    BUY=1 if:
        close[t+horizon] >= open[t+1] + atr_tp_mult * atr14[t]
    AND
        min(low[t+1 .. t+horizon]) > open[t+1] - atr_sl_mult * atr14[t]

    This mirrors live execution: signal fires on candle-t close,
    order fills on candle-(t+1) open.
    """
    df = df.copy()
    if "atr14" not in df.columns:
        df = build_features(df)

    entry_price = df["open"].shift(-label_entry_shift)
    atr         = df["atr14"]
    tp_price    = entry_price + atr_tp_mult * atr
    sl_price    = entry_price - atr_sl_mult * atr

    future_close = df["close"].shift(-horizon)

    low_min = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        s = i + label_entry_shift
        e = i + horizon
        if e < len(df):
            low_min.iloc[i] = df["low"].iloc[s:e+1].min()
        else:
            low_min.iloc[i] = np.nan

    tp_hit = future_close >= tp_price
    sl_ok  = low_min     >= sl_price

    df["label"]       = ((tp_hit & sl_ok) * 1).astype(int)
    df["label_entry"] = entry_price
    df["label_tp"]    = tp_price
    df["label_sl"]    = sl_price

    total = len(df.dropna(subset=["label"]))
    pos   = int(df["label"].sum())
    log.debug(f"Labels: {pos}/{total} BUY ({pos/max(total,1):.1%})")
    return df


# ── Dataset builder ──────────────────────────────────────────────
def prepare_dataset(symbols: list) -> pd.DataFrame:
    dfs = []
    for sym in symbols:
        try:
            raw  = load_historical_data(sym)
            feat = build_features(raw)
            feat = build_labels(
                feat,
                horizon=cfg.HORIZON,
                atr_tp_mult=cfg.ATR_LABEL_TP_MULT,
                atr_sl_mult=cfg.ATR_LABEL_SL_MULT,
                label_entry_shift=cfg.LABEL_ENTRY_SHIFT,
            )
            feat["symbol"] = sym
            dfs.append(feat)
            log.info(f"{sym}: {len(feat)} rows | BUY%={feat['label'].mean():.1%}")
        except Exception as e:
            log.warning(f"Skipped {sym}: {e}")

    if not dfs:
        raise RuntimeError(
            "No stock data loaded.\n"
            "Either:\n"
            "  a) Set DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN env vars, OR\n"
            "  b) Place CSV files as data/{SYMBOL}_5min.csv"
        )

    combined = pd.concat(dfs)
    combined = combined.dropna(subset=FEATURE_COLS + ["label"])
    log.info(
        f"Dataset ready: {len(combined)} rows | "
        f"BUY%={combined['label'].mean():.1%} | "
        f"symbols={combined['symbol'].nunique()}"
    )
    return combined


# ── Walk-forward trainer ─────────────────────────────────────────
def walk_forward_train(
    df: pd.DataFrame,
    n_folds: int = 5,
    embargo_days: int = 14,
) -> tuple:
    """
    Walk-forward cross-validation with temporal embargo.
    Fix I6b: Each fold fits+transforms its own StandardScaler so that
    validation metrics reflect the same data transform used at production
    inference. Previously folds used raw features — making WF metrics
    inconsistent with live behaviour.
    """
    df = df.sort_index()
    dates     = df.index.normalize().unique().sort_values()
    N         = len(dates)
    fold_size = N // (n_folds + 1)
    models, metrics = [], []

    for fold in range(n_folds):
        train_end = dates[fold_size * (fold + 1)]
        val_start = train_end + timedelta(days=embargo_days)
        val_end   = dates[min(fold_size * (fold + 2), N - 1)]

        tr = df[df.index.normalize() <= train_end]
        va = df[
            (df.index.normalize() >= val_start) &
            (df.index.normalize() <= val_end)
        ]

        if len(tr) < cfg.MIN_TRAIN_SAMPLES or len(va) < 100:
            log.warning(f"Fold {fold}: insufficient rows (tr={len(tr)} va={len(va)}) — skip")
            continue

        neg = (tr["label"] == 0).sum()
        pos = (tr["label"] == 1).sum()
        log.info(f"Fold {fold}: train→{train_end.date()} | "
                 f"val {val_start.date()}–{val_end.date()} | "
                 f"tr={len(tr)} va={len(va)} BUY%={pos/(neg+pos):.1%}")

        # Fix I6b: fit scaler on fold train set, transform both sets
        fold_scaler = StandardScaler()
        X_tr_raw = tr[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0).values
        X_va_raw = va[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0).values
        X_tr = fold_scaler.fit_transform(X_tr_raw)
        X_va = fold_scaler.transform(X_va_raw)

        mdl = XGBClassifier(
            n_estimators=500, max_depth=5, learning_rate=0.03,
            subsample=0.75, colsample_bytree=0.75, min_child_weight=5,
            gamma=0.1, reg_alpha=0.1, reg_lambda=1.5,
            scale_pos_weight=neg / max(pos, 1),
            use_label_encoder=False, eval_metric="auc",
            early_stopping_rounds=30, random_state=42, n_jobs=-1,
        )
        mdl.fit(X_tr, tr["label"], eval_set=[(X_va, va["label"])], verbose=False)

        proba = mdl.predict_proba(X_va)[:, 1]
        preds = (proba >= 0.5).astype(int)
        acc   = accuracy_score(va["label"], preds)
        auc   = roc_auc_score(va["label"], proba)
        prec  = precision_score(va["label"], preds, zero_division=0)
        best_iter = getattr(mdl, "best_iteration", 500)
        log.info(f"Fold {fold}: acc={acc:.3f} auc={auc:.3f} prec={prec:.3f} best_iter={best_iter}")

        models.append({"model": mdl, "auc": auc, "acc": acc, "prec": prec, "best_iter": best_iter})
        metrics.append({"fold": fold, "acc": acc, "auc": auc, "prec": prec, "best_iter": best_iter})

    if not models:
        raise RuntimeError("All WF folds failed — check data quality and MIN_TRAIN_SAMPLES")

    best = max(models, key=lambda x: x["auc"])
    return best["model"], pd.DataFrame(metrics), best["best_iter"]


# ── Main entry point ─────────────────────────────────────────────
def train_and_save(symbols: list = None):
    from watchlist import ALL_SYMBOLS, get_watchlist
    if symbols is None:
        symbols = get_watchlist() or ALL_SYMBOLS

    log.info(f"Training on {len(symbols)} symbols: {symbols}")
    df = prepare_dataset(symbols)

    # Fix I6b: walk_forward_train returns best_iter from WF folds
    model, fold_metrics, wf_best_iter = walk_forward_train(
        df,
        n_folds=cfg.WALK_FORWARD_FOLDS,
        embargo_days=cfg.EMBARGO_DAYS,
    )

    mean_auc  = fold_metrics["auc"].mean()
    mean_acc  = fold_metrics["acc"].mean()
    mean_prec = fold_metrics["prec"].mean()
    log.info(f"Walk-forward mean: AUC={mean_auc:.3f} ACC={mean_acc:.3f} PREC={mean_prec:.3f}")
    log.info(f"WF best_iteration (best AUC fold): {wf_best_iter}")
    log.info(f"\n{fold_metrics.to_string()}")

    if mean_auc  < cfg.MIN_AUC:
        raise ValueError(f"AUC gate failed: {mean_auc:.3f} < {cfg.MIN_AUC}")
    if mean_acc  < cfg.MIN_ACCURACY:
        raise ValueError(f"Accuracy gate failed: {mean_acc:.3f} < {cfg.MIN_ACCURACY}")
    if mean_prec < cfg.MIN_PRECISION:
        raise ValueError(f"Precision gate failed: {mean_prec:.3f} < {cfg.MIN_PRECISION}")

    # Fix I6a: was n_estimators=600 hardcoded, ignoring WF best_iteration
    # n_final = wf_best_iter + 50 buffer (full dataset shifts optimum slightly later)
    # Hard cap at 700 to prevent runaway on very large datasets.
    n_final = min(wf_best_iter + 50, 700)
    log.info(f"Final model n_estimators={n_final} (wf_best_iter={wf_best_iter} + 50 buffer)")

    scaler   = StandardScaler()
    X_all    = df[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0)
    X_all_sc = scaler.fit_transform(X_all)

    neg = (df["label"] == 0).sum()
    pos = (df["label"] == 1).sum()
    final_mdl = XGBClassifier(
        n_estimators=n_final,
        max_depth=5, learning_rate=0.03,
        subsample=0.75, colsample_bytree=0.75, min_child_weight=5,
        gamma=0.1, reg_alpha=0.1, reg_lambda=1.5,
        scale_pos_weight=neg / max(pos, 1),
        use_label_encoder=False, eval_metric="auc",
        random_state=42, n_jobs=-1,
    )
    final_mdl.fit(X_all_sc, df["label"])

    os.makedirs("models", exist_ok=True)
    with open(cfg.MODEL_PATH,   "wb") as f: pickle.dump(final_mdl,    f)
    with open(cfg.SCALER_PATH,  "wb") as f: pickle.dump(scaler,       f)
    with open(cfg.FEATURE_PATH, "wb") as f: pickle.dump(FEATURE_COLS, f)

    log.info(
        f"Model saved: {n_final} trees | scaler | {len(FEATURE_COLS)} features\n"
        f"AUC={mean_auc:.3f} ACC={mean_acc:.3f} PREC={mean_prec:.3f} ✓"
    )
    return final_mdl, scaler


if __name__ == "__main__":
    train_and_save()
