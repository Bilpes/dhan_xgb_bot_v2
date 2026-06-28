# ============================================================
# bot/auto_retrain.py — Nightly / scheduled walk-forward
#                        retraining for the live bot
#
# Fully aligned with:
#   trade_policy.py  — ATR_SL_MULT, ATR_TP_MULT, HORIZON
#   data/features.py — build_features(), FEATURE_COLS
#   models/train.py  — identical label logic, same XGB params,
#                      same deployment gate thresholds
#   config/config.py — MODEL_PATH, SCALER_PATH, BACKUP_*,
#                      RETRAIN_LOG, NIFTY50_SECURITY_ID, WATCHLIST
#
# Invocation:
#   python -m bot.auto_retrain            # manual / test
#   cron: 0 18 * * 1-5  python -m bot.auto_retrain
#
# Flow:
#   1. Fetch last RETRAIN_DAYS days of 5-min candles
#      — Broker first, CSV fallback (handles weekends/holidays)
#   2. build_features() — same pipeline as train.py
#   3. _make_atr_labels() — path-dependent label logic
#   4. Walk-forward OOS evaluation (N_SPLITS folds)
#   5. Deployment gate — same MIN_ACC / MIN_AUC / MIN_PREC
#   6. Gate pass  -> backup old model, save new, Telegram alert
#   7. Gate fail  -> keep old model, Telegram warning
#
# Fix log:
#   2026-06-28: Fixed weekend/holiday silent-empty broker bug.
#               Dhan API returns [] on weekends without raising
#               an exception — CSV fallback was never triggered.
#               Now: broker empty → immediately try CSV,
#               only skip symbol if BOTH sources are empty.
# ============================================================

from __future__ import annotations

import logging
import os
import pickle
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.features import build_features, FEATURE_COLS
from bot.trade_policy import ATR_SL_MULT, ATR_TP_MULT, HORIZON
from config.config import (
    WATCHLIST,
    MODEL_PATH,
    SCALER_PATH,
    BACKUP_MODEL_PATH,
    BACKUP_SCALER_PATH,
    RETRAIN_LOG,
    NIFTY50_SECURITY_ID,
)

os.makedirs(ROOT / "logs",   exist_ok=True)
os.makedirs(ROOT / "models", exist_ok=True)

logging.basicConfig(
    level    = logging.INFO,
    format   = "%(asctime)s  %(levelname)-8s  %(message)s",
    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(RETRAIN_LOG, mode="a", encoding="utf-8"),
    ],
)
log = logging.getLogger("auto_retrain")

# ── Rolling retraining window ─────────────────────────────────
RETRAIN_DAYS = int(os.getenv("RETRAIN_DAYS", "90"))
N_SPLITS     = int(os.getenv("N_SPLITS",     "5"))

# ── Deployment gate ───────────────────────────────────────────
MIN_ACC  = 0.53
MIN_AUC  = 0.55
MIN_PREC = 0.50
MIN_ROWS = 500

# ── XGBoost params ─────────────────────────────────────────────
PARAMS = dict(
    objective          = "binary:logistic",
    eval_metric        = "logloss",
    n_estimators       = 500,
    max_depth          = 4,
    learning_rate      = 0.035,
    subsample          = 0.75,
    colsample_bytree   = 0.75,
    colsample_bylevel  = 0.75,
    min_child_weight   = 15,
    gamma              = 0.15,
    reg_alpha          = 0.2,
    reg_lambda         = 1.5,
    random_state       = 42,
    n_jobs             = -1,
    tree_method        = "hist",
    early_stopping_rounds = 30,
)


# ──────────────────────────────────────────────────────────────
#  Data loading  (broker-first, CSV fallback)
#  FIX: Dhan returns empty list silently on weekends/holidays.
#       Previously this was only caught if broker raised an exception.
#       Now: empty broker response → immediately try CSV fallback.
# ──────────────────────────────────────────────────────────────
def _load_csv_for_symbol(
    symbol: str,
    hist_dir: Path,
    cutoff: datetime,
) -> pd.DataFrame | None:
    """
    Load historical CSV for a symbol.
    Searches both data/historical/ and data/ folders.
    Returns filtered DataFrame or None if not found.
    """
    candidates = [
        hist_dir / f"{symbol}_5min.csv",               # data/historical/SYMBOL_5min.csv
        hist_dir.parent / f"{symbol}_5min.csv",        # data/SYMBOL_5min.csv
        hist_dir / f"{symbol}.csv",                    # data/historical/SYMBOL.csv
        hist_dir.parent / f"{symbol}.csv",             # data/SYMBOL.csv
    ]
    for path in candidates:
        if path.exists():
            try:
                df = pd.read_csv(path, parse_dates=["datetime"])
                df = df[df["datetime"] >= cutoff].copy()
                df["symbol"] = symbol
                return df
            except Exception as e:
                log.warning("  %s: CSV read error (%s): %s", symbol, path.name, e)
    return None


def _fetch_recent_candles() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch last RETRAIN_DAYS of 5-min OHLCV for all watchlist stocks + Nifty.
    Strategy per symbol:
        1. Try DhanBroker.get_candles()
        2. If broker returns empty OR raises  → try CSV fallback
        3. If both fail  → skip symbol with warning
    This handles weekends, NSE holidays, and API outages transparently.
    """
    from bot.dhan_api import DhanBroker
    broker   = DhanBroker()
    hist_dir = ROOT / "data" / "historical"
    nifty_path = ROOT / "data" / "raw" / "NIFTY50.csv"
    cutoff   = datetime.now() - timedelta(days=RETRAIN_DAYS)
    frames: list[pd.DataFrame] = []

    log.info("[Retrain] Starting — symbols=%d embargo=%dd folds=%d",
             len(WATCHLIST), RETRAIN_DAYS, N_SPLITS)
    log.info("Training on %d symbols: %s", len(WATCHLIST), list(WATCHLIST.keys()))

    for symbol, sec_id in WATCHLIST.items():
        df_broker = pd.DataFrame()

        # ─ Step 1: Try broker ───────────────────────────────────
        try:
            df_broker = broker.get_candles(sec_id, symbol, days_back=RETRAIN_DAYS)
            if not df_broker.empty:
                if "datetime" not in df_broker.columns:
                    df_broker = df_broker.reset_index().rename(
                        columns={"index": "datetime"})
                df_broker = df_broker[
                    df_broker["datetime"] >= cutoff].copy()
                df_broker["symbol"] = symbol
        except Exception as e:
            log.debug("  %s: broker exception: %s", symbol, e)
            df_broker = pd.DataFrame()

        # ─ Step 2: Use broker data if non-empty ───────────────────
        if not df_broker.empty:
            frames.append(df_broker)
            log.info("  broker ✅  %-14s  %d candles", symbol, len(df_broker))
            continue

        # ─ Step 3: Broker empty/failed → try CSV ─────────────────
        df_csv = _load_csv_for_symbol(symbol, hist_dir, cutoff)
        if df_csv is not None and not df_csv.empty:
            frames.append(df_csv)
            log.info("  csv    📂  %-14s  %d candles (broker empty/offline)",
                     symbol, len(df_csv))
            continue

        # ─ Both failed ──────────────────────────────────────────
        log.warning("  SKIP   ❌  %-14s  no broker data and no CSV found", symbol)

    if not frames:
        raise RuntimeError(
            "No stock data loaded from broker or CSV.\n"
            "  It is currently a weekend/holiday and no CSV files were found.\n"
            f"  Expected CSV location: {hist_dir / 'SYMBOL_5min.csv'}\n"
            "  Make sure data/historical/ contains your 5-min CSV files."
        )

    stock_df = pd.concat(frames, ignore_index=True)

    # ── Nifty index ───────────────────────────────────────────────
    nifty_df = pd.DataFrame()
    try:
        nd = broker.get_candles(
            NIFTY50_SECURITY_ID, "NIFTY50", days_back=RETRAIN_DAYS)
        if not nd.empty:
            if "datetime" not in nd.columns:
                nd = nd.reset_index().rename(columns={"index": "datetime"})
            nifty_df = nd[nd["datetime"] >= cutoff].copy()
            log.info("  Nifty  ✅  broker  %d candles", len(nifty_df))
    except Exception as e:
        log.debug("Nifty broker exception: %s", e)

    if nifty_df.empty and nifty_path.exists():
        nifty_df = pd.read_csv(nifty_path, parse_dates=["datetime"])
        nifty_df = nifty_df[nifty_df["datetime"] >= cutoff].copy()
        log.info("  Nifty  📂  CSV  %d candles", len(nifty_df))

    if nifty_df.empty:
        log.warning("  Nifty unavailable — index features will be 0.")

    return stock_df, nifty_df


# ──────────────────────────────────────────────────────────────
#  ATR path-dependent label
# ──────────────────────────────────────────────────────────────
def _make_atr_labels(feat: pd.DataFrame) -> pd.DataFrame:
    """
    Label = 1 (BUY) if TP hit before SL within HORIZON bars.
    Exactly mirrors the label used during initial training.
    """
    feat   = feat.copy().sort_values(["symbol", "datetime"]).reset_index(drop=True)
    groups = []
    for sym, g in feat.groupby("symbol", sort=False):
        g   = g.reset_index(drop=True)
        c   = g["close"].values
        h   = g["high"].values
        l   = g["low"].values
        atr = g["atr_14"].values
        n   = len(g)
        lbl = np.zeros(n, dtype=int)
        for i in range(n - HORIZON):
            entry = c[i];  a = atr[i]
            if a <= 0 or np.isnan(a):
                continue
            tp = entry + ATR_TP_MULT * a
            sl = entry - ATR_SL_MULT * a
            tp_hit = sl_hit = False
            for j in range(i + 1, min(i + 1 + HORIZON, n)):
                if l[j] <= sl:
                    sl_hit = True;  break
                if h[j] >= tp:
                    tp_hit = True;  break
            lbl[i] = 1 if (tp_hit and not sl_hit) else 0
        g["target"] = lbl
        groups.append(g)
    return pd.concat(groups, ignore_index=True)


# ──────────────────────────────────────────────────────────────
#  Walk-forward OOS evaluation
# ──────────────────────────────────────────────────────────────
def _walk_forward(X: np.ndarray, y: np.ndarray) -> dict:
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    accs, aucs, precs, recalls = [], [], [], []
    for fold, (tr_idx, te_idx) in enumerate(tscv.split(X), 1):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        scale_pos  = max(1.0, (y_tr == 0).sum() / max(1, (y_tr == 1).sum()))
        sc  = StandardScaler()
        Xt  = sc.fit_transform(X_tr)
        Xe  = sc.transform(X_te)
        p   = {**PARAMS, "scale_pos_weight": scale_pos}
        p.pop("early_stopping_rounds", None)
        m   = xgb.XGBClassifier(**p)
        m.fit(Xt, y_tr, verbose=False)
        yp  = m.predict(Xe)
        yq  = m.predict_proba(Xe)[:, 1]
        rep = classification_report(y_te, yp, output_dict=True, zero_division=0)
        accs.append(accuracy_score(y_te, yp))
        aucs.append(roc_auc_score(y_te, yq))
        precs.append(rep.get("1", {}).get("precision", 0.0))
        recalls.append(rep.get("1", {}).get("recall",  0.0))
        log.info("  Fold %d: acc=%.3f AUC=%.3f prec=%.3f recall=%.3f  "
                 "(train=%d test=%d)",
                 fold, accs[-1], aucs[-1], precs[-1], recalls[-1],
                 len(X_tr), len(X_te))
    return {
        "acc":    float(np.mean(accs)),
        "auc":    float(np.mean(aucs)),
        "prec":   float(np.mean(precs)),
        "recall": float(np.mean(recalls)),
    }


# ──────────────────────────────────────────────────────────────
#  Train final model on full data
# ──────────────────────────────────────────────────────────────
def _train_final(X: np.ndarray, y: np.ndarray) -> tuple:
    split       = int(len(X) * 0.85)
    X_tr, X_val = X[:split], X[split:]
    y_tr, y_val = y[:split], y[split:]
    scale_pos   = max(1.0, (y_tr == 0).sum() / max(1, (y_tr == 1).sum()))
    scaler      = StandardScaler()
    X_tr_s      = scaler.fit_transform(X_tr)
    X_val_s     = scaler.transform(X_val)
    p           = {**PARAMS, "scale_pos_weight": scale_pos}
    model       = xgb.XGBClassifier(**p)
    model.fit(
        X_tr_s, y_tr,
        eval_set = [(X_val_s, y_val)],
        verbose  = 50,
    )
    return model, scaler


# ──────────────────────────────────────────────────────────────
#  Deployment gate
# ──────────────────────────────────────────────────────────────
def _gate_passes(metrics: dict, n: int) -> bool:
    checks = {
        f"acc  {metrics['acc']:.3f}  >= {MIN_ACC}":  metrics["acc"]  >= MIN_ACC,
        f"AUC  {metrics['auc']:.3f}  >= {MIN_AUC}":  metrics["auc"]  >= MIN_AUC,
        f"prec {metrics['prec']:.3f} >= {MIN_PREC}":  metrics["prec"] >= MIN_PREC,
        f"rows {n:,} >= {MIN_ROWS}":                  n               >= MIN_ROWS,
    }
    log.info("── Deployment Gate ──")
    all_pass = True
    for desc, ok in checks.items():
        log.info("  %s  %s", "✅" if ok else "❌", desc)
        if not ok:
            all_pass = False
    return all_pass


# ──────────────────────────────────────────────────────────────
#  Telegram helper
# ──────────────────────────────────────────────────────────────
def _notify(msg: str):
    try:
        from bot.telegram_alert import _send
        _send(msg)
    except Exception as e:
        log.warning("Telegram notify failed: %s", e)


# ──────────────────────────────────────────────────────────────
#  Main entry point
# ──────────────────────────────────────────────────────────────
def retrain():
    log.info("=" * 58)
    log.info("  AUTO-RETRAIN  %s", datetime.now().strftime("%Y-%m-%d %H:%M IST"))
    log.info("  Window: %d days | Stocks: %d", RETRAIN_DAYS, len(WATCHLIST))
    log.info("=" * 58)

    # 1. Fetch data
    try:
        stock_df, nifty_df = _fetch_recent_candles()
    except RuntimeError as e:
        log.error("Data fetch failed: %s", e)
        _notify(f"❌ <b>Auto-retrain FAILED</b>\nData error: {e}")
        return

    n_symbols = stock_df["symbol"].nunique()
    log.info("Fetched %d candles across %d symbols", len(stock_df), n_symbols)

    # 2. Build features
    log.info("Building features...")
    try:
        feat = build_features(
            stock_df,
            nifty_df=nifty_df if not nifty_df.empty else None
        )
    except Exception as e:
        log.error("Feature build failed: %s", e)
        _notify(f"❌ <b>Auto-retrain FAILED</b>\nFeature error: {e}")
        return

    # 3. ATR path-dependent labels
    log.info("Creating ATR labels (HORIZON=%d)...", HORIZON)
    feat = feat.drop(columns=["target"], errors="ignore")
    feat = _make_atr_labels(feat)
    feat = feat.dropna(subset=FEATURE_COLS + ["target"]).reset_index(drop=True)

    X = feat[FEATURE_COLS].values.astype(np.float32)
    y = feat["target"].values.astype(int)

    pos_rate = y.mean() * 100
    log.info("Dataset: %d rows | BUY=%.1f%%  HOLD=%.1f%%",
             len(X), pos_rate, 100 - pos_rate)

    if pos_rate > 70 or pos_rate < 10:
        log.warning(
            "Label imbalance %.1f%% BUY — check ATR_TP_MULT/HORIZON in trade_policy.py",
            pos_rate)

    if len(X) < MIN_ROWS:
        log.warning("Too few rows (%d < %d) — retrain skipped.", len(X), MIN_ROWS)
        _notify(f"⚠️ <b>Retrain skipped</b>\nOnly {len(X)} rows (min={MIN_ROWS})")
        return

    # 4. Walk-forward evaluation
    log.info("Walk-forward OOS evaluation (%d folds)...", N_SPLITS)
    metrics = _walk_forward(X, y)
    log.info("OOS: acc=%.3f  AUC=%.3f  prec=%.3f  recall=%.3f",
             metrics["acc"], metrics["auc"], metrics["prec"], metrics["recall"])

    # 5. Deployment gate
    if not _gate_passes(metrics, len(X)):
        msg = (
            f"❌ <b>Retrain gate FAILED — old model kept</b>\n"
            f"Date  : {datetime.now().strftime('%Y-%m-%d')}\n"
            f"acc   : {metrics['acc']:.3f}  (min {MIN_ACC})\n"
            f"AUC   : {metrics['auc']:.3f}  (min {MIN_AUC})\n"
            f"prec  : {metrics['prec']:.3f}  (min {MIN_PREC})\n"
            f"rows  : {len(X):,}\n"
            f"BUY%  : {pos_rate:.1f}%\n"
            f"Tip   : more data or adjust ATR_TP_MULT in trade_policy.py"
        )
        log.warning("Gate FAILED — keeping old model.")
        _notify(msg)
        return

    # 6. Train final model
    log.info("Gate passed — training final model on full dataset...")
    try:
        model, scaler = _train_final(X, y)
    except Exception as e:
        log.error("Final training failed: %s", e)
        _notify(f"❌ <b>Auto-retrain FAILED</b>\nTraining error: {e}")
        return

    # 7. Backup old, deploy new
    try:
        if Path(MODEL_PATH).exists():
            shutil.copy2(MODEL_PATH,  BACKUP_MODEL_PATH)
            shutil.copy2(SCALER_PATH, BACKUP_SCALER_PATH)
            log.info("Old model backed up -> %s", BACKUP_MODEL_PATH)

        with open(MODEL_PATH,  "wb") as f: pickle.dump(model,  f)
        with open(SCALER_PATH, "wb") as f: pickle.dump(scaler, f)
        log.info("New model saved -> %s", MODEL_PATH)
    except Exception as e:
        log.error("Model save failed: %s", e)
        _notify(f"❌ <b>Auto-retrain FAILED</b>\nSave error: {e}")
        return

    # 8. Success notification
    _notify(
        f"✅ <b>Auto-retrain DEPLOYED</b>\n"
        f"Date   : {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        f"Window : {RETRAIN_DAYS}d | {n_symbols} stocks\n"
        f"Rows   : {len(X):,} | BUY%={pos_rate:.1f}%\n"
        f"acc    : {metrics['acc']:.3f}\n"
        f"AUC    : {metrics['auc']:.3f}\n"
        f"prec   : {metrics['prec']:.3f}\n"
        f"recall : {metrics['recall']:.3f}\n"
        f"Model  : {MODEL_PATH}"
    )
    log.info("Retrain complete — new model is live.")


if __name__ == "__main__":
    retrain()
