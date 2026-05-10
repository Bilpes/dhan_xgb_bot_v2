# ============================================================
# models/train.py — XGBoost training for NSE intraday bot
#
# PRODUCTION-GRADE VERSION
# • Zero leakage time-series training
# • Better imbalance handling
# • Confidence-based predictions
# • Safer deployment for live trading
# ============================================================

from __future__ import annotations
import os
import sys
import pickle
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
print("RUNNING FILE:", os.path.abspath(__file__))
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.features import build_features, FEATURE_COLS
from bot.trade_policy import HORIZON

# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────
HIST_DIR    = ROOT / "data" / "historical"
NIFTY_PATH  = ROOT / "data" / "raw" / "NIFTY50.csv"

MODEL_PATH  = ROOT / "models" / "xgb_model.pkl"
SCALER_PATH = ROOT / "models" / "scaler.pkl"
FIMP_PATH   = ROOT / "models" / "feature_importance.csv"

# ─────────────────────────────────────────────────────────────
# Model Hyperparameters
# ─────────────────────────────────────────────────────────────
PARAMS = dict(
    objective="binary:logistic",
    eval_metric="auc",

    # ── Core ───────────────────────────────────────────────
    n_estimators=700,
    max_depth=6,
    learning_rate=0.025,

    # ── Regularisation ─────────────────────────────────────
    subsample=0.80,
    colsample_bytree=0.80,
    colsample_bylevel=0.80,

    min_child_weight=8,
    gamma=0.10,

    reg_alpha=0.30,
    reg_lambda=2.0,

    # ── Stability ──────────────────────────────────────────
    random_state=42,
    n_jobs=-1,
    tree_method="hist",

    # ── Early stopping ─────────────────────────────────────
    early_stopping_rounds=50,
)

# ─────────────────────────────────────────────────────────────
# Training Config
# ─────────────────────────────────────────────────────────────
N_SPLITS            = 5
PREDICTION_THRESHOLD = 0.60

MIN_ACC    = 0.53
MIN_AUC    = 0.55
MIN_PREC   = 0.50
MIN_TRADES = 50

# ─────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────
os.makedirs(ROOT / "models", exist_ok=True)
os.makedirs(ROOT / "logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            ROOT / "logs" / "train.log",
            mode="a",
            encoding="utf-8",
        ),
    ],
)

log = logging.getLogger("train")

# ─────────────────────────────────────────────────────────────
# Load Data
# ─────────────────────────────────────────────────────────────
def load_all_stocks() -> tuple[pd.DataFrame, pd.DataFrame]:

    print("=" * 60)
    print("  XGBoost Training — NSE Intraday Bot")
    print("=" * 60)

    EXCLUDE = {"NIFTY50"}

    frames = []

    for f in sorted(HIST_DIR.glob("*_5min.csv")):

        try:
            symbol = f.stem.replace("_5min", "").upper()

            if symbol in EXCLUDE:
                print(f"  Skipping index → {f.name}")
                continue

            df = pd.read_csv(f, parse_dates=["datetime"])

            df["symbol"] = symbol

            frames.append(df)

            print(
                f"  Loaded stock → {f.name:<30} "
                f"({len(df):,} rows)"
            )

        except Exception as e:
            log.warning("Skipping %s: %s", f.name, e)

    if not frames:
        raise FileNotFoundError(
            f"No stock files found in {HIST_DIR}"
        )

    stock_df = pd.concat(frames, ignore_index=True)

    # ── Nifty context ──────────────────────────────────────
    nifty_df = pd.DataFrame()

    if NIFTY_PATH.exists():

        try:
            nifty_df = pd.read_csv(
                NIFTY_PATH,
                parse_dates=["datetime"],
            )

            print(
                f"  Loaded Nifty → NIFTY50.csv "
                f"({len(nifty_df):,} rows)"
            )

        except Exception as e:
            log.warning("Nifty load failed: %s", e)

    else:
        log.warning("NIFTY50.csv not found")

    return stock_df, nifty_df

# ─────────────────────────────────────────────────────────────
# ATR-style Labels
# ─────────────────────────────────────────────────────────────
def _make_atr_labels(feat: pd.DataFrame) -> pd.DataFrame:

    from bot.trade_policy import TP_PCT, SL_PCT, HORIZON

    feat = (
        feat.copy()
        .sort_values(["symbol", "datetime"])
        .reset_index(drop=True)
    )

    all_frames = []

    for sym, g in feat.groupby("symbol", sort=False):

        g = g.reset_index(drop=True)

        c = g["close"].values
        h = g["high"].values
        l = g["low"].values

        n = len(g)

        labels = np.zeros(n, dtype=int)

        for i in range(n - HORIZON):

            entry = c[i]

            if entry <= 0:
                continue

            tp = entry * (1 + TP_PCT)
            sl = entry * (1 - SL_PCT)

            for j in range(i + 1, min(i + 1 + HORIZON, n)):

                if l[j] <= sl:
                    break

                if h[j] >= tp:
                    labels[i] = 1
                    break

        g["target"] = labels

        all_frames.append(g)

    return pd.concat(all_frames, ignore_index=True)

# ─────────────────────────────────────────────────────────────
# Prepare Dataset
# ─────────────────────────────────────────────────────────────
def prepare_dataset(
    stock_df: pd.DataFrame,
    nifty_df: pd.DataFrame,
):

    print("\nBuilding features...")

    feat = build_features(
        stock_df,
        nifty_df=nifty_df,
    )

    # Replace temporary target
    feat = feat.drop(columns=["target"], errors="ignore")

    feat = _make_atr_labels(feat)

    # Remove infinities
    feat = feat.replace([np.inf, -np.inf], np.nan)

    feat = (
        feat.dropna(subset=FEATURE_COLS + ["target"])
        .reset_index(drop=True)
    )
    feat = feat.sort_values("datetime").reset_index(drop=True)
    
    X = feat[FEATURE_COLS].values.astype(np.float32)
    y = feat["target"].values.astype(int)

    pos_rate = y.mean() * 100

    print(
        f"  Dataset: {len(X):,} rows | "
        f"{len(FEATURE_COLS)} features"
    )

    print(
        f"  BUY={pos_rate:.1f}% | "
        f"HOLD={100 - pos_rate:.1f}%"
    )

    return X, y

# ─────────────────────────────────────────────────────────────
# Walk-forward Evaluation
# ─────────────────────────────────────────────────────────────
def walk_forward_eval(X, y):

    tscv = TimeSeriesSplit(n_splits=N_SPLITS,gap=HORIZON)

    oos_acc = []
    oos_auc = []
    oos_prec = []
    oos_recall = []

    print(f"\nWalk-forward evaluation ({N_SPLITS} folds)...")

    for fold, (tr_idx, te_idx) in enumerate(tscv.split(X), 1):

        X_tr = X[tr_idx]
        X_te = X[te_idx]

        y_tr = y[tr_idx]
        y_te = y[te_idx]

        # ── Dynamic imbalance handling ─────────────────────
        pos = (y_tr == 1).sum()
        neg = (y_tr == 0).sum()

        scale_pos_weight = neg / max(pos, 1)

        scaler = StandardScaler()

        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        params = {
            **PARAMS,
            "scale_pos_weight": scale_pos_weight,
        }

        params.pop("early_stopping_rounds", None)

        model = xgb.XGBClassifier(**params)

        model.fit(
            X_tr_s,
            y_tr,
            verbose=False,
        )

        y_prob = model.predict_proba(X_te_s)[:, 1]

        # Higher confidence threshold
        y_pred = (
            y_prob >= PREDICTION_THRESHOLD
        ).astype(int)

        acc = accuracy_score(y_te, y_pred)

        auc = roc_auc_score(y_te, y_prob)

        rep = classification_report(
            y_te,
            y_pred,
            output_dict=True,
            zero_division=0,
        )

        prec = rep.get("1", {}).get(
            "precision",
            0.0,
        )

        rec = rep.get("1", {}).get(
            "recall",
            0.0,
        )

        trade_count = y_pred.sum()

        avg_conf = (
            y_prob[y_pred == 1].mean()
            if trade_count > 0 else 0
        )

        oos_acc.append(acc)
        oos_auc.append(auc)
        oos_prec.append(prec)
        oos_recall.append(rec)

        print(
            f"  Fold {fold}: "
            f"acc={acc:.3f}  "
            f"AUC={auc:.3f}  "
            f"prec={prec:.3f}  "
            f"recall={rec:.3f}  "
            f"trades={trade_count:,}  "
            f"avg_conf={avg_conf:.3f}"
        )

    return {
        "acc": np.mean(oos_acc),
        "auc": np.mean(oos_auc),
        "prec": np.mean(oos_prec),
        "recall": np.mean(oos_recall),
    }

# ─────────────────────────────────────────────────────────────
# Final Training
# ─────────────────────────────────────────────────────────────
def train_final(X, y):

    # STRICT time split
    split = int(len(X) * 0.80)

    X_tr = X[:split]
    X_val = X[split:]

    y_tr = y[:split]
    y_val = y[split:]

    # ── Imbalance handling ────────────────────────────────
    pos = (y_tr == 1).sum()
    neg = (y_tr == 0).sum()

    scale_pos_weight = neg / max(pos, 1)

    scaler = StandardScaler()

    X_tr_s = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)

    params = {
        **PARAMS,
        "scale_pos_weight": scale_pos_weight,
    }

    model = xgb.XGBClassifier(**params)

    model.fit(
        X_tr_s,
        y_tr,
        eval_set=[(X_val_s, y_val)],
        verbose=50,
    )

    # ── Feature importance ────────────────────────────────
    imp = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": model.feature_importances_,
    })

    imp = imp.sort_values(
        "importance",
        ascending=False,
    )

    imp.to_csv(FIMP_PATH, index=False)

    top10 = imp.head(10)["feature"].tolist()

    print(f"\nTop-10 features:")
    print(top10)

    return model, scaler

# ─────────────────────────────────────────────────────────────
# Deployment Gate
# ─────────────────────────────────────────────────────────────
def deployment_gate(metrics, n_samples):

    checks = {
        f"OOS accuracy >= {MIN_ACC}":
            metrics["acc"] >= MIN_ACC,

        f"OOS AUC >= {MIN_AUC}":
            metrics["auc"] >= MIN_AUC,

        f"OOS precision >= {MIN_PREC}":
            metrics["prec"] >= MIN_PREC,

        f"Samples >= {MIN_TRADES}":
            n_samples >= MIN_TRADES,
    }

    print("\nDeployment Gate")

    all_pass = True

    for desc, ok in checks.items():

        status = "✅" if ok else "❌"

        print(f"  {status} {desc}")

        if not ok:
            all_pass = False

    return all_pass

# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # 1. Load data
    stock_df, nifty_df = load_all_stocks()

    # 2. Build dataset
    X, y = prepare_dataset(
        stock_df,
        nifty_df,
    )

    # 3. Walk-forward evaluation
    metrics = walk_forward_eval(X, y)

    print(
        f"\nOOS Summary → "
        f"acc={metrics['acc']:.3f} | "
        f"AUC={metrics['auc']:.3f} | "
        f"prec={metrics['prec']:.3f} | "
        f"recall={metrics['recall']:.3f}"
    )

    # 4. Deployment gate
    gate_passed = deployment_gate(
        metrics,
        len(y),
    )

    if gate_passed:

        print("\nTraining final model...")

        model, scaler = train_final(X, y)

        # ── Save deployment package ───────────────────────
        model_package = {
            "model": model,
            "threshold": PREDICTION_THRESHOLD,
            "features": FEATURE_COLS,
        }

        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model_package, f)

        with open(SCALER_PATH, "wb") as f:
            pickle.dump(scaler, f)

        print(f"\n✅ Model saved → {MODEL_PATH}")
        print(f"✅ Scaler saved → {SCALER_PATH}")
        print(f"✅ Feature importance → {FIMP_PATH}")

        print("\n🚀 LIVE MODEL READY")

    else:

        print(
            "\n❌ DEPLOYMENT GATE FAILED\n"
            "\nPossible fixes:\n"
            "1. Download more data\n"
            "2. Reduce overfitting\n"
            "3. Tune TP_PCT / SL_PCT\n"
            "4. Reduce noisy features\n"
        )

        sys.exit(1)