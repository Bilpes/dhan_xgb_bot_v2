# ============================================================
#  models/train.py  —  Train XGBoost locally, save model
# ============================================================
"""
Run this script weekly (or whenever you want to retrain):
    python models/train.py

It will:
  1. Load historical OHLCV CSVs from data/historical/
  2. Build features
  3. Train XGBoost with cross-validation
  4. Save model + scaler to models/
  5. Print backtest accuracy + feature importances
"""

import os, sys, pickle, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection  import TimeSeriesSplit
from sklearn.metrics          import accuracy_score, classification_report
import xgboost as xgb

from data.features import build_features, FEATURE_COLS
from config.config import MODEL_PATH, SCALER_PATH


# ── 1. Load data ─────────────────────────────────────────────
def load_historical(data_dir="data/historical"):
    """
    Expects CSVs named like:  HDFCBANK_5min.csv
    Columns required: datetime, open, high, low, close, volume
    You can download these free from:
      - NSEpy  (pip install nsepy)
      - yfinance with .NS suffix
      - Dhan historical data API
    """
    frames = []
    for f in os.listdir(data_dir):
        if not f.endswith(".csv"):
            continue
        path  = os.path.join(data_dir, f)
        df    = pd.read_csv(path, parse_dates=["datetime"], index_col="datetime")
        df.columns = df.columns.str.lower()
        df    = df.sort_index()
        frames.append(df)
        print(f"  Loaded {f}  →  {len(df)} rows")

    if not frames:
        raise FileNotFoundError(
            "No CSVs in data/historical/. "
            "Download Nifty 50 5-min data first (see README)."
        )
    return pd.concat(frames).sort_index()


# ── 2. Build features ─────────────────────────────────────────
def prepare_dataset(df):
    print("\nBuilding features...")
    df = build_features(df)

    X = df[FEATURE_COLS]
    y = df["target"]

    print(f"  Dataset: {len(X)} samples  |  "
          f"Class balance: {y.mean()*100:.1f}% UP signals")
    return X, y


# ── 3. Train with time-series cross-validation ───────────────
def train_model(X, y):
    tscv    = TimeSeriesSplit(n_splits=5)
    scores  = []

    # XGBoost parameters tuned for Nifty 50 5-min data
    params = {
        "objective":        "binary:logistic",
        "eval_metric":      "logloss",
        "n_estimators":     500,
        "max_depth":        4,            # shallow = less overfit
        "learning_rate":    0.05,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 10,           # needs 10 samples per leaf
        "gamma":            0.1,
        "reg_alpha":        0.1,
        "reg_lambda":       1.0,
        "n_jobs":           -1,           # use all CPU cores
        "random_state":     42,
        "tree_method":      "hist",       # fast on CPU
    }

    print("\nTraining with 5-fold TimeSeriesSplit...")
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = xgb.XGBClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        preds  = model.predict(X_val)
        acc    = accuracy_score(y_val, preds)
        scores.append(acc)
        print(f"  Fold {fold+1}: accuracy = {acc*100:.2f}%")

    print(f"\n  Mean CV accuracy: {np.mean(scores)*100:.2f}%  "
          f"(std ±{np.std(scores)*100:.2f}%)")

    # Final model trained on all data
    print("\nTraining final model on full dataset...")
    final_model = xgb.XGBClassifier(**params)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=FEATURE_COLS, index=X.index)

    final_model.fit(X_scaled, y, verbose=False)
    return final_model, scaler


# ── 4. Feature importance ─────────────────────────────────────
def print_importances(model):
    imp = pd.Series(
        model.feature_importances_,
        index=FEATURE_COLS
    ).sort_values(ascending=False)

    print("\nTop 10 feature importances:")
    for feat, val in imp.head(10).items():
        bar = "█" * int(val * 200)
        print(f"  {feat:<25} {bar}  {val:.4f}")


# ── 5. Save ───────────────────────────────────────────────────
def save_artifacts(model, scaler):
    os.makedirs("models", exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    print(f"\nModel saved  →  {MODEL_PATH}")
    print(f"Scaler saved →  {SCALER_PATH}")


# ── Main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  XGBoost Training  —  Nifty 50 Intraday/Swing Bot")
    print("=" * 55)

    df          = load_historical()
    X, y        = prepare_dataset(df)
    model, scaler = train_model(X, y)
    print_importances(model)
    save_artifacts(model, scaler)

    print("\nDone. Run  python bot/live_bot.py  to start trading.")
