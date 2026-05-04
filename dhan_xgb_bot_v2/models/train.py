# ============================================================
#  models/train.py  —  Train XGBoost locally, save model
# ============================================================
import os, sys, pickle, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.preprocessing  import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics         import accuracy_score
import xgboost as xgb

from data.features import build_features, FEATURE_COLS
from config.config import MODEL_PATH, SCALER_PATH


# ── 1. Load data ──────────────────────────────────────────────
def load_historical(data_dir="data/historical"):
    """
    Expects CSVs named like: HDFCBANK_5min.csv, NIFTY50_5min.csv
    Columns required: datetime, open, high, low, close, volume
    """
    stock_frames = []
    nifty_df     = None

    for f in os.listdir(data_dir):
        if not f.endswith(".csv"):
            continue
        path = os.path.join(data_dir, f)
        df   = pd.read_csv(path, parse_dates=["datetime"], index_col="datetime")
        df.columns = df.columns.str.lower()
        df   = df.sort_index()

        # ── Separate Nifty from stock CSVs ───────────────────
        # Name your Nifty file NIFTY50_5min.csv or NIFTY_5min.csv
        if "NIFTY" in f.upper() and "BANK" not in f.upper():
            nifty_df = df
            print(f"  Loaded Nifty → {f}  ({len(df)} rows)")
        else:
            stock_frames.append(df)
            print(f"  Loaded stock → {f}  ({len(df)} rows)")

    if not stock_frames:
        raise FileNotFoundError(
            "No stock CSVs in data/historical/. "
            "Download Nifty 50 stocks 5-min data first."
        )
    if nifty_df is None:
        print("\n  ⚠️  WARNING: NIFTY50_5min.csv not found.")
        print("     Nifty features will default to 0.0 (neutral).")
        print("     Download it and add to data/historical/ for best results.\n")

    return pd.concat(stock_frames).sort_index(), nifty_df


# ── 2. Build features ─────────────────────────────────────────
def prepare_dataset(stock_df, nifty_df):
    print("\nBuilding features...")
    # Pass nifty_df into build_features — it handles None gracefully
    df = build_features(stock_df, nifty_df=nifty_df)

    X = df[FEATURE_COLS]
    y = df["target"]

    print(f"  Dataset : {len(X)} samples")
    print(f"  Balance : {y.mean()*100:.1f}% UP signals")
    return X, y


# ── 3. Train with time-series cross-validation ───────────────
def train_model(X, y):
    tscv   = TimeSeriesSplit(n_splits=5)
    scores = []

    params = {
        "objective":        "binary:logistic",
        "eval_metric":      "logloss",
        "n_estimators":     500,
        "max_depth":        4,
        "learning_rate":    0.05,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 10,
        "gamma":            0.1,
        "reg_alpha":        0.1,
        "reg_lambda":       1.0,
        "n_jobs":           -1,
        "random_state":     42,
        "tree_method":      "hist",
    }

    print("\nTraining with 5-fold TimeSeriesSplit...")
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = xgb.XGBClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        acc = accuracy_score(y_val, model.predict(X_val))
        scores.append(acc)
        print(f"  Fold {fold+1}: accuracy = {acc*100:.2f}%")

    print(f"\n  Mean CV accuracy : {np.mean(scores)*100:.2f}%  "
          f"(±{np.std(scores)*100:.2f}%)")

    # Final model on full dataset
    print("\nTraining final model on full dataset...")
    scaler   = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=FEATURE_COLS,
        index=X.index
    )
    final_model = xgb.XGBClassifier(**params)
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
    print(f"\nModel  saved → {MODEL_PATH}")
    print(f"Scaler saved → {SCALER_PATH}")


# ── Main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  XGBoost Training  —  Nifty 50 Intraday Bot")
    print("=" * 55)

    stock_df, nifty_df = load_historical()
    X, y               = prepare_dataset(stock_df, nifty_df)
    model, scaler      = train_model(X, y)
    print_importances(model)
    save_artifacts(model, scaler)

    print("\nDone. Run  python bot/live_bot.py  to start trading.")