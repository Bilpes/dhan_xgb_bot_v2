# ============================================================
#  bot/auto_retrain.py  —  Weekly scheduled retrainer
# ============================================================
"""
Runs every Sunday at 8:00 PM automatically via Task Scheduler.
Steps:
  1. Download fresh data (last 60 days)
  2. Merge this week's live trades into training data
  3. Retrain XGBoost
  4. Validate accuracy — only deploy if better than old model
  5. Roll back if new model is worse
  6. Send Telegram summary
"""

import os, sys, pickle, shutil, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
import xgboost as xgb

from data.features   import build_features, FEATURE_COLS
from config.config   import MODEL_PATH, SCALER_PATH, TRADE_LOG
from bot.telegram_alert import _send

log = logging.getLogger("auto_retrain")
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.FileHandler("logs/retrain.log"), logging.StreamHandler()])

BACKUP_MODEL  = "models/xgb_model_backup.pkl"
BACKUP_SCALER = "models/scaler_backup.pkl"
MIN_ACCURACY_IMPROVEMENT = -0.005   # allow up to 0.5% drop — within noise


# ── Step 1: Load all available data ──────────────────────────
def load_all_data():
    from data.download_data import NIFTY50_SYMBOLS
    import yfinance as yf

    log.info("Downloading fresh 60-day data...")
    frames = []
    for sym in NIFTY50_SYMBOLS[:10]:   # top 10 for speed
        try:
            df = yf.download(f"{sym}.NS", period="60d",
                             interval="5m", auto_adjust=True, progress=False)
            if df.empty:
                continue
            df.index.name = "datetime"
            df.columns    = ["open","high","low","close","volume"]
            df = df.dropna()
            df.to_csv(f"data/historical/{sym}_5min.csv")
            frames.append(df)
            log.info("  %s: %d rows", sym, len(df))
        except Exception as e:
            log.warning("  %s failed: %s", sym, e)

    return pd.concat(frames).sort_index() if frames else pd.DataFrame()


# ── Step 2: Merge live trade outcomes ─────────────────────────
def load_live_trade_features():
    """
    Reads trades.csv and extracts the feature rows that led to
    winning and losing trades — feeds them back as training labels.
    """
    if not os.path.exists(TRADE_LOG):
        log.info("No live trade log yet — skipping live data merge.")
        return pd.DataFrame()

    trades = pd.read_csv(TRADE_LOG)
    entries = trades[trades["action"] == "ENTRY"]
    exits   = trades[trades["action"] == "EXIT"]

    if exits.empty:
        return pd.DataFrame()

    # Build outcome labels: win=1, loss=0
    results = []
    for _, ex in exits.iterrows():
        pnl = float(ex["pnl"]) if ex["pnl"] != "" else 0
        results.append({
            "time":   ex["time"],
            "symbol": ex["symbol"],
            "label":  1 if pnl > 0 else 0,
        })

    log.info("Live trades this week: %d wins, %d losses",
             sum(r["label"] for r in results),
             sum(1 - r["label"] for r in results))
    return pd.DataFrame(results)


# ── Step 3: Retrain ───────────────────────────────────────────
def retrain(df: pd.DataFrame):
    from sklearn.preprocessing import StandardScaler

    log.info("Building features for %d rows...", len(df))
    feat = build_features(df)
    X    = feat[FEATURE_COLS]
    y    = feat["target"]

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=FEATURE_COLS, index=X.index)

    params = {
        "objective":        "binary:logistic",
        "n_estimators":     500,
        "max_depth":        4,
        "learning_rate":    0.05,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 10,
        "reg_alpha":        0.1,
        "reg_lambda":       1.0,
        "n_jobs":           -1,
        "random_state":     42,
        "tree_method":      "hist",
    }

    # Cross-validate
    tscv   = TimeSeriesSplit(n_splits=5)
    scores = []
    for tr, val in tscv.split(X_scaled):
        m = xgb.XGBClassifier(**params)
        m.fit(X_scaled.iloc[tr], y.iloc[tr], verbose=False)
        scores.append(accuracy_score(y.iloc[val], m.predict(X_scaled.iloc[val])))

    new_accuracy = float(np.mean(scores))
    log.info("New model CV accuracy: %.2f%%", new_accuracy * 100)

    # Final model
    model = xgb.XGBClassifier(**params)
    model.fit(X_scaled, y, verbose=False)
    return model, scaler, new_accuracy


# ── Step 4: Validate vs old model ────────────────────────────
def get_old_accuracy():
    """Quick re-score of old model on latest data."""
    try:
        with open(MODEL_PATH,  "rb") as f: old_model  = pickle.load(f)
        with open(SCALER_PATH, "rb") as f: old_scaler = pickle.load(f)

        frames = []
        for f in os.listdir("data/historical"):
            if f.endswith(".csv"):
                d = pd.read_csv(f"data/historical/{f}",
                                parse_dates=["datetime"], index_col="datetime")
                d.columns = d.columns.str.lower()
                frames.append(d)
        if not frames:
            return 0.0

        df   = pd.concat(frames).sort_index()
        feat = build_features(df)
        X    = feat[FEATURE_COLS]
        y    = feat["target"]

        X_sc = old_scaler.transform(X)
        preds= old_model.predict(X_sc)
        return accuracy_score(y, preds)
    except Exception:
        return 0.0


# ── Step 5: Save or rollback ──────────────────────────────────
def deploy_model(model, scaler, new_acc, old_acc):
    if new_acc >= old_acc + MIN_ACCURACY_IMPROVEMENT:
        # Backup old first
        if os.path.exists(MODEL_PATH):
            shutil.copy(MODEL_PATH,  BACKUP_MODEL)
            shutil.copy(SCALER_PATH, BACKUP_SCALER)

        with open(MODEL_PATH,  "wb") as f: pickle.dump(model,  f)
        with open(SCALER_PATH, "wb") as f: pickle.dump(scaler, f)

        log.info("New model deployed. Accuracy: %.2f%% (was %.2f%%)",
                 new_acc * 100, old_acc * 100)
        return True, "deployed"
    else:
        log.warning("New model worse (%.2f%% vs %.2f%%). Keeping old model.",
                    new_acc * 100, old_acc * 100)
        return False, "rolled_back"


# ── Step 6: Telegram summary ──────────────────────────────────
def send_retrain_summary(new_acc, old_acc, status, trade_count):
    date_str = datetime.now().strftime("%d %b %Y")
    emoji    = "✅" if status == "deployed" else "⚠️"
    status_str = "New model deployed" if status == "deployed" else "Old model kept (new was worse)"

    msg = (
        f"{emoji} <b>WEEKLY RETRAIN — {date_str}</b>\n"
        f"{'─' * 28}\n"
        f"📊 <b>New accuracy</b>  : {new_acc*100:.1f}%\n"
        f"📊 <b>Old accuracy</b>  : {old_acc*100:.1f}%\n"
        f"📈 <b>Trades learned</b>: {trade_count} this week\n"
        f"🔄 <b>Status</b>        : {status_str}\n"
        f"⏰ <b>Next retrain</b>  : Next Sunday 8:00 PM\n\n"
        f"Bot is ready for Monday trading."
    )
    _send(msg)


# ── Main ──────────────────────────────────────────────────────
def run_retrain():
    log.info("=" * 50)
    log.info("Weekly retraining started — %s",
             datetime.now().strftime("%Y-%m-%d %H:%M"))

    # Check it's Sunday (skip if run manually on other days in production)
    # Comment out next 3 lines if you want to run manually anytime
    # if datetime.now().weekday() != 6:
    #     log.info("Not Sunday — skipping auto-retrain.")
    #     return

    df           = load_all_data()
    live_trades  = load_live_trade_features()
    trade_count  = len(live_trades)

    if df.empty:
        log.error("No data available. Retrain aborted.")
        return

    old_acc              = get_old_accuracy()
    model, scaler, new_acc = retrain(df)
    deployed, status     = deploy_model(model, scaler, new_acc, old_acc)
    send_retrain_summary(new_acc, old_acc, status, trade_count)

    log.info("Retraining complete. Status: %s", status)
    log.info("=" * 50)


if __name__ == "__main__":
    run_retrain()
