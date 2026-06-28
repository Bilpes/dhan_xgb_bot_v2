# diagnostics.py — dhan_xgb_bot_v2
# =============================================================
# PATCH 2026-06-28 (audit pass 2):
#   ISSUE-13: Added model_feature_count verification.
#   If the saved model has a different feature count than the current
#   FEATURE_COLS (e.g. old 40-feature model after upgrading to 44),
#   predict_proba will raise a silent shape mismatch error.
#   Now prints a clear actionable message: "Run: python auto_retrain.py"
# =============================================================

import pickle
import logging
import os
import numpy as np
import pandas as pd

import config as cfg
from features import build_features, FEATURE_COLS

log = logging.getLogger("diagnostics")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def check_model_exists() -> bool:
    """Confirm all 3 model artefacts (model, scaler, feature list) exist."""
    missing = []
    for p, name in [
        (cfg.MODEL_PATH,   "model"),
        (cfg.SCALER_PATH,  "scaler"),
        (cfg.FEATURE_PATH, "feature list"),
    ]:
        if not os.path.exists(p):
            missing.append(f"  ✗ {name}: {p}")
    if missing:
        log.warning("Missing model artefacts:\n" + "\n".join(missing))
        log.warning("Run: python auto_retrain.py")
        return False
    log.info("Model artefacts present ✓")
    return True


def check_feature_count() -> bool:
    """
    ISSUE-13 FIX: Verify that the saved model was trained with the same
    number of features as the current FEATURE_COLS list.

    Mismatch scenario:
      - Old model trained with 40 features is on disk.
      - FEATURE_COLS was updated to 44 features (audit pass 1: ORB, beta, etc.).
      - signal_engine.py calls predict_proba with 44-feature vector.
      - XGBoost silently fails or raises ValueError on shape mismatch.

    This check catches the mismatch before the market opens and tells
    the user exactly what to do.
    """
    try:
        with open(cfg.FEATURE_PATH, "rb") as f:
            saved_features = pickle.load(f)
        with open(cfg.MODEL_PATH, "rb") as f:
            model = pickle.load(f)
    except FileNotFoundError:
        log.warning("Model not found — run: python auto_retrain.py")
        return False

    saved_n   = len(saved_features)
    current_n = len(FEATURE_COLS)

    # Check 1: saved feature list vs current FEATURE_COLS
    if saved_n != current_n:
        log.error(
            f"FEATURE COUNT MISMATCH: saved model has {saved_n} features, "
            f"current FEATURE_COLS has {current_n}.\n"
            f"  → The model was trained before the feature upgrade.\n"
            f"  → All predictions will fail with a shape error.\n"
            f"  → Fix: python auto_retrain.py"
        )
        return False

    # Check 2: XGBoost internal n_features_ vs current FEATURE_COLS
    model_n_features = getattr(model, "n_features_in_", None)
    if model_n_features is not None and model_n_features != current_n:
        log.error(
            f"MODEL INTERNAL FEATURE MISMATCH: "
            f"model.n_features_in_={model_n_features}, "
            f"current FEATURE_COLS={current_n}.\n"
            f"  → Fix: python auto_retrain.py"
        )
        return False

    # Check 3: feature names match (order matters for XGBoost)
    mismatched = [
        (i, s, c)
        for i, (s, c) in enumerate(zip(saved_features, FEATURE_COLS))
        if s != c
    ]
    if mismatched:
        log.error(
            f"FEATURE NAME MISMATCH at {len(mismatched)} positions:\n"
            + "\n".join(f"  [{i}] saved='{s}' current='{c}'" for i, s, c in mismatched[:5])
            + "\n  → Fix: python auto_retrain.py"
        )
        return False

    log.info(f"Feature count OK: {current_n} features ✓")
    return True


def check_label_balance(csv_path: str = "logs/signals.csv") -> None:
    """Report recent signal log statistics (action distribution)."""
    if not os.path.exists(csv_path):
        log.info(f"No signal log at {csv_path} — run bot first.")
        return
    try:
        df = pd.read_csv(csv_path)
        total = len(df)
        if total == 0:
            log.info("Signal log is empty.")
            return

        buy_count    = (df["action"] == "BUY").sum()
        hold_count   = (df["action"] == "HOLD").sum()
        reject_dist  = df["reject_reason"].value_counts().head(10)

        log.info(
            f"Signal log: {total} rows | "
            f"BUY={buy_count} ({buy_count/total:.1%}) | "
            f"HOLD/reject={hold_count}"
        )
        log.info(f"Top reject reasons:\n{reject_dist.to_string()}")

    except Exception as e:
        log.warning(f"Signal log read error: {e}")


def run_all_checks() -> bool:
    """
    Run all diagnostics in order.
    Returns True if bot is ready to trade, False if action needed.
    """
    log.info("=" * 55)
    log.info("  DIAGNOSTICS — dhan_xgb_bot_v2")
    log.info("=" * 55)

    ok = True
    ok = check_model_exists()   and ok
    ok = check_feature_count()  and ok

    # Signal log is informational — doesn't block readiness
    check_label_balance()

    if ok:
        log.info("\n✅ Bot is ready. Run: python bot.py")
    else:
        log.info("\n❌ Issues found. Run: python auto_retrain.py, then retry.")

    return ok


if __name__ == "__main__":
    run_all_checks()
