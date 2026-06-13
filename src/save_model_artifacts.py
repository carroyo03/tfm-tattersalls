#!/usr/bin/env python3
"""
Export the best model artifacts from MLflow to ``models/``.

Run from the project root after ``04_Modeling.ipynb`` has been executed::

    uv run python -m src.save_model_artifacts

This script loads the winning models from MLflow (by experiment + metric query)
and writes them to ``models/`` as ``stage1_final_model.joblib`` and
``stage2_final_model.joblib``.  It does NOT re-train anything.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import mlflow
from mlflow.tracking import MlflowClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import TRAIN_MAX_YEAR, VAL_MIN_YEAR, VAL_MAX_YEAR, TEST_MIN_YEAR

MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MLFLOW_URI = (PROJECT_ROOT / "notebooks" / "mlruns").as_uri()
EXPERIMENTS = {
    "stage1_classification": {"order": "metrics.val_pr_auc DESC", "metric": "val_pr_auc", "tie_metric": "val_brier"},
    "stage2_regression":     {"order": "metrics.val_rmse_log ASC", "metric": "val_rmse_log"},
}


def _best_run(client, exp_name: str) -> tuple:
    """Return (run, run_id) for the best run in *exp_name*."""
    exp = client.get_experiment_by_name(exp_name)
    if exp is None:
        raise RuntimeError(f"Experiment '{exp_name}' not found in {MLFLOW_URI}")

    cfg = EXPERIMENTS[exp_name]
    runs = client.search_runs(exp.experiment_id, order_by=[cfg["order"]], max_results=3)
    if not runs:
        raise RuntimeError(f"No runs found for experiment '{exp_name}'")

    best = runs[0]
    # Tiebreaker
    if len(runs) >= 2 and "tie_metric" in cfg:
        pr_diff = best.data.metrics.get(cfg["metric"], 0) - runs[1].data.metrics.get(cfg["metric"], 0)
        if abs(pr_diff) <= 0.001:
            if runs[1].data.metrics.get(cfg["tie_metric"], 1) < best.data.metrics.get(cfg["tie_metric"], 1):
                best = runs[1]
    return best, best.info.run_id


def main() -> None:
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = MlflowClient()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load best runs ──
    clf_run, clf_id = _best_run(client, "stage1_classification")
    reg_run, reg_id = _best_run(client, "stage2_regression")

    print(f"Stage 1 winner: {clf_run.info.run_name}  (run_id={clf_id})")
    print(f"Stage 2 winner: {reg_run.info.run_name}  (run_id={reg_id})")

    # ── Download models ──
    clf_model = mlflow.sklearn.load_model(f"runs:/{clf_id}/model")
    reg_model = mlflow.sklearn.load_model(f"runs:/{reg_id}/model")

    # ── Persist ──
    joblib.dump(clf_model, MODELS_DIR / "stage1_final_model.joblib")
    joblib.dump(reg_model, MODELS_DIR / "stage2_final_model.joblib")

    # ── Derive feature lists from exported parquet (source of truth) ──
    import pyarrow.parquet as pq
    CLF_META = {'sale_year', 'sold_to_third_party', 'lot_uid', 'vendor_buyback', 'lot_not_sold'}
    REG_META = {'sold_to_third_party', 'vendor_buyback', 'sale_year', 'lot_uid',
                'log_price_gns', 'log_year_median_price_prior', 'log_price_detrended'}

    clf_cols = pq.read_schema(DATA_DIR / "classification_ready.parquet").names
    reg_cols = pq.read_schema(DATA_DIR / "regression_ready.parquet").names

    features_clf = [c for c in clf_cols if c not in CLF_META]
    features_reg = [c for c in reg_cols if c not in REG_META]

    print(f"features_clf: {len(features_clf)} → {features_clf}")
    print(f"features_reg: {len(features_reg)} → {features_reg}")

    # ── sigma2_reg: variance of log-residuals on validation set ──
    # Must be logged by 04_Modeling as metric "val_sigma2_log".
    # Fallback: val_rmse_log² (approximate, assumes zero-mean residuals).
    sigma2_reg = reg_run.data.metrics.get("val_sigma2_log", None)
    if sigma2_reg is None:
        rmse = reg_run.data.metrics.get("val_rmse_log", 0)
        sigma2_reg = rmse ** 2
        print(f"⚠ val_sigma2_log not found in MLflow; using val_rmse_log² = {sigma2_reg:.4f}")

    # ── threshold_youden: must be logged by 04_Modeling ──
    thr_youden = clf_run.data.metrics.get("threshold_youden", None)

    metadata = {
        "stage1_model": clf_run.info.run_name,
        "stage1_run_id": clf_id,
        "stage2_model": reg_run.info.run_name,
        "stage2_run_id": reg_id,
        "threshold_youden": thr_youden,
        "sigma2_reg": sigma2_reg,
        "features_clf": features_clf,
        "features_reg": features_reg,
    }
    with open(MODELS_DIR / "final_model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # ── Validate completeness ──
    missing = []
    if thr_youden is None:
        missing.append("threshold_youden")
    if sigma2_reg is None:
        missing.append("sigma2_reg")
    if missing:
        print(f"⚠ WARNING: Missing metadata keys: {missing}. "
              f"Ensure 04_Modeling.ipynb logs these to MLflow.")
    else:
        print("✅ All metadata keys present")

    print(f"✅ Artifacts saved to {MODELS_DIR}/")
    print(f"   stage1_final_model.joblib  ({clf_run.info.run_name})")
    print(f"   stage2_final_model.joblib  ({reg_run.info.run_name})")
    print(f"   final_model_metadata.json")


if __name__ == "__main__":
    main()
