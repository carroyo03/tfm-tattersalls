"""Final reproducibility audit for the TFM project.

Run from the project root:

    python -m src.final_audit

This script intentionally checks the boring things that make the work defensible:
stable lot IDs, coherent modelling universes, sold-only price training, and metric
reproduction from the exported OOT predictions.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation import (
    classification_discrimination,
    expected_calibration_error,
    regression_metrics,
)
from src.sensors import universe_consistency_check


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUT_DIR = PROJECT_ROOT / "outputs" / "analyses"
MODELS_DIR = PROJECT_ROOT / "models"


def _load_parquet(name: str) -> pd.DataFrame:
    path = DATA_DIR / f"{name}.parquet"
    if not path.exists():
        raise AssertionError(f"Missing dataset: {path}")
    return pd.read_parquet(path)


def _require_columns(df: pd.DataFrame, cols: list[str], label: str) -> None:
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise AssertionError(f"{label} missing required columns: {missing}")


def _check_ids(clf: pd.DataFrame, reg: pd.DataFrame, univ: pd.DataFrame) -> str:
    if all("lot_uid" in df.columns for df in (clf, reg, univ)):
        id_cols = ["lot_uid"]
    elif all(
        all(col in df.columns for col in ["sale_name", "sale_year", "day", "lot"])
        for df in (clf, reg, univ)
    ):
        id_cols = ["sale_name", "sale_year", "day", "lot"]
    else:
        raise AssertionError(
            "Model datasets need a stable lot key. Add 'lot_uid' or preserve "
            "('sale_name', 'sale_year', 'day', 'lot') in classification_ready, "
            "regression_ready, and inference_universe."
        )

    for label, df in [("classification_ready", clf), ("regression_ready", reg), ("inference_universe", univ)]:
        n_dup = int(df.duplicated(id_cols).sum())
        if n_dup:
            raise AssertionError(f"{label} has {n_dup} duplicate rows for {id_cols}")

    universe_consistency_check(univ, reg, id_cols=id_cols)
    return ", ".join(id_cols)


def _check_regression_universe(reg: pd.DataFrame) -> None:
    _require_columns(reg, ["log_price_gns"], "regression_ready")

    if "sold_to_third_party" not in reg.columns:
        raise AssertionError(
            "regression_ready must retain sold_to_third_party so the final audit can "
            "verify that Stage 2 trains only on realised third-party sales."
        )
    if "vendor_buyback" not in reg.columns:
        raise AssertionError(
            "regression_ready must retain vendor_buyback so buybacks can be audited "
            "as excluded reserve/RNA observations."
        )

    n_not_sold = int((~reg["sold_to_third_party"].astype(bool)).sum())
    n_buyback = int(reg["vendor_buyback"].astype(bool).sum())
    if n_not_sold or n_buyback:
        raise AssertionError(
            "Stage 2 training must be sold-only for realised market price modelling. "
            f"Found {n_not_sold} non-sold rows, including {n_buyback} vendor buybacks."
        )


def _check_exported_metrics() -> dict[str, float]:
    clf_path = OUT_DIR / "audit_clf_predictions.parquet"
    reg_path = OUT_DIR / "audit_reg_predictions.parquet"
    if not clf_path.exists() or not reg_path.exists():
        raise AssertionError("Missing audit prediction exports from 04_Modeling.ipynb")

    clf = pd.read_parquet(clf_path)
    reg = pd.read_parquet(reg_path)
    _require_columns(clf, ["sold_to_third_party", "prob_stacking"], "audit_clf_predictions")
    # ── Stage 2 fix (2026-06-13): model predicts in log_price_detrended space.
    # log_price_detrended = log_price_gns − log_year_median_price_prior.
    # The audit must compare detrended predictions against detrended true values,
    # NOT against raw log_price_gns (which would produce RMSE ~9.6, bias ~9.5).
    _require_columns(reg, ["log_price_detrended_true", "log_price_pred_stacking"], "audit_reg_predictions")

    disc = classification_discrimination(
        clf["sold_to_third_party"], clf["prob_stacking"], n_boot=0
    )
    reg_met = regression_metrics(
        reg["log_price_detrended_true"], reg["log_price_pred_stacking"], n_boot=0
    )
    return {
        "auc_roc": float(disc["auc_roc"]),
        "auc_pr": float(disc["auc_pr"]),
        "ece": float(expected_calibration_error(clf["sold_to_third_party"], clf["prob_stacking"])),
        "rmse_log": float(reg_met["rmse_log"]),
        "r2_log": float(reg_met["r2_log"]),
        "bias_log": float(reg_met["bias_log"]),
        "n_clf_oot": float(len(clf)),
        "n_reg_oot": float(len(reg)),
    }


def _check_exact_model_artifacts() -> None:
    expected = [
        "stage1_final_model.joblib",
        "stage2_final_model.joblib",
        "final_model_metadata.json",
    ]
    missing = [name for name in expected if not (MODELS_DIR / name).exists()]
    if missing:
        raise AssertionError(
            "Missing exact final model artifacts. Re-run 04_Modeling.ipynb after the "
            f"new final-save cell is present. Missing: {missing}"
        )


def run() -> dict[str, float | str]:
    clf = _load_parquet("classification_ready")
    reg = _load_parquet("regression_ready")
    univ = _load_parquet("inference_universe")

    id_key = _check_ids(clf, reg, univ)
    _check_regression_universe(reg)
    _check_exact_model_artifacts()
    metrics = _check_exported_metrics()

    return {
        "id_key": id_key,
        "n_classification_ready": float(len(clf)),
        "n_regression_ready": float(len(reg)),
        "n_inference_universe": float(len(univ)),
        **metrics,
    }


if __name__ == "__main__":
    results = run()
    print("FINAL AUDIT PASSED")
    for key, value in results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6g}")
        else:
            print(f"{key}: {value}")
