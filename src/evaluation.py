"""Model evaluation metrics and reporting — called from 04_Modeling.ipynb and 05_Model_Audit.ipynb."""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)

from src.data_prep import bootstrap_ci, bootstrap_proportion_ci

_N_BOOT = 2000
_RS = 42


def classification_discrimination(
    y_true: np.ndarray | pd.Series,
    proba: np.ndarray | pd.Series,
    n_boot: int = _N_BOOT,
    random_state: int = _RS,
) -> dict:
    """AUC-ROC, AUC-PR, Brier, log-loss with 95% bootstrap CIs."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(proba).astype(float)

    out = {
        "auc_roc": roc_auc_score(y, p),
        "auc_pr": average_precision_score(y, p),
        "brier": brier_score_loss(y, p),
        "log_loss": log_loss(y, p),
    }

    rng = np.random.default_rng(random_state)
    n = len(y)
    boot = {k: [] for k in ("auc_roc", "auc_pr", "brier")}

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y[idx])) < 2:
            continue
        boot["auc_roc"].append(roc_auc_score(y[idx], p[idx]))
        boot["auc_pr"].append(average_precision_score(y[idx], p[idx]))
        boot["brier"].append(brier_score_loss(y[idx], p[idx]))

    for k, vals in boot.items():
        if vals:
            out[f"{k}_ci_lo"] = np.quantile(vals, 0.025)
            out[f"{k}_ci_hi"] = np.quantile(vals, 0.975)

    return out


def confusion_at_threshold(
    y_true: np.ndarray | pd.Series,
    proba: np.ndarray | pd.Series,
    thr: float,
) -> dict:
    """Confusion matrix and classification metrics at a fixed decision threshold."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(proba).astype(float)
    pred = (p >= thr).astype(int)

    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {
        "threshold": thr,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "f1_weighted": f1_score(y, pred, average="weighted", zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else np.nan,
        "n_predicted_positive": int(pred.sum()),
        "prevalence": float(y.mean()),
    }


def threshold_sweep(
    y_true: np.ndarray | pd.Series,
    proba: np.ndarray | pd.Series,
    thresholds: Optional[Sequence[float]] = None,
) -> pd.DataFrame:
    """F1-weighted, precision, recall across a range of thresholds."""
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.99, 95)
    return pd.DataFrame([confusion_at_threshold(y_true, proba, t) for t in thresholds])


def calibration_curve_data(
    y_true: np.ndarray | pd.Series,
    proba: np.ndarray | pd.Series,
    n_bins: int = 10,
    strategy: str = "quantile",
    n_boot: int = _N_BOOT,
    random_state: int = _RS,
) -> pd.DataFrame:
    """Reliability diagram data: mean predicted prob vs fraction positives per bin, with CIs."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(proba).astype(float)

    if strategy == "quantile":
        bin_edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
        bin_edges[0] -= 1e-9
        bin_edges[-1] += 1e-9
    else:
        bin_edges = np.linspace(p.min() - 1e-9, p.max() + 1e-9, n_bins + 1)

    rows = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (p > lo) & (p <= hi)
        if mask.sum() == 0:
            continue
        _, ci_lo, ci_hi = bootstrap_proportion_ci(
            y[mask], n_boot=n_boot, random_state=random_state
        )
        rows.append({
            "bin_lo": lo, "bin_hi": hi, "n": int(mask.sum()),
            "mean_predicted": float(p[mask].mean()),
            "frac_positives": float(y[mask].mean()),
            "ci_lo": ci_lo, "ci_hi": ci_hi,
        })
    return pd.DataFrame(rows)


def expected_calibration_error(
    y_true: np.ndarray | pd.Series,
    proba: np.ndarray | pd.Series,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error (ECE) using uniform probability bins."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(proba).astype(float)
    n = len(y)
    ece = 0.0
    for lo, hi in zip(np.linspace(0, 1, n_bins + 1)[:-1], np.linspace(0, 1, n_bins + 1)[1:]):
        mask = (p >= lo) & (p < hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(y[mask].mean() - p[mask].mean())
    return ece


def regression_metrics(
    y_true_log: np.ndarray | pd.Series,
    y_pred_log: np.ndarray | pd.Series,
    gns_scale: bool = False,
    log_year_median: Optional[np.ndarray | pd.Series] = None,
    n_boot: int = _N_BOOT,
    random_state: int = _RS,
) -> dict:
    """RMSE, MAE, R², MAPE in log-scale.

    If gns_scale=True, also computes GNS-scale metrics via np.exp() (use when
    y_true_log / y_pred_log are raw log-GNS, i.e. not detrended).
    If log_year_median is provided instead, adds the trend back before exp()
    (use when predictions were produced on a detrended target).
    """
    yt = np.asarray(y_true_log).astype(float)
    yp = np.asarray(y_pred_log).astype(float)
    resid = yt - yp

    out = {
        "rmse_log": float(np.sqrt(mean_squared_error(yt, yp))),
        "mae_log": float(mean_absolute_error(yt, yp)),
        "r2_log": float(r2_score(yt, yp)),
        "mape_log": float(np.mean(np.abs(resid) / (np.abs(yt) + 1e-9))),
        "bias_log": float(resid.mean()),
    }

    rng = np.random.default_rng(random_state)
    n = len(yt)
    boot_rmses = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_rmses.append(np.sqrt(mean_squared_error(yt[idx], yp[idx])))
    out["rmse_log_ci_lo"] = float(np.quantile(boot_rmses, 0.025))
    out["rmse_log_ci_hi"] = float(np.quantile(boot_rmses, 0.975))

    if gns_scale:
        price_true = np.exp(yt)
        price_pred = np.exp(yp)
        out["rmse_gns"] = float(np.sqrt(mean_squared_error(price_true, price_pred)))
        out["mae_gns"] = float(mean_absolute_error(price_true, price_pred))
        out["mape_gns"] = float(np.mean(np.abs(price_true - price_pred) / (price_true + 1e-9)))
    elif log_year_median is not None:
        lym = np.asarray(log_year_median).astype(float)
        price_true = np.exp(yt + lym)
        price_pred = np.exp(yp + lym)
        out["rmse_gns"] = float(np.sqrt(mean_squared_error(price_true, price_pred)))
        out["mae_gns"] = float(mean_absolute_error(price_true, price_pred))
        out["mape_gns"] = float(np.mean(np.abs(price_true - price_pred) / (price_true + 1e-9)))

    return out


def residual_diagnostics(
    df: pd.DataFrame,
    y_true_col: str,
    y_pred_col: str,
    group_cols: Sequence[str],
    n_boot: int = _N_BOOT,
    random_state: int = _RS,
) -> pd.DataFrame:
    """Mean residual (bias), std, RMSE by group with bootstrap CI on bias."""
    df = df.copy()
    df["_resid"] = df[y_true_col] - df[y_pred_col]

    rows = []
    for group_vals, grp in df.groupby(list(group_cols)):
        if not isinstance(group_vals, tuple):
            group_vals = (group_vals,)
        row = dict(zip(group_cols, group_vals))
        row["n"] = len(grp)
        row["bias"] = float(grp["_resid"].mean())
        row["std"] = float(grp["_resid"].std())
        row["rmse"] = float(np.sqrt((grp["_resid"] ** 2).mean()))
        _, ci_lo, ci_hi = bootstrap_ci(
            grp["_resid"].values, stat_func=np.mean,
            n_boot=n_boot, random_state=random_state,
        )
        row["bias_ci_lo"] = ci_lo
        row["bias_ci_hi"] = ci_hi
        rows.append(row)

    return pd.DataFrame(rows)


def temporal_drift(
    df: pd.DataFrame,
    year_col: str,
    metric_fn: Callable[[pd.DataFrame], float],
    baseline_years: Optional[Sequence[int]] = None,
    drift_threshold: float = 0.05,
) -> pd.DataFrame:
    """Metric by year; flags years where metric degrades >drift_threshold vs baseline."""
    rows = []
    for year, grp in df.groupby(year_col):
        try:
            val = metric_fn(grp)
        except Exception:
            val = np.nan
        rows.append({"year": int(year), "metric": val})

    result = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)

    if baseline_years:
        base_val = result.loc[result["year"].isin(baseline_years), "metric"].mean()
    else:
        base_val = result["metric"].iloc[0]

    result["baseline"] = base_val
    result["pct_change"] = (result["metric"] - base_val) / (abs(base_val) + 1e-9)
    result["drift_flag"] = result["pct_change"].abs() > drift_threshold
    return result


def plot_calibration(
    df_calib: pd.DataFrame,
    ax: Optional[matplotlib.axes.Axes] = None,
    title: str = "Calibration (Reliability Diagram)",
    label: str = "Model",
) -> matplotlib.axes.Axes:
    """Reliability diagram: mean predicted probability vs fraction of positives."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration", alpha=0.6)
    ax.errorbar(
        df_calib["mean_predicted"], df_calib["frac_positives"],
        yerr=[
            (df_calib["frac_positives"] - df_calib["ci_lo"]).clip(lower=0),
            (df_calib["ci_hi"] - df_calib["frac_positives"]).clip(lower=0),
        ],
        fmt="o-", capsize=3, label=label, color="#2563eb", ms=5,
    )
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives (observed)")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    return ax


def plot_residuals(
    df_resid: pd.DataFrame,
    group_col: str,
    ax: Optional[matplotlib.axes.Axes] = None,
    title: str = "Residual bias by group",
    rotate_labels: bool = True,
) -> matplotlib.axes.Axes:
    """Bar chart of mean residual (bias) per group with bootstrap CI bands."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    df_plot = df_resid.sort_values(group_col).reset_index(drop=True)
    x = np.arange(len(df_plot))
    colors = ["#dc2626" if b > 0 else "#2563eb" for b in df_plot["bias"]]

    ax.bar(x, df_plot["bias"], color=colors, alpha=0.7)
    ax.errorbar(
        x, df_plot["bias"],
        yerr=[
            (df_plot["bias"] - df_plot["bias_ci_lo"]).clip(lower=0),
            (df_plot["bias_ci_hi"] - df_plot["bias"]).clip(lower=0),
        ],
        fmt="none", color="black", capsize=3, linewidth=1,
    )
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(
        df_plot[group_col].astype(str),
        rotation=45 if rotate_labels else 0, ha="right",
    )
    ax.set_ylabel("Bias (true − predicted log-price)")
    ax.set_title(title)
    return ax
