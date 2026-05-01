"""Fairness slicing and audit aggregators — called from 05_Model_Audit.ipynb."""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd

_N_BOOT = 2000
_RS = 42


def fairness_slice(
    df: pd.DataFrame,
    slice_col: str,
    metric_fn: Callable[[pd.DataFrame], float],
    min_n: int = 30,
    n_boot: int = _N_BOOT,
    random_state: int = _RS,
) -> pd.DataFrame:
    """Metric + 95% bootstrap CI per slice; drops slices with fewer than min_n rows."""
    rows = []
    rng = np.random.default_rng(random_state)

    for val, grp in df.groupby(slice_col, sort=True):
        n = len(grp)
        if n < min_n:
            continue
        try:
            metric_val = metric_fn(grp)
        except Exception:
            metric_val = np.nan

        boot_vals = []
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            try:
                boot_vals.append(metric_fn(grp.iloc[idx]))
            except Exception:
                pass

        ci_lo = float(np.quantile(boot_vals, 0.025)) if len(boot_vals) >= 10 else np.nan
        ci_hi = float(np.quantile(boot_vals, 0.975)) if len(boot_vals) >= 10 else np.nan

        rows.append({
            slice_col: val,
            "n": n,
            "metric": float(metric_val) if not np.isnan(metric_val) else np.nan,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
        })

    return pd.DataFrame(rows).sort_values("metric", ascending=False).reset_index(drop=True)


def slice_disparities(
    df_slices: pd.DataFrame,
    metric_col: str = "metric",
    baseline: str | float = "overall",
    disparity_threshold: float = 2.0,
) -> pd.DataFrame:
    """Absolute and relative gap between each slice metric and a baseline value."""
    df = df_slices.copy()

    if baseline == "overall":
        # Weighted mean by slice size if 'n' column available
        if "n" in df.columns:
            baseline_val = float(np.average(df[metric_col].dropna(), weights=df.loc[df[metric_col].notna(), "n"]))
        else:
            baseline_val = float(df[metric_col].mean())
    elif isinstance(baseline, (int, float)):
        baseline_val = float(baseline)
    else:
        raise ValueError("baseline must be 'overall' or a numeric value")

    df["baseline"] = baseline_val
    df["gap_abs"] = df[metric_col] - baseline_val
    df["gap_rel"] = df["gap_abs"] / (abs(baseline_val) + 1e-9)
    df["flag_disparity"] = df[metric_col].abs() > disparity_threshold * abs(baseline_val)
    return df.sort_values("gap_abs")


def top_bottom_slices(
    df_slices: pd.DataFrame,
    k: int = 10,
    metric_col: str = "metric",
) -> dict[str, pd.DataFrame]:
    """Top-k (best) and bottom-k (worst) slices by metric value."""
    sorted_df = df_slices.sort_values(metric_col).reset_index(drop=True)
    return {
        "bottom_k": sorted_df.head(k).copy(),
        "top_k": sorted_df.tail(k).iloc[::-1].reset_index(drop=True).copy(),
    }
