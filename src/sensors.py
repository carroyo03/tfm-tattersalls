"""Domain invariant sensors — run before training to verify data integrity.

Implements §2.2 of AGENTS.md: temporal_split_validator and encoding_leakage_check.
Call these from the notebook immediately after building splits / encoding features.
"""
from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
import pandas as pd


def temporal_split_validator(
    train: pd.DataFrame,
    val: pd.DataFrame,
    oot: pd.DataFrame,
    year_col: str = "sale_year",
) -> None:
    """Assert strict temporal ordering: max(train) < min(val) <= max(val) < min(oot).

    Raises AssertionError with a descriptive message on any violation.
    Call this immediately after building train/val/OOT splits.
    """
    t_max = int(train[year_col].max())
    v_min = int(val[year_col].min())
    v_max = int(val[year_col].max())
    o_min = int(oot[year_col].min())

    if t_max >= v_min:
        raise AssertionError(
            f"⚠ TEMPORAL LEAKAGE RISK: train max year ({t_max}) "
            f">= val min year ({v_min}). "
            f"Use an expanding-window or strict year-cutoff split."
        )
    if v_max >= o_min:
        raise AssertionError(
            f"⚠ TEMPORAL LEAKAGE RISK: val max year ({v_max}) "
            f">= oot min year ({o_min}). "
            f"Use an expanding-window or strict year-cutoff split."
        )


def encoding_leakage_check(
    df: pd.DataFrame,
    entity_encoded_pairs: Sequence[tuple[str, str]],
    year_col: str = "sale_year",
    target_col: str | None = None,
    m: float = 10.0,
    sample_n: int = 500,
    tol: float = 1e-6,
    random_state: int = 42,
) -> None:
    """Verify that target-encoded columns only use data from prior years.

    Two levels of checking:

    1. **Structural** (always run): within each (year, entity) group, every row
       must share the same encoded value — they all draw from identical prior-year
       history, so any within-group variance indicates row-level leakage.

    2. **Recomputation** (when ``target_col`` is provided): for a random sample
       of rows, recomputes the M-estimate encoding from scratch using only rows
       with ``year < row.year``, then asserts the recomputed value matches the
       stored one within ``tol``.

    Parameters
    ----------
    df:
        DataFrame that contains *both* entity columns and encoded columns
        (i.e. the intermediate feature-engineering dataset, not the model-ready one).
    entity_encoded_pairs:
        List of ``(entity_col, encoded_col)`` tuples, e.g.
        ``[("sire", "sire_target_enc"), ("damsire", "damsire_target_enc")]``.
    year_col:
        Column name for the sale year.
    target_col:
        Target column used for encoding (e.g. ``"log_price_gns"``). When
        provided, enables the recomputation check.
    m:
        Regularisation constant for the M-estimate (must match the value used
        during feature engineering).
    sample_n:
        Number of rows to sample for the (expensive) recomputation check.
    tol:
        Absolute tolerance for the recomputation comparison.
    random_state:
        Seed for the row sample.
    """
    missing_cols = [
        col
        for entity_col, encoded_col in entity_encoded_pairs
        for col in (entity_col, encoded_col)
        if col not in df.columns
    ]
    if missing_cols:
        warnings.warn(
            f"encoding_leakage_check: columns not found in df, skipping: {missing_cols}",
            stacklevel=2,
        )
        entity_encoded_pairs = [
            (e, enc)
            for e, enc in entity_encoded_pairs
            if e in df.columns and enc in df.columns
        ]

    for entity_col, encoded_col in entity_encoded_pairs:
        # ── Check 1: no NaN ──────────────────────────────────────────────────
        n_nan = int(df[encoded_col].isna().sum())
        if n_nan > 0:
            raise AssertionError(
                f"⚠ ENCODING LEAKAGE RISK: '{encoded_col}' has {n_nan} NaN values. "
                f"All rows must have a valid encoding (use global mean as fallback)."
            )

        # ── Check 2: within-(year, entity) consistency ───────────────────────
        group_std = (
            df.groupby([year_col, entity_col])[encoded_col]
            .std()
            .dropna()
        )
        if len(group_std) > 0:
            max_std = float(group_std.max())
            if max_std > tol:
                worst_group = group_std.idxmax()
                raise AssertionError(
                    f"⚠ ENCODING LEAKAGE RISK: '{encoded_col}' has within-group "
                    f"variance (max_std={max_std:.2e}) for group "
                    f"(year={worst_group[0]}, {entity_col}={worst_group[1]}). "
                    f"All rows in the same (year, entity) must share an identical encoding."
                )

    # ── Check 3: M-estimate recomputation (optional) ─────────────────────────
    if target_col is None or target_col not in df.columns:
        return

    global_mean = float(df[target_col].mean())
    rng = np.random.default_rng(random_state)
    years = sorted(df[year_col].unique())
    # skip first year (no prior data to recompute from)
    eligible = df[df[year_col] > years[0]]
    sample_idx = rng.choice(len(eligible), size=min(sample_n, len(eligible)), replace=False)
    sample_df = eligible.iloc[sample_idx]

    for entity_col, encoded_col in entity_encoded_pairs:
        if entity_col not in df.columns or encoded_col not in df.columns:
            continue
        if "sale_rate" in encoded_col:
            # sale-rate encodings use a binary target; skip recomputation unless
            # the caller provides the correct binary target
            continue

        for _, row in sample_df.iterrows():
            year_val = row[year_col]
            entity_val = row[entity_col]
            stored = float(row[encoded_col])

            prior = df[df[year_col] < year_val]
            if len(prior) == 0:
                expected = global_mean
            else:
                entity_prior = prior[prior[entity_col] == entity_val]
                n = len(entity_prior)
                if n == 0:
                    expected = global_mean
                else:
                    entity_mean = float(entity_prior[target_col].mean())
                    expected = (n * entity_mean + m * global_mean) / (n + m)

            if abs(stored - expected) > tol:
                raise AssertionError(
                    f"⚠ ENCODING LEAKAGE RISK: '{encoded_col}' recomputation mismatch "
                    f"(year={year_val}, {entity_col}={entity_val!r}): "
                    f"stored={stored:.6f}, recomputed={expected:.6f}, "
                    f"diff={abs(stored - expected):.2e}"
                )


def universe_consistency_check(
    inference_universe: pd.DataFrame,
    regression_ready: pd.DataFrame,
    id_cols: Sequence[str] = ("sale_year", "day", "lot"),
) -> None:
    """Assert that all regression_ready IDs exist in the inference_universe.

    Verifies: |inference_universe| >= |regression_ready| and
    IDs(regression_ready) ⊂ IDs(inference_universe).
    """
    if len(inference_universe) < len(regression_ready):
        raise AssertionError(
            f"Universe consistency violated: inference_universe has "
            f"{len(inference_universe)} rows but regression_ready has "
            f"{len(regression_ready)} rows."
        )

    id_cols = list(id_cols)
    univ_ids = set(map(tuple, inference_universe[id_cols].values.tolist()))
    reg_ids = set(map(tuple, regression_ready[id_cols].values.tolist()))

    leaked = reg_ids - univ_ids
    if leaked:
        raise AssertionError(
            f"Universe consistency violated: {len(leaked)} IDs in regression_ready "
            f"are NOT in inference_universe. First 5: {list(leaked)[:5]}"
        )
