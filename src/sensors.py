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
    entity_encoded_pairs: Sequence[tuple[str, str] | tuple[str, str, float]],
    year_col: str = "sale_year",
    target_col: str | None = None,
    m: float = 10.0,
    sample_n: int = 500,
    tol: float = 1e-6,
    random_state: int = 42,
    encoding_mask: pd.Series | None = None,
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
        For pairs that use a different regularisation constant, pass a 3-tuple
        ``(entity_col, encoded_col, pair_m)`` — the per-pair ``pair_m`` overrides
        the global ``m`` for that entity.
    year_col:
        Column name for the sale year.
    target_col:
        Target column used for encoding (e.g. ``"log_price_gns"``). When
        provided, enables the recomputation check.
    m:
        Default regularisation constant for the M-estimate. Overridden by
        per-pair ``pair_m`` in 3-tuple entries.
    sample_n:
        Number of rows to sample for the (expensive) recomputation check.
    tol:
        Absolute tolerance for the recomputation comparison.
    random_state:
        Seed for the row sample.
    encoding_mask:
        Boolean mask indicating which rows were used as the encoding base
        (e.g. ``df['sold_to_third_party'] == True`` for price encodings).
        If None, all rows with a non-NaN target are used, which may include
        vendor buyback prices and produce a different global mean than the
        sold-only encoding base used during feature engineering.
    """
    missing_cols = [
        col
        for item in entity_encoded_pairs
        for col in (item[0], item[1])
        if col not in df.columns
    ]
    if missing_cols:
        warnings.warn(
            f"encoding_leakage_check: columns not found in df, skipping: {missing_cols}",
            stacklevel=2,
        )
        entity_encoded_pairs = [
            item for item in entity_encoded_pairs
            if item[0] in df.columns and item[1] in df.columns
        ]

    for item in entity_encoded_pairs:
        if len(item) == 3:
            entity_col, encoded_col, pair_m = item
        else:
            entity_col, encoded_col = item
            pair_m = m

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

    # Determine encoding base: if encoding_mask is provided, use only those rows;
    # otherwise use all rows with a non-NaN target (may include buyback prices).
    encoding_base = df[encoding_mask] if encoding_mask is not None else df[target_col].notna()
    if isinstance(encoding_base, pd.Series):
        encoding_base = df[encoding_base]
    encoding_base = encoding_base.dropna(subset=[target_col])

    global_mean = float(encoding_base[target_col].mean())
    rng = np.random.default_rng(random_state)
    years = sorted(df[year_col].unique())
    # skip first year (no prior data to recompute from)
    eligible = df[df[year_col] > years[0]]
    sample_idx = rng.choice(len(eligible), size=min(sample_n, len(eligible)), replace=False)
    sample_df = eligible.iloc[sample_idx]

    for item in entity_encoded_pairs:
        if len(item) == 3:
            entity_col, encoded_col, pair_m = item
        else:
            entity_col, encoded_col = item
            pair_m = m

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

            prior = encoding_base[encoding_base[year_col] < year_val]
            if len(prior) == 0:
                expected = global_mean
            else:
                entity_prior = prior[prior[entity_col] == entity_val]
                entity_target = entity_prior[target_col].dropna()
                n = len(entity_target)
                if n == 0:
                    expected = global_mean
                else:
                    entity_mean = float(entity_target.mean())
                    expected = (n * entity_mean + pair_m * global_mean) / (n + pair_m)

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
    id_cols: Sequence[str] = ("lot_uid",),
) -> None:
    """Assert that all regression_ready IDs exist in the inference_universe.

    Verifies: |inference_universe| >= |regression_ready| and
    IDs(regression_ready) ⊂ IDs(inference_universe).
    """
    id_cols = list(id_cols)
    missing = [
        col for col in id_cols
        if col not in inference_universe.columns or col not in regression_ready.columns
    ]
    if missing:
        fallback = ["sale_name", "sale_year", "day", "lot"]
        if all(col in inference_universe.columns and col in regression_ready.columns for col in fallback):
            id_cols = fallback
        else:
            raise AssertionError(
                "Universe consistency cannot be checked with a stable key. "
                f"Missing requested ID columns: {missing}. Add 'lot_uid' or preserve "
                "('sale_name', 'sale_year', 'day', 'lot') in all model datasets."
            )

    if len(inference_universe) < len(regression_ready):
        raise AssertionError(
            f"Universe consistency violated: inference_universe has "
            f"{len(inference_universe)} rows but regression_ready has "
            f"{len(regression_ready)} rows."
        )

    for name, frame in [
        ("inference_universe", inference_universe),
        ("regression_ready", regression_ready),
    ]:
        n_dup = int(frame.duplicated(id_cols).sum())
        if n_dup:
            raise AssertionError(
                f"Universe consistency violated: {name} has {n_dup} duplicate "
                f"rows for ID columns {id_cols}. Use a unique lot identifier."
            )

    univ_ids = set(map(tuple, inference_universe[id_cols].values.tolist()))
    reg_ids = set(map(tuple, regression_ready[id_cols].values.tolist()))

    leaked = reg_ids - univ_ids
    if leaked:
        raise AssertionError(
            f"Universe consistency violated: {len(leaked)} IDs in regression_ready "
            f"are NOT in inference_universe. First 5: {list(leaked)[:5]}"
        )
