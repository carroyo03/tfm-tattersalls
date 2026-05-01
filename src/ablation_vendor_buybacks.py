"""Stage 2 ablation: impact of including vendor buybacks in the target-encoding base.

In the feature engineering pipeline (03_FeatureEngineering, Cell 22), the encoding
base ``df_price`` includes BOTH sold_to_third_party AND vendor_buyback lots:

    df_price = df[(df['sold_to_third_party']) | (df['vendor_buyback'])]

Vendor buybacks have a recorded price (the revealed reserve), so ``log_price_gns``
is computed from ``price_gns`` for them (Cell 10 in FE notebook).

This script quantifies the counterfactual: what happens to M-estimate encodings
if vendor buybacks are excluded?  Key questions for the thesis:
  1. How many entity observations are gained?
  2. How much do M-estimate values shift?
  3. Are any entities visible ONLY via buybacks?

Outputs written to outputs/analyses/:
  - ablation_vendor_buybacks_entity_impact.csv
  - ablation_vendor_buybacks_summary.csv
  - outputs/figures/audit/ablation_vendor_buybacks_enc_diff.png

Run from project root:
    python -m src.ablation_vendor_buybacks
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR   = Path("data/processed")
OUT_DIR    = Path("outputs/analyses")
FIG_DIR    = Path("outputs/figures/audit")
M_PRICE    = 10.0   # M-estimate regularisation for price encoding (matches FE notebook)
# (entity_col_in_clean_data, encoded_col_label)
ENTITY_COLS = [
    ("sire_entity",     "sire_target_enc"),
    ("damsire_entity",  "damsire_target_enc"),
    ("consignor_model", "consignor_target_enc"),
]
TARGET_COL = "log_price_gns"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


def m_estimate_global(entity_counts: pd.Series, entity_means: pd.Series,
                      global_mean: float, m: float) -> pd.Series:
    """M-estimate: (n*entity_mean + m*global_mean) / (n + m)."""
    return (entity_counts * entity_means + m * global_mean) / (entity_counts + m)


def compute_encodings(df_base: pd.DataFrame, entity_col: str) -> pd.DataFrame:
    """Compute global M-estimate encoding for each entity using the full df_base."""
    global_mean = float(df_base[TARGET_COL].mean())
    stats = (
        df_base.groupby(entity_col)[TARGET_COL]
        .agg(n="count", mean_target="mean")
        .reset_index()
        .rename(columns={entity_col: "entity"})
    )
    stats["enc"] = m_estimate_global(stats["n"], stats["mean_target"],
                                     global_mean, M_PRICE)
    stats["global_mean"] = global_mean
    return stats


def run_ablation() -> dict:
    """Compare encodings computed with vs without vendor buybacks."""
    clean = pd.read_parquet(DATA_DIR / "clean_data.parquet")

    # Replicate feature engineering Cell 10: compute log_price_gns for buybacks
    vb_mask = (clean["vendor_buyback"] == True) & clean["price_gns"].notna()
    clean.loc[vb_mask, TARGET_COL] = np.log(clean.loc[vb_mask, "price_gns"])

    df_with    = clean[(clean["sold_to_third_party"]) | (clean["vendor_buyback"])].copy()
    df_with    = df_with[df_with[TARGET_COL].notna()]
    df_without = clean[clean["sold_to_third_party"]].copy()
    df_without = df_without[df_without[TARGET_COL].notna()]

    print(f"Encoding base WITH buybacks   : {len(df_with):,} rows")
    print(f"Encoding base WITHOUT buybacks: {len(df_without):,} rows")
    print(f"Vendor buyback observations   : {len(df_with) - len(df_without):,}")

    results = []
    summary_rows = []

    for entity_col, enc_label in ENTITY_COLS:
        if entity_col not in clean.columns:
            print(f"  Skipping {entity_col} — not in clean_data")
            continue

        enc_with    = compute_encodings(df_with,    entity_col)
        enc_without = compute_encodings(df_without, entity_col)

        merged = enc_with.merge(enc_without, on="entity", how="outer",
                                suffixes=("_with", "_without"))

        # Entities with ZERO obs when buybacks are excluded fall back to global mean
        gm_without = float(df_without[TARGET_COL].mean())
        merged["n_with"]            = merged["n_with"].fillna(0).astype(int)
        merged["n_without"]         = merged["n_without"].fillna(0).astype(int)
        merged["enc_with"]          = merged["enc_with"].fillna(merged["global_mean_with"])
        merged["enc_without"]       = merged["enc_without"].fillna(gm_without)
        merged["global_mean_with"]  = merged["global_mean_with"].fillna(float(df_with[TARGET_COL].mean()))

        merged["obs_gained"]   = merged["n_with"] - merged["n_without"]
        merged["enc_abs_diff"] = (merged["enc_with"] - merged["enc_without"]).abs()
        merged["entity_col"]   = enc_label
        results.append(merged)

        # Entities visible ONLY via buybacks
        buyback_only = merged[merged["n_without"] == 0]

        summary_rows.append({
            "entity_col":                    enc_label,
            "n_entities_with":               int((merged["n_with"] > 0).sum()),
            "n_entities_without":            int((merged["n_without"] > 0).sum()),
            "n_entities_buyback_only":       len(buyback_only),
            "total_obs_gained":              int(merged["obs_gained"].sum()),
            "median_obs_gained_per_entity":  float(merged["obs_gained"].median()),
            "mean_enc_abs_diff":             float(merged["enc_abs_diff"].mean()),
            "p95_enc_abs_diff":              float(merged["enc_abs_diff"].quantile(0.95)),
            "max_enc_abs_diff":              float(merged["enc_abs_diff"].max()),
            "global_mean_with":              float(df_with[TARGET_COL].mean()),
            "global_mean_without":           gm_without,
        })

    df_impact  = pd.concat(results, ignore_index=True)
    df_summary = pd.DataFrame(summary_rows)

    df_impact.to_csv(OUT_DIR / "ablation_vendor_buybacks_entity_impact.csv", index=False)
    df_summary.to_csv(OUT_DIR / "ablation_vendor_buybacks_summary.csv", index=False)

    # ── Figure ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, len(ENTITY_COLS), figsize=(15, 4), sharey=False)
    for ax, (entity_col, enc_label) in zip(axes, ENTITY_COLS):
        sub = df_impact[df_impact["entity_col"] == enc_label]
        nonzero = sub[sub["enc_abs_diff"] > 0]
        ax.hist(nonzero["enc_abs_diff"], bins=40, color="#2563eb",
                edgecolor="white", alpha=0.8)
        mean_diff = float(sub["enc_abs_diff"].mean())
        ax.axvline(mean_diff, color="red", lw=1.5, ls="--",
                   label=f"mean={mean_diff:.4f}")
        n_bo = int((sub["n_without"] == 0).sum())
        ax.set_title(f"{enc_label}\n(buyback-only entities: {n_bo})", fontsize=10)
        ax.set_xlabel("|enc_with − enc_without|  (log-GNS)")
        ax.set_ylabel("# entities")
        ax.legend(fontsize=8)

    fig.suptitle(
        "Stage 2 Ablation: M-estimate shift when vendor buybacks are removed from encoding base",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ablation_vendor_buybacks_enc_diff.png", dpi=150)
    plt.close(fig)

    return {"summary": df_summary, "impact": df_impact}


if __name__ == "__main__":
    out = run_ablation()
    print("\n=== Ablation Summary ===")
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 120)
    print(out["summary"].to_string(index=False))
    print(f"\nDetailed entity impact → {OUT_DIR}/ablation_vendor_buybacks_entity_impact.csv")
    print(f"Figure → {FIG_DIR}/ablation_vendor_buybacks_enc_diff.png")
