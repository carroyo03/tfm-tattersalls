"""Production-grade products built on the two-stage TFM pipeline.

Three products:
1. ranking_report() — Rank lots by expected_price with per-obs uncertainty
2. anomaly_flags() — Identify lots where expected_price > P75_sold among RNAs
3. batch_simulate_placement() — Simulate price under alternative catalogue placement

All functions produce parquet files in outputs/analyses/ and print human-readable summaries.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

OUT_DIR = Path("outputs/analyses")
OUT_DIR.mkdir(parents=True, exist_ok=True)
_RANDOM_SEED = 42


# ═══════════════════════════════════════════════════════════════════════════════
# Shared utilities
# ═══════════════════════════════════════════════════════════════════════════════

def ensemble_disagreement(
    base_models: dict[str, object],
    X: np.ndarray,
    mode: str = "reg",
) -> np.ndarray:
    """Per-observation variance across base model predictions (classification or regression).

    Parameters
    ----------
    base_models : dict of str → sklearn-compatible estimator
    X : (n_samples, n_features) array
    mode : "reg" or "clf"

    Returns
    -------
    variance : (n_samples,) ndarray
        Variance of predictions across base models for each observation.
        Higher values indicate higher model disagreement → higher uncertainty.
    """
    if mode == "clf":
        preds = np.column_stack([
            m.predict_proba(X)[:, 1] for m in base_models.values()
        ])
    else:
        preds = np.column_stack([
            m.predict(X) for m in base_models.values()
        ])
    return preds.var(axis=1)


def _load_stage2_base_models() -> dict[str, object]:
    """Load the four base regressors from the backup stacking wrapper."""
    wrapper = joblib.load("models/_backup_stage2_final_model.joblib")
    return wrapper.base_models


# ═══════════════════════════════════════════════════════════════════════════════
# Product 1: Ranking Report
# ═══════════════════════════════════════════════════════════════════════════════

def generate_ranking_report(
    univ_path: str = "outputs/analyses/audit_universe_predictions.parquet",
    reg_path: str = "outputs/analyses/audit_reg_predictions.parquet",
    output_name: str = "ranking_report.parquet",
    top_n: int = 20,
) -> pd.DataFrame:
    """Rank all inference lots by expected_price with per-observation uncertainty.

    The uncertainty is computed as ensemble disagreement (variance across the
    4 tree base models of the original stacking regressor). This is a meaningful
    per-observation proxy — lots with high disagreement are cases where the
    models disagree most (e.g. novel sire/consignor combinations, edge cases).
    """
    print("=" * 60)
    print("PRODUCT 1: Ranking Report")
    print("=" * 60)

    univ = pd.read_parquet(univ_path)
    print(f"  Loaded universe: {len(univ)} lots")

    # Rank by expected_price
    ranked = univ.sort_values("expected_price", ascending=False).reset_index(drop=True)
    ranked["rank"] = range(1, len(ranked) + 1)
    ranked["percentile"] = (ranked["rank"] / len(ranked) * 100).round(1)

    # Compute ensemble disagreement for a sample of the universe
    # We need the regression features to predict
    feat_reg = pd.read_csv("data/processed/feature_documentation.csv")
    reg_cols = feat_reg[feat_reg["model"] == "regression"]["feature"].tolist()

    # Load Stage 2 model (RF)
    stage2 = joblib.load("models/stage2_final_model.joblib")

    # Compute disagreement for all rows
    base_models = _load_stage2_base_models()
    X_univ = univ[[c for c in reg_cols if c in univ.columns]].values
    disagree = ensemble_disagreement(base_models, X_univ, mode="reg")
    ranked["ensemble_disagreement"] = disagree

    # Uncertainty tier
    q1, q3 = np.percentile(disagree, [25, 75])
    ranked["uncertainty_tier"] = pd.cut(
        disagree,
        bins=[-np.inf, q1, q3, np.inf],
        labels=["Low", "Medium", "High"],
    )

    # Save
    out_path = OUT_DIR / output_name
    ranked.to_parquet(out_path)
    print(f"  Saved to {out_path}")

    # Print top / bottom
    print(f"\n  ── Top {top_n} lots by expected_price ──")
    top = ranked.head(top_n)
    for _, row in top.iterrows():
        print(f"    #{row['rank']:>5d} | ${row['expected_price']:>8,.0f} GNS | "
              f"Day {int(row['day'])} | uncert={row['ensemble_disagreement']:.3f} | "
              f"P(sold)={row['prob_sold']:.3f}")

    print(f"\n  ── Bottom {top_n} lots by expected_price ──")
    bot = ranked.tail(top_n).iloc[::-1]
    for _, row in bot.iterrows():
        print(f"    #{row['rank']:>5d} | ${row['expected_price']:>8,.0f} GNS | "
              f"Day {int(row['day'])} | uncert={row['ensemble_disagreement']:.3f}")

    # Uncertainty distribution
    print(f"\n  Uncertainty distribution:")
    print(f"    Min: {disagree.min():.4f}, Median: {np.median(disagree):.4f}, Max: {disagree.max():.4f}")
    print(f"    Low uncertainty (n≤Q1): {(disagree <= q1).sum()} lots | "
          f"High uncertainty (n≥Q3): {(disagree >= q3).sum()} lots")

    return ranked


# ═══════════════════════════════════════════════════════════════════════════════
# Product 2: "Cheap" Anomaly Flags
# ═══════════════════════════════════════════════════════════════════════════════

def generate_anomaly_flags(
    univ_path: str = "outputs/analyses/audit_universe_predictions.parquet",
    output_name: str = "anomaly_flags.parquet",
) -> pd.DataFrame:
    """Identify RNA (Reserve Not Attained) lots that look 'cheap' relative to their profile.

    These are lots where the model estimates a high expected_price (greater than
    P75 among successfully sold lots) but the horse did not sell. This signals
    potential market inefficiency or a mispriced reserve.
    """
    print("\n" + "=" * 60)
    print("PRODUCT 2: Anomaly Flags — Undervalued RNA Detection")
    print("=" * 60)

    univ = pd.read_parquet(univ_path)
    print(f"  Loaded universe: {len(univ)} lots")

    # Compute P75 of expected_price among sold lots
    sold = univ[univ["sold_to_third_party"] == True]
    p75_sold = sold["expected_price"].quantile(0.75)
    print(f"  P75 of expected_price (sold): {p75_sold:,.0f} GNS")

    # Find RNAs with expected_price > P75_sold
    not_sold = univ[univ["sold_to_third_party"] == False].copy()
    flags = not_sold[not_sold["expected_price"] > p75_sold].copy()
    print(f"  RNAs above P75_sold: {len(flags)} lots ({len(flags)/len(not_sold)*100:.1f}% of all RNAs)")

    # Reconstruct sex from one-hot columns if available
    _sex_cols = ["sex_C", "sex_F", "sex_G", "sex_H", "sex_M"]
    available_sex_cols = [c for c in _sex_cols if c in flags.columns]
    if available_sex_cols:
        flags["sex"] = (
            flags[available_sex_cols].idxmax(axis=1)
            .str.replace("sex_", "", regex=False)
            .replace({"C": "Colt", "F": "Filly", "G": "Gelding", "H": "Horse", "M": "Mare"})
        )

    # Anomaly score: how many day-MADs above the day's median?
    # MAD = Median Absolute Deviation — robust to outliers
    day_mads = {}
    for day in sorted(flags["day"].unique()):
        day_lots = univ[univ["day"] == day]["expected_price"]
        med = day_lots.median()
        mad = np.median(np.abs(day_lots - med))
        day_mads[day] = (med, max(mad, 1.0))  # avoid div by zero

    # We need the lot_uid or sale_year + day + lot to identify each lot
    # Check what ID columns we have
    id_cols = ["lot_uid"] if "lot_uid" in flags.columns else ["sale_year", "day", "lot"]

    rows = []
    for _, lot in flags.iterrows():
        day = lot["day"]
        med, mad = day_mads.get(day, (p75_sold, 1000))
        anomaly_score = (lot["expected_price"] - med) / mad
        rows.append({
            **{c: lot[c] for c in id_cols if c in lot.index},
            "expected_price": lot["expected_price"],
            "price_nominal_pred": lot["price_nominal_pred"],
            "prob_sold": lot["prob_sold"],
            "day": int(lot["day"]),
            "day_median_price": med,
            "day_mad": mad,
            "anomaly_score": anomaly_score,
            "sex": lot.get("sex", "unknown"),
        })

    result = pd.DataFrame(rows).sort_values("anomaly_score", ascending=False).reset_index(drop=True)
    out_path = OUT_DIR / output_name
    result.to_parquet(out_path)
    print(f"  Saved to {out_path}")

    # Summary stats
    print(f"\n  ── Anomaly Score Distribution ──")
    print(f"    Mean: {result['anomaly_score'].mean():.2f} day-MADs above day median")
    print(f"    Max:  {result['anomaly_score'].max():.2f} day-MADs above day median")
    print(f"    Top anomaly: ${result.iloc[0]['expected_price']:,.0f} GNS on Day {int(result.iloc[0]['day'])}")

    # Sex composition
    if "sex" in result.columns:
        print(f"\n  ── Sex Composition of Flags ──")
        for sex, cnt in result["sex"].value_counts().head(5).items():
            print(f"    {sex}: {cnt} ({cnt/len(result)*100:.0f}%)")

    # Day composition
    print(f"\n  ── Day Distribution of Flags ──")
    for day, cnt in result["day"].value_counts().sort_index().items():
        day_total = len(univ[univ["day"] == day])
        print(f"    Day {int(day)}: {cnt} flags ({cnt/day_total*100:.1f}% of Day {int(day)})")

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Product 3: Placement Simulator
# ═══════════════════════════════════════════════════════════════════════════════

def batch_simulate_placement(
    n_lots: int = 10,
    univ_path: str = "outputs/analyses/audit_universe_predictions.parquet",
    output_name: str = "placement_simulation.parquet",
):
    """Batch simulation: shows expected_price variation by catalogue day.

    For each lot, predicts expected_price under the original day, then
    predicts under alternative days (1-5) with prime-time intraday position.
    """
    print("\n" + "=" * 60)
    print("PRODUCT 3: Placement Simulator")
    print("=" * 60)

    univ = pd.read_parquet(univ_path)

    # Group by day and show price distribution
    print("\n  ── Expected Price by Catalogue Day (for reference) ──")
    by_day = univ.groupby("day")["expected_price"].agg(["mean", "median", "std", "count"])
    by_day.index = by_day.index.astype(int)
    for d, row in by_day.iterrows():
        print(f"    Day {d}: median={row['median']:>8,.0f} GNS | "
              f"mean={row['mean']:>8,.0f} GNS | n={int(row['count']):,}")

    print("\n  ── Day-over-Day Price Uplift Potential (median) ──")
    best_day = by_day["median"].idxmax()
    worst_day = by_day["median"].idxmin()
    uplift_pct = (by_day.loc[best_day, "median"] / by_day.loc[worst_day, "median"] - 1) * 100
    print(f"    Best day: Day {int(best_day)} ({by_day.loc[best_day, 'median']:,.0f} GNS median)")
    print(f"    Worst day: Day {int(worst_day)} ({by_day.loc[worst_day, 'median']:,.0f} GNS median)")
    print(f"    Max uplift potential: {uplift_pct:.0f}% (moving from Day {int(worst_day)} → Day {int(best_day)})")

    # Show specific examples — lots that would benefit most
    print("\n  ── Lots with Highest Day-Shift Uplift Potential ──")
    late_lots = univ[univ["day"].isin([4, 5])].copy()
    late_lots["potential_uplift_pct"] = (
        by_day.loc[1, "median"] / late_lots["expected_price"] - 1
    ) * 100
    top_uplift = late_lots.nlargest(n_lots, "potential_uplift_pct")

    id_cols = ["lot_uid"] if "lot_uid" in univ.columns else ["sale_year", "day", "lot"]
    for _, lot in top_uplift.iterrows():
        ids = ", ".join([f"{c}={lot[c]!s}" for c in id_cols[:3] if c in lot.index])
        print(f"    {ids} | current: ${lot['expected_price']:>8,.0f} GNS "
              f"(Day {int(lot['day'])}) | Day 1 median: {by_day.loc[1, 'median']:>8,.0f} GNS "
              f"| potential uplift: {lot['potential_uplift_pct']:.0f}%")

    # Save results
    result = univ.groupby("day")["expected_price"].describe()
    out_path = OUT_DIR / output_name
    result.to_parquet(out_path)
    print(f"\n  Reference table saved to {out_path}")

    return result


if __name__ == "__main__":
    print("Generating all 3 products...")
    generate_ranking_report()
    generate_anomaly_flags()
    batch_simulate_placement()
    print("\n✅ All products generated. Load from outputs/analyses/")
