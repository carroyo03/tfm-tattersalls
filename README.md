# Predictive Modelling for Horse Auction Prices
**Master's Thesis (TFM) — Master in Big Data Science & Artificial Intelligence**  
*Universidad de Navarra · idealista*

---

## 1. Overview & Research Focus

This project develops a two-stage predictive pipeline for the *Tattersalls Autumn Horses in Training Sale* (2009–2025, 17 editions, 26,076 catalogued lots). The pipeline addresses two sequential questions:

1. **Will this horse sell to a third party?** — binary classification → P(sold\_to\_third\_party)
2. **What price will it fetch?** — regression on `log_price_detrended` (= `log_price_gns − log_year_median_price_prior`), applied to all offered lots

The core challenges driving methodological choices:

- **Selection bias** — prices are only cleanly observed for lots sold to third parties. Vendor buybacks/RNAs indicate that the reserve was not met, so they are analysed as non-transactions rather than true market prices.
- **Temporal drift** — 17 years of macroeconomic shift (+78% nominal prices, GBP/EUR volatility, BoE rate cycles) require strict temporal validation and a detrended regression target.
- **High-cardinality entities** — ~997 unique sires, ~840 consignors, with ~90% and ~70% rotation between early (2009–2015) and recent (2021–2025) periods. Cold-start risk at inference time.
- **Log-normal price distribution** — skewness 6.98 in raw scale, 0.03 in log scale. Regression target is `log_price_detrended` (`log_price_gns − log_year_median_price_prior`).

---

## 2. Key EDA Findings

| Finding | Result |
|---|---|
| **Price distribution** | Log-normal (skewness 6.98 raw → 0.03 log) — target: `log_price_gns` |
| **Strongest predictor** | Catalogue day: Days 1–2 median ~17,000 gns vs Days 4–5 ~4,000 gns (3.4×, permutation p<0.0001) |
| **Sex premium** | Colts 17,000 > Geldings 13,000 > Fillies 7,000 gns (diff 0.887 log-units, p<0.0001) |
| **Intraday structure** | Price peaks at lot positions 0.6–0.8 within day ("prime time"); clearance rate flat (~85–90%) |
| **Nominal vs real growth** | +78% nominal (2009–2025), ~+20% real — ~58 pp explained by CPIH inflation |
| **Entity rotation** | Top sires ~90% rotated between periods; top consignors ~70% — cold-start risk |
| **Post-Brexit stability** | Day 1–2 price premium structurally stable across 4- and 5-day sale editions |

---

## 3. Modelling Architecture

### Stage 1 — Classification: P(sold\_to\_third\_party)

Binary classifier predicting whether a catalogued lot will sell to a third party.

| Decision | Choice | Reason |
|---|---|---|
| **Target** | `sold_to_third_party` (binary) | Economically meaningful; vendor buyback vs. not-sold distinction driven by unobservable reserve price |
| **Universe** | All offered lots (~18,989, withdrawn excluded) | Withdrawn lots never faced the market |
| **Class imbalance** | ~87% positive — `class_weight='balanced'` + PR-AUC optimisation | Accuracy misleading; F1-weighted is the primary metric |
| **Final model selected in modeling** | **Stacking ensemble (RF · XGB · LGBM · CatBoost) with LR meta-learner** | PR-AUC tied (0.938) with Random Forest; stacking wins ROC-AUC (0.6521 vs 0.6461) and calibration |

### Stage 2 — Regression: log\_price\_detrended

Price regression trained on lots with observable prices, applied to the full offered universe.
Target is **detrended**: `log_price_detrended = log_price_gns − log_year_median_price_prior`.
At prediction time, the trend is re-added: `log_price_nominal = pred_detrended + log_year_median_price_prior`.

| Decision | Choice | Reason |
|---|---|---|
| **Training set** | sold\_to\_third\_party only (~16.5k rows) | Realised third-party market price; excludes reserve-not-met/buyback observations |
| **Inference set** | All offered lots (`inference_universe`) | Counterfactual fair-value for sold, buyback, and not-sold lots |
| **Target** | `log_price_detrended` | Removes +78% nominal drift; model focuses on relative horse value |
| **Detrending** | `log_price_gns − log_year_median_price_prior` | Absorbs 78% nominal drift; re-added at prediction time |
| **Final model selected in modeling** | **Stacking ensemble (RF · XGB · LGBM · CatBoost) with Ridge meta-learner** | Wins validation RMSE (1.146 vs RF 1.158, **−1%**). Margin is modest; stacking chosen for robustness |

### Dataset Flow

```
Raw CSV (26,076 lots)
  └─ notebooks/01_Data_Preparation      →  clean_data.parquet
       └─ notebooks/02_EDA_Analysis     →  autumn_horses_modeling_ready.csv
            └─ notebooks/03_FeatureEngineering  →  classification_ready   (20 selected features, Stage 1)
                                               →  regression_ready       (12 selected features, Stage 2 train)
                                               →  inference_universe     (selected features, Stage 2 predict)
                 └─ notebooks/04_Modeling       →  stacking predictions + SHAP
                      └─ notebooks/05_Model_Audit  →  fairness · calibration · RNA · drift
```

---

## 4. Dataset & Analytical Universe

| Outcome | N | % of total | Treatment |
|---|---|---|---|
| Sold to third party | 16,531 | 63.4% | Stage 1 positive class · Stage 2 training ✅ |
| Withdrawn before ring | 7,081 | 27.2% | **Excluded** from all modelling universes |
| Vendor buyback | 1,383 | 5.3% | Stage 1 negative class · Stage 2 inference/audit only (reserve not met) |
| Not sold on the day | 1,081 | 4.1% | Stage 1 negative class · Stage 2 inference only (price predicted) |

**On vendor buybacks in the regression target:** the final specification treats buybacks/RNAs as non-transactions. They are retained in the broader inference/audit universe, but excluded from Stage 2 training to prevent downward bias in expected values for high-quality lots.

---

## 5. Feature Engineering Decisions

| EDA Conclusion | Implementation | Notes |
|---|---|---|
| Day 1–2 premium (stable post-Brexit) | `day`, `day_normalized`, `intraday_position`, `is_prime_time` | Temporal normalisation for 4- vs 5-day editions |
| Sire quality signal | `sire_target_enc`, `sire_sale_rate_enc`, `sire_frequency` | M-estimate encoding with expanding window (anti-leakage) |
| Consignor reputation | `consignor_target_enc`, `consignor_volume`, `consignor_price_tier`, `consignor_sale_rate_enc` | Root-entity normalisation (removes Ltd, Racing, etc.) |
| Macro context | `gbp_eur_rate`, `boe_base_rate`, `year_*_prior` | LOO expanding window to prevent temporal leakage |
| Colour, foaled month — noise | Excluded from feature sets | Marginal effect <5%; captured by other variables |
| Sire-dam combo — leakage risk | Excluded from regression; novelty proxy in classification | 90%+ singletons |

**Target encoding strategy**: M-estimate with expanding temporal window — each observation's encoding uses only data from years prior to its sale year. Global mean shrinkage factor `m=10` (price and sale-rate).

**Feature selection**: Features reduced from 55 to **20 (classification)** and **12 (regression)** via cumulative permutation importance + redundancy diagnostic. See `03_FeatureEngineering` §8.5–8.7.

**Model Performance Snapshot from Modeling Run** (test OOT 2022–2025):

| Stage | Metric | Value | Notes |
|---|---|---|---|
| Classification (Stage 1) | AUC-ROC OOT | **0.641** | Final model (stacking classifier: RF·XGB·LGBM·CB → LR meta) |
| Classification | AUC-PR OOT | **0.928** | Primary metric given class imbalance |
| Classification | ECE | **0.023** | Expected Calibration Error — stacking is natively well-calibrated |
| Regression (Stage 2) | RMSE_log OOT | **1.133** | Detrended target (log_price_gns − log_year_median_price_prior) |
| Regression | R²_log OOT | **0.270** | ~27% price variance explained by catalogue features |
| Regression | Bias_log OOT | −0.139 | Slight underprediction — consistent with post-COVID market regime shift |

Threshold selected by Youden's J on validation: **≈0.888**.

**Raw-scale benchmark vs hedonic OLS (Ridge)** (nominal GNS, OOT 2022–2025):

The stacking ensemble outperforms the hedonic Ridge baseline across all raw-scale metrics.
R²_raw is low because exp(log) amplifies upper-tail errors — the relevant metric is R²_log (0.270).
MdAPE (~68%) better describes the typical error because MAPE is inflated by the lowest price decile
(<2k GNS), where the model overpredicts (these are mainly retirements/liquidations, not standard market transactions).

The global MAPE (219%) is dominated by the lowest decile (<2k GNS). In the 22k–75k GNS segment
where ~60% of the market operates, the MAPE is 37–49%. The MdAPE (68.6%) better describes the typical error.

**SHAP Interpretability via Surrogate LGBM**: SHAP values are computed on LightGBM models trained
imitating the stacking ensembles. Surrogate fidelity R²=0.9966 (CLF) / 0.9995 (REG) on OOT.
This is standard practice for explaining stacked ensembles (Hasnat et al., 2025; Choudhary et al., 2025).

**Fundamental limit of catalogue-only data**: An R²_log of 0.25 is what can be explained from
catalogue features alone (pedigree, consignor reputation, catalogue position, macro context).
The remaining 75% of price variance is driven by unobservable factors: physical conformation,
biomechanics, veterinary inspection findings, temperament, and buyer–day demand dynamics.
These features require video analysis and computer vision, which is the natural next step
beyond this work. An et al. (2026) demonstrate that ML models using clinical (OCD surgery)
and pedigree data can meaningfully predict racehorse performance, suggesting that analogous
clinical variables could improve auction price models beyond catalogue features alone.

---

## 6. Repository Structure

```text
.
├── notebooks/
│   ├── 01_Data_Preparation.ipynb       # Load, clean, define outcomes, export clean_data.parquet
│   ├── 02_EDA_Analysis.ipynb           # Exploratory analysis: market, pedigree, temporal dynamics
│   ├── 03_FeatureEngineering.ipynb     # Two-stage pipeline prep: feature engineering + 3 exports
│   ├── 04_Modeling.ipynb               # Stage 1 classifier + Stage 2 regressor
│   └── 05_Model_Audit.ipynb            # Model audit: fairness, calibration, SHAP, RNA paradox
│
├── src/
│   ├── data_prep.py                    # Data loading, cleaning, parsing, bootstrap CI
│   ├── evaluation.py                   # Discrimination, calibration, residuals, drift metrics
│   ├── audit.py                        # Fairness slices, disparity analysis
│   ├── sensors.py                      # Temporal split validation, leakage checks, invariants
│   ├── features.py                     # Temporal target encoding (M-estimate expanding window)
│   ├── model_wrappers.py               # Persist exact stacked final models
│   ├── final_audit.py                  # Reproducibility gate for final artefacts
│   ├── save_model_artifacts.py         # Export best MLflow models → models/
│   ├── placement_products.py           # Ranking report, anomaly flags, placement simulation
│   └── ablation_vendor_buybacks.py     # Stage 2 ablation: with vs. without buybacks
│
├── tests/
│   └── test_smoke.py                   # Smoke tests for evaluation and audit modules
│
├── tasks/
│   ├── todo.md                         # Active task tracking (per-session)
│   └── lessons.md                      # Lessons learned log
│
├── outputs/
│   ├── analyses/                       # CSVs, parquets from audit and ablation runs
│   ├── figures/                        # All figures (EDA, audit) — PDF + PNG
│   ├── reports/                        # Auto-generated reports
│   └── memo_defensa/                   # Defence presentation materials
│
├── models/                             # Trained model artifacts (.joblib via MLflow)
├── data/                               # Raw CSVs + processed parquets (gitignored)
├── raw/                                # Experimental notebooks, papers, design decisions
├── pyproject.toml                      # Project dependencies and config
├── uv.lock                             # Locked dependency tree (reproducible)
└── README.md
```

---

## 7. Development Setup

This project uses `uv` for reproducible environment management.

```bash
# Clone and sync — installs exact dependency tree from uv.lock
uv sync

# Or create venv first, then sync
uv venv .venv
source .venv/bin/activate       # macOS / Linux
uv sync
```

**Key dependencies**: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, `statsmodels`,
`scikit-learn>=1.9.0`, `xgboost`, `lightgbm`, `catboost`, `optuna`, `mlflow`, `shap`

**Notebook execution order**: `01` → `02` → `03` → `04` → `05`  
Each notebook reads from `data/processed/` and writes back to it.

---

## 8. Key Findings & Audit Results

**Model performance** is moderate by design — auction price is inherently hard to predict from catalogue features alone (information asymmetry, undisclosed reserve prices, buyer-day demand). An AUC-ROC of 0.63 and R²_log of 0.25 represents the frontier of catalogue-only predictive power.

The remaining ~75% of price variance cannot be recovered from catalogue data alone. Features like physical conformation, biomechanics (length of stride, muscle mass, joint angles), veterinary inspection findings, temperament, and buyer–day demand dynamics are required. 

**SHAP importance** (via LGBM surrogate trained to imitate the stacking ensemble): computed per-run in `05_Model_Audit.ipynb` §9. The strongest signals across both stages are the catalogue-day effects (`consignor_day_interaction`, `intraday_position`), entity target encodings (`sire_target_enc`, `consignor_target_enc`), and sex-based interactions (`sex_sire_interaction`, `sex_day_interaction`). See `outputs/figures/audit/09_shap_*.{pdf,png}` for the full beeswarm and importance plots.

**Temporal drift**: 80% of features show KS-test drift (p < 0.05) between 2009–2021 and 2022–2025. Model requires annual expanding-window retraining before each October sale.

**Fairness slices**: Day 5 AUC-ROC = 0.47 (near-random) — model does not distinguish what sells at the tail of the sale, consistent with higher variance in end-of-sale lot quality.

**RNA paradox**: ~2,462 RNA lots (13% of offered universe) represent lots that did not meet their reserve. The hurdle model correctly identifies RNA lots as having systematically lower expected prices than sold lots because `P(sold)` discounts their nominal predictions. 362 RNA lots with expected_price > P75 of sold lots are flagged as potential high-value opportunities — these are predominantly geldings on catalogue days 3–4 where reserve prices may have been set too aggressively. See §10 in `05_Model_Audit` for the full permutation-test analysis.

**Ablation (vendor buybacks)**: Buybacks are kept as a sensitivity/audit topic because they expose reserve behaviour and affect high-cardinality encodings, but the final price model uses sold-to-third-party only for unbiased expected price mapping.

**Leakage audit**: PASSED — sensors validated no temporal leakage in target encoding, macro features, or train/OOT splits.

---

## Citation

## Regenerate SHAP figures (production)

Interpretability figures (SHAP) are computed on a **surrogate LGBM** trained to imitate
the stacking ensemble. This is necessary because SHAP TreeExplainer requires a single tree model, not
an ensemble of ensembles. Surrogate fidelity is verified before computing SHAP.

Open the notebook `notebooks/05_Model_Audit.ipynb` and execute section §9 (SHAP Interpretability).
The figures are automatically saved in `outputs/figures/audit/`.

The script `scripts/generate_shap_production.py` has been removed; notebook 05 is the canonical
source for SHAP generation, with inline surrogate fidelity checks.


If you use this work, please cite:

```bibtex
@mastersthesis{arroyo2025predictive,
  title  = {Predictive Modelling for Horse Auction Prices},
  author = {Arroyo Lorenzo, Carlos G.},
  school = {Universidad de Navarra},
  year   = {2026},
  note   = {Master's Thesis — Big Data Science \& AI, in collaboration with idealista}
}
```

---

## License

This project is released under the [MIT License](LICENSE).

---

*Thesis supervised by Stella Salvatierra (Universidad de Navarra) and Daniel del Pozo Salinas (idealista).*
