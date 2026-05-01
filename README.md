# Predictive Modelling for Horse Auction Prices
**Master's Thesis (TFM) — Master in Big Data Science & Artificial Intelligence**  
*Universidad de Navarra · idealista*

---

## 1. Overview & Research Focus

This project develops a two-stage predictive pipeline for the *Tattersalls Autumn Horses in Training Sale* (2009–2025, 17 editions, 26,076 catalogued lots). The pipeline addresses two sequential questions:

1. **Will this horse sell to a third party?** — binary classification → P(sold\_to\_third\_party)
2. **What price will it fetch?** — regression on `log(price_gns)`, applied to all offered lots

The core challenges driving methodological choices:

- **Selection bias** — prices are only observed for lots that sell or are bought back. Training a regression on sold lots only produces biased coefficient estimates (Heckman, 1979). Vendor buybacks (whose reserve prices are revealed) are included in the regression training set to partially mitigate this.
- **Temporal drift** — 17 years of macroeconomic shift (+78% nominal prices, GBP/EUR volatility, BoE rate cycles) require strict temporal validation and a detrended regression target.
- **High-cardinality entities** — ~997 unique sires, ~840 consignors, with ~90% and ~70% rotation between early (2009–2015) and recent (2021–2025) periods. Cold-start risk at inference time.
- **Log-normal price distribution** — skewness 6.98 in raw scale, 0.03 in log scale. Regression target is always `log_price_gns`.

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
| **Final model** | Stacking ensemble (LR · RF · XGB · LGBM · CatBoost) | Single models AUC 0.585–0.611; stacking OOT AUC-ROC **0.6205** (+0.018 over best single) |

### Stage 2 — Regression: log(price\_gns)

Price regression trained on lots with observable prices, applied to the full offered universe.

| Decision | Choice | Reason |
|---|---|---|
| **Training set** | sold\_to\_third\_party + vendor\_buyback (~17,914 rows) | Both have real price data; buyback price = reserve revealed |
| **Inference set** | All offered lots (`inference_universe`, ~18,989 rows) | Counterfactual fair-value for unsold and bought-back lots |
| **Target** | `log_price_gns` | Skewness correction; RMSE in log-scale ≈ RMSLE |
| **Detrending** | `log_price_gns − log_year_median_price_prior` | Absorbs 78% nominal drift; re-added at prediction time |
| **Final model** | Stacking ensemble (Ridge · LGBM · XGBoost · CatBoost) | Ridge baseline RMSE_log 1.297 → stacking **1.142** (−11.7%); R²_log **0.250** OOT |

### Dataset Flow

```
Raw CSV (26,076 lots)
  └─ notebooks/01_Data_Preparation      →  clean_data.parquet
       └─ notebooks/02_EDA_Analysis     →  autumn_horses_modeling_ready.csv
            └─ notebooks/03_FeatureEngineering  →  classification_ready   (Stage 1)
                                               →  regression_ready       (Stage 2 train)
                                               →  inference_universe     (Stage 2 predict)
                 └─ notebooks/04_Modeling       →  stacking predictions + SHAP
                      └─ notebooks/05_Model_Audit  →  fairness · calibration · RNA · drift
```

---

## 4. Dataset & Analytical Universe

| Outcome | N | % of total | Treatment |
|---|---|---|---|
| Sold to third party | 16,531 | 63.4% | Stage 1 positive class · Stage 2 training ✅ |
| Withdrawn before ring | 7,081 | 27.2% | **Excluded** from all modelling universes |
| Vendor buyback | 1,383 | 5.3% | Stage 1 negative class · Stage 2 training ✅ (reserve price revealed) |
| Not sold on the day | 1,081 | 4.1% | Stage 1 negative class · Stage 2 inference only (price predicted) |

**On vendor buybacks in the regression training set:** the `vendor_buyback` flag is retained in `regression_ready` to enable an ablation study in `04_Modeling` (train with vs. without buybacks and compare OOT RMSE). The buyback price represents the vendor's reserve — a systematic upper bound on what the market was willing to pay — so its effect on the regression should be assessed empirically.

---

## 5. Feature Engineering Decisions

| EDA Conclusion | Implementation | Notes |
|---|---|---|
| Day 1–2 premium (stable post-Brexit) | `day`, `day_normalized`, `intraday_position`, `is_prime_time` | Temporal normalisation for 4- vs 5-day editions |
| Sire quality signal | `sire_target_enc`, `sire_global_median_gns`, `sire_career_stage`, `sire_premium_ratio` | M-estimate encoding with expanding window (anti-leakage) |
| Consignor reputation | `consignor_target_enc`, `consignor_volume`, `consignor_price_tier`, `consignor_sale_rate_enc` | Root-entity normalisation (removes Ltd, Racing, etc.) |
| Macro context | `gbp_eur_rate`, `boe_base_rate`, `year_*_prior` | LOO expanding window to prevent temporal leakage |
| Colour, foaled month — noise | Excluded from feature sets | Marginal effect <5%; captured by other variables |
| Sire-dam combo — leakage risk | Excluded from regression; novelty proxy in classification | 90%+ singletons |

**Target encoding strategy**: M-estimate with expanding temporal window — each observation's encoding uses only data from years prior to its sale year. Global mean shrinkage factor `m=10` (price) / `m=20` (dam entity, high cardinality).

**Final Model Performance (Stacking Ensemble, OOT 2022–2025)**:

| Stage | Metric | Value | Notes |
|---|---|---|---|
| Classification (Stage 1) | AUC-ROC OOT | **0.6205** | +0.018 vs. best single model (RF 0.611) |
| Classification | AUC-PR OOT | **0.9212** | Primary metric given class imbalance |
| Classification | ECE (calibration) | **0.014** | Excellent calibration |
| Classification | F1-weighted OOT @ thr=0.888 | **0.7093** | Conservative threshold; precision 0.904 |
| Regression (Stage 2) | RMSE_log OOT | **1.1417** [1.120–1.168 CI 95%] | −11.7% vs. Ridge baseline (1.297) |
| Regression | R²_log OOT | **0.2499** | ~25% price variance explained by catalogue features |
| Regression | MAE_gns OOT | ~23,859 GNS | Relevant for 75% of market (lots < 35k GNS) |

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
│   ├── features.py                     # Feature engineering utilities
│   ├── modeling.py                     # Model training & hyperparameter tuning
│   ├── evaluation.py                   # Discrimination, calibration, residuals, drift metrics
│   ├── audit.py                        # Fairness slices, disparity analysis
│   ├── sensors.py                      # Temporal split validation, leakage checks, invariants
│   ├── save_models.py                  # MLflow model persistence
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
├── requirements.txt
└── README.md
```

---

## 7. Development Setup

This project uses `uv` for reproducible environment management.

```bash
# Create and activate virtual environment
uv venv .venv
source .venv/bin/activate       # macOS / Linux
.venv\Scripts\activate          # Windows

# Install dependencies (recommended — resolves all extras correctly)
uv sync

# Alternative: pip-based install
uv pip install -r requirements.txt
```

**Key dependencies**: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, `statsmodels`,
`scikit-learn`, `lightgbm`, `optuna`, `optuna-integration[mlflow]`, `mlflow`, `shap`

**Notebook execution order**: `01` → `02` → `03` → `04` → `05`  
Each notebook reads from `data/processed/` and writes back to it.

---

## 8. Key Findings & Audit Results

**Model performance** is moderate by design — auction price is inherently hard to predict from catalogue features alone (information asymmetry, undisclosed reserve prices, buyer-day demand). An AUC of 0.62 and R² of 0.25 align with comparable Heckman-corrected benchmarks in similar markets (≤ 0.65 AUC, ≤ 0.35 R² OOT).

**SHAP importance** (Stage 1): `day`, `intraday_position`, `sire_target_enc`, `consignor_target_enc`, `year_sale_rate_prior`. (Stage 2): `sire_target_enc`, `sire_global_median_gns`, `day`, `consignor_target_enc`. Note: `day` is a partial proxy for latent lot quality (consignors place better horses on Days 1–2) — endogeneity documented in thesis.

**Temporal drift**: 80% of features show KS-test drift (p < 0.05) between 2009–2021 and 2022–2025. Model requires annual expanding-window retraining before each October sale.

**Fairness slices**: Day 5 AUC-ROC = 0.47 (near-random) — model does not distinguish what sells at the tail of the sale, consistent with higher variance in end-of-sale lot quality.

**RNA paradox**: 2,462 RNA lots (13% of offered universe). Permutation test sold vs. RNA expected_price: diff = +579 GNS, **p = 0.1212** (not significant). 482 historically high-value RNAs (expected_price > 30,190 GNS) identified — Geldings 59%, concentrated in Days 2–3.

**Ablation (vendor buybacks)**: Including buybacks in Stage 2 training adds 1,383 observations; marginal impact on global RMSE but improves encoding stability for high-cardinality sires and consignors (mean `sire_target_enc` shift = 0.034 log-units).

**Leakage audit**: PASSED — sensors validated no temporal leakage in target encoding, macro features, or train/OOT splits.

---

## Citation

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
