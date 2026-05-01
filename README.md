# Predictive Modelling for Horse Auction Prices
**Master's Thesis (TFM) — Master in Big Data Science & Artificial Intelligence**  
*Universidad de Navarra · idealista*

> **Status**: Data Preparation ✅ · EDA ✅ · Feature Engineering ✅ · Modelling 🔄 in progress  
> This repository contains code and reproducible analysis only. The final thesis document is submitted separately.

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
| **Primary model** | HistGradientBoostingClassifier | Handles missing values natively; strong OOT baseline (AUC-ROC 0.617) |

### Stage 2 — Regression: log(price\_gns)

Price regression trained on lots with observable prices, applied to the full offered universe.

| Decision | Choice | Reason |
|---|---|---|
| **Training set** | sold\_to\_third\_party + vendor\_buyback (~17,914 rows) | Both have real price data; buyback price = reserve revealed |
| **Inference set** | All offered lots (`inference_universe`, ~18,989 rows) | Counterfactual fair-value for unsold and bought-back lots |
| **Target** | `log_price_gns` | Skewness correction; RMSE in log-scale ≈ RMSLE |
| **Detrending** | `log_price_gns − log_year_median_price_prior` | Absorbs 78% nominal drift; re-added at prediction time |
| **Primary model** | LightGBM / XGBoost | Gradient boosting for tabular data with temporal features |

### Dataset Flow

```
Raw CSV (26,076 lots)
  └─ 01_Data_Preparation  →  clean_data.parquet
       └─ 02_EDA_Analysis  →  autumn_horses_modeling_ready.csv
            └─ 03_FeatureEngineering  →  classification_ready   (Stage 1 train/eval)
                                      →  regression_ready       (Stage 2 train/eval)
                                      →  inference_universe     (Stage 2 predict all)
                                           └─ 04_Modeling  →  predictions + SHAP
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

**Feature Selection Baseline**:

| Stage | Metric | Value | Threshold | Status |
|---|---|---|---|---|
| Classification (Stage 1) | AUC-ROC OOT | 0.617 | > 0.60 | ✅ |
| Regression (Stage 2) | CV RMSE train | 1.085 | < 1.10 | ✅ |
| Regression | OOT RMSE | 1.132 | — | ⚠️ drift |
| Regression | OOT R² | 0.279 | — | ⚠️ drift |

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

## 8. Modeling Roadmap

- **`04_Modeling.ipynb`**: HistGradientBoostingClassifier (Stage 1) → probability calibration (Platt / Isotonic) → threshold optimisation by F1-weighted; LightGBM / XGBoost (Stage 2) → temporal detrending → OOT evaluation; ablation with vs. without vendor buybacks; SHAP values for both stages
- **`05_Model_Audit.ipynb`**: Fairness slices by sex, consignor tier, and sale day; calibration curves; SHAP global/local explainability; RNA (Regression Naming Artefact) paradox analysis; temporal gap assessment between training and deployment window

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
