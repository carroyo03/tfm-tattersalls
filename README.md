# Predictive Modelling for Horse Auction Prices
**Master's Thesis (TFM) — Master in Big Data Science & Artificial Intelligence**  
*Universidad de Navarra · idealista*

---

## 1. Overview & Research Focus

This project develops a two-stage predictive pipeline for the *Tattersalls Autumn Horses in Training Sale* (2009–2025, 17 editions, 26,076 catalogued lots). The pipeline addresses two sequential questions:

1. **Will this horse sell to a third party?** — binary classification → P(sold\_to\_third\_party)
2. **What price will it fetch?** — regression on `log(price_gns)`, applied to all offered lots

The core challenges driving methodological choices:

- **Selection bias** — prices are only cleanly observed for lots sold to third parties. Vendor buybacks/RNAs indicate that the reserve was not met, so they are analysed as non-transactions rather than treated as realised market prices in the final regression target.
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
| **Final model selected in modeling** | **Stacking ensemble (RF · XGB · LGBM · CatBoost) with LR meta-learner** | PR-AUC tied (0.938) with Random Forest; stacking wins ROC-AUC (0.6521 vs 0.6461) and Brier (0.0894 vs 0.1796, **2× better calibration**). Selected by Brier tiebreaker (difference <= 0.001). No separate calibration needed |

### Stage 2 — Regression: log(price\_gns)

Price regression trained on lots with observable prices, applied to the full offered universe.

| Decision | Choice | Reason |
|---|---|---|
| **Training set** | sold\_to\_third\_party only (~16.5k rows) | Realised third-party market price; excludes reserve-not-met/buyback observations |
| **Inference set** | All offered lots (`inference_universe`) | Counterfactual fair-value for sold, buyback, and not-sold lots |
| **Target** | `log_price_gns` | Skewness correction; RMSE in log-scale ≈ RMSLE |
| **Detrending** | `log_price_gns − log_year_median_price_prior` | Absorbs 78% nominal drift; re-added at prediction time |
| **Final model selected in modeling** | **Stacking ensemble (RF · XGB · LGBM · CatBoost) with Ridge meta-learner** | Wins validation RMSE (1.146 vs RF 1.158, **−1%**). Margin is modest; stacking chosen for robustness and methodological coherence. Ridge baseline is 1.297, but the relevant comparison is against RF (best individual) |

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
| Vendor buyback | 1,383 | 5.3% | Stage 1 negative class · Stage 2 inference/audit only (reserve not met) |
| Not sold on the day | 1,081 | 4.1% | Stage 1 negative class · Stage 2 inference only (price predicted) |

**On vendor buybacks in the regression target:** the final specification treats buybacks/RNAs as non-transactions. They are retained in the broader inference/audit universe, but excluded from Stage 2 price training because a reserve-not-met amount is not equivalent to a realised third-party hammer price.

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

**Model Performance Snapshot from Modeling Run** (test OOT 2022–2025):

| Stage | Metric | Value | Notes |
|---|---|---|---|
| Classification (Stage 1) | AUC-ROC OOT | **0.6329** | Final model (stacking), Brier 0.1034 (natively calibrated) |
| Classification | AUC-PR OOT | **0.9254** | Primary metric given class imbalance |
| Classification | F1 @ Youden thr=0.893 | **0.7567** | Selected by Youden’s J on validation |
| Regression (Stage 2) | RMSE_log OOT | **1.1540** | –14.5% vs Ridge (hedónico baseline): RMSE_raw 70,658 → 60,413 GNS |
| Regression | R²_log OOT | **0.2473** | ~25% price variance explained by catalogue features |
| Regression | MAPE / MdAPE OOT | 219% / 68.6% | High MAPE driven by low-price tail (<2k GNS); MdAPE=68.6% is representative of the median lot |

**Raw-scale benchmark vs hedónico OLS (Ridge)**:

| Metric | Ridge (hedónico) | Stacking ensemble | Mejora |
|---|---|---|---|
| R² raw GNS | –0.2266 | **0.1033** | +145.6% |
| RMSE raw GNS | 70,658 | **60,413** | –14.5% |
| MAE raw GNS | 31,660 | **25,466** | –19.6% |
| MAE / mediana | 211.1% | 169.8% | –19.6% |

R² raw GNS es negativo para Ridge porque la transformación exp(log) amplifica errores en la cola superior.
La métrica relevante es R²_log (0.2473). Aun así, stacking supera al modelo hedónico en todas las
métricas en escala real.

**Error por decil de precio** (test OOT, stacking):

| Decil | Rango (GNS) | RMSE | MAPE |
|---|---|---|---|
| 0 (baratos) | 1k – 2k | 17,383 | 1,054% |
| 1 | 2.5k – 5k | 17,773 | 370% |
| 2 | 5.5k – 7k | 17,705 | 196% |
| 3 | 7.5k – 10k | 17,869 | 146% |
| 4 | 10.5k – 15k | 16,683 | 93% |
| 5 | 16k – 21k | 17,327 | 73% |
| 6 | 22k – 29k | 15,118 | 49% |
| 7 | 30k – 42k | 15,738 | 37% |
| 8 | 43k – 75k | 26,159 | 38% |
| 9 (caros) | 78k – 1.3M | 187,675 | 70% |

El MAPE global (219%) está dominado por el decil más bajo (<2k GNS). En el segmento 22k–75k GNS
donde opera ~60% del mercado, el MAPE es 37–49%. El MdAPE (68.6%) describe mejor el error típico.

**SHAP Interpretability via Surrogate LGBM**: SHAP values are computed on LightGBM models trained
imitating the stacking ensembles. Surrogate fidelity R²=0.9966 (CLF) / 0.9995 (REG) on OOT.
This is standard practice for explaining stacked ensembles (Hasnat et al., 2025; Choudhary et al., 2025).

**Fundamental limit of catalogue-only data**: An R²_log of 0.25 is what can be explained from
catalogue features alone (pedigree, consignor reputation, catalogue position, macro context).
The remaining 75% of price variance is driven by unobservable factors: physical conformation,
biomechanics, veterinary inspection findings, temperament, and buyer–day demand dynamics.
These features require video analysis and computer vision, which is the natural next step
beyond this work. See An et al. (2026) for evidence that clinical and biomechanical variables
significantly improve equine prediction models.

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
│   ├── model_wrappers.py               # Persist exact stacked final models
│   ├── final_audit.py                  # Reproducibility gate for final artefacts
│   ├── save_models.py                  # Legacy model persistence helper
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

**Model performance** is moderate by design — auction price is inherently hard to predict from catalogue features alone (information asymmetry, undisclosed reserve prices, buyer-day demand). An AUC of 0.63 and R²_log of 0.25 align with comparable Heckman-corrected benchmarks in similar markets (≤ 0.65 AUC, ≤ 0.35 R² OOT). The stacking ensemble improves over the hedonic OLS baseline (Ridge) by +145% R² raw, –14.5% RMSE raw, and –19.6% MAE raw.

The remaining ~75% of price variance cannot be recovered from catalogue data alone. Features like physical conformation, biomechanics (length of stride, muscle mass, joint angles), veterinary inspection findings, and temperament require video and computer vision analysis — a natural extension beyond this work (cf. An et al. 2026, who show that biomechanical variables significantly improve equine performance prediction).

**SHAP importance** (via LGBM surrogate trained to imitate the stacking ensemble): (Stage 1) `day`, `intraday_position`, `sire_target_enc`, `consignor_target_enc`, `year_sale_rate_prior`. (Stage 2): `sire_target_enc`, `sire_global_median_gns`, `day`, `consignor_target_enc`. Note: `day` is a partial proxy for latent lot quality (consignors place better horses on Days 1–2) — endogeneity documented in thesis. Surrogate fidelity is verified (R² > 0.95 on OOT) as standard practice in the literature (Hasnat et al., 2025; Choudhary et al., 2025).

**Temporal drift**: 80% of features show KS-test drift (p < 0.05) between 2009–2021 and 2022–2025. Model requires annual expanding-window retraining before each October sale.

**Fairness slices**: Day 5 AUC-ROC = 0.47 (near-random) — model does not distinguish what sells at the tail of the sale, consistent with higher variance in end-of-sale lot quality.

**RNA paradox**: 2,462 RNA lots (13% of offered universe). Permutation test sold vs. RNA expected_price: diff = +579 GNS, **p = 0.1212** (not significant). 482 historically high-value RNAs (expected_price > 30,190 GNS) identified — Geldings 59%, concentrated in Days 2–3.

**Ablation (vendor buybacks)**: Buybacks are kept as a sensitivity/audit topic because they expose reserve behaviour and affect high-cardinality encodings, but the final price model uses sold-to-third-party lots only.

**Leakage audit**: PASSED — sensors validated no temporal leakage in target encoding, macro features, or train/OOT splits.

---

## Citation

## Regenerar figuras SHAP (producción)

Las figuras de interpretabilidad (SHAP) se calculan sobre **surrogate LGBM** entrenado para imitar
el stacking ensemble. Esto es necesario porque SHAP TreeExplainer requiere un único árbol, no
un ensemble de ensembles. La fidelidad del surrogate se verifica antes de computar SHAP.

Abre el notebook `notebooks/05_Model_Audit.ipynb` y ejecuta la sección §9 (SHAP Interpretability).
Las figuras se guardan automáticamente en `outputs/figures/audit/`.

El script `scripts/generate_shap_production.py` ha sido eliminado; el notebook 05 es la fuente
canónica para la generación de SHAP, con verificaciones de fidelidad del surrogate inline.


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
