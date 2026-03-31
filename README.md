# Predictive Modeling for Horse Auction Prices
**Master's Thesis (TFM) — Master in Big Data Science & Artificial Intelligence**  
*Universidad de Navarra · idealista*

> **Status**: Exploratory Data Analysis (EDA) ✅ completed · Predictive modeling 🔄 in progress  
> **Note**: This repository contains only the code and reproducible analysis. The final thesis document is submitted separately to the university.

---

## 1. Overview & Research Focus

This project develops predictive models for horse auction pricing, using the *Tattersalls Autumn Horses in Training Sale* (2009–2025, 17 editions) as the primary dataset. The goal is to identify which factors most influence sale prices and to build interpretable models that can assist stakeholders in valuation decisions.

The thesis tackles critical challenges in auction forecasting:

- **Severe target skewness** — the price distribution is log-normal (skewness 6.98), driven by a small number of elite lots (the "superstar effect" in bloodstock).
- **Temporal structure** — 17 years of macroeconomic drift, stallion career cycles, and shifting consignor dominance require temporal validation, not random k-fold.
- **High-cardinality categorical variables** — ~997 unique sires and ~840 consignors, with ~90% and ~70% rotation between the 2009–2015 and 2021–2025 periods respectively.
- **Endogenous features** — catalogue day (`day`) is a powerful predictor but partially encodes latent quality chosen by consignors, not a causal lever.

---

## 2. Key EDA Findings

| Finding | Result |
|---|---|
| **Price distribution** | Log-normal (skewness 6.98 raw, 0.03 in log scale) — regression target: `log_price_gns` |
| **Strongest signal** | Catalogue day: Days 2–3 median 17,000 gns vs Days 4–5 4,000 gns (3.4×, permutation p<0.0001) |
| **Sex effect** | Colts 17,000 > Geldings 13,000 > Fillies 7,000 gns (permutation diff 0.887 log-units, p<0.0001) |
| **Intra-day structure** | "Prime time" effect: median price peaks at lot positions 0.6–0.8 within the day; clearance rate remains flat (~80–90%), suggesting a quality-selection mechanism rather than buyer fatigue |
| **Nominal vs real growth** | +60% nominal price growth (2009–2025) but only ~+2% real — 58 percentage points explained by inflation (ONS CPIH01, deflated to constant GBP) |
| **Entity rotation** | Top sires rotated ~90% between periods (Acclamation/Oasis Dream → Kodiac/Dark Angel); top consignors ~70% — cold-start risk at evaluation time |

---

## 3. Methodology & Rigor

- **Problem framing**: Two-stage pipeline — (1) classify `sold_to_third_party`, (2) regress `log_price_gns` on sold lots only. Unsold lots have no price; conflating them inflates noise.
- **Validation**: Strict temporal split — train on 2009–2021, evaluate on 2022–2025. Random k-fold would be optimistic by ~0.5 log-units due to market drift.
- **Evaluation metric**: RMSE on log-scale (equivalent to RMSLE on the original scale). Symmetric penalty for over- and underestimation.
- **High-cardinality encoding**: Target encoding with leave-one-out regularization (`category_encoders.TargetEncoder`), fitted strictly on the training split before the temporal cutoff to prevent temporal leakage.
- **Model explainability**: SHAP values to decompose feature contributions. `day` will be flagged explicitly as partially encoding latent quality, not direct causality.
- **Anti-leakage**: `purchaser`, `sale_outcome`, and `price_euros` excluded from the feature matrix (post-outcome variables not available at prediction time).

---

## 4. Dataset & Analytical Universe

The raw dataset covers **26,076 catalogued horses** across 17 editions. Four distinct sale outcomes are defined and treated separately:

| Outcome | N | % | Treatment |
|---|---|---|---|
| Sold to third party | 16,531 | 63.4% | Primary regression target |
| Withdrawn (before ring) | 7,081 | 27.2% | **Excluded** from analytical universe (`df_offered`) |
| Vendor buyback | 1,383 | 5.3% | **Included** as sold — real transacted price exists; exclusion would bias the lower tail downward |
| Unsold on the day | 1,081 | 4.1% | Included in classification target, excluded from regression |

> **Note on vendor buybacks**: rows with `price_gns` populated but `sold_to_third_party == False` are intentional — these are vendor buybacks treated as sales. This is a deliberate methodological decision, not a data quality issue.

---

## 5. Repository Structure

```text
.
├── 01_EDA.ipynb                        # EDA, feature engineering & conclusions
├── 02_modeling.ipynb                   # Predictive modeling pipeline (in progress)
├── data/                               # Ignored by git
│   ├── raw/
│   │   ├── Autumn Horses In Training Sale 2009-2025.csv
│   │   └── STALLIONS_EUR.csv
│   └── processed/
│       ├── autumn_horses_modeling_ready.csv
│       ├── autumn_horses_feature_roles.csv
│       └── top15_sires_enriched.csv
├── 01_EDA_files/                       # Exported plots from EDA notebook
├── requirements.txt
└── thesis_memory/                      # Final document drafts (ignored by git)
```

---

## 6. Development Setup

This project uses `uv` for reproducible and fast environment management.

```bash
# 1. Create virtual environment
uv venv .venv

# 2. Activate
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

# 3. Install dependencies
uv pip install -r requirements.txt
```

**Key dependencies**: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, `statsmodels`, `scikit-learn`, `category_encoders`, `shap`, `catboost`, `onspy`

---

## 7. Modeling Strategy (Planned)

| Decision | Strategy | Reason |
|---|---|---|
| **Problem framing** | Two-stage: classify → regress | Unsold lots have no price |
| **Price target** | `log_price_gns` | Skewness 6.98 raw vs 0.03 in log |
| **Validation** | Temporal split 2009–2021 / 2022–2025 | Market drift makes k-fold optimistic |
| **Strongest features** | `day`, `sex`, `sale_year`, `lot_norm` | Largest permutation-test gaps; no leakage |
| **High-cardinality entities** | Target encoding + LOO regularization | 997/840 levels; ~90%/70% rotation between periods |
| **Inflation** | `price_real_gns` for temporal analysis; `price_gns` for cross-sectional models | Nominal +60%, real +2% over 2009–2025 |
| **Leaky variables** | Exclude `purchaser`, `sale_outcome`, `price_euros` | Not available at prediction time |

---

*Thesis supervised by Stella Salvatierra (Universidad de Navarra) and Daniel del Pozo Salinas (idealista).*