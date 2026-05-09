# Repository Structure — Single Source of Truth

All referenced paths are relative to repo root.
Update this file after every notebook run or meaningful code change.

---

## PROJECT MAP

```
Wellbore-tvt-Modeling/
│
├── README.md                          → pointer to context/README.md
├── CURSOR_RULES.md                    → AI agent workflow rules
├── requirements.txt                   → full dependency set
├── .gitignore
│
├── configs/                           → yaml experiment configs
│   ├── baseline.yaml                  → interpolation baseline
│   ├── xgboost.yaml                   → XGBoost regression
│   ├── lgbm.yaml                      → LightGBM regression
│   ├── bayesian_gp.yaml               → NumpPyro Gaussian Process
│   ├── bnn.yaml                       → Bayesian Neural Network (Flax+NumpPyro)
│   └── ensemble.yaml                  → Weighted ensemble
│
├── context/                           → all project documentation
│   ├── README.md                      → how to install and run
│   ├── structure.md                   → THIS FILE — repo map + artifact index
│   ├── development_rules.md           → pointer to rules
│   ├── PROJECT_BRIEF.md               → competition objective + success criteria
│   ├── GLOSSARY.md                    → domain terms and acronyms
│   ├── DATASETS.md                    → data sources, schemas, governance
│   ├── INTERFACES.md                  → function + script I/O contracts
│   ├── ASSUMPTIONS.md                 → modeling assumptions + caveats
│   ├── DECISIONS.md                   → architecture + design decisions
│   ├── CHANGELOG.md                   → milestone log
│   ├── STATUS.md                      → current state + next steps
│   └── RISKS.md                       → known risks + mitigations
│
├── data/
│   ├── raw/                           → immutable Kaggle download (gitignored)
│   │   ├── train/
│   │   ├── test/
│   │   └── sample_submission.csv
│   ├── interim/                       → cached transforms (gitignored)
│   └── processed/                     → final feature matrices (gitignored)
│
├── notebooks/
│   ├── eda/
│   │   ├── 01_data_exploration.ipynb  → load data, schema checks, distributions
│   │   └── 02_feature_engineering.ipynb → feature analysis, typewell correlation
│   ├── modeling/
│   │   ├── 03_baseline.ipynb          → interpolation baseline + CV
│   │   ├── 04_classical_ml.ipynb      → XGBoost / LightGBM
│   │   ├── 05_bayesian_gp.ipynb       → NumpPyro GP
│   │   ├── 06_bnn.ipynb               → Bayesian Neural Network
│   │   └── 07_ensemble.ipynb          → model comparison + ensemble
│   └── reporting/
│       └── 08_results_summary.ipynb   → leaderboard history + final analysis
│
├── src/                               → importable Python package
│   ├── io/
│   │   └── loaders.py                 → well data loaders, discovery
│   ├── preprocessing/
│   │   ├── features.py                → feature engineering pipeline
│   │   └── transforms.py             → scaling, imputation utilities
│   ├── modeling/
│   │   ├── baseline.py                → interpolation baseline
│   │   ├── classical.py               → XGBoost / LightGBM wrappers
│   │   ├── bayesian_gp.py             → NumpPyro GP model
│   │   ├── bnn.py                     → Flax + NumpPyro BNN
│   │   └── ensemble.py                → ensemble combination
│   ├── evaluation/
│   │   ├── metrics.py                 → RMSE + other metrics
│   │   └── validation.py              → cross-validation (leave-one-well-out)
│   ├── viz/
│   │   └── plots.py                   → well path, TVT, GR visualization
│   └── utils/
│       ├── paths.py                   → Kaggle/local path detection
│       ├── seeds.py                   → RNG seed management
│       ├── logging_utils.py           → structured logging helpers
│       └── run_bundle.py              → run metadata writer
│
├── scripts/
│   ├── download_data.py               → Kaggle API download
│   ├── train.py                       → model training CLI
│   ├── predict.py                     → generate submission CSV
│   └── eval.py                        → cross-validation evaluation
│
├── tests/
│   ├── test_loaders.py                → data loader unit tests
│   ├── test_features.py               → feature engineering unit tests
│   ├── test_metrics.py                → metric function unit tests
│   └── test_smoke.py                  → end-to-end smoke test
│
└── artifacts/
    ├── figures/                       → exported plots (never hand-edited)
    ├── tables/                        → exported tables (CSV/HTML)
    ├── models/                        → serialized model files
    ├── reports/                       → auto-generated reports
    └── logs/
        ├── runs/                      → per-run metadata bundles
        ├── prompts/                   → LLM prompt/response snapshots
        └── provenance/                → file hashes/manifests
```

---

## ARTIFACT INDEX

### Figures
*(populated as notebooks are run)*

### Tables
*(populated as notebooks are run)*

### Models
*(populated as training runs complete)*

---

## RESULTS LOG

### notebooks/eda/01_data_exploration.ipynb
**Status:** Not yet run — stub only.
**Purpose:** Load all train/test wells; inspect schemas; plot GR distributions; verify id mapping.
**Inputs:** `data/raw/train/`, `data/raw/test/`, `data/raw/sample_submission.csv`
**Outputs:** TBD
**Key findings:** TBD
**Next steps:** Complete EDA then move to feature engineering.

---

### notebooks/eda/02_feature_engineering.ipynb
**Status:** Not yet run — stub only.
**Purpose:** Build and analyze feature set; typewell GR correlation; lag/rolling features.
**Inputs:** `data/raw/train/`, `data/raw/test/`
**Outputs:** TBD
**Next steps:** Feed validated features into modeling notebooks.

---

### notebooks/modeling/03_baseline.ipynb
**Status:** Not yet run — stub only.
**Purpose:** Interpolation / forward-fill baseline; measure RMSE on masked evaluation zone.

---

### notebooks/modeling/04_classical_ml.ipynb
**Status:** Not yet run — stub only.
**Purpose:** XGBoost and LightGBM with full feature set; CV RMSE comparison.

---

### notebooks/modeling/05_bayesian_gp.ipynb
**Status:** Not yet run — stub only.
**Purpose:** NumpPyro Gaussian Process; posterior mean as prediction; uncertainty visualization.

---

### notebooks/modeling/06_bnn.ipynb
**Status:** Not yet run — stub only.
**Purpose:** Flax + NumpPyro Bayesian Neural Network; SVI inference.

---

### notebooks/modeling/07_ensemble.ipynb
**Status:** Not yet run — stub only.
**Purpose:** Compare all models; build weighted ensemble; select final submission.

---

### notebooks/reporting/08_results_summary.ipynb
**Status:** Not yet run — stub only.
**Purpose:** Leaderboard history, model comparison table, final submission analysis.

---

## CURRENT STATUS

**Phase:** Repository setup complete. Awaiting data download and EDA.
**Next action:** `python scripts/download_data.py` then run `notebooks/eda/01_data_exploration.ipynb`.
