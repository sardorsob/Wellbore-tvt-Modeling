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
├── .agents/
│   └── AGENTS.md                      → repo-local AI workflow and commit rules
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
│   ├── EXPERIMENTS.md                 → human-readable experiment registry
│   ├── STATUS.md                      → current state + next steps
│   └── RISKS.md                       → known risks + mitigations
│
├── kaggle/                            → competition operations and submission tracking
│   ├── COMPETITION.md                 → objective, metric, deadlines, local source of truth
│   ├── RULES_SUMMARY.md               → known rules and verification checklist
│   ├── CV_STRATEGY.md                 → local validation plan and leakage guardrails
│   ├── FEATURES.md                    → test-safe and train-only feature log
│   ├── SUBMISSIONS.md                 → every Kaggle submission log entry
│   └── LEADERBOARD_NOTES.md           → public/private leaderboard interpretation notes
│   └── linear_ridge_submission/        → Kaggle-ready private notebook upload folder
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
│   │   ├── 01_phase0_phase1_data_and_schema.ipynb → download verify + schema / ID / eval-zone checks
│   │   ├── 02_phase2_geological_domain.ipynb      → TVT/GR/typewell/spatial (stub)
│   │   ├── 03_phase3_feature_signal.ipynb           → feature correlations / lags (stub)
│   │   └── 04_phase4_validation.ipynb             → CV + baselines (stub)
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
*(Phase 0–1 structural checks produced in-notebook tables only; export PNGs per `CURSOR_RULES.md` when Phase 2 plotting runs.)*

- `artifacts/figures/model__interpolation_baseline_residuals__2026-06-21__1216__interpolation_baseline__14bef53.png` — residual strip for stateless interpolation baseline on grouped masked-zone validation.
- `artifacts/figures/model__linear_rmse_comparison__2026-06-21__1218__linear_ols_ridge_lasso__14bef53.png` — RMSE bar comparison across OLS/Ridge/Lasso candidates.
- `artifacts/figures/model__linear_best_residuals__2026-06-21__1218__linear_ols_ridge_lasso__14bef53.png` — residual strip for best linear candidate (`ridge_alpha_10`).

### Tables
*(populated as notebooks are run)*

- **In-notebook only (01 Phase 0–1):** `zone_df`-style eval-zone summary and weighted missingness (`miss_summary`) — reproduce from `notebooks/eda/01_phase0_phase1_data_and_schema.ipynb`. Optional export: `artifacts/tables/eda__phase01_missingness__YYYY-MM-DD.csv` when saved from notebook.
- `artifacts/tables/model__interpolation_baseline_metrics__2026-06-21__1216__interpolation_baseline__14bef53.csv` — 3-fold grouped masked-zone metrics for interpolation baseline.
- `artifacts/tables/model__linear_ols_ridge_lasso_metrics__2026-06-21__1218__linear_ols_ridge_lasso__14bef53.csv` — candidate summary for OLS/Ridge/Lasso sweep.
- `artifacts/tables/model__linear_ols_ridge_lasso_fold_metrics__2026-06-21__1218__linear_ols_ridge_lasso__14bef53.csv` — fold-level metrics for each linear candidate.
- `artifacts/tables/model_comparison.csv` — latest reportable baseline-vs-linear comparison rows.
- `artifacts/reports/submission__linear_ridge_alpha_10__2026-06-21.csv` — locally validated Kaggle-format submission generated by the self-contained Ridge notebook.

### Models
*(populated as training runs complete)*

- `artifacts/models/linear/model__linear_ridge_alpha_10__2026-06-21__1218__linear_ols_ridge_lasso__14bef53.pkl` — ignored local model artifact for best first linear baseline.

---

## RESULTS LOG

### Workflow / Kaggle operations setup
**Status:** Added — local workflow reference and Kaggle tracking docs.
**Purpose:** Align this project with `sardorsob/sardors-agentic-workflows` while keeping copied workflow material out of Git. Establish places to track competition rules, CV strategy, features, submissions, leaderboard notes, and experiments.
**Inputs:** `context/_external/sardors-agentic-workflows/` (ignored local reference), existing project context files.
**Outputs:** `.agents/AGENTS.md`, `kaggle/*.md`, `context/EXPERIMENTS.md`, `.gitignore` entry for `context/_external/`.
**Findings:** GitHub connector could not read the private workflow repo, but local Git access could. The reference copy is kept under an ignored path and should not be committed.

---

### notebooks/eda/01_phase0_phase1_data_and_schema.ipynb
**Status:** **Run** — Phase 0–1 checks completed on downloaded Kaggle data (outputs captured below and in `context/ASSUMPTIONS.md` / `DECISIONS.md`).
**Purpose:** Verify download counts; schema spot-check; Assumption A1 ID mapping; D002 formation columns; eval-zone layout (1C–1D); missingness audit (1E).
**Inputs:** `data/raw/train/`, `data/raw/test/`, `data/raw/sample_submission.csv`

**Findings (concise):**

| Item | Result |
|---|---|
| Train wells | **773** (`discover_wells(train/)`) |
| Test wells | **3** |
| `sample_submission.csv` rows | **14,151** (= total eval-zone rows to predict) |
| Example train horizontal columns | `MD, X, Y, Z, ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA, TVT, GR, TVT_input` (+ `row_idx` added by loader). Example typewell: `TVT, GR, Geology`. |
| Phase 1A (A1) | **Pass:** every submission `row_idx` exists on the matching test horizontal-well index. |
| Phase 1B (A3 / D002) | **Pass:** no formation columns `{ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA}` in test CSV headers. |
| Phase 1C — eval zone shape | **773 / 773** wells: `nan_index_contiguous == True`, `nan_index_gaps == 0`. **100%** of wells satisfy `max(MD_known) ≤ min(MD_eval)` (trailing MD block after known zone). Per-well `eval_frac` ranges roughly **0.65–0.81** (eval-zone fraction of rows). |
| Phase 1D (A2) | **Pass:** rows with `TVT_input` NaN match submission indices exactly per test well. |
| Phase 1E — missingness (weighted mean across **all train rows**) | `TVT_input`: **~74.3%** missing overall (expected: NaN in eval zone only). `GR`: **~29.6%** all rows; **~23.9%** in known zone; **~31.6%** in eval zone. `ANCC`: **~0.90%** missing (equal in known vs eval). `EGFDL`: **~0.12%** missing. All other listed trajectory/target/formation columns in audit: **0%** missing in aggregate. |

**Outputs:** Context updates (this file, ASSUMPTIONS, DECISIONS, DATASETS, STATUS); figures not yet exported.
**Next steps:** Phase 2 notebook — TVT vs MD, GR, typewell correlation, spatial XY, dip proxies; save figures under `artifacts/figures/` per naming convention.

---

### notebooks/eda/02_phase2_geological_domain.ipynb
**Status:** Stub — Phase 2 (geology / GR / typewell / spatial / dip).
**Purpose:** TVT vs MD, GR distributions, typewell correlation, XY scatter, dTVT/dMD.
**Inputs:** `data/raw/train/`, `data/raw/test/`
**Outputs:** TBD

---

### notebooks/eda/03_phase3_feature_signal.ipynb
**Status:** Stub — Phase 3 (feature correlations, lag analysis, train/test drift).
**Purpose:** Correlate engineered features with TVT; lag-1 residual analysis.
**Inputs:** Train/processed features via `src/preprocessing/features.py`
**Outputs:** TBD

---

### notebooks/eda/04_phase4_validation.ipynb
**Status:** Stub — Phase 4 (leave-one-well-out CV + baseline benchmarks).
**Purpose:** Lock validation protocol; interpolation vs naive mean RMSE.
**Inputs:** `src/modeling/baseline.py`, configs
**Outputs:** TBD

---

### notebooks/modeling/03_baseline.ipynb
**Status:** Superseded by actual path `notebooks/modeling/00_baseline/00_interpolation_baseline.ipynb`.

---

### notebooks/modeling/00_baseline/00_interpolation_baseline.ipynb
**Status:** Run — stateless interpolation/forward-fill baseline.
**Purpose:** Establish the first naive RMSE anchor before regression models.
**Inputs:** `data/raw/train/`
**Validation:** 3 grouped well folds; score rows where `TVT_input` is missing and `TVT` is present.
**Run ID:** `2026-06-21__1216__interpolation_baseline__14bef53`
**Result:** CV RMSE mean **15.9003**, CV RMSE std **0.6071**, overall RMSE **15.9099** across **3,783,989** target-zone rows.
**Outputs:** metrics CSV and residual PNG listed above; run bundle under `artifacts/logs/runs/2026-06-21__1216__interpolation_baseline__14bef53/`.

---

### notebooks/modeling/01_linear/01_ols_ridge_lasso.ipynb
**Status:** Run — first regression baseline sweep.
**Purpose:** Compare OLS, Ridge, and Lasso using simple test-safe numeric trajectory/GR/known-context features.
**Inputs:** `data/raw/train/`
**Validation:** 3 grouped well folds; train on known-zone rows, score held-out wells' target-zone rows; cap each fold training sample at **25,000** deterministic rows.
**Run ID:** `2026-06-21__1218__linear_ols_ridge_lasso__14bef53`
**Result:** Best candidate **`ridge_alpha_10`** with CV RMSE mean **15.7660**, CV RMSE std **0.5331**, MAE **11.1123**, mean R2 **0.999365**.
**Comparison:** Improves over interpolation baseline by about **0.1343 RMSE**.
**Outputs:** candidate metrics CSV, fold metrics CSV, RMSE comparison PNG, best residual PNG, ignored local model pickle, and run bundle under `artifacts/logs/runs/2026-06-21__1218__linear_ols_ridge_lasso__14bef53/`.

---

### kaggle/linear_ridge_submission/linear_ridge_submission.ipynb
**Status:** Pushed to Kaggle as private notebook version 2, completed, and submitted to the competition. Submission `53923555` scored **15.905** public LB. Version 1 errored due to an assumed input mount path.
**Purpose:** Self-contained Kaggle submission notebook for the first `ridge_alpha_10` baseline.
**Inputs:** Competition dataset attached by `kernel-metadata.json` as `rogii-wellbore-geology-prediction`.
**Local verification:** Ran top-to-bottom locally and wrote `artifacts/reports/submission__linear_ridge_alpha_10__2026-06-21.csv` with **14,151** finite predictions matching `sample_submission.csv` id order.
**Kaggle URL:** `https://www.kaggle.com/code/sardorbsobirov/wellbore-linear-ridge-baseline`
**Leaderboard:** Submission `53923555` public score **15.905**.

---

### notebooks/modeling/02_trees/*
**Status:** Not yet run.
**Purpose:** Extra Trees, Random Forest, XGBoost, LightGBM, CatBoost with richer features and CV RMSE comparison.

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

**Phase:** Phase 0–1 **EDA structural verification complete** on local download. Geological / signal EDA (Phase 2 notebook) not started.
**Workflow posture:** Use `sardorsob/sardors-agentic-workflows` data-science gates: data gate, split/leakage gate, baseline gate, experiment loop, evaluation gate, artifact gate, handoff gate. Local reference copy lives under ignored `context/_external/`.
**Next action:** Run `notebooks/eda/02_phase2_geological_domain.ipynb`; export plots to `artifacts/figures/` and refresh this log.
