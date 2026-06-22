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
│   ├── linear_ridge_submission/        → Kaggle-ready private notebook upload folder
│   ├── random_forest_submission/       → Kaggle-ready private notebook upload folder
│   └── elasticnet_submission/          → Kaggle-ready private notebook upload folder
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
│   │   ├── 02_phase2_geological_domain.ipynb      → TVT/GR/typewell/spatial (run)
│   │   ├── 03_phase3_feature_signal.ipynb           → feature signal / probes (run)
│   │   └── 04_phase4_validation.ipynb             → validation diagnostics (run)
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
- `artifacts/figures/model__elasticnet_huber_quantile_rmse_comparison__2026-06-21__1412__linear_elasticnet_huber_quantile__448574b.png` — RMSE bar comparison across ElasticNet, Huber, and median Quantile candidates.
- `artifacts/figures/model__elasticnet_huber_quantile_residuals__2026-06-21__1412__linear_elasticnet_huber_quantile__448574b.png` — residual strip for best robust/regularized linear candidate.
- `artifacts/figures/model__random_forest_residuals__2026-06-21__1310__tree_random_forest__4eb4a55.png` — residual strip for the first Random Forest baseline.
- `artifacts/figures/model__random_forest_feature_importance__2026-06-21__1310__tree_random_forest__4eb4a55.png` — feature-importance bar chart for the first Random Forest baseline.
- `artifacts/figures/model__extra_trees_residuals__2026-06-21__1312__tree_extra_trees__4eb4a55.png` — residual strip for the first Extra Trees baseline.
- `artifacts/figures/model__extra_trees_feature_importance__2026-06-21__1312__tree_extra_trees__4eb4a55.png` — feature-importance bar chart for the first Extra Trees baseline.
- `artifacts/figures/model__knn_regression_rmse_comparison__2026-06-21__1348__neighbor_knn_regression__31e8e1a.png` — sampled CV RMSE comparison across KNN candidates.
- `artifacts/figures/model__knn_regression_residuals__2026-06-21__1348__neighbor_knn_regression__31e8e1a.png` — sampled residual strip for the best KNN candidate.
- `artifacts/figures/eda__phase2_eval_fraction__2026-06-21.png` - train/test evaluation-zone fraction distribution.
- `artifacts/figures/eda__phase2_gr_distribution__2026-06-21.png` - GR distribution and missingness comparison.
- `artifacts/figures/eda__phase2_tvt_md_samples__2026-06-21.png` - sample TVT versus MD profiles.
- `artifacts/figures/eda__phase2_boundary_extrapolation__2026-06-21.png` - boundary extrapolation diagnostics.
- `artifacts/figures/eda__phase2_typewell_alignment__2026-06-21.png` - horizontal/typewell GR alignment diagnostics.
- `artifacts/figures/eda__phase2_xy_footprint__2026-06-21.png` - XY footprint overview.
- `artifacts/figures/eda__phase2_test_similarity__2026-06-21.png` - local visible test similarity to train descriptors.
- `artifacts/figures/eda__phase3_feature_signal_rank__2026-06-21.png` - top feature-signal ranks by target-zone delta association.
- `artifacts/figures/eda__phase3_group_probe_rmse__2026-06-21.png` - diagnostic linear-probe RMSE by candidate feature family.
- `artifacts/figures/eda__phase3_boundary_delta_curve__2026-06-21.png` - binned target-zone TVT delta versus boundary distance.
- `artifacts/figures/eda__phase3_gr_missing_signal__2026-06-21.png` - target-zone delta by GR availability.
- `artifacts/figures/eda__phase3_typewell_residual_signal__2026-06-21.png` - typewell residual versus target-zone delta.
- `artifacts/figures/eda__phase4_model_fold_rmse__2026-06-21.png` - historical local CV fold RMSE by model.
- `artifacts/figures/eda__phase4_fold_balance__2026-06-21.png` - scaled fold-balance diagnostics.
- `artifacts/figures/eda__phase4_cv_vs_leaderboard__2026-06-21.png` - local CV versus public leaderboard RMSE.
- `artifacts/figures/eda__phase4_endpoint_vs_slope__2026-06-21.png` - endpoint-hold versus direct slope-projection well RMSE.
- `artifacts/figures/eda__phase4_descriptor_pca__2026-06-21.png` - current fold coverage in descriptor PCA space.
- `artifacts/figures/model__gam_spline_rmse_comparison__2026-06-21.png` - RMSE comparison across GAM-style spline variants and baselines.
- `artifacts/figures/model__gam_spline_fold_rmse__2026-06-21.png` - fold RMSE for selected GAM-style variants and baselines.
- `artifacts/figures/model__gam_spline_best_residuals__2026-06-21.png` - residual scatter for the best GAM-style variant.
- `artifacts/figures/model__gam_spline_residual_by_boundary__2026-06-21.png` - residuals versus boundary distance for a GAM-style variant.

### Tables
*(populated as notebooks are run)*

- **In-notebook only (01 Phase 0–1):** `zone_df`-style eval-zone summary and weighted missingness (`miss_summary`) — reproduce from `notebooks/eda/01_phase0_phase1_data_and_schema.ipynb`. Optional export: `artifacts/tables/eda__phase01_missingness__YYYY-MM-DD.csv` when saved from notebook.
- `artifacts/tables/model__interpolation_baseline_metrics__2026-06-21__1216__interpolation_baseline__14bef53.csv` — 3-fold grouped masked-zone metrics for interpolation baseline.
- `artifacts/tables/model__linear_ols_ridge_lasso_metrics__2026-06-21__1218__linear_ols_ridge_lasso__14bef53.csv` — candidate summary for OLS/Ridge/Lasso sweep.
- `artifacts/tables/model__linear_ols_ridge_lasso_fold_metrics__2026-06-21__1218__linear_ols_ridge_lasso__14bef53.csv` — fold-level metrics for each linear candidate.
- `artifacts/tables/model__elasticnet_huber_quantile_summary__2026-06-21__1412__linear_elasticnet_huber_quantile__448574b.csv` — candidate summary for ElasticNet, Huber, and median Quantile sweep.
- `artifacts/tables/model__elasticnet_huber_quantile_fold_metrics__2026-06-21__1412__linear_elasticnet_huber_quantile__448574b.csv` — fold-level metrics for each robust/regularized linear candidate.
- `artifacts/tables/model__random_forest_metrics__2026-06-21__1310__tree_random_forest__4eb4a55.csv` — fold-level metrics for the Random Forest baseline.
- `artifacts/tables/model__extra_trees_metrics__2026-06-21__1312__tree_extra_trees__4eb4a55.csv` — fold-level metrics for the Extra Trees baseline.
- `artifacts/tables/model__knn_regression_metrics__2026-06-21__1348__neighbor_knn_regression__31e8e1a.csv` — fold-level sampled validation metrics for KNN candidates.
- `artifacts/tables/model__knn_regression_summary__2026-06-21__1348__neighbor_knn_regression__31e8e1a.csv` — candidate summary for the first KNN sweep.
- `artifacts/tables/eda__phase2_train_well_summary__2026-06-21.csv` - per-train-well geometry, missingness, TVT, and trajectory summary.
- `artifacts/tables/eda__phase2_test_well_summary__2026-06-21.csv` - per-visible-test-well geometry, missingness, and trajectory summary.
- `artifacts/tables/eda__phase2_typewell_summary__2026-06-21.csv` - typewell GR and geology-label summary.
- `artifacts/tables/eda__phase2_formation_summary__2026-06-21.csv` - train-only formation depth summary for generative context.
- `artifacts/tables/eda__phase2_typewell_alignment__2026-06-21.csv` - horizontal/typewell GR alignment summary.
- `artifacts/tables/eda__phase2_eval_zone_tvt_summary__2026-06-21.csv` - train evaluation-zone TVT summary.
- `artifacts/tables/eda__phase2_test_train_similarity__2026-06-21.csv` - visible test/train descriptor nearest-neighbor check.
- `artifacts/tables/eda__phase2_findings__2026-06-21.csv` - concise findings table used by the Phase 2 report.
- `artifacts/tables/eda__phase3_candidate_feature_sample_preview__2026-06-21.csv` - bounded per-well preview of Phase 3 candidate features.
- `artifacts/tables/eda__phase3_candidate_feature_signal__2026-06-21.csv` - candidate feature correlations and mutual-information scores.
- `artifacts/tables/eda__phase3_feature_group_probe_metrics__2026-06-21.csv` - fold-level diagnostic linear-probe metrics by feature group.
- `artifacts/tables/eda__phase3_feature_group_probe_summary__2026-06-21.csv` - feature-group probe summary table.
- `artifacts/tables/eda__phase3_visible_test_drift__2026-06-21.csv` - visible test/train descriptor drift diagnostics.
- `artifacts/tables/eda__phase3_gr_missing_signal__2026-06-21.csv` - target-zone TVT delta by GR missingness.
- `artifacts/tables/eda__phase3_findings__2026-06-21.csv` - concise findings table used by the Phase 3 report.
- `artifacts/tables/eda__phase4_endpoint_baselines_by_well__2026-06-21.csv` - endpoint-hold and direct slope-projection baselines by well.
- `artifacts/tables/eda__phase4_fold_assignments__2026-06-21.csv` - current sorted and descriptor-balanced fold assignments.
- `artifacts/tables/eda__phase4_fold_descriptor_summary__2026-06-21.csv` - fold-level descriptor and baseline summary.
- `artifacts/tables/eda__phase4_fold_variability__2026-06-21.csv` - fold-balance variability summary.
- `artifacts/tables/eda__phase4_model_fold_metrics__2026-06-21.csv` - prior model fold metrics in a common format.
- `artifacts/tables/eda__phase4_cv_leaderboard_gap__2026-06-21.csv` - local CV versus public leaderboard gap table.
- `artifacts/tables/eda__phase4_well_descriptor_pca__2026-06-21.csv` - descriptor PCA coordinates by well.
- `artifacts/tables/eda__phase4_findings__2026-06-21.csv` - concise findings table used by the Phase 4 report.
- `artifacts/tables/model__gam_spline_fold_metrics__2026-06-21.csv` - fold-level metrics for the first GAM-style additive spline sweep.
- `artifacts/tables/model__gam_spline_summary__2026-06-21.csv` - candidate summary for the first GAM-style additive spline sweep.
- `artifacts/tables/model__gam_spline_variants__2026-06-21.csv` - feature-family definition table for GAM-style variants.
- `artifacts/tables/model__gam_spline_residual_sample__2026-06-21.csv` - bounded residual sample for selected GAM-style variants.
- `artifacts/tables/model__gam_spline_findings__2026-06-21.csv` - concise findings table used by the GAM-style report.
- `artifacts/tables/model_comparison.csv` — latest reportable baseline, linear, tree, neighbor, and GAM-style comparison rows.
- `artifacts/reports/submission__linear_ridge_alpha_10__2026-06-21.csv` — locally validated Kaggle-format submission generated by the self-contained Ridge notebook.

### Models
*(populated as training runs complete)*

- `artifacts/models/linear/model__linear_ridge_alpha_10__2026-06-21__1218__linear_ols_ridge_lasso__14bef53.pkl` — ignored local model artifact for best first linear baseline.
- `artifacts/models/linear/model__elasticnet_huber_quantile__2026-06-21__1412__linear_elasticnet_huber_quantile__448574b.pkl` — ignored local model artifact for best ElasticNet/Huber/Quantile sweep candidate.
- `artifacts/models/trees/model__random_forest__2026-06-21__1310__tree_random_forest__4eb4a55.pkl` — ignored local model artifact for first Random Forest baseline.
- `artifacts/models/trees/model__extra_trees__2026-06-21__1312__tree_extra_trees__4eb4a55.pkl` — ignored local model artifact for first Extra Trees baseline.
- `artifacts/models/neighbors/model__knn_regression__2026-06-21__1348__neighbor_knn_regression__31e8e1a.pkl` — ignored local model artifact for first KNN baseline.

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

**Outputs:** Context updates (this file, ASSUMPTIONS, DECISIONS, DATASETS, STATUS); Phase 2 now exports figures under `artifacts/figures/`.
**Next steps:** Use Phase 2 findings to guide Phase 3 feature-signal EDA and Phase 4 validation diagnostics.

---

### notebooks/eda/02_phase2_geological_domain.ipynb
**Status:** Run - Phase 2 geology / GR / typewell / spatial / dip EDA complete.
**Purpose:** TVT vs MD, GR distributions, typewell correlation, XY scatter, dTVT/dMD, visible-test caveats, and feature-engineering implications.
**Inputs:** `data/raw/train/`, `data/raw/test/`
**Run ID:** `2026-06-21__1506__eda_phase2_geological_domain__542b1c1`
**Findings:** Train evaluation-zone fraction median **0.740**; visible local test fraction range **0.674-0.796**. Median horizontal/typewell GR correlation at known TVT positions **0.812**. Mean GR missingness **0.294** in train and **0.215** in visible local test files. The visible local test wells overlap train wells exactly (`000d7d20`, `00bbac68`, `00e12e8b`), so they are useful for submission-shape validation but not hidden-test distribution inference.
**Outputs:** Phase 2 tables, figures, report, and run bundle listed in the artifact index.

---

### notebooks/eda/03_phase3_feature_signal.ipynb
**Status:** Run - Phase 3 feature-signal EDA complete.
**Purpose:** Screen boundary/endpoint, recent-slope, GR availability, typewell residual, and trajectory candidates before richer feature engineering and boosted-tree tuning.
**Inputs:** `data/raw/train/`, `data/raw/test/`
**Run ID:** `2026-06-21__1847__eda_phase3_feature_signal__ebaa6c3`
**Findings:** Endpoint hold diagnostic RMSE **15.5815**; all-candidate diagnostic linear-probe RMSE **14.6236**; last-50 slope projection RMSE **104.5322**. Top absolute Spearman feature for target-zone delta is `tvt_slope_last_10` (**0.199**); top mutual-information feature is `z_slope_last_100` (**0.514**). Raw typewell residual `gr_minus_tw_proxy` is weak as a row-level signal (**0.019** Spearman).
**Outputs:** Phase 3 tables, figures, report, and run bundle listed in the artifact index.

---

### notebooks/eda/04_phase4_validation.ipynb
**Status:** Run - Phase 4 validation EDA complete.
**Purpose:** Audit local validation with endpoint and slope baselines, current versus descriptor-balanced grouped folds, historical model fold metrics, and public leaderboard gaps.
**Inputs:** `data/raw/train/`, Phase 2 descriptor artifacts, prior model metric artifacts, Kaggle leaderboard log entries.
**Run ID:** `2026-06-21__1928__eda_phase4_validation__203566d`
**Findings:** Full-row endpoint-hold RMSE **15.9099**; direct last-50 slope projection RMSE **112.4817**. Current sorted folds have eval-row CV **0.003** and endpoint-RMSE range **1.466**. Random Forest beats Ridge publicly by **0.032** RMSE while local CV ranks it worse by **0.300**.
**Outputs:** Phase 4 tables, figures, report, and run bundle listed in the artifact index.

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

### notebooks/modeling/01_linear/02_elasticnet_huber_quantile.ipynb
**Status:** Run — second linear-family sweep.
**Purpose:** Compare ElasticNet, Huber, and median Quantile regression using the same simple test-safe numeric features.
**Inputs:** `data/raw/train/`
**Validation:** 3 grouped well folds; train on known-zone rows, score held-out wells' target-zone rows; cap each fold training sample at **25,000** deterministic rows.
**Run ID:** `2026-06-21__1412__linear_elasticnet_huber_quantile__448574b`
**Result:** Best candidate **`elasticnet_alpha_0p001_l1_0p2`** with CV RMSE mean **15.7907**, CV RMSE std **0.5575**, MAE **11.2119**, mean R2 **0.999363**.
**Comparison:** Close to but worse than `ridge_alpha_10` by about **0.0247 RMSE** locally. The Kaggle submission scored **16.091**, worse than both Ridge and Random Forest.
**Outputs:** candidate summary CSV, fold metrics CSV, RMSE comparison PNG, residual PNG, ignored local model pickle, and run bundle under `artifacts/logs/runs/2026-06-21__1412__linear_elasticnet_huber_quantile__448574b/`.

---

### notebooks/modeling/01_linear/04_glm_distribution_diagnostics.ipynb
**Status:** Run - GLM/robust distribution diagnostic complete.
**Purpose:** Test Gaussian, robust, quantile, Gamma, Tweedie, and Poisson-style response assumptions before Bayesian modeling.
**Run ID:** `2026-06-21__2147__glm_distribution_diagnostics__107a7d7`
**Results:** Best candidate `robust_delta_huber`; sampled target-zone grouped CV RMSE mean **14.7928**, RMSE std **0.4716**. Gaussian Ridge on the same signed delta target scored **14.8398**; endpoint hold scored **15.5782**.
**Decision:** Robust signed-delta modeling is promising and should be considered for a self-contained Kaggle submission. Beta and Dirichlet are not valid target geometries for this scalar TVT task.

---

### notebooks/modeling/06_bayesian/01_bayesian_linear_robust.ipynb
**Status:** Run - first NumPyro/JAX Bayesian linear uncertainty baseline complete.
**Purpose:** Compare Normal, Student-t, and heteroskedastic Student-t Bayesian linear models on slope-core target-zone features before HSGP.
**Run ID:** `2026-06-22__1211__bayesian_linear_robust__3855aa6`
**Results:** Best candidate `bayes_student_t_heteroskedastic`; sampled target-zone grouped CV RMSE mean **14.9752**, RMSE std **0.5352**. Empirical 80% coverage **0.783** and 95% coverage **0.933**.
**Decision:** Keep as an uncertainty and likelihood-calibration baseline. Do not package for Kaggle yet because posterior-mean RMSE trails Huber GLM and tuned GAM local diagnostics.

---

### notebooks/modeling/06_bayesian/02_bayesian_linear_tuning.ipynb
**Status:** Run - Bayesian linear tuning sweep complete.
**Purpose:** Tune fixed-tail Student-t, Laplace, heteroskedastic scale, and AutoDelta/AutoNormal guide choices before HSGP residual modeling.
**Run ID:** `2026-06-22__1231__bayesian_linear_tuning__4b66b91`
**Results:** Best candidate `student_t_fixed3_autodelta`; sampled target-zone grouped CV RMSE mean **14.8449**, RMSE std **0.5205**. Empirical 80% coverage **0.769** with mean width **31.12** TVT units.
**Decision:** Keep as the best Bayesian linear point-RMSE baseline so far, but move next to HSGP or other structured residual models rather than more linear-only Bayesian tuning.

---

### notebooks/modeling/06_bayesian/03_bayesian_posterior_diagnostics_scale.ipynb
**Status:** Run - posterior diagnostics and targeted scale sweep complete.
**Purpose:** Diagnose posterior predictive calibration, row-level scale behavior, residual bins, and well-level signed bias before HSGP.
**Run ID:** `2026-06-22__1254__bayesian_posterior_diagnostics_scale__6644c20`
**Results:** Best candidate `hetero_boundary_progress_autodelta`; sampled target-zone grouped CV RMSE mean **14.8626**, RMSE std **0.5019**. Empirical 80% coverage **0.780** with mean width **31.95** TVT units, posterior SD vs absolute error Spearman **0.315**, and well-level absolute mean residual q90 **18.60** TVT units.
**Decision:** Use these diagnostics to design HSGP next. Targeted scale helps relative to the same-notebook fixed reference, but it does not beat the tuned Bayesian fixed Student-t baseline.

---

### kaggle/linear_ridge_submission/linear_ridge_submission.ipynb
**Status:** Pushed to Kaggle as private notebook version 2, completed, and submitted to the competition. Submission `53923555` scored **15.905** public LB. Version 1 errored due to an assumed input mount path.
**Purpose:** Self-contained Kaggle submission notebook for the first `ridge_alpha_10` baseline.
**Inputs:** Competition dataset attached by `kernel-metadata.json` as `rogii-wellbore-geology-prediction`.
**Local verification:** Ran top-to-bottom locally and wrote `artifacts/reports/submission__linear_ridge_alpha_10__2026-06-21.csv` with **14,151** finite predictions matching `sample_submission.csv` id order.
**Kaggle URL:** `https://www.kaggle.com/code/sardorbsobirov/wellbore-linear-ridge-baseline`
**Leaderboard:** Submission `53923555` public score **15.905**.

---

### kaggle/elasticnet_submission/elasticnet_submission.ipynb
**Status:** Pushed to Kaggle as private notebook version 1, completed, and submitted to the competition. Submission `53924934` scored **16.091** public LB.
**Purpose:** Self-contained Kaggle submission notebook for the best ElasticNet candidate from the robust-linear sweep.
**Inputs:** Competition dataset attached by `kernel-metadata.json` as `rogii-wellbore-geology-prediction`.
**Local verification:** Ran top-to-bottom locally and wrote `artifacts/reports/submission__elasticnet_alpha_0p001_l1_0p2__2026-06-21.csv` with **14,151** finite predictions matching `sample_submission.csv` id order.
**Kaggle URL:** `https://www.kaggle.com/code/sardorbsobirov/wellbore-elasticnet-baseline`
**Leaderboard:** Submission `53924934` public score **16.091**.
**Decision:** Rejected as a promotion candidate; the public score is worse than Ridge and Random Forest.

---

### kaggle/random_forest_submission/random_forest_submission.ipynb
**Status:** Pushed to Kaggle as private notebook version 1, completed, and submitted to the competition. Submission `53924036` scored **15.873** public LB.
**Purpose:** Self-contained Kaggle submission notebook for the first Random Forest baseline.
**Inputs:** Competition dataset attached by `kernel-metadata.json` as `rogii-wellbore-geology-prediction`.
**Local verification:** Ran top-to-bottom locally and wrote `artifacts/reports/submission__random_forest__2026-06-21.csv` with **14,151** finite predictions matching `sample_submission.csv` id order.
**Kaggle URL:** `https://www.kaggle.com/code/sardorbsobirov/wellbore-random-forest-baseline`
**Leaderboard:** Submission `53924036` public score **15.873**.
**Decision:** Current best public LB, but local CV ranked it worse than Ridge, so validation design needs review before treating the gain as robust.

---

### notebooks/modeling/02_trees/*
**Status:** Partially run — Random Forest and Extra Trees complete; XGBoost, LightGBM, and CatBoost remain stubs.
**Purpose:** Extra Trees, Random Forest, XGBoost, LightGBM, CatBoost with richer features and CV RMSE comparison.
**Run IDs:** `2026-06-21__1310__tree_random_forest__4eb4a55`, `2026-06-21__1312__tree_extra_trees__4eb4a55`
**Results:** Random Forest CV RMSE mean **16.0655**; Extra Trees CV RMSE mean **52.2415**.
**Decision:** Random Forest is a useful nonlinear baseline and now beats Ridge on public LB, despite worse local CV; naive Extra Trees should not be pursued without feature changes.

---

### notebooks/modeling/03_svr_neighbors/02_knn_regression.ipynb
**Status:** Run — first KNN sweep complete on sampled target-zone validation.
**Purpose:** Test whether global feature-space neighbors are useful before trying SVR, radius-neighbor, clustered-neighbor, XGBoost, or LightGBM models.
**Run ID:** `2026-06-21__1348__neighbor_knn_regression__31e8e1a`
**Results:** Best candidate `knn_k31_distance`; sampled CV RMSE mean **448.4148**, sampled CV RMSE std **33.1216**.
**Decision:** Do not pursue vanilla global KNN on the current simple scaled features; revisit neighbors only with radius constraints, clusters, or richer local geologic features.

---

### notebooks/modeling/04_gam/01_gam_spline_baseline.ipynb
**Status:** Run - first GAM-style additive spline sweep complete.
**Purpose:** Test smooth additive boundary, trajectory, GR, slope, and typewell feature families before boosted-tree tuning.
**Run ID:** `2026-06-21__2007__gam_spline_baseline__8046b3f`
**Results:** Best candidate `boundary_trajectory_slopes`; sampled target-zone grouped CV RMSE mean **14.8719**, RMSE std **0.5337**. Sampled endpoint-hold RMSE **15.5782**; direct slope projection RMSE **104.6157**.
**Decision:** Keep as first additive-smooth modeling baseline and consider a self-contained Kaggle notebook before full boosted-tree tuning.

---

### notebooks/modeling/04_gam/02_gam_spline_tuning.ipynb
**Status:** Run - tuned GAM-style additive spline sweep complete.
**Purpose:** Tune target definition, spline complexity, Ridge regularization, and controlled feature-family additions before deciding whether to submit a GAM model.
**Run ID:** `2026-06-21__2037__gam_spline_tuning__4e31e0e`
**Results:** Best candidate `core_fine__delta_boundary__k7__d3__a60`; sampled target-zone grouped CV RMSE mean **14.8687**, RMSE std **0.5012**. Improvement over first GAM baseline is only **0.0032** RMSE.
**Decision:** Treat as current best GAM candidate but pause GAM micro-tuning unless new feature families are introduced. Submit only if a GAM-specific public-LB read is valuable before boosted trees.

---

### kaggle/gam_spline_tuning_submission/gam_spline_tuning_submission.ipynb
**Status:** Pushed to Kaggle as private notebook version 1, completed, and submitted to the competition. Submission `53932883` scored **15.055** public LB.
**Purpose:** Self-contained Kaggle submission notebook for the fine-tuned GAM-style spline candidate.
**Inputs:** Competition dataset attached by `kernel-metadata.json` as `rogii-wellbore-geology-prediction`.
**Local verification:** Ran top-to-bottom locally and wrote `artifacts/reports/submission__gam_spline_tuning__2026-06-21.csv` with **14,151** finite predictions matching `sample_submission.csv` id order.
**Kaggle URL:** `https://www.kaggle.com/code/sardorbsobirov/wellbore-fine-tuned-gam-spline`
**Leaderboard:** Submission `53932883` public score **15.055**.
**Decision:** Current best public LB. Use the slope-core feature logic as the baseline for boosted trees and future ensembles.

---

### kaggle/glm_huber_submission/glm_huber_submission.ipynb
**Status:** Pushed to Kaggle as private notebook version 1, completed, and submitted to the competition. Submission `53952950` scored **15.464** public LB.
**Purpose:** Self-contained Kaggle submission notebook for the robust signed-delta Huber GLM candidate.
**Inputs:** Competition dataset attached by `kernel-metadata.json` as `rogii-wellbore-geology-prediction`.
**Local verification:** Ran top-to-bottom locally and wrote `artifacts/reports/submission__glm_huber__2026-06-22.csv` with **14,151** finite predictions matching `sample_submission.csv` id order.
**Kaggle URL:** `https://www.kaggle.com/code/sardorbsobirov/wellbore-huber-glm-slope-core`
**Leaderboard:** Submission `53952950` public score **15.464**.
**Decision:** Useful robust-likelihood validation read, but not promoted over the fine-tuned GAM public LB **15.055**.

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

**Phase:** Bayesian posterior diagnostics complete - structural EDA and Phases 2-4 are complete; fine-tuned GAM remains best public LB; robust signed-delta GLM is the best local diagnostic.
**Workflow posture:** Use `sardorsob/sardors-agentic-workflows` data-science gates: data gate, split/leakage gate, baseline gate, experiment loop, evaluation gate, artifact gate, handoff gate. Local reference copy lives under ignored `context/_external/`.
**Next action:** Design the HSGP residual model using the tuned fixed-tail Student-t mean and boundary/progress posterior diagnostics, then investigate the Huber/GAM local-public mismatch and run LightGBM/XGBoost with the same notebook-local slope-core feature logic.
