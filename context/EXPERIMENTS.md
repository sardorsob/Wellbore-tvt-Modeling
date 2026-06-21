# Experiments

This file is the human-readable registry for reportable runs.

## 2026-06-21__1216__interpolation_baseline__14bef53

- Task: baseline modeling
- Purpose: Establish a naive interpolation/forward-fill RMSE anchor.
- Config: notebook-local config in `notebooks/modeling/00_baseline/00_interpolation_baseline.ipynb`
- Data version: local Kaggle download in `data/raw/train/`
- Split / CV: 3 grouped well folds; score rows where `TVT_input` is missing and `TVT` is present.
- Model: `InterpolationBaseline`
- Primary metric: CV RMSE mean 15.9003; CV RMSE std 0.6071; overall RMSE 15.9099.
- Secondary metrics: fold MAE and R2 in `artifacts/tables/model__interpolation_baseline_metrics__2026-06-21__1216__interpolation_baseline__14bef53.csv`
- Artifacts: metrics CSV, residual PNG, run bundle under `artifacts/logs/runs/2026-06-21__1216__interpolation_baseline__14bef53/`
- Decision: selected as the current naive baseline.
- Notes: Stateless and very fast; useful as a sanity floor, not a competitive model.

## 2026-06-21__1218__linear_ols_ridge_lasso__14bef53

- Task: first regression baseline
- Purpose: Compare OLS, Ridge, and Lasso on simple test-safe numeric features.
- Config: notebook-local config in `notebooks/modeling/01_linear/01_ols_ridge_lasso.ipynb`
- Data version: local Kaggle download in `data/raw/train/`
- Split / CV: 3 grouped well folds; train known-zone rows; score target-zone rows; deterministic cap of 25,000 training rows per fold.
- Model: best candidate `ridge_alpha_10`
- Primary metric: CV RMSE mean 15.7660; CV RMSE std 0.5331.
- Secondary metrics: MAE 11.1123; mean R2 0.999365.
- Artifacts: candidate metrics CSV, fold metrics CSV, RMSE comparison PNG, residual PNG, ignored local model pickle, run bundle under `artifacts/logs/runs/2026-06-21__1218__linear_ols_ridge_lasso__14bef53/`
- Decision: selected as current best local model, improving over interpolation by about 0.1343 RMSE.
- Notes: Still intentionally simple; next model should improve features and try tree-based methods.

## Entry Format

```text
## <run_id>

- Task:
- Purpose:
- Config:
- Data version:
- Split / CV:
- Model:
- Primary metric:
- Secondary metrics:
- Artifacts:
- Decision:
- Notes:
```
