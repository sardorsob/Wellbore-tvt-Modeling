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

## 2026-06-21__1412__linear_elasticnet_huber_quantile__448574b

- Task: second linear-family sweep
- Purpose: Compare ElasticNet, Huber, and median Quantile regression before moving into Bayesian models.
- Config: notebook-local config in `notebooks/modeling/01_linear/02_elasticnet_huber_quantile.ipynb`
- Data version: local Kaggle download in `data/raw/train/`
- Split / CV: 3 grouped well folds; train known-zone rows; score target-zone rows; deterministic cap of 25,000 training rows per fold.
- Model: best candidate `elasticnet_alpha_0p001_l1_0p2`
- Primary metric: CV RMSE mean 15.7907; CV RMSE std 0.5575.
- Secondary metrics: MAE 11.2119; mean R2 0.999363.
- Artifacts: candidate summary CSV, fold metrics CSV, RMSE comparison PNG, residual PNG, ignored local model pickle, run bundle under `artifacts/logs/runs/2026-06-21__1412__linear_elasticnet_huber_quantile__448574b/`
- Decision: do not submit to Kaggle yet; it is close to Ridge but worse by about 0.0247 local RMSE.
- Notes: Huber and median Quantile landed near the interpolation baseline at 15.9003; stronger ElasticNet regularization was much worse at 23.1799.

## 2026-06-21__1310__tree_random_forest__4eb4a55

- Task: first tree baseline
- Purpose: Test a nonlinear bagged-tree baseline before deeper linear tuning.
- Config: notebook-local config in `notebooks/modeling/02_trees/01_random_forest.ipynb`
- Data version: local Kaggle download in `data/raw/train/`
- Split / CV: 3 grouped well folds; train known-zone rows; score target-zone rows; deterministic cap of 30,000 training rows per fold.
- Model: `RandomForestRegressor`
- Primary metric: CV RMSE mean 16.0655; CV RMSE std 0.5282.
- Secondary metrics: MAE 11.4252; mean R2 0.999341.
- Artifacts: fold metrics CSV, residual PNG, feature-importance PNG, ignored local model pickle, run bundle under `artifacts/logs/runs/2026-06-21__1310__tree_random_forest__4eb4a55/`
- Decision: keep as a nonlinear baseline, but do not promote over Ridge because local RMSE is worse by about 0.2995.
- Notes: Uses the same simple test-safe features as the linear sweep; better tree results likely need richer well-relative/geologic features or boosted trees.

## 2026-06-21__1312__tree_extra_trees__4eb4a55

- Task: first tree baseline
- Purpose: Compare a more randomized tree ensemble against Random Forest on the same feature set.
- Config: notebook-local config in `notebooks/modeling/02_trees/02_extra_trees.ipynb`
- Data version: local Kaggle download in `data/raw/train/`
- Split / CV: 3 grouped well folds; train known-zone rows; score target-zone rows; deterministic cap of 30,000 training rows per fold.
- Model: `ExtraTreesRegressor`
- Primary metric: CV RMSE mean 52.2415; CV RMSE std 1.3594.
- Secondary metrics: MAE 36.5310; mean R2 0.992999.
- Artifacts: fold metrics CSV, residual PNG, feature-importance PNG, ignored local model pickle, run bundle under `artifacts/logs/runs/2026-06-21__1312__tree_extra_trees__4eb4a55/`
- Decision: do not pursue this naive Extra Trees setup without feature changes; it is far worse than interpolation, Ridge, and Random Forest.
- Notes: The randomized split behavior appears poorly matched to the smooth trailing-zone extrapolation task with only the current simple features.

## 2026-06-21__1348__neighbor_knn_regression__31e8e1a

- Task: first neighbor baseline
- Purpose: Test whether local feature-space neighbors are useful before trying SVR, radius-neighbor, clustered-neighbor, XGBoost, or LightGBM models.
- Config: notebook-local config in `notebooks/modeling/03_svr_neighbors/02_knn_regression.ipynb`
- Data version: local Kaggle download in `data/raw/train/`
- Split / CV: 3 grouped well folds; train known-zone rows; score a deterministic sample of 30,000 target-zone validation rows per fold for runtime control.
- Model: best candidate `knn_k31_distance`
- Primary metric: sampled CV RMSE mean 448.4148; sampled CV RMSE std 33.1216.
- Secondary metrics: sampled MAE 352.3168; sampled mean R2 0.483437.
- Artifacts: candidate fold metrics CSV, candidate summary CSV, RMSE comparison PNG, residual PNG, ignored local model pickle, run bundle under `artifacts/logs/runs/2026-06-21__1348__neighbor_knn_regression__31e8e1a/`
- Decision: do not pursue vanilla KNN on the current simple scaled features; it is orders of magnitude worse than Ridge and Random Forest.
- Notes: Runtime required sampled validation, so the score is not directly comparable to full-row tree/linear CV, but it is far enough away to reject this baseline. If revisiting neighbors, use well-local/stratified features, radius constraints, or clustered local models rather than global KNN.

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
