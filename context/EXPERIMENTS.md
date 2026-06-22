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

## 2026-06-21__1847__eda_phase3_feature_signal__ebaa6c3

- Task: Phase 3 feature-signal EDA
- Purpose: Screen boundary/endpoint, recent-slope, GR availability, typewell, and trajectory candidate features before boosted-tree tuning.
- Config: notebook-local config in `notebooks/eda/03_phase3_feature_signal.ipynb`
- Data version: local Kaggle download in `data/raw/train/` and visible example files in `data/raw/test/`
- Split / CV: 3 grouped well folds on sampled target-zone rows for diagnostic linear probes; not a submission training protocol.
- Primary result: all-candidate diagnostic probe RMSE **14.6236** vs endpoint-hold diagnostic RMSE **15.5815**.
- Secondary results: last-50 slope projection RMSE **104.5322**; top absolute Spearman target-zone delta feature `tvt_slope_last_10` (**0.199**); top mutual-information feature `z_slope_last_100` (**0.514**).
- Artifacts: bounded candidate feature sample preview, feature signal CSV, feature-group probe metrics/summary, visible-test drift table, GR missingness signal table, diagnostic figures, report, and run bundle under `artifacts/logs/runs/2026-06-21__1847__eda_phase3_feature_signal__ebaa6c3/`
- Decision: implement a richer well-relative feature builder before serious LightGBM/XGBoost tuning.
- Notes: Visible local test drift remains diagnostic only because those wells overlap train.

## 2026-06-21__1928__eda_phase4_validation__203566d

- Task: Phase 4 validation EDA
- Purpose: Audit the local validation protocol before richer feature engineering and boosted-tree tuning.
- Config: notebook-local config in `notebooks/eda/04_phase4_validation.ipynb`
- Data version: local Kaggle download in `data/raw/train/`, plus Phase 2/3 artifact tables and prior model metrics.
- Split / CV: current sorted grouped well folds compared with descriptor-balanced grouped folds; no model submission trained.
- Primary result: full-row endpoint-hold RMSE **15.9099**; direct last-50 slope projection RMSE **112.4817**.
- Secondary results: current sorted fold eval-row CV **0.003** and endpoint-RMSE range **1.466**; descriptor-balanced fold eval-row CV **0.001** and endpoint-RMSE range **2.428**. Random Forest public LB beats Ridge by **0.032**, while local CV ranks Random Forest worse by **0.300**.
- Artifacts: endpoint baseline table, fold assignments, fold descriptor summaries, fold variability, model fold metrics, CV/leaderboard gap table, descriptor PCA table, figures, report, and run bundle under `artifacts/logs/runs/2026-06-21__1928__eda_phase4_validation__203566d/`
- Decision: require endpoint-hold baseline, grouped fold-level metrics, descriptor-balance reporting, and leaderboard-gap interpretation in future reportable modeling notebooks.
- Notes: Public LB remains useful as an external check, but should not be the optimization target.

## 2026-06-21__2007__gam_spline_baseline__8046b3f

- Task: first GAM-style baseline
- Purpose: Test smooth additive spline models before LightGBM/XGBoost.
- Config: notebook-local config in `notebooks/modeling/04_gam/01_gam_spline_baseline.ipynb`
- Data version: local Kaggle download in `data/raw/train/`
- Split / CV: 3 grouped well folds on sampled target-zone rows; deterministic cap of 140 target rows per well and 80,000 training rows per fold.
- Model: best candidate `boundary_trajectory_slopes`, implemented as sklearn `SplineTransformer` plus Ridge.
- Primary metric: sampled target-zone CV RMSE mean **14.8719**; RMSE std **0.5337**.
- Secondary metrics: MAE mean **10.4898**; mean R2 **0.999423**. Sampled endpoint-hold RMSE **15.5782**; direct slope projection RMSE **104.6157**.
- Artifacts: fold metrics CSV, summary CSV, variant table, residual sample, residual/RMSE figures, report, model comparison row, and run bundle under `artifacts/logs/runs/2026-06-21__2007__gam_spline_baseline__8046b3f/`
- Decision: keep as the first additive-smooth modeling baseline. Consider a self-contained Kaggle notebook before moving fully to boosted trees.
- Notes: Because this trains on sampled target-zone rows, compare directionally with older known-zone-trained Ridge/Random Forest results.

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
