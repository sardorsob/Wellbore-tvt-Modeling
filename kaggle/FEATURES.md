# Features

## Test-Safe Raw Columns

- `MD`
- `X`
- `Y`
- `Z`
- `GR`
- `TVT_input`

## Train-Only Columns

Do not use these in train/test parity feature sets:

- `ANCC`
- `ASTNU`
- `ASTNL`
- `EGFDU`
- `EGFDL`
- `BUDA`
- `TVT`

## Current Engineered Feature Families

Defined in `src/preprocessing/features.py`:

- `row_idx`
- `tvt_input_lag_*`
- `tvt_input_rolling_*`
- GR rolling statistics
- MD and trajectory derivatives
- cumulative horizontal displacement
- typewell GR interpolation at TVT proxy positions
- typewell formation labels for analysis
- known-zone flag
- normalized within-well MD position

## Candidate Feature Families From Phase 2 EDA

These are not yet promoted into reportable modeling runs unless listed in the feature log below.

- Boundary and endpoint features: last known `TVT_input`, last known `MD`, distance past known/evaluation boundary, and normalized evaluation-zone progress.
- Recent-window shape features: last-10/25/50 known-zone slopes, recent TVT range, endpoint smoothness, and local curvature proxies.
- GR availability features: row-level `GR` missingness, rolling GR missing fraction, known/eval-zone GR coverage summaries, and imputation source flags.
- Typewell alignment features: robust horizontal/typewell GR correlation, smoothed typewell GR at TVT proxy positions, local alignment-window scores, and geology-label summaries used only when train/test-safe.
- Spatial/trajectory features: XY displacement, trajectory derivatives, and interactions with boundary distance.

## Phase 3 Feature-Signal Notes

- Endpoint hold is a strong diagnostic baseline: sampled target-zone grouped RMSE **15.5815**.
- Direct last-50 known-zone slope projection is unstable: sampled target-zone grouped RMSE **104.5322**.
- Combining candidate feature families is promising: diagnostic all-candidate linear probe RMSE **14.6236**.
- Strongest univariate target-zone delta signals in Phase 3: `tvt_slope_last_10` by absolute Spearman (**0.199**) and `z_slope_last_100` by mutual information (**0.514**).
- Raw typewell residual `gr_minus_tw_proxy` is weak as a row-level signal: Spearman delta correlation **0.019**. Prefer robust typewell summaries over raw residuals.

## GAM-Style Feature Notes

- First GAM-style additive spline run used notebook-local features only, with no dependency on local precomputed artifacts.
- Best variant used boundary, trajectory, and recent-slope features: `boundary_trajectory_slopes`.
- Best sampled target-zone grouped RMSE was **14.8719**, beating sampled endpoint hold (**15.5782**).
- Tuned GAM sweep improved the slope-core feature family only slightly: `core_fine__delta_boundary__k7__d3__a60` reached sampled target-zone grouped RMSE **14.8687**.
- GR and typewell feature-family additions did not beat slope-core splines in the tuned additive Ridge-spline setup.
- For any Kaggle submission, copy the inline feature construction into the Kaggle notebook and rerun there.

## GLM Distribution Notes

- The GLM diagnostic reused the same slope-core boundary/trajectory/recent-slope feature family without spline expansion.
- Best candidate was `robust_delta_huber`, scoring sampled target-zone grouped RMSE **14.7928**.
- The signed delta target is only **53.3%** positive and has heavy tails, so robust continuous losses are better matched than Beta, Dirichlet, Poisson, or raw-TVΤ Gamma-style assumptions.
- If promoted to Kaggle, the Huber notebook should use the same notebook-local slope-core feature construction as the fine-tuned GAM submission.

## Bayesian Linear Notes

- First NumPyro/JAX Bayesian linear run reused the slope-core boundary/trajectory/recent-slope feature family.
- Best candidate was `bayes_student_t_heteroskedastic`, with sampled target-zone grouped RMSE **14.9752**.
- Posterior predictive calibration was useful for a first pass: 80% coverage **0.783**, 95% coverage **0.933**.
- Treat this as an uncertainty baseline for HSGP, not as a Kaggle submission candidate yet.
- Bayesian linear tuning reused the same slope-core feature family and improved point RMSE to **14.8449** with `student_t_fixed3_autodelta`.
- Heteroskedastic Bayesian variants did not beat the fixed-scale Student-t AutoDelta candidate in this sweep; reserve extra scale structure for models that also add residual geometry.
- The Kaggle Bayesian linear submission uses the same slope-core feature family and a SciPy fixed-nu Student-t MAP objective as a dependency-light analogue of the NumPyro AutoDelta winner.
- Targeted posterior-scale diagnostics found boundary/progress scale features (`distance_log1p`, `eval_progress`, `progress_squared`, `distance_progress`) were the best heteroskedastic scale family, reaching RMSE **14.8626** and 80% coverage **0.780**.
- Posterior SD only moderately ranks hard rows (Spearman **0.315** with absolute error), and well-level mean residual bias remains material, so HSGP/hierarchical structure should target residual mean structure rather than only row-level scale.
- First HSGP residual baseline used standardized `distance_log1p` and `eval_progress` as the best coordinate set. A small **25**-basis model improved the same-run linear reference, while a **49**-basis version and 3D horizontal variant did not help.
- Additive HSGP tuning improved the GP result by using separate one-dimensional smooth terms. The best coordinate set was `distance_log1p`, `eval_progress`, and `abs_projected_delta_10`, reaching sampled RMSE **14.8235**.
- Additive HSGP SVI needed a lower learning rate and clipped gradients; keep that optimizer setup for future GP notebooks with multiple independent length/amplitude terms.

## Feature Log

Record feature additions here when they are used in reportable experiments:

| Date | Feature family | Files | Reason | Leakage notes |
|---|---|---|---|---|
| 2026-06-21 | Initial documented feature families | `src/preprocessing/features.py` | Existing scaffold orientation | Train-only formation columns excluded by `get_feature_columns`. |
| 2026-06-21 | Simple linear baseline features | `notebooks/modeling/01_linear/01_ols_ridge_lasso.ipynb` | First OLS/Ridge/Lasso sweep | Uses raw trajectory, clipped/imputed GR, forward-filled `TVT_input`, known-zone flag, MD position, and trajectory deltas. Trains only on known-zone rows and scores target-zone rows. |
| 2026-06-21 | Simple robust linear features | `src/preprocessing/tabular.py`, `notebooks/modeling/01_linear/02_elasticnet_huber_quantile.ipynb`, `kaggle/elasticnet_submission/elasticnet_submission.ipynb` | ElasticNet / Huber / median Quantile sweep and ElasticNet Kaggle submission | Reuses the same 15 test-safe numeric features with train-fold standardization. No train-only formation columns. |
| 2026-06-21 | Simple tabular tree baseline features | `src/preprocessing/tabular.py`, `notebooks/modeling/02_trees/01_random_forest.ipynb`, `notebooks/modeling/02_trees/02_extra_trees.ipynb`, `kaggle/random_forest_submission/random_forest_submission.ipynb` | First Random Forest / Extra Trees sweep and Random Forest Kaggle submission | Reuses the same 15 test-safe numeric features; no train-only formation columns. Tree training uses deterministic row caps on known-zone rows and scores target-zone rows. |
| 2026-06-21 | Simple tabular KNN features | `src/preprocessing/tabular.py`, `notebooks/modeling/03_svr_neighbors/02_knn_regression.ipynb` | First KNN sweep | Reuses the same 15 test-safe numeric features with train-fold standardization. KNN scored deterministic target-zone validation samples for runtime control; no train-only formation columns. |
| 2026-06-21 | Phase 2 EDA feature candidates | `notebooks/eda/02_phase2_geological_domain.ipynb`, `artifacts/reports/eda__phase2_geological_domain__2026-06-21.md` | Guide the next richer feature builder before LightGBM/XGBoost and Bayesian models | Candidate-only; do not use train-only formation columns in submission models. Local visible test wells overlap train, so they are not valid hidden-test drift evidence. |
| 2026-06-21 | Phase 3 feature-signal candidates | `notebooks/eda/03_phase3_feature_signal.ipynb`, `artifacts/reports/eda__phase3_feature_signal__2026-06-21.md` | Rank boundary, slope, GR, typewell, and trajectory feature families before boosted-tree tuning | Candidate-only; diagnostic probes train on sampled target-zone rows and are not a submission training protocol. |
| 2026-06-21 | GAM-style additive spline features | `notebooks/modeling/04_gam/01_gam_spline_baseline.ipynb`, `artifacts/reports/model__gam_spline_baseline__2026-06-21.md` | Test smooth additive boundary/trajectory/slope structure before boosted trees | Feature construction is notebook-local for submission portability. Result is sampled target-zone CV, not yet public LB. |
| 2026-06-21 | Tuned GAM slope-core features | `notebooks/modeling/04_gam/02_gam_spline_tuning.ipynb`, `artifacts/reports/model__gam_spline_tuning__2026-06-21.md` | Tune spline complexity and Ridge regularization before deciding whether to submit a GAM notebook | Best candidate still uses slope-core boundary/trajectory/recent-slope features. GR/typewell additions did not help under additive Ridge splines. |
| 2026-06-22 | GLM robust signed-delta features | `notebooks/modeling/01_linear/04_glm_distribution_diagnostics.ipynb`, `artifacts/reports/model__glm_distribution_diagnostics__2026-06-22.md` | Test whether distribution/loss choice improves the slope-core feature baseline before Bayesian modeling | Same slope-core feature family as the GAM submission. Beta and Dirichlet are invalid target geometries for scalar TVT. |
| 2026-06-22 | Bayesian slope-core uncertainty features | `notebooks/modeling/06_bayesian/01_bayesian_linear_robust.ipynb`, `artifacts/reports/model__bayesian_linear_robust__2026-06-22.md` | Establish Bayesian Normal/Student-t/heteroskedastic baseline before HSGP | Same slope-core feature family; posterior mean trails Huber/GAM but intervals provide HSGP calibration guidance. |
| 2026-06-22 | Bayesian tuned slope-core features | `notebooks/modeling/06_bayesian/02_bayesian_linear_tuning.ipynb`, `artifacts/reports/model__bayesian_linear_tuning__2026-06-22.md` | Tune Bayesian likelihood tails, guides, and heteroskedastic scale before HSGP | Same slope-core feature family; fixed-tail Student-t AutoDelta is best Bayesian linear point-RMSE candidate but still not a Kaggle submission candidate over GAM. |
| 2026-06-22 | Bayesian linear Kaggle submission features | `kaggle/bayesian_linear_submission/bayesian_linear_submission.ipynb`, `artifacts/reports/submission__bayesian_linear_student_t__2026-06-22.csv` | Submit a Kaggle-safe Bayesian linear Student-t MAP baseline | Same slope-core feature family; SciPy Student-t MAP avoids NumPyro dependency risk with internet disabled and scored public LB **15.512**. |
| 2026-06-22 | Bayesian targeted scale diagnostics | `notebooks/modeling/06_bayesian/03_bayesian_posterior_diagnostics_scale.ipynb`, `artifacts/reports/model__bayesian_posterior_scale__2026-06-22.md` | Diagnose posterior predictive calibration and targeted heteroskedastic scale before HSGP | Boundary/progress scale improves same-notebook fixed-scale reference, but residual mean structure remains the next Bayesian target. |
| 2026-06-22 | Bayesian HSGP residual coordinates | `notebooks/modeling/06_bayesian/04_hsgp_residual_baseline.ipynb`, `artifacts/reports/model__bayesian_hsgp_residual__2026-06-22.md` | Test first HSGP residual mean structure over boundary/progress geometry | `distance_log1p` plus `eval_progress` with 25 basis functions is the first HSGP reference; increasing basis count or adding horizontal distance did not improve this pass. |
| 2026-06-22 | Bayesian additive HSGP residual coordinates | `notebooks/modeling/06_bayesian/05_hsgp_additive_tuning.ipynb`, `artifacts/reports/model__bayesian_hsgp_additive__2026-06-22.md` | Test additive one-dimensional HSGP residual terms before Huber/GAM residual HSGP | Best geometry uses `distance_log1p`, `eval_progress`, and `abs_projected_delta_10`; future HSGP notebooks should keep clipped lower-step SVI and try this residual around a stronger Huber/GAM mean. |
