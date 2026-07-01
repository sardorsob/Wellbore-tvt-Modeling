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

## 2026-06-30 Stratigraphic Feature Backlog

Deep domain review and the latest public misses point to stratigraphic-state inference as the next feature target. Direct formation-distance features are powerful locally but not hidden-test-safe, so the next feature screen should approximate that signal with known-zone evidence only.

Candidate feature groups:

- Typewell/log alignment: DTW distance to typewell windows, best lag, stretch/compression, segment correlation, local motif mismatch, and best-vs-second alignment margin.
- Pseudo-top uncertainty: soft zone/top probabilities, entropy, candidate margin, inferred distance to likely top/boundary, and near-boundary indicators.
- Boundary-state continuity: endpoint `TVT_input`, endpoint slope/curvature, target-zone progress, distance past boundary, and monotonicity or local-jump diagnostics.
- Analog stability: nearest log-plus-trajectory analog id for diagnostics, analog distance, analog-delta magnitude, and whether the nearest analog changes under feature ablation.
- Reliability/stress flags: GR missingness, known-zone GR coverage, short-known-zone flags, extrapolation distance, and high-trajectory-curvature flags.

Implementation rule: every feature must be recomputed inside the notebook and inside each CV fold. Direct train-only formation columns remain oracle diagnostics only.

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
- HSGP residual-on-mean diagnostics found the same additive coordinate set improves a robust Huber signed-delta mean from **14.4172** to **14.3586** local RMSE. The reconstructed GAM residual path was unstable in one fold, so the next Kaggle-safe residual notebook should prioritize Huber + additive Hilbert residuals.

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
| 2026-06-22 | Huber mean plus additive Hilbert residual | `notebooks/modeling/06_bayesian/06_hsgp_residual_on_gam_huber.ipynb`, `artifacts/reports/model__hsgp_residual_on_gam_huber__2026-06-22.md` | Test residual HSGP corrections on strong GAM/Huber means | Best local result is Huber + additive Hilbert residual over `distance_log1p`, `eval_progress`, and `abs_projected_delta_10`; use this for the next Kaggle-safe residual submission candidate. |
| 2026-06-30 | Stratigraphic-state feature backlog | `kaggle/FEATURES.md`, `context/ASSUMPTIONS.md`, `kaggle/CV_STRATEGY.md` | Convert domain research into the next hidden-test-safe feature direction | Candidate-only. Must use fold-safe known-zone evidence; direct formation columns remain oracle diagnostics only. |
| 2026-06-30 | First stratigraphic-state feature screen | `src/preprocessing/stratigraphic.py`, `notebooks/modeling/08_alignment/07_stratigraphic_state_feature_screen.ipynb`, `artifacts/reports/model__stratigraphic_state_feature_screen__2026-06-30.md` | Test known-zone state, typewell/log alignment uncertainty, and fold-safe analog descriptors | Known-zone state features improved same-notebook Huber safe-base by **0.0391** RMSE but did not clear promotion gates; keep as candidate features only. |
| 2026-06-30 | Exact XGBoost known-state injection | `notebooks/modeling/02_trees/10_xgboost_known_state_features.ipynb`, `artifacts/reports/model__xgboost_known_state_features__2026-06-30.md` | Test whether known-zone state summaries improve the current inference-safe XGBoost recipe when added directly | Raw state features worsened the exact recipe; best legal remained `safe_typewell` at **13.4334** RMSE, while best state-augmented candidate was **13.5189**. Do not package. |
| 2026-06-30 | Gated Bayesian state-regime features | `notebooks/modeling/06_bayesian/07_gated_bayesian_regime.ipynb`, `kaggle/gated_bayesian_regime_submission/gated_bayesian_regime_submission.ipynb`, `artifacts/reports/model__gated_bayesian_regime__2026-06-30.md` | Use known-zone state/typewell descriptors as a Bayesian soft gate over residual experts | Conservative gate improved same-notebook Huber from **14.5053** to **14.1072** strict RMSE; package as curiosity probe, not active baseline until public score returns. |
| 2026-06-30 | Pseudo-top uncertainty features | `notebooks/modeling/08_alignment/08_pseudo_top_uncertainty.ipynb`, `artifacts/reports/model__pseudo_top_uncertainty__2026-06-30.md` | Infer hidden-test-safe formation-top elevation, signed offset, nearest-top margin, entropy, and ensemble spread from legal row/typewell/state predictors | Direct formation columns are labels/oracle diagnostics only. Fold-safe proxy tops reached mean top RMSE **35.0660** and nearest-top agreement **0.5804**, but raw pseudo-top tree features did not beat safe/typewell XGBoost, so use as gated/prior evidence next. |
| 2026-06-30 | Pseudo-top gated residual regime | `notebooks/modeling/08_alignment/09_pseudo_top_gated_regime.ipynb`, `artifacts/reports/model__pseudo_top_gated_regime__2026-06-30.md` | Use pseudo-top uncertainty as residual shrinkage/gate evidence rather than raw additive predictors | Best gate scored **13.4694** strict RMSE and **13.7665** analog stress RMSE. It improved stress behavior but did not beat strict safe/typewell reference **13.4506**, so do not package as-is. |
| 2026-06-30 | Pseudo-top interaction XGBoost | `notebooks/modeling/08_alignment/10_pseudo_top_interaction_xgboost.ipynb`, `artifacts/reports/model__pseudo_top_interaction_xgboost__2026-06-30.md` | Test pseudo-top uncertainty as controlled interactions inside the primary XGBoost mean | Best interaction candidate scored **13.4815** strict RMSE, worse than same-notebook safe/typewell **13.4745**. This closes the first flat pseudo-top tree-feature path; use mixture/prior structure next. |
| 2026-07-01 | Pseudo-top Bayesian mixture regime | `notebooks/modeling/06_bayesian/08_pseudo_top_mixture_regime.ipynb`, `artifacts/reports/model__pseudo_top_mixture_regime__2026-07-01.md` | Convert fold-safe pseudo-top distances into soft formation-regime probabilities and fit shrinkage-controlled MAP residual priors | Best candidate scored **13.4437** strict RMSE and **13.7593** analog stress RMSE, the first pseudo-top-family result to edge past strict safe/typewell **13.4506**. Gain is too small to package yet. |
| 2026-07-01 | Pseudo-top regime mean | `notebooks/modeling/06_bayesian/09_pseudo_top_regime_mean.ipynb`, `artifacts/reports/model__pseudo_top_regime_mean__2026-07-01.md` | Blend per-formation weighted Ridge MAP means by fold-safe pseudo-top probabilities and confidence | Best candidate scored **13.0928** strict RMSE and **13.4006** analog stress RMSE, a large local gain over the prior mixture-prior path. Package next with hidden-test-safe recomputation. |
| 2026-07-01 | Pseudo-top regime-mean tuning | `notebooks/modeling/06_bayesian/10_pseudo_top_regime_mean_tuning.ipynb`, `artifacts/reports/model__pseudo_top_regime_mean_tuning__2026-07-01.md` | Tune Ridge alpha and confidence schedules around the direct regime-mean winner | Best tuned candidate scored **13.0910** strict RMSE but worsened stress to **13.4065**, so keep the first regime-mean package and move to a structurally different next model. |
| 2026-07-01 | XGBoost regime-mean stack | `notebooks/modeling/02_trees/11_xgboost_regime_mean_stack.ipynb`, `artifacts/reports/model__xgboost_regime_mean_stack__2026-07-01.md` | Use fold-safe regime-mean predictions as boosted meta-features and residual baselines | Best stack scored **13.0769** strict RMSE but worsened stress to **13.4468**. Keep as diagnostic; next residual stack needs stress/uncertainty gating. |
| 2026-07-01 | Stress-gated regime-mean stack | `notebooks/modeling/02_trees/12_xgboost_regime_stress_gated_stack.ipynb`, `artifacts/reports/model__xgboost_regime_stress_gated_stack__2026-07-01.md` | Gate residual corrections by pseudo-top confidence, margin, and transfer-stress descriptors | Gates produced a tradeoff but no promotion candidate; best compromise **13.0853** strict / **13.4079** stress still trailed original regime-mean stress **13.4006**. |
| 2026-07-01 | Pseudo-top probability-scale tuning | `notebooks/modeling/08_alignment/11_pseudo_top_probability_scale_tuning.ipynb`, `artifacts/reports/model__pseudo_top_probability_scale_tuning__2026-07-01.md` | Tune pseudo-top softmax scale before regime-mean fitting | Original scale **50** remained best on strict **13.0928** and stress **13.4006**; stop simple probability-scale sweeps. |
| 2026-07-01 | Pseudo-top regime mean more rows | `notebooks/modeling/06_bayesian/11_pseudo_top_regime_mean_more_rows.ipynb`, `artifacts/reports/model__pseudo_top_regime_mean_more_rows__2026-07-01.md` | Increase target-row cap from 120 to 220 in the direct regime-mean recipe | More rows worsened strict to **13.1185** and stress to **13.4745**; keep the lighter 120-row package. |
| 2026-07-01 | Alignment-aware pseudo-top regime mean | `notebooks/modeling/08_alignment/12_alignment_aware_pseudo_top_regime_mean.ipynb`, `artifacts/reports/model__alignment_aware_pseudo_top_regime_mean__2026-07-01.md` | Add local GR/typewell DTW, correlation, and log-window features to pseudo-top and regime-mean predictors | Strict improved slightly to **13.0838**, but stress worsened to **13.4303** and proxy-top quality did not improve; diagnostic only. |
| 2026-07-01 | Bayesian regime partial pooling | `notebooks/modeling/06_bayesian/12_bayesian_regime_partial_pooling.ipynb`, `artifacts/reports/model__bayesian_regime_partial_pooling__2026-07-01.md` | Use pseudo-top probabilities and probability-by-feature interactions in shared Ridge residual/direct partial-pooling layers | Best new partial-pooling layers worsened; best overall was the independent regime reference at **13.0953** strict / **13.4038** stress. |
| 2026-07-01 | XGBoost local sequence stack | `notebooks/modeling/02_trees/13_xgboost_local_sequence_stack.ipynb`, `artifacts/reports/model__xgboost_local_sequence_stack__2026-07-01.md` | Use local GR/typewell window summaries and sequence contrasts as stack features around the regime mean | Worsened to **13.1613** strict / **13.4561** stress; do not use flat local-window summaries in the package. |
| 2026-07-01 | Pseudo-top state revision | `notebooks/modeling/08_alignment/13_pseudo_top_state_revision.ipynb`, `artifacts/reports/model__pseudo_top_state_revision__2026-07-01.md` | Learn fold-safe residual corrections for pseudo-top elevations before forming regime probabilities | Improves stress to **13.3883** but worsens strict to **13.1194**; continue only as lighter/gated state correction. |
| 2026-07-01 | Gated pseudo-top state revision | `notebooks/modeling/08_alignment/14_pseudo_top_state_revision_gated.ipynb`, `artifacts/reports/model__pseudo_top_state_revision_gated__2026-07-01.md` | Compare base, global, uncertainty-gated, stress-gated, and hybrid top residual corrections before regime probabilities | Best strict remains unchanged base **13.0928**; best stress **13.3710** worsens strict to **13.1309**. |
| 2026-07-01 | XGBoost raw-window sequence | `notebooks/modeling/02_trees/14_xgboost_raw_window_sequence.ipynb`, `artifacts/reports/model__xgboost_raw_window_sequence__2026-07-01.md` | Add raw local GR/typewell motif bins and bin-wise differences to the pseudo-top/regime-mean recipe | Worsened to **13.1664** strict / **13.4758** stress; do not continue flat raw-window tabular features. |
