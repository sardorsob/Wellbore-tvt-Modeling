# Project Status

**Last updated:** 2026-06-22
**Current phase:** Hierarchical modeling roadmap drafted

---

## Done

- [x] Repository structure created per `rules_templet.md`
- [x] Full requirements.txt (JAX, NumpPyro, Flax, Optax, XGBoost, LightGBM, PyMC, ArviZ)
- [x] Context documentation (GLOSSARY, DATASETS, INTERFACES, ASSUMPTIONS, DECISIONS, RISKS)
- [x] `src/` package skeleton (io, preprocessing, modeling, evaluation, viz, utils)
- [x] Config files for all modeling approaches
- [x] CLI scripts (download_data, train, predict, eval)
- [x] Unit test stubs
- [x] Notebook stubs (EDA through reporting)
- [x] Download competition data (`python scripts/download_data.py`)
- [x] Phase 0-1 EDA notebook (`notebooks/eda/01_phase0_phase1_data_and_schema.ipynb`) - counts, schema spot-check, 1A-1E
- [x] Verify row index to submission id mapping (**A1** - pass)
- [x] Verify TVT_input NaN rows match submission ids (**A2** - pass)
- [x] Check test file columns (**A3** / **D002** - formation columns absent in test; decision unchanged)
- [x] Add local agent rules and Kaggle tracking docs from `sardorsob/sardors-agentic-workflows` guidance
- [x] Run interpolation baseline (`2026-06-21__1216__interpolation_baseline__14bef53`) - CV RMSE mean **15.9003**
- [x] Run first OLS/Ridge/Lasso sweep (`2026-06-21__1218__linear_ols_ridge_lasso__14bef53`) - best **ridge_alpha_10**, CV RMSE mean **15.7660**
- [x] Run ElasticNet/Huber/Quantile sweep (`2026-06-21__1412__linear_elasticnet_huber_quantile__448574b`) - best **elasticnet_alpha_0p001_l1_0p2**, CV RMSE mean **15.7907**
- [x] Push and submit ElasticNet Kaggle notebook version 1 (`ref=53924934`) - public LB **16.091**
- [x] Create and locally validate self-contained Kaggle Ridge submission notebook
- [x] Push private Kaggle notebook version 1: `https://www.kaggle.com/code/sardorbsobirov/wellbore-linear-ridge-baseline`
- [x] Push fixed private Kaggle notebook version 2 with robust input discovery
- [x] Submit Kaggle notebook version 2 output to competition (`ref=53923555`) - public LB **15.905**
- [x] Run first Random Forest baseline (`2026-06-21__1310__tree_random_forest__4eb4a55`) - CV RMSE mean **16.0655**
- [x] Run first Extra Trees baseline (`2026-06-21__1312__tree_extra_trees__4eb4a55`) - CV RMSE mean **52.2415**
- [x] Push and submit Random Forest Kaggle notebook version 1 (`ref=53924036`) - public LB **15.873**
- [x] Run first KNN baseline (`2026-06-21__1348__neighbor_knn_regression__31e8e1a`) - sampled CV RMSE mean **448.4148**
- [x] Run Phase 2 geological/domain EDA (`2026-06-21__1506__eda_phase2_geological_domain__542b1c1`) - long trailing-zone extrapolation, GR missingness, typewell alignment, and local-test overlap documented
- [x] Run Phase 3 feature-signal EDA (`2026-06-21__1847__eda_phase3_feature_signal__ebaa6c3`) - endpoint hold, feature-family probes, GR availability, typewell residual signal, and visible-test caveats documented
- [x] Run Phase 4 validation EDA (`2026-06-21__1928__eda_phase4_validation__203566d`) - endpoint baseline, fold balance, local/public mismatch, and validation protocol documented
- [x] Run first GAM-style additive spline baseline (`2026-06-21__2007__gam_spline_baseline__8046b3f`) - best `boundary_trajectory_slopes`, sampled target-zone CV RMSE mean **14.8719**
- [x] Run tuned GAM-style additive spline sweep (`2026-06-21__2037__gam_spline_tuning__4e31e0e`) - best `core_fine__delta_boundary__k7__d3__a60`, sampled target-zone CV RMSE mean **14.8687**
- [x] Push and submit fine-tuned GAM Kaggle notebook version 1 (`ref=53932883`) - public LB **15.055**
- [x] Run GLM distribution diagnostic sweep (`2026-06-21__2147__glm_distribution_diagnostics__107a7d7`) - best `robust_delta_huber`, sampled target-zone CV RMSE mean **14.7928**
- [x] Push and submit Huber GLM Kaggle notebook version 1 (`ref=53952950`) - public LB **15.464**
- [x] Run NumPyro/JAX Bayesian linear robust baseline (`2026-06-22__1211__bayesian_linear_robust__3855aa6`) - best `bayes_student_t_heteroskedastic`, sampled target-zone CV RMSE mean **14.9752**, 80% coverage **0.783**
- [x] Run NumPyro/JAX Bayesian linear tuning sweep (`2026-06-22__1231__bayesian_linear_tuning__4b66b91`) - best `student_t_fixed3_autodelta`, sampled target-zone CV RMSE mean **14.8449**, 80% coverage **0.769**
- [x] Run Bayesian posterior diagnostics and targeted scale sweep (`2026-06-22__1254__bayesian_posterior_diagnostics_scale__6644c20`) - best `hetero_boundary_progress_autodelta`, sampled target-zone CV RMSE mean **14.8626**, 80% coverage **0.780**
- [x] Push and submit Bayesian linear Student-t MAP Kaggle notebook version 1 (`ref=53957370`) - public LB **15.512**
- [x] Run first NumPyro/JAX HSGP residual baseline (`2026-06-22__1358__bayesian_hsgp_residual_baseline__bd24319`) - best `hsgp_bp_m5_autodelta`, sampled target-zone CV RMSE mean **14.8997**
- [x] Run additive NumPyro/JAX HSGP residual tuning (`2026-06-22__1425__bayesian_hsgp_additive_tuning__48601fe`) - best `additive_dps_m8_se_autodelta`, sampled target-zone CV RMSE mean **14.8235**
- [x] Push self-contained additive HSGP Student-t MAP Kaggle notebook version 1 and submit output (`ref=53958582`) - public LB **15.255**
- [x] Run HSGP residuals on GAM/Huber means (`2026-06-22__1519__bayesian_hsgp_residual_on_gam_huber__fadbdba`) - best `huber_plus_hsgp_dps_m8`, sampled target-zone CV RMSE mean **14.3586**
- [x] Push self-contained Huber + additive Hilbert residual Kaggle notebook version 1 and submit output (`ref=53960276`) - public LB **14.514**

## In progress

- [ ] Export artifact figures per `CURSOR_RULES.md`
- [ ] Verify Kaggle competition rules page before first submission (`kaggle/RULES_SUMMARY.md`)
- [ ] Use public LB **14.514** as the current external baseline; avoid leaderboard-only tuning
- [ ] Follow the staged hierarchy plan in `context/HIERARCHICAL_MODELING_ROADMAP.md`
- [ ] Investigate why Huber GLM beat GAM locally but scored worse publicly
- [ ] Investigate why local CV ranked Random Forest worse than Ridge while public LB ranked it better
- [ ] Defer SVR/radius-neighbor work unless new well-local features make neighbor geometry plausible

## Next up

1. Create `notebooks/modeling/07_hierarchical/` and run `00_residual_bias_audit.ipynb` before any hierarchical HSGP implementation.
2. Prototype empirical Bayes/contextual well offsets only if the residual audit shows material well-level bias that is explainable by test-safe descriptors.
3. Add contextual hierarchy to the Huber mean before combining it with the additive Hilbert residual.
4. Defer full hierarchical HSGP until simpler partial-pooling notebooks prove transferable signal.
5. Investigate why the Huber + Hilbert residual gain transferred externally while the Huber-only public score lagged GAM.
6. Investigate why the reconstructed spline GAM mean was unstable in `06_hsgp_residual_on_gam_huber.ipynb` despite the public GAM notebook scoring best externally.
7. Refine validation before making many leaderboard submissions from small public-LB gains
8. Revisit neighbors only with radius constraints, clusters, or richer local geologic features

## Blockers

- None currently

## Competition timeline

| Milestone | Date |
|---|---|
| Start | 2026-05-05 |
| Entry deadline | 2026-07-29 |
| Team merger deadline | 2026-07-29 |
| Final submission | 2026-08-05 |

## Leaderboard history

| Date | Model | Local RMSE | LB RMSE | Notes |
|---|---|---|---|---|
| 2026-06-21 | Ridge alpha 10 | 15.7660 | 15.905 | Submission `53923555`; first external baseline |
| 2026-06-21 | Random Forest | 16.0655 | 15.873 | Submission `53924036`; current best public LB, but local CV disagrees with Ridge ordering |
| 2026-06-21 | ElasticNet alpha 0.001 l1 0.2 | 15.7907 | 16.091 | Submission `53924934`; rejected, worse than Ridge and Random Forest externally |
| 2026-06-22 | Fine-tuned GAM spline | 14.8687 | 15.055 | Submission `53932883`; new best public LB using slope-core boundary/trajectory/recent-slope features |
| 2026-06-22 | Huber GLM slope-core | 14.7928 | 15.464 | Submission `53952950`; worse than GAM externally despite best local diagnostic |
| 2026-06-22 | Bayesian linear Student-t MAP | 14.8449 | 15.512 | Submission `53957370`; useful Bayesian baseline, but worse than Huber and GAM externally |
| 2026-06-22 | Additive HSGP Student-t MAP | 14.8235 | 15.255 | Submission `53958582`; beats Bayesian linear and Huber externally, but not fine-tuned GAM |
| 2026-06-22 | Huber + additive Hilbert residual | 14.3586 | 14.514 | Submission `53960276`; new best public LB, but still far from top public range |

## Local diagnostic model history

| Date | Model | Local diagnostic RMSE | Notes |
|---|---|---:|---|
| 2026-06-21 | GAM-style additive spline `boundary_trajectory_slopes` | 14.8719 | Sampled target-zone grouped CV; not yet a Kaggle submission |
| 2026-06-21 | Tuned GAM-style spline `core_fine__delta_boundary__k7__d3__a60` | 14.8687 | Sampled target-zone grouped CV; +0.0032 RMSE vs first GAM, so useful but nearly plateaued |
| 2026-06-22 | Robust signed-delta Huber GLM `robust_delta_huber` | 14.7928 | Sampled target-zone grouped CV; supports robust/Student-t likelihood direction before Bayesian modeling |
| 2026-06-22 | Bayesian heteroskedastic Student-t linear `bayes_student_t_heteroskedastic` | 14.9752 | Sampled target-zone grouped CV; useful uncertainty baseline, not a point-RMSE promotion candidate |
| 2026-06-22 | Bayesian fixed Student-t linear `student_t_fixed3_autodelta` | 14.8449 | Sampled target-zone grouped CV; best Bayesian linear point-RMSE result so far, but still trails Huber GLM locally |
| 2026-06-22 | Bayesian boundary/progress scale `hetero_boundary_progress_autodelta` | 14.8626 | Sampled target-zone grouped CV; improves same-notebook fixed scale reference and calibrates uncertainty, but does not beat tuned Bayesian fixed Student-t |
| 2026-06-22 | Bayesian HSGP residual `hsgp_bp_m5_autodelta` | 14.8997 | Sampled target-zone grouped CV; improves same-notebook linear reference but does not beat tuned Bayesian fixed Student-t |
| 2026-06-22 | Bayesian additive HSGP residual `additive_dps_m8_se_autodelta` | 14.8235 | Sampled target-zone grouped CV; best HSGP so far and beats tuned Bayesian fixed Student-t locally, but still trails robust Huber GLM |
| 2026-06-22 | Huber + additive Hilbert residual `huber_plus_hsgp_dps_m8` | 14.3586 | Sampled target-zone grouped CV; strongest local diagnostic so far, but needs Kaggle public validation because Huber-only previously underperformed externally |
