# Project Status

**Last updated:** 2026-06-28
**Current phase:** XGBoost inference-safe submission is current best public LB

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
- [x] Run hierarchical residual-bias audit (`2026-06-22__hierarchical_residual_bias_audit`) - median absolute well mean residual **6.890**, p90 **18.929**, max **56.454** TVT units
- [x] Run hierarchical empirical Bayes offset screen (`2026-06-23__hierarchical_empirical_bayes_offsets`) - best `global_bias_shrink_050`, sampled RMSE **14.7378**, only **0.0008** better than same-sample no-offset baseline
- [x] Run hierarchical residual-shape screen (`2026-06-23__hierarchical_residual_shape_screen`) - best `row_linear_ridge_a100_s025`, sampled RMSE **14.7382**, only **0.0003** better than same-sample no-shape baseline
- [x] Run contextual hierarchical Huber interaction screen (`2026-06-23__contextual_hierarchical_huber_screen`) - best `context_poly2_ridge_a2000_s025`, sampled RMSE **14.7322**, **0.0064** better than same-sample no-context baseline
- [x] Run strict-CV LightGBM screen (`2026-06-26__lightgbm_strict_cv`) - best `lgbm_medium_l2`, strict-hybrid RMSE **13.2363**, analog-distance stress RMSE **13.4453**
- [x] Push and submit inference-safe LightGBM Kaggle notebook version 1 (`ref=54097453`) - public LB **14.687**
- [x] Run strict-CV XGBoost screen (`2026-06-27__xgboost_strict_cv`) - best `xgb_pseudohuber_shallow_robust`, strict-hybrid RMSE **12.7504**, analog-distance stress RMSE **12.9116**
- [x] Push and submit inference-safe XGBoost Kaggle notebook version 1 (`ref=54129475`) - public LB **14.472**
- [x] Run inference-safe XGBoost tuning screen (`2026-06-28__xgboost_inference_safe_tuning`) - best legal `pseudo_shallow_more_trees__safe_typewell`, strict-hybrid RMSE **13.4178**, analog-distance stress RMSE **13.7533**; do not package yet
- [x] Run nested XGBoost residual-stacking screen (`2026-06-28__xgboost_residual_stacking`) - best `ridge_residual_s025`, strict-hybrid RMSE **13.2548**, analog-distance stress RMSE **13.5632**; promote to Kaggle package next
- [x] Push and submit XGBoost Ridge residual-stack Kaggle notebook version 2 (`ref=54157692`) - public LB **14.583**, worse than current best **14.472**

## In progress

- [ ] Export artifact figures per `CURSOR_RULES.md`
- [ ] Verify Kaggle competition rules page before first submission (`kaggle/RULES_SUMMARY.md`)
- [ ] Use public LB **14.472** as the current external baseline; XGBoost submission `54129475` is the current best public result
- [x] Build validation mismatch audit after analog-gate public score **14.603** failed to beat Huber + Hilbert **14.514**
- [ ] Follow D014 staged hierarchy plan: residual audit, contextual offsets, contextual Huber hierarchy, then Hilbert residual
- [ ] Investigate why Huber GLM beat GAM locally but scored worse publicly
- [ ] Investigate why local CV ranked Random Forest worse than Ridge while public LB ranked it better
- [ ] Defer SVR/radius-neighbor work unless new well-local features make neighbor geometry plausible

## Next up

1. Keep XGBoost strict inference-safe submission `54129475` / public LB **14.472** as the active benchmark.
2. Do not continue generic residual stacking immediately; the Ridge residual package scored **14.583** publicly despite local strict/stress gains.
3. Do not package the current proxy-formation XGBoost tuning pass; proxy features scored worse than safe base locally.
4. Use the oracle direct-formation gap as a design target for better stratigraphic/typewell features, not as a Kaggle-safe feature set.
5. Defer full hierarchical HSGP until alignment or stronger geologic descriptors give hierarchy something transferable to pool over.
6. Investigate why the Huber + Hilbert residual gain transferred externally while the Huber-only public score lagged GAM.
7. Investigate why the reconstructed spline GAM mean was unstable in `06_hsgp_residual_on_gam_huber.ipynb` despite the public GAM notebook scoring best externally.
8. Refine validation before making many leaderboard submissions from small public-LB gains.
9. Revisit neighbors only with radius constraints, clusters, or richer local geologic features.

## Validation / boosted-tree task ladder

- [x] `notebooks/modeling/09_validation/00_local_public_mismatch_audit.ipynb`: compare local diagnostics, public scores, analog transfer behavior, and promotion decisions.
- [x] `notebooks/modeling/09_validation/01_stricter_cv_fold_design.ipynb`: design cluster/far-analog/high-extrapolation folds that better punish fragile analog gains.
- [x] `notebooks/modeling/02_trees/06_lightgbm_strict_cv.ipynb`: tune LightGBM only after strict folds and feature families are fixed.
- [x] `notebooks/modeling/02_trees/07_xgboost_strict_cv.ipynb`: compare XGBoost against LightGBM under the same strict folds.
- [ ] `notebooks/modeling/02_trees/08_xgboost_inference_safe_tuning.ipynb`: test XGBoost without train-only formation columns and add hidden-test-safe analog/typewell proxies.
- [x] `kaggle/lightgbm_strict_submission/`: package and submit the inference-safe LightGBM strict-CV candidate as an external transfer check.
- [x] `kaggle/xgboost_strict_submission/`: package and submit the inference-safe XGBoost strict-CV candidate as an external transfer check.

## Hierarchical task ladder

- [x] `notebooks/modeling/07_hierarchical/00_residual_bias_audit.ipynb`: export well residual summary, descriptor correlations, fold summary, and markdown report.
- [x] `notebooks/modeling/07_hierarchical/01_empirical_bayes_offsets.ipynb`: test global shrinkage, contextual well intercepts, contextual progress slopes, and residual correction shrinkage.
- [x] `notebooks/modeling/07_hierarchical/02_contextual_hierarchical_huber.ipynb`: fold contextual well effects into the robust signed-delta Huber mean.
- [x] `notebooks/modeling/07_hierarchical/03_hierarchical_residual_slope.ipynb`: model residual trend over `eval_progress` and `distance_log1p` with conservative partial pooling.
- [ ] `notebooks/modeling/07_hierarchical/04_contextual_hierarchy_hilbert_residual.ipynb`: combine the best contextual hierarchy with additive Hilbert residuals over `distance_log1p`, `eval_progress`, and `abs_projected_delta_10`.
- [ ] `notebooks/modeling/07_hierarchical/05_multitask_gp_screen.ipynb`: screen low-rank/coregionalized GP ideas using descriptor-based tasks, not raw well IDs.
- [ ] `notebooks/modeling/07_hierarchical/06_hierarchical_hsgp_full.ipynb`: attempt full hierarchical HSGP only after the simpler hierarchy notebooks prove transferable signal.

## Alignment / geologic feature task ladder

- [x] `notebooks/modeling/08_alignment/00_typewell_alignment_feature_screen.ipynb`: screen fold-safe typewell/analog features against Huber and Huber + Hilbert references.
- [x] `notebooks/modeling/08_alignment/01_multi_analog_zone_blend.ipynb`: test distance-weighted multi-analog transfer and pseudo-zone-specific blending.
- [x] `notebooks/modeling/08_alignment/02_pseudo_zone_huber.ipynb`: test pseudo-stratigraphic zones and zone-specific robust means with nearest-analog delta as a model feature.
- [x] `notebooks/modeling/08_alignment/03_progress_gated_analog.ipynb`: test late-zone-gated nearest-analog corrections and tuned shrinkage schedules.
- [x] `notebooks/modeling/08_alignment/04_spatial_residual_correlation.ipynb`: test whether residuals justify regression kriging or spatial GP residuals.
- [x] `notebooks/modeling/08_alignment/05_analog_selection_refinement.ipynb`: refine nearest analog selection using log-shape, late-zone performance, and descriptor constraints.
- [x] `notebooks/modeling/08_alignment/06_alignment_hsgp_residual.ipynb`: add log-plus-trajectory analog features to the Huber + additive Hilbert residual path.
- [x] `kaggle/huber_hilbert_analog_submission/`: package a self-contained Kaggle candidate with direct analog-gated correction.

## Hierarchical promotion gates

- Continue past residual audit only if well-level residual bias or slope bias is material and partly explainable by test-safe descriptors.
- Promote empirical Bayes offsets only if grouped CV improves by at least **0.05 RMSE** and no fold gets materially worse.
- Package a hierarchy + Hilbert residual Kaggle notebook only if local CV improves by at least **0.10 RMSE** or fixes a known public-generalization issue.
- Every hierarchy notebook should export fold metrics, candidate summary, residual sample, well-level diagnostics, markdown report, model comparison row, and run bundle.

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
| 2026-06-24 | Huber + additive Hilbert + analog gate | 14.6383 diagnostic residual-sample | 14.603 | Submission `53999839`; failed to beat Huber + Hilbert `14.514`, so do not promote |
| 2026-06-27 | LightGBM strict-CV inference-safe | 13.2363 strict-hybrid; 13.4453 analog-distance stress | 14.687 | Submission `54097453`; did not beat Huber + Hilbert `14.514` |
| 2026-06-28 | XGBoost strict-CV inference-safe | 12.7504 strict-hybrid; 12.9116 analog-distance stress | 14.472 | Submission `54129475`; current best public LB |
| 2026-06-29 | XGBoost Ridge residual stack | 13.2548 strict-hybrid; 13.5632 analog-distance stress | 14.583 | Submission `54157692`; failed to beat XGBoost strict inference-safe `14.472` |

## 2026-06-30 Domain Research Pivot

Current public-best baseline remains XGBoost strict inference-safe at **14.472**. The latest research pass and public misses indicate that the next bottleneck is not another generic model class, but hidden-test-safe recovery of stratigraphic state.

Immediate next work:

- Build a stratigraphic-transfer audit notebook before another Kaggle submission.
- Measure typewell/log-alignment quality with DTW/correlation/lag/stretch features using known-zone rows only.
- Add pseudo-top or pseudo-zone uncertainty features rather than hard pseudo-formation labels.
- Re-score XGBoost/LightGBM/Huber candidates on `strict_hybrid_fold`, `analog_distance_fold`, and high-stress slices for poor alignment, high extrapolation, and short known zones.
- Package only if the new feature family clears both global strict-CV and stress-slice promotion gates.

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
| 2026-06-26 | LightGBM strict CV `lgbm_medium_l2` | 13.2363 | Strict sampled CV on `strict_hybrid_fold`; analog-distance stress test RMSE **13.4453**; promising but needs reproducibility/tuning before Kaggle |
| 2026-06-27 | XGBoost strict CV `xgb_pseudohuber_shallow_robust` | 12.7504 | Strict sampled CV on `strict_hybrid_fold`; analog-distance stress test RMSE **12.9116**; strongest local diagnostic so far, but uses train-only formation-distance features |

## Validation artifacts

| Date | Artifact | Recommendation | Notes |
|---|---|---|---|
| 2026-06-26 | `validation__strict_cv_fold_assignments__2026-06-26.csv` | Use `strict_hybrid_fold` for first strict-CV modeling | Row imbalance **1.000**, well imbalance **1.024**; use `analog_distance_fold` as strictest diagnostic stress test |
