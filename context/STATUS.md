# Project Status

**Last updated:** 2026-06-21
**Current phase:** Feature engineering loop - Phase 3 feature-signal EDA complete

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

## In progress

- [ ] Richer well-relative feature builder for boosted-tree and validation experiments
- [ ] Export artifact figures per `CURSOR_RULES.md`
- [ ] Verify Kaggle competition rules page before first submission (`kaggle/RULES_SUMMARY.md`)
- [ ] Use public LB **15.873** as the current external baseline; avoid leaderboard-only tuning
- [ ] Investigate why local CV ranked Random Forest worse than Ridge while public LB ranked it better
- [ ] Defer SVR/radius-neighbor work unless new well-local features make neighbor geometry plausible

## Next up

1. Implement a richer well-relative feature builder based on Phase 3: endpoint hold, boundary distance, target-zone progress, trajectory, recent slopes, GR availability, and cautious typewell summaries.
2. Phase 4 validation review: inspect whether held-out folds resemble the scoring setup and explain the Ridge vs Random Forest public/local disagreement.
3. Try boosted-tree baseline (XGBoost or LightGBM) on richer well-relative features, then tune.
4. Refine validation before making many leaderboard submissions from small public-LB gains
5. Revisit neighbors only with radius constraints, clusters, or richer local geologic features
6. Set up Bayesian GP (NumpPyro) once the feature/split baseline is more stable

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
