# Kaggle Submissions

## Notebook Versions And Submissions

- Timestamp: 2026-06-21
- Run ID: `2026-06-21__1218__linear_ols_ridge_lasso__14bef53`
- Git SHA: `6bebd36`
- Config: self-contained Ridge notebook, `alpha=10.0`, `MAX_TRAIN_ROWS=25_000`
- Local CV: best local linear model `ridge_alpha_10`, CV RMSE mean `15.7660`, CV RMSE std `0.5331`
- Local submission file: `artifacts/reports/submission__linear_ridge_alpha_10__2026-06-21.csv`
- Kaggle notebook: `https://www.kaggle.com/code/sardorbsobirov/wellbore-linear-ridge-baseline`
- Kaggle kernel version: 1
- Status: `KernelWorkerStatus.ERROR`
- Public LB: not submitted
- Notes: Notebook was pushed as a private Kaggle Code version. It errored because the competition data was not found at the initially assumed mount path.
- Decision: dry run / execution check, not a leaderboard-tuning signal.

- Timestamp: 2026-06-21
- Run ID: `2026-06-21__1218__linear_ols_ridge_lasso__14bef53`
- Git SHA: `6bebd36`
- Config: self-contained Ridge notebook, `alpha=10.0`, `MAX_TRAIN_ROWS=25_000`, robust `/kaggle/input` data discovery
- Local CV: best local linear model `ridge_alpha_10`, CV RMSE mean `15.7660`, CV RMSE std `0.5331`
- Local submission file: `artifacts/reports/submission__linear_ridge_alpha_10__2026-06-21.csv`
- Kaggle notebook: `https://www.kaggle.com/code/sardorbsobirov/wellbore-linear-ridge-baseline`
- Kaggle kernel version: 2
- Status: `KernelWorkerStatus.COMPLETE`
- Public LB: `15.905`
- Submission ref: `53923555`
- Submission status: `SubmissionStatus.COMPLETE`
- Notes: Fixes input path discovery by scanning `/kaggle/input` for `sample_submission.csv` with sibling `train/` and `test/`.
- Decision: first valid leaderboard baseline. Public LB is close to local CV (`15.905` vs `15.7660`), so the grouped local validation is directionally plausible but slightly optimistic.

- Timestamp: 2026-06-21
- Run ID: `2026-06-21__1310__tree_random_forest__4eb4a55`
- Git SHA: `264168e`
- Config: self-contained Random Forest notebook, `n_estimators=60`, `max_depth=18`, `min_samples_leaf=8`, `max_features=0.85`, `MAX_TRAIN_ROWS=60_000`, robust `/kaggle/input` data discovery
- Local CV: Random Forest CV RMSE mean `16.0655`, CV RMSE std `0.5282`
- Local submission file: `artifacts/reports/submission__random_forest__2026-06-21.csv`
- Kaggle notebook: `https://www.kaggle.com/code/sardorbsobirov/wellbore-random-forest-baseline`
- Kaggle kernel version: 1
- Status: `KernelWorkerStatus.COMPLETE`
- Public LB: `15.873`
- Submission ref: `53924036`
- Submission status: `SubmissionStatus.COMPLETE`
- Notes: Local CV ranked this behind Ridge, but public LB improved over the Ridge submission (`15.873` vs `15.905`). This is a validation-warning signal, not proof that the forest will generalize better privately.
- Decision: current best public LB. Keep Random Forest as the best external baseline while investigating why local grouped CV under-ranked it.

- Timestamp: 2026-06-21
- Run ID: `2026-06-21__1412__linear_elasticnet_huber_quantile__448574b`
- Git SHA: `cbadf04`
- Config: self-contained ElasticNet notebook, `alpha=0.001`, `l1_ratio=0.2`, `MAX_TRAIN_ROWS=100_000`, robust `/kaggle/input` data discovery
- Local CV: best robust-linear candidate `elasticnet_alpha_0p001_l1_0p2`, CV RMSE mean `15.7907`, CV RMSE std `0.5575`
- Local submission file: `artifacts/reports/submission__elasticnet_alpha_0p001_l1_0p2__2026-06-21.csv`
- Kaggle notebook: `https://www.kaggle.com/code/sardorbsobirov/wellbore-elasticnet-baseline`
- Kaggle kernel version: 1
- Status: `KernelWorkerStatus.COMPLETE`
- Public LB: `16.091`
- Submission ref: `53924934`
- Submission status: `SubmissionStatus.COMPLETE`
- Notes: External score is worse than both Ridge (`15.905`) and Random Forest (`15.873`), despite local CV being close to Ridge.
- Decision: rejected as a promotion candidate; useful evidence that this ElasticNet variant does not generalize better to the public test wells.

- Timestamp: 2026-06-22
- Run ID: `2026-06-21__2037__gam_spline_tuning__4e31e0e`
- Git SHA: `7c13dfa`
- Config: self-contained fine-tuned GAM-style spline notebook, slope-core boundary/trajectory/recent-slope features, `SplineTransformer(n_knots=7, degree=3)`, `Ridge(alpha=60.0)`, `MAX_TARGET_ROWS_PER_WELL=140`, robust `/kaggle/input` data discovery
- Local CV: tuned GAM candidate `core_fine__delta_boundary__k7__d3__a60`, sampled target-zone grouped CV RMSE mean `14.8687`, CV RMSE std `0.5012`
- Local submission file: `artifacts/reports/submission__gam_spline_tuning__2026-06-21.csv`
- Kaggle notebook: `https://www.kaggle.com/code/sardorbsobirov/wellbore-fine-tuned-gam-spline`
- Kaggle kernel version: 1
- Status: `KernelWorkerStatus.COMPLETE`
- Public LB: `15.055`
- Submission ref: `53932883`
- Submission status: `SubmissionStatus.COMPLETE`
- Notes: First self-contained target-zone-trained GAM submission. The public score improves substantially over Random Forest (`15.873`) and Ridge (`15.905`), even though Phase 2 tuning only improved the first GAM local diagnostic by `0.0032` RMSE.
- Decision: current best public LB. Promote slope-core boundary/trajectory/recent-slope feature logic as the next modeling baseline for LightGBM/XGBoost and ensemble work.

Use this format for every submission:

```text
- Timestamp:
- Run ID:
- Git SHA:
- Config:
- Local CV:
- Public LB:
- Submission file:
- Kaggle notebook:
- Notes:
- Decision:
```
