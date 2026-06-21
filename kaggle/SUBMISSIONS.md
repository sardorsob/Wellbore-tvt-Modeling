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
