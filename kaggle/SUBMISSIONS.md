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

- Timestamp: 2026-06-22
- Run ID: `2026-06-21__2147__glm_distribution_diagnostics__107a7d7`
- Git SHA: `023c66b`
- Config: self-contained Huber GLM notebook, slope-core boundary/trajectory/recent-slope features, `HuberRegressor(alpha=0.0001, epsilon=1.35, max_iter=1000)`, `MAX_TARGET_ROWS_PER_WELL=140`, robust `/kaggle/input` data discovery
- Local CV: GLM diagnostic candidate `robust_delta_huber`, sampled target-zone grouped CV RMSE mean `14.7928`, CV RMSE std `0.4716`
- Local submission file: `artifacts/reports/submission__glm_huber__2026-06-22.csv`
- Kaggle notebook: `https://www.kaggle.com/code/sardorbsobirov/wellbore-huber-glm-slope-core`
- Kaggle kernel version: 1
- Status: `KernelWorkerStatus.COMPLETE`
- Public LB: `15.464`
- Submission ref: `53952950`
- Submission status: `SubmissionStatus.COMPLETE`
- Notes: Public score beats Ridge, Random Forest, and ElasticNet, but is worse than the fine-tuned GAM spline (`15.055`) despite stronger local diagnostic RMSE.
- Decision: useful external check for robust signed-delta modeling, but not the current public-LB leader. Keep fine-tuned GAM as the external baseline and investigate local/public mismatch before promoting Huber.

- Timestamp: 2026-06-22
- Run ID: `2026-06-22__1231__bayesian_linear_tuning__4b66b91`
- Git SHA: `f3d4b83`
- Config: self-contained Bayesian linear Student-t MAP notebook, slope-core boundary/trajectory/recent-slope features, fixed `nu=3`, Student-t MAP objective with Normal priors, `MAX_TARGET_ROWS_PER_WELL=140`, robust `/kaggle/input` data discovery
- Local CV: tuned Bayesian candidate `student_t_fixed3_autodelta`, sampled target-zone grouped CV RMSE mean `14.8449`, CV RMSE std `0.5205`
- Local submission file: `artifacts/reports/submission__bayesian_linear_student_t__2026-06-22.csv`
- Kaggle notebook: `https://www.kaggle.com/code/sardorbsobirov/wellbore-bayesian-linear-student-t`
- Kaggle kernel version: 1
- Status: `KernelWorkerStatus.COMPLETE`
- Public LB: `15.512`
- Submission ref: `53957370`
- Submission status: `SubmissionStatus.COMPLETE`
- Notes: External score is slightly worse than Huber GLM (`15.464`) and much worse than fine-tuned GAM (`15.055`). The Kaggle notebook uses a SciPy Student-t MAP implementation to avoid NumPyro dependency risk with internet disabled.
- Decision: keep as the first valid Bayesian-linear public baseline and likelihood check, but do not promote it over GAM/Huber. Continue to HSGP for Bayesian residual mean structure.

- Timestamp: 2026-06-22
- Run ID: `2026-06-22__1425__bayesian_hsgp_additive_tuning__48601fe`
- Git SHA: `1cffe9a`
- Config: self-contained additive HSGP Student-t MAP notebook, slope-core boundary/trajectory/recent-slope features plus additive Hilbert sine bases over `distance_log1p`, `eval_progress`, and `abs_projected_delta_10`; `HSGP_M=8`, `HSGP_ELL=3.0`, `MAX_TARGET_ROWS_PER_WELL=140`, robust `/kaggle/input` data discovery
- Local CV: additive HSGP candidate `additive_dps_m8_se_autodelta`, sampled target-zone grouped CV RMSE mean `14.8235`, CV RMSE std `0.5186`
- Local submission file: `artifacts/reports/submission__hsgp_additive_student_t__2026-06-22.csv`
- Kaggle notebook: `https://www.kaggle.com/code/sardorbsobirov/wellbore-additive-hsgp-student-t`
- Kaggle kernel version: 1
- Status: `KernelWorkerStatus.COMPLETE`
- Public LB: `15.255`
- Submission ref: `53958582`
- Submission status: `SubmissionStatus.COMPLETE`
- Notes: The Kaggle-safe additive HSGP approximation improves over Bayesian linear (`15.512`) and Huber GLM (`15.464`) externally, but remains worse than the fine-tuned GAM spline (`15.055`).
- Decision: useful public validation that additive HSGP residual geometry helps. Do not promote over the fine-tuned GAM; continue planning Huber/GAM residual HSGP locally.

- Timestamp: 2026-06-22
- Run ID: `2026-06-22__1519__bayesian_hsgp_residual_on_gam_huber__fadbdba`
- Git SHA: `029feb8`
- Config: self-contained Huber + additive Hilbert residual notebook, expanded slope-core features with raw trajectory terms, `HuberRegressor(alpha=0.0001, epsilon=1.35)`, additive Hilbert sine bases over `distance_log1p`, `eval_progress`, and `abs_projected_delta_10`; `HSGP_M=8`, `HSGP_ELL=3.0`, `MAX_TARGET_ROWS_PER_WELL=140`, robust `/kaggle/input` data discovery
- Local CV: diagnostic candidate `huber_plus_hsgp_dps_m8`, sampled target-zone grouped CV RMSE mean `14.3586`, CV RMSE std `0.5713`
- Local submission file: `artifacts/reports/submission__huber_hsgp_residual__2026-06-22.csv`
- Kaggle notebook: `https://www.kaggle.com/code/sardorbsobirov/wellbore-huber-hsgp-residual`
- Kaggle kernel version: 1
- Status: `KernelWorkerStatus.COMPLETE`
- Public LB: `14.514`
- Submission ref: `53960276`
- Submission status: `SubmissionStatus.COMPLETE`
- Notes: The Huber + additive Hilbert residual package improves the public LB from the fine-tuned GAM's **15.055** to **14.514**, and beats the prior additive HSGP score **15.255**. The local-to-public ordering is still imperfect, but this is the first meaningful external gain since GAM.
- Decision: promote as the current best public submission. Continue to hierarchical/well-level structure because the score is still far from the top public range discussed by the user.

- Timestamp: 2026-06-23
- Run ID: `2026-06-23__alignment_hsgp_residual`
- Git SHA: package commit containing this entry
- Config: self-contained Huber + additive Hilbert residual notebook plus direct log-plus-trajectory nearest-analog gate; `HuberRegressor(alpha=0.0001, epsilon=1.35)`, additive Hilbert sine bases over `distance_log1p`, `eval_progress`, and `abs_projected_delta_10`, analog descriptors from recent GR shape/typewell lag/TVT span/XY span/vertical drop, progress-gated analog shrink schedule `z0=0.025`, `z1=0.025`, `z2=0.075`, `z3=0.100`
- Local CV: residual-sample diagnostic candidate `direct_analog_gate`, sampled RMSE `14.6383` vs same-sample Huber + Hilbert baseline `14.7385`; note this is not directly comparable to the full local reference `14.3586`
- Local submission file: `artifacts/reports/submission__huber_hilbert_analog__2026-06-23.csv`
- Kaggle notebook: `sardorbsobirov/wellbore-huber-hilbert-analog`
- Kaggle kernel version: 1
- Status: `KernelWorkerStatus.COMPLETE`
- Public LB: `14.603`
- Submission ref: `53999839`
- Submission status: `SubmissionStatus.COMPLETE`
- Notes: This packages the first above-gate alignment idea into a rerunnable code-competition notebook. The analog layer is recomputed inside the notebook from training wells only and writes `/kaggle/working/submission.csv` on Kaggle. Kaggle version 1 completed and produced **14,151** prediction rows.
- Decision: do not promote over Huber + additive Hilbert residual (`14.514`). Treat the result as validation-mismatch evidence because the bounded local residual-sample gain did not transfer publicly.

- Timestamp: 2026-06-27
- Run ID: `2026-06-26__lightgbm_strict_cv`
- Git SHA: package commit containing this entry
- Config: self-contained LightGBM strict-CV package, `objective="huber"`, `n_estimators=1200`, `learning_rate=0.03`, `num_leaves=63`, `max_depth=10`, `reg_lambda=2.0`, `MAX_TARGET_ROWS_PER_WELL=220`; inference-safe row geometry and descriptor features only, with local-only formation columns excluded for Kaggle hidden-test compatibility.
- Local CV: strict-CV candidate `lgbm_medium_l2`, `strict_hybrid_fold` RMSE **13.2363**, analog-distance stress RMSE **13.4453**; same-row last-known hold RMSE about **15.58**.
- Local submission file: Kaggle kernel output `/kaggle/working/submission.csv`; local validation output `kaggle/lightgbm_strict_submission/artifacts/reports/submission__lightgbm_strict__2026-06-26.csv`.
- Kaggle notebook: `https://www.kaggle.com/code/sardorbsobirov/wellbore-lightgbm-strict-cv`
- Kaggle kernel version: 1
- Status: `KernelWorkerStatus.COMPLETE`
- Public LB: `14.687`
- Submission ref: `54097453`
- Submission status: `SubmissionStatus.COMPLETE`
- Notes: Kaggle version 1 completed successfully, used `/kaggle/input/competitions/rogii-wellbore-geology-prediction`, trained on **170,060** sampled rows with **31** inference-safe features, and produced **14,151** prediction rows with no missing `tvt` values.
- Decision: do not promote over Huber + additive Hilbert public LB **14.514**. The inference-safe LightGBM package scored worse than the current baseline, which confirms that the full strict-CV tree result does not transfer automatically after removing local-only formation-distance features.

- Timestamp: 2026-06-28
- Run ID: `2026-06-27__xgboost_strict_cv`
- Git SHA: package commit containing this entry
- Config: self-contained XGBoost strict-CV package, `objective="reg:pseudohubererror"`, `n_estimators=800`, `learning_rate=0.04`, `max_depth=4`, `min_child_weight=120`, `reg_lambda=8.0`, `MAX_TARGET_ROWS_PER_WELL=220`; inference-safe row geometry and descriptor features only, with local-only formation columns excluded for Kaggle hidden-test compatibility.
- Local CV: full-feature strict-CV candidate `xgb_pseudohuber_shallow_robust`, `strict_hybrid_fold` RMSE **12.7504**, analog-distance stress RMSE **12.9116**; this Kaggle package is the inference-safe feature subset version.
- Local submission file: Kaggle kernel output `/kaggle/working/submission.csv`; local validation output `kaggle/xgboost_strict_submission/artifacts/reports/submission__xgboost_strict__2026-06-27.csv`.
- Kaggle notebook: `https://www.kaggle.com/code/sardorbsobirov/wellbore-xgboost-strict-cv`
- Kaggle kernel version: 1
- Status: `KernelWorkerStatus.COMPLETE`
- Public LB: `14.472`
- Submission ref: `54129475`
- Submission status: `SubmissionStatus.COMPLETE`
- Notes: Kaggle version 1 completed successfully, used `/kaggle/input/competitions/rogii-wellbore-geology-prediction`, trained on **170,060** sampled rows with **31** inference-safe features, and produced **14,151** prediction rows with no missing `tvt` values.
- Decision: promote as the current best public submission. It improves over Huber + additive Hilbert **14.514** despite using only inference-safe features, so XGBoost pseudo-Huber tree tuning is the next priority.

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
