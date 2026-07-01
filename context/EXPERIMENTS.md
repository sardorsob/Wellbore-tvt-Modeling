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

## 2026-06-21__2037__gam_spline_tuning__4e31e0e

- Task: tuned GAM-style spline sweep
- Purpose: Improve the first GAM candidate before deciding whether to spend a Kaggle submission attempt on this model family.
- Config: notebook-local config in `notebooks/modeling/04_gam/02_gam_spline_tuning.ipynb`
- Data version: local Kaggle download in `data/raw/train/`
- Split / CV: 3 grouped well folds on sampled target-zone rows; deterministic cap of 140 target rows per well and 80,000 training rows per fold.
- Model: best candidate `core_fine__delta_boundary__k7__d3__a60`, implemented as sklearn `SplineTransformer` plus Ridge on slope-core boundary/trajectory/recent-slope features.
- Primary metric: sampled target-zone CV RMSE mean **14.8687**; RMSE std **0.5012**.
- Secondary metrics: MAE mean **10.4828**; mean R2 **0.999423**. Sampled endpoint-hold RMSE **15.5782**; improvement over first GAM baseline **0.0032** RMSE.
- Artifacts: fold metrics CSV, candidate summary CSV, feature-family and target summaries, residual sample, residual/RMSE figures, report, model comparison row, and run bundle under `artifacts/logs/runs/2026-06-21__2037__gam_spline_tuning__4e31e0e/`
- Decision: keep as the current best GAM candidate, but treat the tiny gain over Phase 1 as a plateau signal. Submit only if we want a GAM-specific public-LB read; otherwise proceed to LightGBM/XGBoost with the same slope-core features.
- Notes: GR and typewell feature-family additions under this additive Ridge-spline setup did not beat the slope-core family.

## 2026-06-21__2147__glm_distribution_diagnostics__107a7d7

- Task: GLM distribution diagnostics
- Purpose: Test whether a general linear model with a different response distribution improves the slope-core target-zone diagnostic before Bayesian modeling.
- Config: notebook-local config in `notebooks/modeling/01_linear/04_glm_distribution_diagnostics.ipynb`
- Data version: local Kaggle download in `data/raw/train/`
- Split / CV: 3 grouped well folds on sampled target-zone rows; deterministic cap of 140 target rows per well and 60,000 training rows per fold.
- Model: best candidate `robust_delta_huber`, a Huber robust linear model on signed `delta_tvt_from_boundary` using slope-core boundary/trajectory/recent-slope features.
- Primary metric: sampled target-zone CV RMSE mean **14.7928**; RMSE std **0.4716**.
- Secondary metrics: Gaussian Ridge on the same delta target scored **14.8398**; median Quantile scored **14.8842**; endpoint hold scored **15.5782**. Raw positive-response Gamma/Tweedie models were much worse.
- Artifacts: fold metrics CSV, summary CSV, candidate table, family-validity table, target diagnostics, residual sample, diagnostic figures, report, model comparison row, and run bundle under `artifacts/logs/runs/2026-06-21__2147__glm_distribution_diagnostics__107a7d7/`
- Decision: treat robust signed-delta GLM as a promising next Kaggle candidate and as evidence for Student-t/robust Gaussian likelihoods in later Bayesian models.
- Notes: Beta and Dirichlet were logged as invalid target geometries rather than fit, because the competition predicts one scalar depth response per row.

## 2026-06-22__1211__bayesian_linear_robust__3855aa6

- Task: Bayesian linear robust baseline
- Purpose: Establish a NumPyro/JAX Bayesian linear baseline with posterior predictive uncertainty before moving to Hilbert-space Gaussian process residual models.
- Config: notebook-local config in `notebooks/modeling/06_bayesian/01_bayesian_linear_robust.ipynb`
- Data version: local Kaggle download in `data/raw/train/`
- Split / CV: 3 grouped well folds on sampled target-zone rows; deterministic cap of 140 target rows per well and 60,000 training rows per fold.
- Model: best candidate `bayes_student_t_heteroskedastic`, a Student-t Bayesian linear model with row-level scale depending on boundary distance, evaluation progress, horizontal distance, and normalized MD.
- Primary metric: sampled target-zone CV RMSE mean **14.9752**; RMSE std **0.5352**.
- Secondary metrics: empirical 80% interval coverage **0.783** with mean width **32.72** TVT units; empirical 95% coverage **0.933** with mean width **57.79** TVT units. Endpoint hold scored **15.5782**; Huber GLM local reference scored **14.7928**.
- Artifacts: fold metrics CSV, summary CSV, posterior diagnostics, residual sample, diagnostic figures, report, model comparison row, and run bundle under `artifacts/logs/runs/2026-06-22__1211__bayesian_linear_robust__3855aa6/`
- Decision: keep as the first Bayesian uncertainty baseline, but do not package for Kaggle yet because posterior-mean RMSE trails Huber GLM and tuned GAM diagnostics.
- Notes: The heteroskedastic Student-t model improves uncertainty calibration and is a useful prior/likelihood reference before HSGP.

## 2026-06-22__1231__bayesian_linear_tuning__4b66b91

- Task: Bayesian linear tuning
- Purpose: Test whether likelihood-tail, guide, and heteroskedastic-scale choices can improve the first NumPyro/JAX Bayesian linear baseline before moving to HSGP residual structure.
- Config: notebook-local config in `notebooks/modeling/06_bayesian/02_bayesian_linear_tuning.ipynb`
- Data version: local Kaggle download in `data/raw/train/`
- Split / CV: 3 grouped well folds on sampled target-zone rows; deterministic cap of 140 target rows per well and 60,000 training rows per fold.
- Model: best candidate `student_t_fixed3_autodelta`, a fixed-nu Student-t Bayesian linear model fit with AutoDelta/MAP-style variational inference on slope-core boundary/trajectory/recent-slope features.
- Primary metric: sampled target-zone CV RMSE mean **14.8449**; RMSE std **0.5205**.
- Secondary metrics: empirical 80% interval coverage **0.769** with mean width **31.12** TVT units. The tuned Bayesian best improves the first Bayesian baseline by **0.1303** RMSE but still trails the Huber GLM local reference by **0.0521** RMSE.
- Artifacts: fold metrics CSV, summary CSV, posterior diagnostics, residual sample, diagnostic figures, report, model comparison row, and run bundle under `artifacts/logs/runs/2026-06-22__1231__bayesian_linear_tuning__4b66b91/`
- Decision: keep as the best Bayesian linear point-RMSE baseline, but move next to HSGP or other structured residual models instead of further linear-only tuning.
- Notes: Fixed-tail Student-t with AutoDelta beat AutoNormal and heteroskedastic variants for point RMSE in this sweep, while still providing posterior predictive interval diagnostics.

## 2026-06-22__1254__bayesian_posterior_diagnostics_scale__6644c20

- Task: Bayesian posterior predictive diagnostics and targeted scale modeling
- Purpose: Diagnose residual bias, coverage, posterior SD/error alignment, and whether a focused heteroskedastic Student-t scale model improves the tuned Bayesian linear baseline before HSGP.
- Config: notebook-local config in `notebooks/modeling/06_bayesian/03_bayesian_posterior_diagnostics_scale.ipynb`
- Data version: local Kaggle download in `data/raw/train/`
- Split / CV: 3 grouped well folds on sampled target-zone rows; deterministic cap of 140 target rows per well and 60,000 training rows per fold.
- Model: best candidate `hetero_boundary_progress_autodelta`, a fixed-nu Student-t Bayesian linear model with boundary/progress-driven row-level scale fit by AutoDelta.
- Primary metric: sampled target-zone CV RMSE mean **14.8626**; RMSE std **0.5019**.
- Secondary metrics: empirical 80% interval coverage **0.780** with mean width **31.95** TVT units; posterior SD vs absolute error Spearman **0.315**; 90th percentile absolute well-level mean residual **18.60** TVT units.
- Artifacts: fold metrics CSV, summary CSV, posterior diagnostics, residual sample, bin diagnostics, well diagnostics, diagnostic figures, report, model comparison row, and run bundle under `artifacts/logs/runs/2026-06-22__1254__bayesian_posterior_diagnostics_scale__6644c20/`
- Decision: use the diagnostics to design HSGP next. Do not spend more notebook cycles on standalone Bayesian-linear heteroskedastic scale tuning unless HSGP results show a specific scale failure.
- Notes: Boundary/progress scale improves the same-notebook fixed Student-t reference by **0.0503** RMSE, but remains **0.0176** worse than the prior tuned Bayesian best and **0.0697** worse than Huber GLM locally.

## 2026-06-22__1358__bayesian_hsgp_residual_baseline__bd24319

- Task: Bayesian HSGP residual baseline
- Purpose: Test whether a Hilbert-space Gaussian process residual over boundary/progress geometry improves the slope-core Bayesian linear Student-t baseline before trying richer GP or hierarchical models.
- Config: notebook-local config in `notebooks/modeling/06_bayesian/04_hsgp_residual_baseline.ipynb`
- Data version: local Kaggle download in `data/raw/train/`
- Split / CV: 3 grouped well folds on sampled target-zone rows; deterministic cap of 140 target rows per well and 50,000 training rows per fold.
- Model: best candidate `hsgp_bp_m5_autodelta`, a fixed-nu Student-t Bayesian linear model with a squared-exponential HSGP residual over standardized `distance_log1p` and `eval_progress`, using 25 basis functions.
- Primary metric: sampled target-zone CV RMSE mean **14.8997**; RMSE std **0.5109**.
- Secondary metrics: empirical 80% interval coverage **0.767** with mean width **31.10** TVT units; HSGP alpha mean **0.770**, length mean **0.802**, residual-function SD mean **1.995** TVT units. The best HSGP improved the same-notebook linear reference by **0.0310** RMSE but trailed the prior tuned Bayesian best by **0.0547** RMSE.
- Artifacts: fold metrics CSV, summary CSV, posterior diagnostics, residual sample, diagnostic figures, report, model comparison row, and run bundle under `artifacts/logs/runs/2026-06-22__1358__bayesian_hsgp_residual_baseline__bd24319/`
- Decision: keep as the first HSGP baseline, but do not submit to Kaggle. Next GP work should test additive/coordinate-specific HSGP variants rather than only larger basis counts.
- Notes: A 49-basis 2D HSGP and a 48-basis 3D boundary/progress/horizontal HSGP both underperformed the smaller 25-basis boundary/progress model.

## 2026-06-22__1425__bayesian_hsgp_additive_tuning__48601fe

- Task: Bayesian additive HSGP residual tuning
- Purpose: Test additive one-dimensional HSGP residual terms over boundary distance, evaluation progress, trajectory distance, and slope-instability coordinates before moving to Huber/GAM residual HSGP.
- Config: notebook-local config in `notebooks/modeling/06_bayesian/05_hsgp_additive_tuning.ipynb`
- Data version: local Kaggle download in `data/raw/train/`
- Split / CV: 3 grouped well folds on sampled target-zone rows; deterministic cap of 140 target rows per well and 50,000 training rows per fold.
- Model: best candidate `additive_dps_m8_se_autodelta`, a fixed-nu Student-t Bayesian linear model with additive squared-exponential HSGP residuals over standardized `distance_log1p`, `eval_progress`, and `abs_projected_delta_10`, using 24 total one-dimensional basis functions.
- Primary metric: sampled target-zone CV RMSE mean **14.8235**; RMSE std **0.5186**.
- Secondary metrics: empirical 80% interval coverage **0.765** with mean width **30.97** TVT units; the best additive HSGP improved the same-notebook linear reference by **0.0698** RMSE and the first joint HSGP by **0.0762** RMSE, but still trailed the Huber GLM local reference by **0.0307** RMSE.
- Artifacts: fold metrics CSV, summary CSV, posterior diagnostics, residual sample, diagnostic figures, report, model comparison row, and run bundle under `artifacts/logs/runs/2026-06-22__1425__bayesian_hsgp_additive_tuning__48601fe/`
- Decision: keep as the best HSGP result so far, but do not submit to Kaggle yet. Proceed to Huber/GAM residual HSGP because the additive residual geometry helps while the mean model still appears limiting.
- Notes: Early additive SVI trials produced non-finite predictions; the stable run used lower-step clipped Adam and removed the unstable AutoNormal candidate from this first sweep.

## 2026-06-22__1519__bayesian_hsgp_residual_on_gam_huber__fadbdba

- Task: HSGP residuals on GAM and Huber means
- Purpose: Test whether additive Hilbert-space residual corrections improve the current public-leading GAM direction and the strongest local robust Huber direction.
- Config: notebook-local config in `notebooks/modeling/06_bayesian/06_hsgp_residual_on_gam_huber.ipynb`
- Data version: local Kaggle download in `data/raw/train/`
- Split / CV: 3 grouped well folds from the shared `np.array_split` protocol on sampled target-zone rows; deterministic cap of 140 target rows per well and 80,000 training rows per fold.
- Model: best candidate `huber_plus_hsgp_dps_m8`, a Huber signed-delta mean model plus Student-t MAP additive Hilbert residual terms over `distance_log1p`, `eval_progress`, and `abs_projected_delta_10`.
- Primary metric: sampled target-zone CV RMSE mean **14.3586**; RMSE std **0.5713**.
- Secondary metrics: Huber mean-only scored **14.4172**, so the additive residual HSGP improved the same-notebook Huber reference by **0.0585** RMSE. The reconstructed GAM mean was unstable in this notebook and was not promoted.
- Artifacts: fold metrics CSV, summary CSV, residual diagnostics, residual sample, diagnostic figures, report, model comparison row, and run bundle under `artifacts/logs/runs/2026-06-22__1519__bayesian_hsgp_residual_on_gam_huber__fadbdba/`
- Decision: package a Kaggle-safe Huber + additive Hilbert residual notebook next if we want a public-LB read. Do not package the GAM-residual variant without first reproducing the exact public-GAM behavior in this notebook.
- Notes: During development, an incorrect same-fold train/validation split and a missing trajectory-feature subset produced misleading intermediate results; both were corrected before recording the final metrics.

## Planned - 07_hierarchical modeling track

- Task: staged hierarchical modeling
- Purpose: Test whether persistent residual structure from the current Huber + additive Hilbert residual model is transferable through contextual well-level partial pooling.
- Config: notebook-local configs under `notebooks/modeling/07_hierarchical/`
- Data version: local Kaggle download in `data/raw/train/`; submission notebooks must remain self-contained and rerunnable on hidden test wells.
- Split / CV: grouped well folds with endpoint-hold baseline, fold diagnostics, and descriptor summaries.
- Model: start with residual-bias audit and empirical Bayes/contextual offsets; defer full hierarchical HSGP until simpler hierarchy proves useful.
- Primary metric: grouped target-zone RMSE against current local diagnostic reference **14.3586** and public LB reference **14.514**.
- Secondary metrics: fold RMSE spread, well-level signed residual bias, residual trend over `eval_progress`, descriptor correlation with residual bias, and public/generalization rationale.
- Artifacts: each notebook exports fold metrics, candidate summary, residual sample, well-level diagnostics, markdown report, model comparison row, and run bundle.
- Decision: no Kaggle submission until local diagnostics show a meaningful gain or a clear public-generalization hypothesis.
- Notes: Hierarchy should use test-safe descriptors rather than raw well-ID effects, because hidden Kaggle wells may be unseen.

## 2026-06-22__hierarchical_residual_bias_audit

- Task: hierarchical residual-bias audit
- Purpose: Diagnose whether the current Huber + additive Hilbert residual model leaves well-level residual structure that justifies contextual partial pooling.
- Config: notebook-local config in `notebooks/modeling/07_hierarchical/00_residual_bias_audit.ipynb`
- Data version: local Kaggle download in `data/raw/train/`, plus sampled out-of-fold residuals from `model__hsgp_residual_on_gam_huber_residual_sample__2026-06-22.csv`
- Split / CV: inherited sampled grouped well-fold residuals from `huber_plus_hsgp_dps_m8`; diagnostic-only, no refit and no submission.
- Model: current best Huber + additive Hilbert residual candidate audited for well-level signed bias and descriptor correlations.
- Primary metric: median absolute well mean residual **6.890** TVT units; p90 **18.929**; max **56.454** across **773** wells.
- Secondary metrics: strongest descriptor signal is max boundary distance vs. MAE Spearman **0.175**; fold mean residuals are **-0.052**, **0.408**, and **0.450**.
- Artifacts: well summary, descriptor correlations, fold summary, findings, comparison row, markdown report, and run bundle under `artifacts/logs/runs/2026-06-22__hierarchical_residual_bias_audit/`
- Decision: continue to `01_empirical_bayes_offsets.ipynb`, but make the first correction small, contextual, and shrinkage-controlled.
- Notes: Descriptor signal is real but weak; this supports testing contextual offsets, not raw well-ID random effects.

## 2026-06-23__hierarchical_empirical_bayes_offsets

- Task: hierarchical empirical Bayes offset screen
- Purpose: Test whether conservative contextual residual offsets improve the current Huber + additive Hilbert residual predictions.
- Config: notebook-local config in `notebooks/modeling/07_hierarchical/01_empirical_bayes_offsets.ipynb`
- Data version: local Kaggle download in `data/raw/train/`, plus sampled out-of-fold residuals and Stage 1 well descriptors.
- Split / CV: fold-held-out residual sample; each offset arm trains only on other folds and scores the held-out fold.
- Model: best candidate `global_bias_shrink_050`, a 50% shrunk global residual correction. Descriptor-based well intercept, progress-slope, robust intercept, and row-context Ridge arms were also tested.
- Primary metric: sampled RMSE **14.7378**, only **0.0008** better than the same-sample no-offset baseline (**14.7385**).
- Secondary metrics: descriptor-based arms worsened sampled RMSE by **0.0069** to **0.0219**; median absolute well-bias delta for the best arm was **-0.0891**.
- Artifacts: fold metrics, summary, residual sample, well diagnostics, findings, comparison row, markdown report, and run bundle under `artifacts/logs/runs/2026-06-23__hierarchical_empirical_bayes_offsets/`
- Decision: do not promote empirical Bayes contextual offsets to Kaggle or to contextual Huber hierarchy yet.
- Notes: The full local reference RMSE **14.3586** comes from the prior model's broader diagnostic, while this screen uses the bounded residual sample whose no-offset RMSE is **14.7385**. Interpret deltas against the same-sample baseline.

## 2026-06-23__hierarchical_residual_shape_screen

- Task: hierarchical residual-shape screen
- Purpose: Test whether row/progress/boundary/slope-instability residual-shape corrections improve the current Huber + additive Hilbert residual predictions.
- Config: notebook-local config in `notebooks/modeling/07_hierarchical/03_hierarchical_residual_slope.ipynb`
- Data version: local Kaggle download in `data/raw/train/`, plus sampled out-of-fold residuals from the current best model.
- Split / CV: fold-held-out residual sample; each correction trains only on other folds and scores the held-out fold.
- Model: best candidate `row_linear_ridge_a100_s025`, a regularized linear residual-shape correction over `eval_progress`, `distance_log1p`, and `abs_projected_delta_10` with 75% shrinkage to zero.
- Primary metric: sampled RMSE **14.7382**, only **0.0003** better than the same-sample no-shape baseline (**14.7385**).
- Secondary metrics: binned progress/distance/slope-instability corrections and spline/polynomial variants did not clear the hierarchy gate; median absolute well-bias delta for the best arm was **-0.0065**.
- Artifacts: fold metrics, summary, residual sample, well diagnostics, findings, comparison row, markdown report, and run bundle under `artifacts/logs/runs/2026-06-23__hierarchical_residual_shape_screen/`
- Decision: do not promote residual-shape corrections to Kaggle or full hierarchy.
- Notes: This is another useful negative result: shallow post-hoc residual corrections are not producing meaningful same-sample RMSE gains.

## 2026-06-23__contextual_hierarchical_huber_screen

- Task: contextual hierarchical Huber interaction screen
- Purpose: Test whether test-safe well descriptors can modulate row-level residual shape through regularized contextual interactions.
- Config: notebook-local config in `notebooks/modeling/07_hierarchical/02_contextual_hierarchical_huber.ipynb`
- Data version: local Kaggle download in `data/raw/train/`, plus sampled out-of-fold residuals and Stage 1 well descriptors.
- Split / CV: fold-held-out residual sample; each contextual correction trains only on other folds and scores the held-out fold.
- Model: best candidate `context_poly2_ridge_a2000_s025`, a quadratic contextual residual surface over row coordinates and well descriptors with strong Ridge regularization and 75% shrinkage to zero.
- Primary metric: sampled RMSE **14.7322**, **0.0064** better than the same-sample no-context baseline (**14.7385**).
- Secondary metrics: descriptor main effects, linear descriptor-by-row interactions, and robust contextual Huber variants did not clear the hierarchy gate; median absolute well-bias delta for the best arm was **0.0143**.
- Artifacts: fold metrics, summary, residual sample, well diagnostics, findings, comparison row, markdown report, and run bundle under `artifacts/logs/runs/2026-06-23__contextual_hierarchical_huber_screen/`
- Decision: do not promote contextual hierarchy yet because the gain is below the **0.05** RMSE gate.
- Notes: This is weak positive signal, not a production candidate. It argues for stronger features or mean-model redesign rather than deeper post-hoc residual stacking.

## Planned - 08_alignment geologic feature track

- Task: typewell and analog alignment feature screen
- Purpose: Test whether research-guided geologic alignment features explain residual structure better than shallow post-hoc hierarchy.
- Config: notebook-local config in `notebooks/modeling/08_alignment/00_typewell_alignment_feature_screen.ipynb`
- Data version: local Kaggle download in `data/raw/train/`; submission candidates must recompute all features inside the notebook on hidden test wells.
- Split / CV: grouped well folds; alignment references and analog templates must be built from training folds only.
- Model: start with robust signed-delta Huber/Ridge means plus alignment descriptors; only add Hilbert residuals after a same-sample gain is confirmed.
- Primary metric: grouped target-zone RMSE against the same-notebook no-alignment Huber reference and the current local diagnostic reference **14.3586**.
- Secondary metrics: fold RMSE spread, alignment-feature ablations, descriptor leakage assertions, well-level residual bias change, and public-generalization rationale.
- Artifacts: fold metrics, candidate summary, feature-family ablation table, residual sample, alignment diagnostics, markdown report, model comparison row, and run bundle.
- Decision: promote alignment features only if they improve same-sample RMSE by at least **0.05** or produce a clear external-generalization hypothesis.
- Notes: Candidate features include best-lag GR/typewell correlation, known-zone trajectory similarity, boundary/progress pseudo-zone labels, typewell-transferred TVT deltas, and distance-to-analog summaries. DTW-style features are allowed only if computed fold-safely and without target-zone TVT leakage.

## 2026-06-23__typewell_alignment_feature_screen

- Task: typewell and analog alignment feature screen
- Purpose: Test whether research-guided typewell, lag-alignment, and nearest-analog features explain residuals left by the current Huber + additive Hilbert model.
- Config: notebook-local config in `notebooks/modeling/08_alignment/00_typewell_alignment_feature_screen.ipynb`
- Data version: local Kaggle download in `data/raw/train/`, plus sampled out-of-fold residuals from `model__hsgp_residual_on_gam_huber_residual_sample__2026-06-22.csv`
- Split / CV: fold-held-out residual sample; each analog template is built from training folds only and excludes the validation fold.
- Model: best candidate `analog_direct_s010`, a 10 percent shrink toward the nearest analog well's transferred target-zone delta.
- Primary metric: sampled RMSE **14.6966**, improving the same-sample no-alignment baseline (**14.7385**) by **0.0419** RMSE.
- Secondary metrics: fold RMSEs **14.2058**, **14.9742**, and **14.9098**; median absolute well-bias delta **-0.0988**; strongest feature/residual correlation `analog_delta_minus_baseline_delta` **0.0680**.
- Artifacts: fold metrics, summary, residual sample, well diagnostics, descriptors, feature correlations, findings, comparison row, markdown report, and run bundle under `artifacts/logs/runs/2026-06-23__typewell_alignment_feature_screen/`
- Decision: do not promote to Kaggle yet because the gain is just below the **0.05** feature-screen gate, but continue alignment modeling because this is much stronger than prior shallow hierarchy screens.
- Notes: The result supports multi-analog, distance-weighted, pseudo-zone, or robust-blended analog features next; direct fixed-shrink transfer is probably too crude to be the final form.

## 2026-06-23__multi_analog_zone_blend

- Task: multi-analog and pseudo-zone blend screen
- Purpose: Test whether the nearest-analog transfer signal improves when using distance-weighted multi-analog deltas or pseudo-zone-specific blend coefficients.
- Config: notebook-local config in `notebooks/modeling/08_alignment/01_multi_analog_zone_blend.ipynb`
- Data version: local Kaggle download in `data/raw/train/`, residual sample from `model__hsgp_residual_on_gam_huber_residual_sample__2026-06-22.csv`, and descriptors from `model__typewell_alignment_feature_descriptors__2026-06-23.csv`
- Split / CV: fold-held-out residual sample; each analog set is built from training folds only, and training wells exclude themselves when estimating analog deltas.
- Model: best candidate `weighted_k12_s150`, a 15 percent fixed shrink toward a distance-weighted average of the 12 nearest analog wells.
- Primary metric: sampled mean-fold RMSE **14.7034**, with same-notebook global-baseline delta **-0.0405** RMSE.
- Secondary metrics: fold RMSEs **14.3908**, **14.8355**, and **14.8840**; median absolute well-bias delta **-0.0870**.
- Artifacts: fold metrics, summary, residual sample, well diagnostics, beta table, findings, comparison row, markdown report, and run bundle under `artifacts/logs/runs/2026-06-23__multi_analog_zone_blend/`
- Decision: do not promote to Kaggle; the multi-analog blend confirms analog signal but does not beat the simpler nearest-analog direct-transfer screen.
- Notes: Avoid simply increasing analog count. Next alignment work should improve analog selection or feed nearest-analog delta into the robust mean model rather than applying a fixed post-hoc residual correction.

## 2026-06-23__pseudo_zone_huber

- Task: pseudo-zone Huber with nearest-analog delta
- Purpose: Test whether nearest-analog transfer should be learned through robust pseudo-zone residual heads instead of fixed shrinkage.
- Config: notebook-local config in `notebooks/modeling/08_alignment/02_pseudo_zone_huber.ipynb`
- Data version: residual sample from `model__hsgp_residual_on_gam_huber_residual_sample__2026-06-22.csv` and nearest-analog gap from `model__multi_analog_zone_blend_residual_sample__2026-06-23.csv`
- Split / CV: fold-held-out residual sample; residual heads train on other folds only.
- Model: best candidate `ridge_zone_a100_s050`, a Ridge residual head over row geometry, nearest-analog gap, pseudo-zone indicators, and interactions with 50 percent shrinkage.
- Primary metric: sampled RMSE **14.7004**, improving the same-sample no-analog-head baseline (**14.7385**) by **0.0381** RMSE.
- Secondary metrics: zone RMSE deltas were `z0` **-0.0032**, `z1` **-0.0133**, `z2` **-0.0629**, and `z3` **-0.0543**; median absolute well-bias delta **-0.0374**.
- Artifacts: fold metrics, summary, residual sample, well diagnostics, zone summary, findings, comparison row, markdown report, and run bundle under `artifacts/logs/runs/2026-06-23__pseudo_zone_huber/`
- Decision: do not promote to Kaggle; learned pseudo-zone heads do not beat fixed nearest-analog shrink, but the late-zone gains justify a progress-gated analog correction screen.
- Notes: The signal is not uniformly useful across the target zone. Treat nearest-analog correction as a late extrapolation aid, not a near-boundary correction.

## 2026-06-23__progress_gated_analog

- Task: progress-gated nearest-analog correction
- Purpose: Test whether late-zone-specific shrinkage schedules improve the nearest-analog transfer signal identified by the pseudo-zone Huber diagnostics.
- Config: notebook-local config in `notebooks/modeling/08_alignment/03_progress_gated_analog.ipynb`
- Data version: residual sample from `model__hsgp_residual_on_gam_huber_residual_sample__2026-06-22.csv` and nearest-analog gap from `model__multi_analog_zone_blend_residual_sample__2026-06-23.csv`
- Split / CV: fold-held-out residual sample; schedules are evaluated by fold against the same residual baseline.
- Model: best candidate `grid_025_025_075_100`, a monotone progress-gated shrinkage schedule with `z0=0.025`, `z1=0.025`, `z2=0.075`, and `z3=0.100`.
- Primary metric: sampled RMSE **14.6966**, improving the same-sample no-gate baseline (**14.7385**) by **0.0420** RMSE.
- Secondary metrics: zone RMSE deltas were `z0` **-0.0012**, `z1` **-0.0016**, `z2` **-0.0482**, and `z3` **-0.0682**; median absolute well-bias delta **-0.0888**.
- Artifacts: fold metrics, summary, residual sample, well diagnostics, zone summary, findings, comparison row, markdown report, and run bundle under `artifacts/logs/runs/2026-06-23__progress_gated_analog/`
- Decision: do not promote to Kaggle yet; progress gating confirms the late-zone analog pattern but only marginally beats fixed nearest-analog shrink.
- Notes: Further schedule tuning is unlikely to be the next large gain. Prefer spatial residual-correlation diagnostics or better analog selection criteria.

## 2026-06-23__spatial_residual_correlation

- Task: spatial residual-correlation screen
- Purpose: Test whether residuals from the current Huber + additive Hilbert model justify regression kriging, spatial KNN well-bias correction, or spatial GP residual modeling.
- Config: notebook-local config in `notebooks/modeling/08_alignment/04_spatial_residual_correlation.ipynb`
- Data version: local Kaggle download in `data/raw/train/`, plus residual sample from `model__hsgp_residual_on_gam_huber_residual_sample__2026-06-22.csv`
- Split / CV: fold-held-out residual sample; spatial KNN references are built from training folds only.
- Model: best candidate `spatial_knn_k12_s200`, a conservative 20 percent shrink toward the distance-weighted mean residual of 12 nearest wells in spatial/trajectory descriptor space.
- Primary metric: sampled RMSE **14.7144**, improving the same-sample no-spatial baseline (**14.7385**) by **0.0241** RMSE.
- Secondary metrics: nearest-neighbor well-mean residual correlation **0.0422**; median absolute well-bias delta **-0.0073**; semivariance is highest in the farthest descriptor-distance bin.
- Artifacts: fold metrics, summary, residual sample, well diagnostics, neighbor diagnostics, variogram tables, findings, comparison row, markdown report, and run bundle under `artifacts/logs/runs/2026-06-23__spatial_residual_correlation/`
- Decision: do not promote to kriging or spatial GP yet; the current descriptor-space residual correlation is too weak.
- Notes: Spatial modeling may still be useful with better geology-aware coordinates, but the next practical move should refine analog selection rather than build a full spatial residual model.

## 2026-06-23__analog_selection_refinement

- Task: analog selection refinement
- Purpose: Test whether nearest-analog transfer improves when analogs are selected by log-shape, typewell-alignment, trajectory, or hybrid descriptor geometry rather than the first broad descriptor set.
- Config: notebook-local config in `notebooks/modeling/08_alignment/05_analog_selection_refinement.ipynb`
- Data version: residual sample from `model__hsgp_residual_on_gam_huber_residual_sample__2026-06-22.csv` and descriptors from `model__typewell_alignment_feature_descriptors__2026-06-23.csv`
- Split / CV: fold-held-out residual sample; analogs are selected from training folds only.
- Model: best candidate `log_plus_trajectory_k1_progress_gate`, using one nearest analog selected by recent GR shape, typewell alignment, TVT span, XY span, and vertical drop, then applying the progress-gated analog schedule.
- Primary metric: sampled RMSE **14.6383**, improving the same-sample no-refined-analog baseline (**14.7385**) by **0.1003** RMSE.
- Secondary metrics: family bests were `log_plus_trajectory` **14.6383**, `baseline_hybrid` **14.6402**, `log_shape` **14.6508**, `late_extrapolation` **14.6857**, `trajectory_depth` **14.6863**, and `typewell_quality` **14.6931**. Zone deltas for the best candidate were `z0` **-0.0053**, `z1` **-0.0214**, `z2` **-0.1147**, and `z3` **-0.1906**.
- Artifacts: fold metrics, summary, residual sample, well diagnostics, zone summary, selection diagnostics, family summary, findings, comparison row, markdown report, and run bundle under `artifacts/logs/runs/2026-06-23__analog_selection_refinement/`
- Decision: promote this modeling idea to the next pipeline stage because it clears the **0.05** RMSE gate. Do not submit the residual-sample correction directly; implement the same fold-safe logic in a full Huber/Hilbert or alignment-HSGP notebook.
- Notes: During execution, a duplicate `last_known_tvt` descriptor column caused a pandas duplicate-label arithmetic error. The notebook now de-duplicates descriptor columns before merge.

## 2026-06-23__alignment_hsgp_residual

- Task: alignment-aware Hilbert residual screen
- Purpose: Test whether the above-gate log-plus-trajectory analog selector still improves after the current Huber + additive Hilbert residual model, and whether learned smooth/RBF residual heads improve over direct analog transfer.
- Config: notebook-local config in `notebooks/modeling/08_alignment/06_alignment_hsgp_residual.ipynb`
- Data version: residual sample from `model__hsgp_residual_on_gam_huber_residual_sample__2026-06-22.csv` and analog-selection residual sample from `model__analog_selection_refinement_residual_sample__2026-06-23.csv`
- Split / CV: fold-held-out residual sample; analog signal comes from the fold-safe log-plus-trajectory selection screen.
- Model: best candidate `direct_analog_gate`, applying the refined nearest-analog progress-gated correction directly on top of the current Huber + Hilbert predictions.
- Primary metric: sampled RMSE **14.6383**, improving the same-sample Huber + Hilbert baseline (**14.7385**) by **0.1003** RMSE.
- Secondary metrics: learned analog Ridge, Huber, polynomial, and RBF residual heads all trailed direct transfer. Zone deltas were `z0` **-0.0053**, `z1` **-0.0214**, `z2` **-0.1147**, and `z3` **-0.1906**; median absolute well-bias delta **-0.0963**.
- Artifacts: fold metrics, summary, residual sample, well diagnostics, zone summary, findings, comparison row, markdown report, and run bundle under `artifacts/logs/runs/2026-06-23__alignment_hsgp_residual/`
- Decision: package a Kaggle-safe direct analog-gated Huber + Hilbert candidate next. Do not add an extra learned residual head yet.
- Notes: This is still a residual-sample diagnostic; Kaggle packaging must recompute analog descriptors and nearest-analog transfer inside the notebook.

## 2026-06-26__validation_mismatch_audit

- Task: local/public validation mismatch audit
- Purpose: Determine whether the next move should be deeper hierarchy, particle/state-space models, or validation repair after the analog-gated Huber + Hilbert candidate failed to improve public LB.
- Config: notebook-local config in `notebooks/modeling/09_validation/00_local_public_mismatch_audit.ipynb`
- Data version: existing model summary, fold, well, selection, and Kaggle submission tracker artifacts through analog submission `53999839`.
- Split / CV: diagnostic audit only; no new predictive fit.
- Model: no model fit. The audit compares public transfer, local diagnostic basis, analog distance bins, well-bias deltas, and promotion decisions.
- Primary metric: analog-gated candidate public LB **14.603**, worse than Huber + additive Hilbert public baseline **14.514** by **0.089** RMSE.
- Secondary metrics: bounded residual-sample analog gain was **0.1003 RMSE**, but mean analog distance-bin well-bias improvement share was only about **0.555**.
- Artifacts: transfer table, analog distance-bin table, worst-well sample, findings, promotion decisions, markdown report, and run bundle under `artifacts/logs/runs/2026-06-26__validation_mismatch_audit/`
- Decision: pause deeper hierarchy and particle/state-space modeling. Build stricter CV folds before LightGBM/XGBoost or further Bayesian complexity.
- Notes: The analog result is useful negative evidence: feature construction may still matter, but the current bounded residual sample is too optimistic for promotion decisions.

## 2026-06-26__strict_cv_fold_design

- Task: strict CV fold design
- Purpose: Create reusable well-level folds that stress descriptor clusters, far-analog wells, and high-extrapolation wells before boosted-tree tuning.
- Config: notebook-local config in `notebooks/modeling/09_validation/01_stricter_cv_fold_design.ipynb`
- Data version: local raw train horizontal wells, typewell alignment descriptors, analog-selection diagnostics, and alignment well diagnostics.
- Split / CV: no model fit. The notebook exports four deterministic fold assignment columns: `descriptor_cluster_fold`, `analog_distance_fold`, `extrapolation_fold`, and `strict_hybrid_fold`.
- Model: no prediction model. KMeans descriptor clusters, analog-distance quantile buckets, extrapolation-stress quantile buckets, and a greedy balanced hybrid assignment.
- Primary metric: recommended default `strict_hybrid_fold` has row imbalance **1.000**, well imbalance **1.024**, and mean nearest-training descriptor distance **2.436**.
- Secondary metrics: strictest diagnostic design is `analog_distance_fold` with strictness score **7.326**; `descriptor_cluster_fold` score **6.659**; `extrapolation_fold` score **5.945**; `strict_hybrid_fold` score **5.774**.
- Artifacts: fold assignments, well descriptors, fold summaries, design scores, nearest-train distances, stress-well sample, findings, markdown report, and run bundle under `artifacts/logs/runs/2026-06-26__strict_cv_fold_design/`
- Decision: use `strict_hybrid_fold` for first LightGBM/XGBoost strict-CV modeling, then stress-test promising candidates with `analog_distance_fold`, `descriptor_cluster_fold`, and `extrapolation_fold`.
- Notes: The fold table is intentionally reusable by downstream notebooks; do not recompute arbitrary random folds inside boosted-tree notebooks.

## 2026-06-26__lightgbm_strict_cv

- Task: LightGBM strict-CV screen
- Purpose: Test whether boosted trees can learn nonlinear feature interactions under the stricter validation protocol after the analog-gate public miss.
- Config: notebook-local config in `notebooks/modeling/02_trees/06_lightgbm_strict_cv.ipynb`
- Data version: local raw train horizontal wells, strict fold assignments, typewell/alignment descriptors, and deterministic target-row sampling capped at **160** target rows per well.
- Split / CV: primary evaluation on `strict_hybrid_fold`; stress evaluation for the best candidate on `analog_distance_fold`.
- Model: best candidate `lgbm_medium_l2`, LightGBM Huber objective over signed TVT delta from each well's last known TVT.
- Primary metric: sampled `strict_hybrid_fold` RMSE **13.2363**, std **0.8224**, versus same-row last-known hold RMSE **15.5734**.
- Secondary metrics: `analog_distance_fold` stress RMSE **13.4453**, std **0.8340**, versus same-row last-known hold RMSE **15.5833**. Fold RMSEs were **12.8483**, **13.0894**, and **14.3981**.
- Artifacts: summary, fold metrics, analog-distance stress metrics, feature importance, zone and analog-bin summaries, well diagnostics, feature preview, findings, markdown report, model comparison row, and run bundle under `artifacts/logs/runs/2026-06-26__lightgbm_strict_cv/`
- Decision: continue boosted-tree modeling. A Kaggle-safe LightGBM package was submitted as an external transfer check at user request; promotion still depends on the public score and comparison against Huber + additive Hilbert public LB **14.514**.
- Notes: The Kaggle package excludes local-only formation-distance columns and uses **31** inference-safe row/descriptor features. Kaggle version 1 completed and submission `54097453` is pending public score.

## 2026-06-27__lightgbm_strict_kaggle_package

- Task: boosted-tree Kaggle packaging
- Purpose: Convert the strict-CV LightGBM candidate into a self-contained code-competition notebook that trains and writes `/kaggle/working/submission.csv`.
- Config: `kaggle/lightgbm_strict_submission/lightgbm_strict_submission.ipynb`, `LGBMRegressor(objective="huber", n_estimators=1200, learning_rate=0.03, num_leaves=63, max_depth=10, min_child_samples=80, reg_lambda=2.0)`, deterministic target-row sampling capped at **220** rows per well.
- Data version: Kaggle competition input on remote run; local mirror under `data/raw/` for validation.
- Split / CV: no new CV fit in the package; training uses all available training wells and recomputed descriptors, while the local score basis remains `2026-06-26__lightgbm_strict_cv`.
- Model: `lightgbm_medium_l2_inference_safe`
- Primary metric: Kaggle submission `54097453` public LB **14.687**.
- Secondary metrics: local package execution and Kaggle version 1 both produced **14,151** rows with `id,tvt`, no missing predictions; Kaggle log reports **170,060** training rows and **31** features.
- Artifacts: Kaggle package under `kaggle/lightgbm_strict_submission/`, local validation output under `kaggle/lightgbm_strict_submission/artifacts/reports/`, downloaded remote output under `.tmp_kaggle_config/lightgbm_strict_output/`.
- Decision: do not promote over Huber + additive Hilbert public LB **14.514**. The package is still useful negative evidence because the inference-safe LightGBM score is much worse than the full-feature strict-CV local result.
- Notes: The package hardens data discovery for both `/kaggle/input` and local parent-directory validation, and adds a fallback for analog-distance descriptors so hidden reruns do not fail on empty descriptor columns. The public miss strengthens the case for explicit inference-safe tree feature engineering rather than direct packaging of full-feature strict-CV winners.

## 2026-06-27__xgboost_strict_cv

- Task: XGBoost strict-CV screen
- Purpose: Test whether the boosted-tree gain from LightGBM survives a different tree learner and robust objective under the same strict validation protocol.
- Config: notebook-local config in `notebooks/modeling/02_trees/07_xgboost_strict_cv.ipynb`; candidates included squared-error medium/deep regularized trees and pseudo-Huber medium/shallow robust trees.
- Data version: local raw train horizontal wells, strict fold assignments, typewell/alignment descriptors, formation-distance features, and deterministic target-row sampling capped at **160** target rows per well.
- Split / CV: primary evaluation on `strict_hybrid_fold`; stress evaluation for the best candidate on `analog_distance_fold`.
- Model: best candidate `xgb_pseudohuber_shallow_robust`, XGBoost histogram trees with pseudo-Huber objective over signed TVT delta from each well's last known TVT.
- Primary metric: sampled `strict_hybrid_fold` RMSE **12.7504**, std **0.7300**, versus same-row last-known hold RMSE **15.5734**.
- Secondary metrics: `analog_distance_fold` stress RMSE **12.9116**, std **0.9691**, versus same-row last-known hold RMSE **15.5833**. This beats the LightGBM strict reference by **0.4859** RMSE and the LightGBM stress reference by **0.5337** RMSE.
- Artifacts: summary, fold metrics, analog-distance stress metrics, feature importance, zone and analog-bin summaries, residual sample, well diagnostics, feature preview, findings, markdown report, model comparison row, and run bundle under `artifacts/logs/runs/2026-06-27__xgboost_strict_cv/`.
- Decision: promote boosted-tree work, but do not package this local notebook directly because top features include train-only formation-distance columns. Next step is inference-safe XGBoost/LightGBM tuning that replaces or removes formation-distance features.
- Notes: Top features for the best candidate include `md_from_start`, `extrapolation_stress_score`, `distance_log1p`, `row_idx_original`, `abs_dist_to_BUDA`, and `dist_to_BUDA`.

## 2026-06-28__xgboost_strict_kaggle_package

- Task: boosted-tree Kaggle packaging
- Purpose: Convert the XGBoost strict-CV candidate into a self-contained code-competition notebook that trains and writes `/kaggle/working/submission.csv`.
- Config: `kaggle/xgboost_strict_submission/xgboost_strict_submission.ipynb`, `XGBRegressor(objective="reg:pseudohubererror", n_estimators=800, learning_rate=0.04, max_depth=4, min_child_weight=120, subsample=0.80, colsample_bytree=0.80, reg_lambda=8.0)`, deterministic target-row sampling capped at **220** rows per well.
- Data version: Kaggle competition input on remote run; local mirror under `data/raw/` for validation.
- Split / CV: no new CV fit in the package; training uses all available training wells and recomputed descriptors, while the local score basis remains `2026-06-27__xgboost_strict_cv`.
- Model: `xgb_pseudohuber_shallow_robust_inference_safe`
- Primary metric: Kaggle submission `54129475` public LB **14.472**.
- Secondary metrics: local package execution and Kaggle version 1 both produced **14,151** rows with `id,tvt`, no missing predictions; Kaggle log reports **170,060** training rows and **31** inference-safe features.
- Artifacts: Kaggle package under `kaggle/xgboost_strict_submission/`, local validation output under `kaggle/xgboost_strict_submission/artifacts/reports/`, downloaded remote output under `.tmp_kaggle_config/xgboost_strict_output/`.
- Decision: promote as the current best public submission. The inference-safe XGBoost package beats Huber + additive Hilbert **14.514** and LightGBM inference-safe **14.687**, so boosted-tree tuning should continue.
- Notes: This package intentionally mirrors the LightGBM inference-safe feature set so public comparison isolates model engine/objective differences rather than feature availability. The public gain is small but important because it transfers without train-only formation features.

## 2026-06-28__xgboost_inference_safe_tuning

- Task: inference-safe XGBoost tuning
- Purpose: Test whether the full-feature XGBoost gain can be recovered with Kaggle-legal typewell features and fold-safe predicted formation-distance proxies.
- Config: notebook-local config in `notebooks/modeling/02_trees/08_xgboost_inference_safe_tuning.ipynb`; deterministic target-row sampling capped at **120** rows per well, bounded XGBoost thread count, pseudo-Huber and squared-error candidates.
- Data version: local raw train horizontal wells, strict fold assignments, typewell/alignment descriptors, and fold-safe proxy formation features derived from safe row/typewell inputs.
- Split / CV: primary evaluation on `strict_hybrid_fold`; best legal candidate stress-tested on `analog_distance_fold`; direct formation features retained only as an oracle diagnostic.
- Model: best legal candidate `pseudo_shallow_more_trees__safe_typewell`, XGBoost histogram trees with pseudo-Huber objective over signed TVT delta from each well's last known TVT.
- Primary metric: sampled `strict_hybrid_fold` RMSE **13.4178**, versus safe-base RMSE **13.4399**.
- Secondary metrics: `analog_distance_fold` stress RMSE **13.7533**. Fold-safe proxy-formation features scored **13.5503**, worse than safe base by **0.1103** RMSE. Oracle direct-formation diagnostic scored **12.9597**, leaving a **0.4581** RMSE gap from the best legal model.
- Artifacts: feature preview, proxy quality, summary, fold metrics, analog-distance stress metrics, feature importance, residual sample, well diagnostics, findings, markdown report, model comparison row, and run bundle under `artifacts/logs/runs/2026-06-28__xgboost_inference_safe_tuning/`.
- Decision: do not package this notebook to Kaggle yet. Best legal tuning improved local strict CV only marginally, and proxy-formation features underperformed safe base.
- Notes: The experiment confirms direct formation distance is valuable, but the current proxy approach does not transfer that signal. Next work should try residual stacking or a stronger hidden-test-safe structural proxy instead of more of the same proxy-formation recipe.

## 2026-06-28__xgboost_residual_stacking

- Task: inference-safe XGBoost residual stacking
- Purpose: Test whether nested out-of-fold residual models can improve the current inference-safe XGBoost path without leaking held-out fold residuals.
- Config: notebook-local config in `notebooks/modeling/02_trees/09_xgboost_residual_stacking.ipynb`; deterministic target-row sampling capped at **120** rows per well; base model `pseudo_shallow_more_trees` safe/typewell XGBoost; residual stackers included progress-bin bias, Ridge, Huber, and small XGBoost residual heads.
- Data version: local raw train horizontal wells, strict fold assignments, typewell/alignment descriptors, and inference-safe row/typewell features only.
- Split / CV: nested OOF stacking under `strict_hybrid_fold`, followed by the same nested design under `analog_distance_fold`.
- Model: best stacker `ridge_residual_s025`, a median-imputed and standardized Ridge residual correction shrunk to **25%** of predicted residual.
- Primary metric: `strict_hybrid_fold` RMSE **13.2548**, improving base-only XGBoost **13.4178** by **0.1630** RMSE.
- Secondary metrics: `analog_distance_fold` stress RMSE **13.5632**, improving stress base-only **13.7424** by **0.1792** RMSE. Reproducibility rerun matched the key metrics.
- Artifacts: summary, fold metrics, analog-distance stress metrics, zone and analog-bin summaries, well diagnostics, findings, markdown report, model comparison row, and run bundle under `artifacts/logs/runs/2026-06-28__xgboost_residual_stacking/`.
- Decision: promoted to Kaggle packaging for one external probe, but do not promote after the public result. Kaggle submission `54157692` scored **14.583**, worse than the current XGBoost strict inference-safe public LB **14.472**.
- Notes: The residual signal is strongest outside the earliest progress zone; late target-zone rows improved by roughly **0.23** RMSE in the strict split, while the first quartile slightly worsened. The public miss suggests residual correction is still validation-fragile even when it clears `analog_distance_fold`.

## 2026-06-30__stratigraphic_state_feature_screen

- Task: stratigraphic-state feature screen
- Purpose: Test whether hidden-test-safe known-zone state, typewell/log alignment uncertainty, and fold-safe nearest-analog descriptors recover part of the train-only formation-distance signal.
- Config: `notebooks/modeling/08_alignment/07_stratigraphic_state_feature_screen.ipynb`; deterministic target-row sampling capped at **90** rows per well; sequence length **32**; typewell window count **18**; feature sets `safe_base`, `strat_known_state`, `strat_alignment_uncertainty`, `strat_alignment_plus_boundary`, `strat_full`, and `oracle_direct_formations`.
- Data version: local raw train horizontal wells, typewell files, strict fold assignments, and reusable helpers in `src/preprocessing/stratigraphic.py`.
- Split / CV: primary evaluation on `strict_hybrid_fold`; secondary rows also exported for `analog_distance_fold`.
- Model: best non-oracle candidate `huber_delta__strat_known_state`, with XGBoost squared-error and histogram-gradient consumers retained as screen diagnostics only.
- Primary metric: `strict_hybrid_fold` RMSE **14.9922**, improving same-notebook Huber safe-base **15.0313** by **0.0391** RMSE.
- Secondary metrics: lightweight XGBoost squared-error `strat_known_state` improved over its same-model safe-base by **0.7738** RMSE, but the consumer scored far worse than the current public-best XGBoost recipe and is not comparable as a promotion candidate.
- Artifacts: summary, fold metrics, stress-slice diagnostics, feature importance, prediction sample, markdown report, and run bundle under `artifacts/logs/runs/2026-06-30__stratigraphic_state_feature_screen/`.
- Decision: do not package for Kaggle. Keep known-zone state features as candidate inputs for the next exact XGBoost recipe test or Bayesian stratigraphic-regime model.
- Notes: Naively adding typewell alignment uncertainty and analog descriptors worsened the Huber screen. Use those quantities as gating/shrinkage signals in a future Bayesian or tree interaction design rather than as a broad additive feature dump.

## 2026-06-30__xgboost_known_state_features

- Task: exact XGBoost known-state feature injection
- Purpose: Test whether the directional known-zone state signal from the lightweight stratigraphic screen improves the current inference-safe XGBoost recipe when added directly.
- Config: `notebooks/modeling/02_trees/10_xgboost_known_state_features.ipynb`; deterministic target-row sampling capped at **120** rows per well; same strict-hybrid and analog-distance folds as the inference-safe XGBoost tuning path; added known-zone `TVT`, `GR`, trajectory span, and depth-drop summaries.
- Data version: local raw train horizontal wells, strict fold assignments, typewell descriptors, fold-safe proxy formation diagnostics, and known-zone state features computed only from rows available before the target-zone gap.
- Split / CV: primary evaluation on `strict_hybrid_fold`; best legal candidate stress-tested on `analog_distance_fold`; direct formation features retained only as an oracle diagnostic.
- Model: best legal candidate `pseudo_shallow_more_trees__safe_typewell`, XGBoost histogram trees with pseudo-Huber objective over signed TVT delta.
- Primary metric: `strict_hybrid_fold` RMSE **13.4334**. The best state-augmented candidate was `square_medium_regularized__safe_typewell_known_state` at **13.5189**, so raw known-state injection worsened the exact recipe.
- Secondary metrics: `analog_distance_fold` stress RMSE **13.8243** for the best legal candidate. Direct-formation oracle diagnostic scored **12.9353**, leaving a **0.4981** RMSE legal-oracle gap.
- Artifacts: feature preview, proxy quality, summary, fold metrics, analog-distance stress metrics, feature importance, residual sample, well diagnostics, findings, markdown report, model comparison row, and run bundle under `artifacts/logs/runs/2026-06-30__xgboost_known_state_features/`.
- Decision: do not package for Kaggle. The notebook is a clean negative screen for raw known-state XGBoost features.
- Notes: This result does not invalidate stratigraphic-state modeling; it says the state evidence probably needs gating, mixture structure, or Bayesian shrinkage rather than flat tree features.

## 2026-06-30__gated_bayesian_regime

- Task: gated Bayesian stratigraphic-regime model
- Purpose: Test whether known-zone state and typewell descriptors can control residual shrinkage and mixture weights after raw XGBoost state features failed.
- Config: `notebooks/modeling/06_bayesian/07_gated_bayesian_regime.ipynb`; deterministic target-row sampling capped at **120** rows per well; Huber signed-delta mean; additive Hilbert residual expert; Ridge state residual expert; Student-t MAP soft gate over zero, Hilbert, state, and blended residual experts.
- Data version: local raw train horizontal wells, strict fold assignments, typewell descriptor artifact, and known-zone-only state features.
- Split / CV: primary evaluation on `strict_hybrid_fold`; package validation used a self-contained all-train notebook that recomputes descriptors inside the Kaggle notebook.
- Model: best candidate `huber_plus_gate_conservative`, with gate prior scale **0.25** and shrink prior **0.25**.
- Primary metric: `strict_hybrid_fold` RMSE **14.1072**, improving same-notebook Huber mean-only **14.5053** by **0.3981** RMSE.
- Secondary metrics: fixed 25% HSGP residual scored **14.4838**; fixed 25% state residual scored **14.3282**; the conservative gate learned average weights of **0.177** zero, **0.202** HSGP, **0.584** state, and **0.037** blend.
- Artifacts: summary, fold metrics, gate diagnostics, gate weight sample, zone/analog-bin summaries, well diagnostics, residual sample, findings, markdown report, model comparison row, run bundle, and self-contained Kaggle package.
- Decision: package as a curiosity public-score probe, not as a promoted baseline.
- Notes: Local execution took about **28.6 minutes** for 3-fold CV; the self-contained Kaggle package executed locally in about **9 minutes** and wrote **14,151** finite predictions.

## 2026-06-30__pseudo_top_uncertainty

- Task: pseudo-formation/top uncertainty feature screen
- Purpose: Convert train-only formation-surface columns into fold-safe latent top-elevation, signed-offset, nearest-top, margin, entropy, and ensemble-spread features.
- Config: `notebooks/modeling/08_alignment/08_pseudo_top_uncertainty.ipynb`; deterministic target-row sampling capped at **120** rows per well; two-seed XGBoost squared-error top-elevation ensembles for six named formations; downstream XGBoost and Huber consumers.
- Data version: local raw train horizontal wells, strict fold assignments, typewell descriptors, and direct formation columns used only as fold-local proxy labels and oracle diagnostics.
- Split / CV: primary evaluation on `strict_hybrid_fold`; legal candidates also scored on `analog_distance_fold`; oracle direct-formation candidates are diagnostic only.
- Model: best legal overall `xgb_square_regularized__safe_typewell`; best pseudo-top candidate `xgb_pseudohuber_known_state_recipe__safe_pseudo_top_uncertainty`.
- Primary metric: best legal strict-hybrid RMSE **13.4506**; best pseudo-top strict-hybrid RMSE **13.5891**.
- Secondary metrics: best legal analog-distance stress RMSE **13.8870**; mean pseudo-top elevation RMSE **35.0660**; nearest pseudo-top agreement **0.5804** under `strict_hybrid_fold`.
- Artifacts: feature preview, top-label summary, proxy quality, summary, fold metrics, zone/analog-bin summaries, well diagnostics, feature importance, findings, markdown report, model comparison row, and run bundle under `artifacts/logs/runs/2026-06-30__pseudo_top_uncertainty/`.
- Decision: do not package. Raw pseudo-top feature injection underperforms the simpler safe/typewell XGBoost recipe.
- Notes: Pseudo-top inference is meaningful enough for state reasoning, but the next attempt should use margin/entropy/spread as gated shrinkage or Bayesian-regime evidence instead of flat tree columns.

## 2026-06-30__pseudo_top_gated_regime

- Task: pseudo-top gated residual-regime model
- Purpose: Test whether pseudo-top uncertainty improves transfer when it controls residual shrinkage rather than entering as raw tree predictors.
- Config: `notebooks/modeling/08_alignment/09_pseudo_top_gated_regime.ipynb`; deterministic target-row sampling capped at **120** rows per well; two-seed squared-error pseudo-top proxy ensembles; safe/typewell XGBoost and Huber mean models; Ridge/Huber residual experts; fixed and uncertainty-gated shrinkage schedules.
- Data version: local raw train horizontal wells, strict fold assignments, typewell descriptors, and direct formation columns used only as fold-local pseudo-top labels.
- Split / CV: primary evaluation on `strict_hybrid_fold`; all candidates also scored on `analog_distance_fold`.
- Model: best candidate `xgb_pseudohuber_mean__ridge_pseudo_state_residual__uncertainty_gate_conservative`.
- Primary metric: strict-hybrid RMSE **13.4694**, improving same-notebook mean-only **13.4745** by **0.0051** RMSE.
- Secondary metrics: analog-distance stress RMSE **13.7665**, better than the prior safe/typewell stress reference **13.8870** but still not enough to clear strict-CV promotion versus **13.4506**.
- Artifacts: feature preview, summary, fold metrics, gate diagnostics, zone/analog-bin summaries, well diagnostics, findings, markdown report, model comparison row, and run bundle under `artifacts/logs/runs/2026-06-30__pseudo_top_gated_regime/`.
- Decision: iterate before package. The gate is a useful stabilizer but not a point-RMSE breakthrough.
- Notes: Next pseudo-top work should move uncertainty into the primary mean learner or a Bayesian prior/mixture, not only a post-hoc residual correction.

## 2026-06-30__pseudo_top_interaction_xgboost

- Task: pseudo-top interaction XGBoost mean model
- Purpose: Test whether pseudo-top uncertainty helps the primary XGBoost mean when expressed as controlled interactions with progress, distance, analog stress, and typewell alignment error.
- Config: `notebooks/modeling/08_alignment/10_pseudo_top_interaction_xgboost.ipynb`; deterministic target-row sampling capped at **120** rows per well; two-seed squared-error pseudo-top proxy ensembles; XGBoost pseudo-Huber, squared-error, and shallow interaction candidates.
- Data version: local raw train horizontal wells, strict fold assignments, typewell descriptors, and direct formation columns used only as fold-local pseudo-top labels.
- Split / CV: primary evaluation on `strict_hybrid_fold`; all candidates also scored on `analog_distance_fold`.
- Model: best overall `xgb_square_regularized__safe_typewell`; best interaction candidate `xgb_pseudohuber_reference__pseudo_interactions`.
- Primary metric: best overall strict-hybrid RMSE **13.4745**; best interaction strict-hybrid RMSE **13.4815**.
- Secondary metrics: best overall analog-distance stress RMSE **13.8284**; best interaction stress RMSE **13.8681**.
- Artifacts: feature preview, pseudo-top proxy quality, summary, fold metrics, zone/analog-bin summaries, well diagnostics, feature importance, findings, markdown report, model comparison row, and run bundle under `artifacts/logs/runs/2026-06-30__pseudo_top_interaction_xgboost/`.
- Decision: do not package. Controlled pseudo-top interactions did not beat the same-notebook safe/typewell baseline.
- Notes: Flat tree feature engineering around pseudo-top appears exhausted; use Bayesian/mixture-regime priors next.

## 2026-07-01__pseudo_top_mixture_regime

- Task: pseudo-top Bayesian/MAP mixture-regime model
- Purpose: Test whether pseudo-top uncertainty is more useful as soft stratigraphic-regime prior evidence than as raw/interacted XGBoost features.
- Config: `notebooks/modeling/06_bayesian/08_pseudo_top_mixture_regime.ipynb`; deterministic target-row sampling capped at **120** rows per well; two-seed squared-error pseudo-top proxy ensembles; softmax formation probabilities; Ridge MAP residual priors over formation probability, progress, distance, and near-top basis terms; fixed and confidence-weighted shrinkage schedules.
- Data version: local raw train horizontal wells, strict fold assignments, typewell descriptors, and direct formation columns used only as fold-local pseudo-top labels.
- Split / CV: primary evaluation on `strict_hybrid_fold`; all candidates also scored on `analog_distance_fold`.
- Model: best candidate `xgb_pseudohuber_mean__ridge_alpha650__mixture_conf_s050`.
- Primary metric: strict-hybrid RMSE **13.4437**, improving prior strict safe/typewell reference **13.4506** by **0.0069** RMSE.
- Secondary metrics: analog-distance stress RMSE **13.7593**; best mean-only candidate **13.4745**.
- Artifacts: feature preview, pseudo-top proxy quality, top-label summary, summary, fold metrics, mixture diagnostics, zone/analog-bin summaries, well diagnostics, findings, markdown report, model comparison row, and run bundle under `artifacts/logs/runs/2026-07-01__pseudo_top_mixture_regime/`.
- Decision: iterate before package. This is the best pseudo-top direction so far, but the strict gain is below the promotion margin.
- Notes: Next Bayesian iteration should model the TVT mean directly by stratigraphic regime or calibrate mixture priors against public-transfer stress slices.

## 2026-07-01__pseudo_top_regime_mean

- Task: direct pseudo-top stratigraphic-regime mean model
- Purpose: Test whether pseudo-top probabilities help more when they define the primary TVT mean rather than a residual correction.
- Config: `notebooks/modeling/06_bayesian/09_pseudo_top_regime_mean.ipynb`; deterministic target-row sampling capped at **120** rows per well; two-seed squared-error pseudo-top proxy ensembles; soft formation probabilities; global safe/typewell XGBoost/Ridge means; per-formation weighted Ridge MAP means; fixed and confidence-weighted blends back to the global mean.
- Data version: local raw train horizontal wells, strict fold assignments, typewell descriptors, and direct formation columns used only as fold-local pseudo-top labels.
- Split / CV: primary evaluation on `strict_hybrid_fold`; all candidates also scored on `analog_distance_fold`.
- Model: best candidate `xgb_square_global__regime_ridge_alpha80__confidence_s100`.
- Primary metric: strict-hybrid RMSE **13.0928**, improving the pseudo-top mixture prior **13.4437** by **0.3510** RMSE and strict safe/typewell reference **13.4506** by **0.3579** RMSE.
- Secondary metrics: analog-distance stress RMSE **13.4006**; best fixed blend **13.1196** strict RMSE.
- Artifacts: summary, fold metrics, regime diagnostics, zone/analog-bin summaries, well diagnostics, findings, markdown report, model comparison row, and run bundle under `artifacts/logs/runs/2026-07-01__pseudo_top_regime_mean/`.
- Decision: package for Kaggle after local package validation. This is the first pseudo-top-family result to clear the local promotion gate by a meaningful margin.
- Notes: Public-transfer risk remains. The package must recompute pseudo-top proxies and regime means from all training wells and hidden-test-safe inputs only.

## 2026-07-01__pseudo_top_regime_mean_tuning

- Task: pseudo-top regime-mean tuning
- Purpose: Test whether denser Ridge alpha and blend-schedule tuning can materially improve the direct pseudo-top regime-mean candidate.
- Config: `notebooks/modeling/06_bayesian/10_pseudo_top_regime_mean_tuning.ipynb`; deterministic target-row sampling capped at **120** rows per well; two-seed squared-error pseudo-top proxy ensembles; two XGBoost global means; regime Ridge alphas **40**, **80**, **140**, and **220**; fixed, confidence, confidence-power, and margin-heavy blend schedules.
- Data version: local raw train horizontal wells, strict fold assignments, typewell descriptors, and direct formation columns used only as fold-local pseudo-top labels.
- Split / CV: primary evaluation on `strict_hybrid_fold`; all candidates also scored on `analog_distance_fold`.
- Model: best candidate `xgb_square_global__regime_ridge_alpha40__confidence_s100`.
- Primary metric: strict-hybrid RMSE **13.0910**, only **0.0018** better than the first regime-mean baseline **13.0928**.
- Secondary metrics: analog-distance stress RMSE **13.4065**, slightly worse than the first regime-mean baseline **13.4006**.
- Artifacts: summary, fold metrics, regime diagnostics, zone/analog-bin summaries, well diagnostics, findings, markdown report, model comparison row, and run bundle under `artifacts/logs/runs/2026-07-01__pseudo_top_regime_mean_tuning/`.
- Decision: do not package over the first regime-mean candidate. The strict gain is negligible and stress behavior is slightly worse.
- Notes: Move next to boosted stacking, pseudo-top inference tuning, or a structurally different Bayesian/partial-pooling regime model.

## 2026-07-01__xgboost_regime_mean_stack

- Task: XGBoost/regression stack on top of pseudo-top regime mean
- Purpose: Test whether fold-safe regime-mean predictions can improve a boosted/meta model after simple Ridge/blend tuning saturated.
- Config: `notebooks/modeling/02_trees/11_xgboost_regime_mean_stack.ipynb`; deterministic target-row sampling capped at **120** rows per well; two-seed squared-error pseudo-top proxy ensembles; fold-safe direct regime-mean OOF predictions; XGBoost direct meta model; XGBoost and Ridge residual heads with conservative shrinkage.
- Data version: local raw train horizontal wells, strict fold assignments, typewell descriptors, and direct formation columns used only as fold-local pseudo-top labels.
- Split / CV: primary evaluation on `strict_hybrid_fold`; all candidates also scored on `analog_distance_fold`.
- Model: best candidate `blend_plus_ridge_residual_s15`.
- Primary metric: strict-hybrid RMSE **13.0769**, improving the direct regime-mean reference **13.0928** by **0.0159** RMSE.
- Secondary metrics: analog-distance stress RMSE **13.4468**, worse than the direct regime-mean reference **13.4006**.
- Artifacts: summary, fold metrics, regime diagnostics placeholder, zone/analog-bin summaries, well diagnostics, findings, markdown report, model comparison row, and run bundle under `artifacts/logs/runs/2026-07-01__xgboost_regime_mean_stack/`.
- Decision: iterate before package. The strict gain is real but small, and stress behavior worsened.
- Notes: Next stack work should gate residual corrections by pseudo-top confidence, analog stress, or high-stress slices rather than applying a uniform shrinkage.

## 2026-07-01__xgboost_regime_stress_gated_stack

- Task: stress-gated XGBoost/regression stack on top of pseudo-top regime mean
- Purpose: Test whether pseudo-top confidence, margin, analog stress, and hybrid gates can keep the strict gain from residual stacking while reducing analog-distance stress damage.
- Config: `notebooks/modeling/02_trees/12_xgboost_regime_stress_gated_stack.ipynb`; same fold-safe regime-mean OOF setup as `11_xgboost_regime_mean_stack`; XGBoost/Ridge residual heads; uniform, confidence, margin, low-stress, hybrid, and stress-floor residual gates.
- Data version: local raw train horizontal wells, strict fold assignments, typewell descriptors, and direct formation columns used only as fold-local pseudo-top labels.
- Split / CV: primary evaluation on `strict_hybrid_fold`; all candidates also scored on `analog_distance_fold`.
- Model: best strict candidate `blend_plus_ridge_residual_s15_uniform`; best lower-stress compromise `blend_plus_ridge_residual_s15_margin`.
- Primary metric: best strict RMSE **13.0769**; best lower-stress compromise strict RMSE **13.0853**.
- Secondary metrics: best strict stress RMSE **13.4468**; lower-stress compromise stress RMSE **13.4079**; original regime mean stress reference **13.4006**.
- Artifacts: summary, fold metrics, regime diagnostics placeholder, zone/analog-bin summaries, well diagnostics, findings, markdown report, model comparison row, and run bundle under `artifacts/logs/runs/2026-07-01__xgboost_regime_stress_gated_stack/`.
- Decision: diagnostic only. Stress gates do not beat the original regime mean on both strict and stress.
- Notes: Move next to pseudo-top inference tuning or partial-pooling regime means.

## 2026-07-01__pseudo_top_probability_scale_tuning

- Task: pseudo-top probability-scale tuning for direct regime means
- Purpose: Test whether sharper or smoother pseudo-top softmax probabilities improve the direct regime-mean model.
- Config: `notebooks/modeling/08_alignment/11_pseudo_top_probability_scale_tuning.ipynb`; deterministic target-row sampling capped at **120** rows per well; two-seed squared-error pseudo-top proxy ensembles; probability scales **25**, **35**, **50**, **75**, **100**, and **150**; fixed global XGBoost and per-formation Ridge `alpha=80` regime means.
- Data version: local raw train horizontal wells, strict fold assignments, typewell descriptors, and direct formation columns used only as fold-local pseudo-top labels.
- Split / CV: primary evaluation on `strict_hybrid_fold`; all candidates also scored on `analog_distance_fold`.
- Model: best candidate `xgb_square_global__regime_ridge_alpha80__scale50__confidence_s100`.
- Primary metric: strict-hybrid RMSE **13.0928**, unchanged from the first direct regime-mean baseline.
- Secondary metrics: analog-distance stress RMSE **13.4006**, also best among tested scales.
- Artifacts: summary, fold metrics, regime diagnostics, zone/analog-bin summaries, well diagnostics, findings, markdown report, model comparison row, and run bundle under `artifacts/logs/runs/2026-07-01__pseudo_top_probability_scale_tuning/`.
- Decision: diagnostic only. The original scale **50** is already the best tested probability calibration.
- Notes: Move next to a structurally different pseudo-top inference model or Bayesian partial pooling; do not continue simple scale sweeps.

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
