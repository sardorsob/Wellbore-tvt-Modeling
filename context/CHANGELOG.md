# Changelog

## 2026-06-22

- Added and ran `notebooks/modeling/07_hierarchical/00_residual_bias_audit.ipynb`; exported well summary, descriptor correlations, fold summary, findings, comparison row, report, and run bundle.
- Recorded A29: residual bias exists after Huber + additive Hilbert residuals, but descriptor signal is modest, so the next empirical Bayes offsets should be conservative.
- Distributed the hierarchical modeling plan across existing context files instead of keeping a standalone roadmap document.
- Added D014 for the staged contextual hierarchy decision and R8/R9 for hierarchy-specific risks.
- Updated `context/STATUS.md` with the hierarchical task ladder, promotion gates, and next modeling steps.
- Added assumption A28 and a planned `07_hierarchical` experiments entry.

## 2026-06-23

- Added self-contained `kaggle/huber_hilbert_analog_submission/` notebook and locally validated a 14,151-row submission CSV.
- Added and ran `notebooks/modeling/08_alignment/06_alignment_hsgp_residual.ipynb`.
- Recorded A43: direct analog transfer carries the alignment signal; learned residual heads do not improve on it.
- Added and ran `notebooks/modeling/08_alignment/05_analog_selection_refinement.ipynb`.
- Recorded A42: log-plus-trajectory nearest-analog selection clears the alignment promotion gate with **0.1003** same-sample RMSE gain.
- Added and ran `notebooks/modeling/08_alignment/04_spatial_residual_correlation.ipynb`.
- Recorded A41: spatial residual correlation is weak under current descriptors, so kriging/spatial GP is deferred.
- Added and ran `notebooks/modeling/08_alignment/03_progress_gated_analog.ipynb`.
- Recorded A40: progress-gated analog correction confirms the late-zone pattern but does not unlock a larger gain.
- Added and ran `notebooks/modeling/08_alignment/02_pseudo_zone_huber.ipynb`.
- Recorded A39: learned pseudo-zone analog heads help late target-zone rows but do not beat fixed nearest-analog shrink.
- Added and ran `notebooks/modeling/08_alignment/01_multi_analog_zone_blend.ipynb`.
- Recorded A38: distance-weighted multi-analog transfer is useful but does not beat the simpler nearest-analog direct-transfer screen.
- Added and ran `notebooks/modeling/08_alignment/00_typewell_alignment_feature_screen.ipynb`.
- Recorded A37: nearest-analog target-zone transfer improved the same residual sample by **0.0419** RMSE, just below the feature-screen promotion gate.
- Updated status and experiments to continue alignment modeling with multi-analog or pseudo-zone analog features instead of submitting the first direct-transfer screen.
- Added and ran `notebooks/modeling/07_hierarchical/01_empirical_bayes_offsets.ipynb`.
- Recorded A30: simple contextual well offsets do not improve enough; best same-sample gain was only **0.0008** RMSE and descriptor-based offsets worsened RMSE.
- Updated status and experiments to mark empirical Bayes offsets as a negative screen, not a promotion candidate.
- Added and ran `notebooks/modeling/07_hierarchical/03_hierarchical_residual_slope.ipynb`.
- Recorded A31: shallow residual-shape corrections do not improve enough; best same-sample gain was only **0.0003** RMSE.
- Updated status and experiments to mark residual-shape hierarchy as a negative screen, not a promotion candidate.
- Added and ran `notebooks/modeling/07_hierarchical/02_contextual_hierarchical_huber.ipynb`.
- Recorded A32: contextual descriptor interactions show weak positive signal (**0.0064** same-sample RMSE gain), but still fail the hierarchy promotion gate.

## [Unreleased]

### Changed
- Recorded the 2026-06-23 research pass across geologic alignment, well-top recommender systems, DTW/log alignment, HSGP/GP kernels, multitask GP, Student-t GP likelihoods, and geostatistical residual modeling; updated assumptions, risks, decisions, and task ladders to pivot from shallow residual hierarchy to fold-safe typewell/analog feature screening.
- Pushed and submitted the self-contained Huber + additive Hilbert residual notebook; public LB scored **14.514**, the current best external result.
- Ran HSGP residual diagnostics on GAM/Huber means and recorded that Huber + additive Hilbert residuals reach **14.3586** local RMSE, while the reconstructed GAM residual path was unstable.
- Pushed and submitted the self-contained additive HSGP Student-t MAP notebook; public LB scored **15.255**, better than Bayesian linear and Huber but worse than fine-tuned GAM.
- Ran additive NumPyro/JAX HSGP residual tuning and recorded that one-dimensional boundary/progress/slope-instability smooth terms beat the first joint HSGP but still trail the Huber GLM local diagnostic.
- Ran the first NumPyro/JAX HSGP residual baseline and recorded that a small boundary/progress HSGP improves its same-notebook linear reference but not the prior tuned Bayesian best.
- Pushed and submitted the self-contained Bayesian linear Student-t MAP notebook; public LB scored **15.512**, worse than Huber GLM and fine-tuned GAM.
- Ran Bayesian posterior predictive diagnostics with targeted heteroskedastic scale models and recorded that boundary/progress scale helps calibration but does not beat the tuned fixed Student-t Bayesian baseline.
- Ran Bayesian linear tuning and recorded that fixed-tail Student-t AutoDelta improves the first Bayesian baseline but still trails the Huber GLM local diagnostic.
- Ran the first NumPyro/JAX Bayesian linear robust baseline and recorded that heteroskedastic Student-t improves uncertainty calibration but not point-RMSE versus Huber/GAM.
- Pushed and submitted the self-contained Huber GLM slope-core notebook; public LB scored **15.464**, worse than the fine-tuned GAM despite better local diagnostic RMSE.
- Ran GLM distribution diagnostics and recorded that robust signed-delta Huber regression beats the tuned GAM local diagnostic, while Beta/Dirichlet are invalid target geometries.
- Pushed and submitted the self-contained fine-tuned GAM spline Kaggle notebook; public LB improved to **15.055**.
- Ran a tuned GAM-style spline sweep and recorded that slope-core splines plateau around **14.8687** sampled target-zone CV RMSE.
- Ran the first GAM-style additive spline baseline and recorded that the best boundary/trajectory/slope additive model beats sampled endpoint hold under target-zone grouped CV.
- Completed Phase 4 validation EDA and recorded the validation protocol for endpoint-hold baselines, fold descriptor balance, model fold metrics, and public leaderboard gap interpretation.
- Completed Phase 3 feature-signal EDA and recorded the resulting assumptions for endpoint hold, unstable direct slope extrapolation, additive feature-family signal, GR availability, and cautious typewell residual use.
- Completed Phase 2 geological/domain EDA and recorded the resulting assumptions for long trailing-zone extrapolation, GR missingness, typewell alignment, train-only formation caveats, and local-test overlap.
- Recorded the ElasticNet Kaggle submission and leaderboard result (`ref=53924934`, public LB `16.091`), including the decision not to promote it.
- Ran the ElasticNet / Huber / median Quantile linear-family sweep and recorded that the best ElasticNet candidate did not beat Ridge locally.
- Ran the first KNN neighbor baseline and recorded the decision to reject vanilla global KNN on the current simple feature space.
- Recorded the Random Forest Kaggle submission and leaderboard result (`ref=53924036`, public LB `15.873`), including the validation mismatch against Ridge.
- Added first tree-baseline experiment results for Random Forest and Extra Trees, including exported metrics, plots, run bundles, and comparison-table rows.
- Logged Phase 0-1 structural EDA results from `notebooks/eda/01_phase0_phase1_data_and_schema.ipynb` into `structure.md`, `ASSUMPTIONS.md`, `DECISIONS.md`, `DATASETS.md`, and `STATUS.md`.
- Generated and ran first baseline modeling notebooks; updated structure, status, experiments, and feature log with run IDs and RMSEs.

### Added
- Self-contained inference-safe LightGBM strict-CV Kaggle package under `kaggle/lightgbm_strict_submission/`; Kaggle version 1 completed with 14,151 prediction rows and submission `54097453` is pending public score.
- LightGBM strict-CV notebook, stress-fold diagnostics, feature importance, reports, model-comparison row, and run bundle. Best `lgbm_medium_l2` scored **13.2363** on `strict_hybrid_fold` and **13.4453** on `analog_distance_fold`.
- Strict CV fold-design notebook, fold-assignment tables, descriptor/stress diagnostics, report, and run bundle. Recommended `strict_hybrid_fold` for first boosted-tree strict CV and `analog_distance_fold` as the strictest diagnostic stress test.
- Validation mismatch audit notebook, exported transfer/analog/decision tables, run bundle, and report after analog-gate public LB **14.603** failed to beat Huber + Hilbert **14.514**.
- Kaggle notebook version 1 submission for Huber + additive Hilbert plus direct analog gate (`ref=53999839`); Kaggle run completed with 14,151 prediction rows and public LB **14.603**.
- Self-contained Huber + additive Hilbert residual Kaggle submission notebook, kernel metadata, and local validation CSV.
- HSGP residual-on-GAM/Huber notebook, residual diagnostics, figures, report, model-comparison row, and run bundle.
- Self-contained additive HSGP Student-t MAP Kaggle submission notebook, kernel metadata, and local validation CSV.
- Additive HSGP residual tuning notebook, posterior diagnostics, residual figures, report, model-comparison row, and run bundle.
- HSGP residual baseline notebook, posterior diagnostics, residual figures, report, model-comparison row, and run bundle.
- Self-contained Bayesian linear Student-t MAP Kaggle submission notebook, kernel metadata, and local validation CSV.
- Bayesian posterior diagnostics and targeted-scale notebook, bin/well diagnostics, calibration figures, report, model-comparison row, and run bundle.
- Bayesian linear tuning notebook, posterior diagnostics, residual/coverage figures, report, model-comparison row, and run bundle.
- Bayesian linear robust notebook, posterior predictive diagnostics, calibration figures, report, model-comparison row, and run bundle.
- Self-contained Huber GLM Kaggle submission notebook, kernel metadata, and local validation CSV.
- GLM distribution diagnostic notebook, family-validity table, target diagnostics, residual/RMSE figures, report, model-comparison row, and run bundle.
- Self-contained fine-tuned GAM spline Kaggle submission notebook, kernel metadata, and local validation CSV.
- Tuned GAM-style spline notebook, candidate/fold/family/target summaries, residual diagnostics, report, model-comparison row, and run bundle.
- GAM-style modeling folder, executable additive spline baseline notebook, fold metrics, summary tables, residual diagnostics, report, model-comparison row, and run bundle.
- Phase 4 EDA notebook execution artifacts: endpoint baseline table, fold assignment and descriptor-balance tables, model fold metrics, CV/leaderboard gap table, descriptor PCA table, diagnostic figures, findings report, and run bundle.
- Phase 3 EDA notebook execution artifacts: bounded candidate feature sample preview, feature signal table, diagnostic feature-family probe tables, visible-test drift table, GR missingness table, diagnostic figures, findings report, and run bundle.
- Phase 2 EDA notebook execution artifacts: train/test/typewell/formation summary tables, diagnostic figures, findings table, markdown report, and run bundle.
- Self-contained Kaggle ElasticNet submission notebook and locally validated submission CSV.
- Scikit-learn linear model factory, focused tests, executable ElasticNet/Huber/Quantile notebook, and reportable run artifacts.
- KNN neighbor model factory, focused tests, executable KNN notebook, and reportable KNN run artifacts.
- Self-contained Kaggle Random Forest submission notebook and kernel metadata under `kaggle/random_forest_submission/`.
- Locally validated Random Forest submission CSV under `artifacts/reports/`.
- Reusable simple tabular feature builder and deterministic row sampler for train/test-safe model baselines.
- Random Forest and Extra Trees model factory utilities plus focused tests.
- Self-contained Kaggle Ridge submission notebook and kernel metadata under `kaggle/linear_ridge_submission/`.
- Locally validated Ridge submission CSV under `artifacts/reports/`.
- First Kaggle code-competition submission log entry for notebook version 2 (`ref=53923555`, public LB `15.905`).
- Agent/project rules for deleting one-off notebook generators and logging Bayesian generative assumptions.
- NumPy OLS/Ridge/Lasso baseline model utilities and regression metric/validation helpers.
- Tests for metrics, validation masks/folds, linear models, and run-bundle writing.
- Run-bundle fallback config serialization when PyYAML is unavailable.
- Local agent rules in `.agents/AGENTS.md`, including the rule to never add AI co-author trailers to commits.
- Ignored local workflow reference path `context/_external/`.
- Kaggle operations docs under `kaggle/`: competition summary, rules summary, CV strategy, feature log, submission log, and leaderboard notes.
- `context/EXPERIMENTS.md` as the reportable run registry.
- Full repository scaffold per `rules_templet.md`
- `requirements.txt` with JAX, NumpPyro, Flax, XGBoost, LightGBM, PyMC, ArviZ
- `src/` package: io, preprocessing, modeling, evaluation, viz, utils
- All config files (baseline, xgboost, lgbm, bayesian_gp, bnn, ensemble)
- CLI scripts: download_data, train, predict, eval
- Context docs: PROJECT_BRIEF, GLOSSARY, DATASETS, INTERFACES, ASSUMPTIONS, DECISIONS, RISKS, STATUS
- Notebook stubs for EDA, modeling (6 approaches), and reporting
- Unit test stubs (loaders, features, metrics, smoke)

---

## Milestone format

```text
## [YYYY-MM-DD] - <Short milestone name>
### Added / Changed / Fixed / Removed
- ...

### Results
- Local CV RMSE: ...
- Kaggle LB RMSE: ...
- Run ID: ...
```
