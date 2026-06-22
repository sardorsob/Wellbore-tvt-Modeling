# Changelog

## [Unreleased]

### Changed
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
