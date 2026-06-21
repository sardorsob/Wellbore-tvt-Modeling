# Changelog

## [Unreleased]

### Changed
- Ran the ElasticNet / Huber / median Quantile linear-family sweep and recorded that the best ElasticNet candidate did not beat Ridge locally.
- Ran the first KNN neighbor baseline and recorded the decision to reject vanilla global KNN on the current simple feature space.
- Recorded the Random Forest Kaggle submission and leaderboard result (`ref=53924036`, public LB `15.873`), including the validation mismatch against Ridge.
- Added first tree-baseline experiment results for Random Forest and Extra Trees, including exported metrics, plots, run bundles, and comparison-table rows.
- Logged Phase 0-1 structural EDA results from `notebooks/eda/01_phase0_phase1_data_and_schema.ipynb` into `structure.md`, `ASSUMPTIONS.md`, `DECISIONS.md`, `DATASETS.md`, and `STATUS.md`.
- Generated and ran first baseline modeling notebooks; updated structure, status, experiments, and feature log with run IDs and RMSEs.

### Added
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
