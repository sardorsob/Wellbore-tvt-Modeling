# Changelog

## [Unreleased]

### Changed
- Logged Phase 0–1 structural EDA results from `notebooks/eda/01_phase0_phase1_data_and_schema.ipynb` into `structure.md`, `ASSUMPTIONS.md`, `DECISIONS.md`, `DATASETS.md`, and `STATUS.md`.

### Added
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

```
## [YYYY-MM-DD] — <Short milestone name>
### Added / Changed / Fixed / Removed
- ...

### Results
- Local CV RMSE: ...
- Kaggle LB RMSE: ...
- Run ID: ...
```
