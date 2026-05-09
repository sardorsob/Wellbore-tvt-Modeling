# Changelog

## [Unreleased]

### Added
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
