# Project Status

**Last updated:** 2026-05-08
**Current phase:** Setup / Data Exploration

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

## In progress

- [ ] Download competition data (`python scripts/download_data.py`)
- [ ] Run EDA notebook (`notebooks/eda/01_data_exploration.ipynb`)
- [ ] Verify row index → submission id mapping (Assumption A1)
- [ ] Check test file columns (Assumption A3 / Decision D002)

## Next up

1. Complete EDA → update `context/structure.md` with findings
2. Build and validate interpolation baseline
3. Train XGBoost with hand-crafted features
4. Set up Bayesian GP (NumpPyro)
5. First Kaggle Notebook commit + submission

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
| — | — | — | — | No submissions yet |
