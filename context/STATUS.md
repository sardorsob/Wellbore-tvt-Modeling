# Project Status

**Last updated:** 2026-05-08
**Current phase:** Data Exploration — structural gate **complete**; geology/signal EDA next

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
- [x] Download competition data (`python scripts/download_data.py`)
- [x] Phase 0–1 EDA notebook (`notebooks/eda/01_phase0_phase1_data_and_schema.ipynb`) — counts, schema spot-check, 1A–1E
- [x] Verify row index → submission id mapping (**A1** — pass)
- [x] Verify TVT_input NaN rows ↔ submission ids (**A2** — pass)
- [x] Check test file columns (**A3** / **D002** — formation columns absent in test; decision unchanged)
- [x] Add local agent rules and Kaggle tracking docs from `sardorsob/sardors-agentic-workflows` guidance
- [x] Run interpolation baseline (`2026-06-21__1216__interpolation_baseline__14bef53`) — CV RMSE mean **15.9003**
- [x] Run first OLS/Ridge/Lasso sweep (`2026-06-21__1218__linear_ols_ridge_lasso__14bef53`) — best **ridge_alpha_10**, CV RMSE mean **15.7660**

## In progress

- [ ] Phase 2 geological / domain EDA (`notebooks/eda/02_phase2_geological_domain.ipynb`)
- [ ] Export artifact figures per `CURSOR_RULES.md`
- [ ] Verify Kaggle competition rules page before first submission (`kaggle/RULES_SUMMARY.md`)

## Next up

1. Phase 2 notebook (TVT vs MD, GR, typewell correlation, XY, dip) → figures under `artifacts/figures/`
2. Expand feature EDA / leakage review for linear and tree features
3. Train tree baseline with hand-crafted features
4. Set up Kaggle Notebook for reproducible local-best baseline
5. Set up Bayesian GP (NumpPyro)
6. First Kaggle Notebook commit + submission

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
