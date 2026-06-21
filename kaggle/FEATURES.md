# Features

## Test-Safe Raw Columns

- `MD`
- `X`
- `Y`
- `Z`
- `GR`
- `TVT_input`

## Train-Only Columns

Do not use these in train/test parity feature sets:

- `ANCC`
- `ASTNU`
- `ASTNL`
- `EGFDU`
- `EGFDL`
- `BUDA`
- `TVT`

## Current Engineered Feature Families

Defined in `src/preprocessing/features.py`:

- `row_idx`
- `tvt_input_lag_*`
- `tvt_input_rolling_*`
- GR rolling statistics
- MD and trajectory derivatives
- cumulative horizontal displacement
- typewell GR interpolation at TVT proxy positions
- typewell formation labels for analysis
- known-zone flag
- normalized within-well MD position

## Candidate Feature Families From Phase 2 EDA

These are not yet promoted into reportable modeling runs unless listed in the feature log below.

- Boundary and endpoint features: last known `TVT_input`, last known `MD`, distance past known/evaluation boundary, and normalized evaluation-zone progress.
- Recent-window shape features: last-10/25/50 known-zone slopes, recent TVT range, endpoint smoothness, and local curvature proxies.
- GR availability features: row-level `GR` missingness, rolling GR missing fraction, known/eval-zone GR coverage summaries, and imputation source flags.
- Typewell alignment features: robust horizontal/typewell GR correlation, smoothed typewell GR at TVT proxy positions, local alignment-window scores, and geology-label summaries used only when train/test-safe.
- Spatial/trajectory features: XY displacement, trajectory derivatives, and interactions with boundary distance.

## Feature Log

Record feature additions here when they are used in reportable experiments:

| Date | Feature family | Files | Reason | Leakage notes |
|---|---|---|---|---|
| 2026-06-21 | Initial documented feature families | `src/preprocessing/features.py` | Existing scaffold orientation | Train-only formation columns excluded by `get_feature_columns`. |
| 2026-06-21 | Simple linear baseline features | `notebooks/modeling/01_linear/01_ols_ridge_lasso.ipynb` | First OLS/Ridge/Lasso sweep | Uses raw trajectory, clipped/imputed GR, forward-filled `TVT_input`, known-zone flag, MD position, and trajectory deltas. Trains only on known-zone rows and scores target-zone rows. |
| 2026-06-21 | Simple robust linear features | `src/preprocessing/tabular.py`, `notebooks/modeling/01_linear/02_elasticnet_huber_quantile.ipynb`, `kaggle/elasticnet_submission/elasticnet_submission.ipynb` | ElasticNet / Huber / median Quantile sweep and ElasticNet Kaggle submission | Reuses the same 15 test-safe numeric features with train-fold standardization. No train-only formation columns. |
| 2026-06-21 | Simple tabular tree baseline features | `src/preprocessing/tabular.py`, `notebooks/modeling/02_trees/01_random_forest.ipynb`, `notebooks/modeling/02_trees/02_extra_trees.ipynb`, `kaggle/random_forest_submission/random_forest_submission.ipynb` | First Random Forest / Extra Trees sweep and Random Forest Kaggle submission | Reuses the same 15 test-safe numeric features; no train-only formation columns. Tree training uses deterministic row caps on known-zone rows and scores target-zone rows. |
| 2026-06-21 | Simple tabular KNN features | `src/preprocessing/tabular.py`, `notebooks/modeling/03_svr_neighbors/02_knn_regression.ipynb` | First KNN sweep | Reuses the same 15 test-safe numeric features with train-fold standardization. KNN scored deterministic target-zone validation samples for runtime control; no train-only formation columns. |
| 2026-06-21 | Phase 2 EDA feature candidates | `notebooks/eda/02_phase2_geological_domain.ipynb`, `artifacts/reports/eda__phase2_geological_domain__2026-06-21.md` | Guide the next richer feature builder before LightGBM/XGBoost and Bayesian models | Candidate-only; do not use train-only formation columns in submission models. Local visible test wells overlap train, so they are not valid hidden-test drift evidence. |
