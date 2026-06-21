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

## Feature Log

Record feature additions here when they are used in reportable experiments:

| Date | Feature family | Files | Reason | Leakage notes |
|---|---|---|---|---|
| 2026-06-21 | Initial documented feature families | `src/preprocessing/features.py` | Existing scaffold orientation | Train-only formation columns excluded by `get_feature_columns`. |
| 2026-06-21 | Simple linear baseline features | `notebooks/modeling/01_linear/01_ols_ridge_lasso.ipynb` | First OLS/Ridge/Lasso sweep | Uses raw trajectory, clipped/imputed GR, forward-filled `TVT_input`, known-zone flag, MD position, and trajectory deltas. Trains only on known-zone rows and scores target-zone rows. |
| 2026-06-21 | Simple tabular tree baseline features | `src/preprocessing/tabular.py`, `notebooks/modeling/02_trees/01_random_forest.ipynb`, `notebooks/modeling/02_trees/02_extra_trees.ipynb`, `kaggle/random_forest_submission/random_forest_submission.ipynb` | First Random Forest / Extra Trees sweep and Random Forest Kaggle submission | Reuses the same 15 test-safe numeric features; no train-only formation columns. Tree training uses deterministic row caps on known-zone rows and scores target-zone rows. |
| 2026-06-21 | Simple tabular KNN features | `src/preprocessing/tabular.py`, `notebooks/modeling/03_svr_neighbors/02_knn_regression.ipynb` | First KNN sweep | Reuses the same 15 test-safe numeric features with train-fold standardization. KNN scored deterministic target-zone validation samples for runtime control; no train-only formation columns. |
