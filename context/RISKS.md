# Risks

## R1 — Kaggle 9-hour runtime limit

**Risk:** Bayesian GP / BNN with HMC may exceed the 9-hour notebook limit.
**Likelihood:** High for full MCMC on large datasets.
**Impact:** Cannot submit Bayesian model as the primary approach.
**Mitigation:**
- Use SVI (variational inference) instead of HMC for BNN — 10-100x faster.
- For GP, use sparse/inducing-point approximations (NumpPyro `SparseGP`).
- Profile runtime locally before committing Kaggle Notebook.
- Have a fast classical ML fallback (XGBoost) ready.

---

## R2 — Feature mismatch between train and test

**Risk:** Training on columns absent from test causes silent KeyError or zero-imputed features.
**Likelihood:** Medium — formation columns (`ANCC`, etc.) are train-only per competition docs.
**Impact:** Submission failure or large performance drop.
**Mitigation:**
- Explicitly define test-safe feature list in `configs/` and `src/preprocessing/features.py`.
- Load test file columns at inference time and assert feature parity.
- Unit test: `test_features.py::test_feature_list_matches_test_schema`.

---

## R3 — Row index mapping error

**Risk:** Predicted TVT is assigned to the wrong submission id due to index offset.
**Likelihood:** Low but catastrophic if wrong.
**Impact:** Zero score (all predictions off by ≥1 row).
**Mitigation:**
- Always `preserve index` when loading CSVs (`index_col=0` or reset after load).
- Validate that all ids in sample_submission.csv resolve to rows in test files (see `tests/test_loaders.py`).
- Add assertion before saving submission.

---

## R4 — Data quality: GR missingness / outliers

**Risk:** Gamma ray log contains NaN or extreme values that distort features.
**Likelihood:** Medium (wireline logs often have gaps).
**Impact:** Feature NaN propagation into model → silent prediction failure.
**Mitigation:**
- Impute GR NaN with rolling median during preprocessing.
- Clip GR to [0, 300] API units (physical range).
- Log imputation statistics in EDA notebook.

---

## R5 — JAX + Windows compatibility

**Risk:** JAX has limited Windows support; certain GPU/XLA features may not work.
**Likelihood:** Medium (JAX CPU on Windows is fine; GPU via WSL or Kaggle is better).
**Impact:** Slow local development for JAX-based models.
**Mitigation:**
- Use JAX CPU for local development; validate on Kaggle GPU.
- Alternatively, use WSL2 for local JAX GPU.
- NumpPyro SVI on CPU is fast enough for development.

---

## R6 — Overfitting to training wells

**Risk:** With many features and small number of wells, models overfit training wells.
**Likelihood:** High.
**Impact:** Poor generalization to test wells.
**Mitigation:**
- Use leave-one-well-out cross-validation for model selection.
- Strong regularization on all models.
- Bayesian models with proper priors provide natural regularization.
- Monitor train vs. CV RMSE gap.

---

## R7 — Kaggle Notebook kernel restart clears variables

**Risk:** Kaggle Notebook runtime restarts between cells erase in-memory state.
**Likelihood:** Low during a single committed run.
**Impact:** Submission not generated.
**Mitigation:**
- Notebook writes intermediate results to disk.
- Final cell checks `/kaggle/working/submission.csv` exists and is valid.
