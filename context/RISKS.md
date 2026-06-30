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

---

## R8 - Hierarchical well effects fail on hidden wells

**Risk:** Raw well-ID random effects or multitask GP task effects can improve local grouped validation but fail on hidden Kaggle wells that were not observed during training.
**Likelihood:** High if hierarchy is keyed directly by train well ID.
**Impact:** Public/private leaderboard regression despite attractive local RMSE.
**Mitigation:**
- Make hierarchy contextual: infer well-level corrections from test-safe known-zone summaries, boundary geometry, trajectory descriptors, GR availability, and residual diagnostics.
- Start with residual-bias audits and empirical Bayes offsets before full hierarchical HSGP.
- Require grouped-well validation and fold-level residual diagnostics before Kaggle submission.
- Shrink uncertain well-level corrections toward zero or the global mean.

---

## R9 - Full hierarchical HSGP becomes too slow or opaque

**Risk:** A full hierarchical Hilbert-space GP with multiple additive terms, well-level priors, and Student-t likelihood can be slow enough that iteration quality drops, or complex enough that failures are hard to diagnose.
**Likelihood:** Medium to high.
**Impact:** Wasted notebook cycles, non-finite optimization, or leaderboard submissions with unclear failure modes.
**Mitigation:**
- Keep the first hierarchy notebooks simple: residual audit, empirical Bayes offsets, contextual Huber hierarchy, then Hilbert residual.
- Use the stable lower-step clipped optimization settings from additive HSGP tuning.
- Do not increase basis count or hierarchy depth without a diagnostic reason.
- Package for Kaggle only after local diagnostics show a meaningful gain or a clear public-generalization hypothesis.

---

## R10 - Alignment and analog-well features can leak fold or hidden-test information

**Risk:** Typewell, nearest-well, DTW, pseudo-zone, or residual-kriging features can accidentally use held-out wells, visible test rows, hidden test structure, or target-zone TVT values when building analog descriptors.
**Likelihood:** Medium to high if implemented quickly.
**Impact:** Inflated local CV, public/private leaderboard regression, or invalid Kaggle submission logic.
**Mitigation:**
- Build alignment references inside each grouped fold from training wells only.
- For Kaggle submissions, derive test-well descriptors only from test-safe known-zone rows and competition-provided auxiliary files.
- Never use target-zone `TVT` or residuals from the scored well to construct prediction features.
- Add notebook assertions that alignment maps, pseudo-zone templates, and residual-spatial references exclude validation well IDs during CV.

---

## R11 - Pseudo-stratigraphic features can become confident wrong geology

**Risk:** Hidden-test-safe pseudo-top, pseudo-zone, or typewell-alignment features may assign the wrong formation-relative state with high confidence, especially when GR motifs repeat or the target well is structurally novel.
**Likelihood:** High if pseudo-labels are used as hard categorical features.
**Impact:** Large public/private regressions, because a tree or GP can amplify wrong boundary proximity and wrong formation-distance signals.
**Mitigation:**
- Prefer soft/uncertainty features over hard pseudo-labels: best score, second-best score, score margin, alignment entropy, DTW distance, lag, stretch, and near-boundary flags.
- Stress-test models on wells with poor typewell correlation, high analog distance, short known zones, high extrapolation distance, and high GR missingness.
- Keep direct train-only formation columns only in oracle diagnostics; never use them in submission feature sets.
- Add fold assertions that pseudo-state models are fit without held-out wells and use only known-zone evidence for validation/test wells.
