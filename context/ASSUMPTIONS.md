# Modeling Assumptions

## A1 — Row index = submission id index

**Assumption:** The integer in `{WELLNAME}_{row_index}` matches the 0-based (or original) CSV row index of the horizontal well file.
**Basis:** sample_submission.csv id format; competition documentation.
**Risk:** If Kaggle reindexes files, id mapping breaks.
**Test:** Verify that all sample_submission ids resolve to valid rows in test horizontal_well files.

**Verification (notebook `notebooks/eda/01_phase0_phase1_data_and_schema.ipynb`, Phase 1A):** **Passed.** For all **3** test wells, every `row_idx` appearing in `sample_submission.csv` for that well exists in the corresponding horizontal-well DataFrame index loaded with `load_horizontal_well()` (default CSV index preserved).

---

## A2 — TVT_input NaN marks evaluation zone

**Assumption:** Rows where `TVT_input` is NaN are exactly the rows to predict; rows where `TVT_input` is non-NaN are "known" and can be used as context.
**Basis:** Competition description: "TVT_input is a copy of TVT provided as a feature, but it contains NaN values for the evaluation zone."
**Risk:** Some NaN rows may be non-evaluation (data quality). Some non-NaN rows may be in the evaluation zone (partial reveal).
**Test:** Count NaN in TVT_input vs. ids in sample_submission per well; they should match.

**Verification (Phase 1D):** **Passed.** For each test well, the set of DataFrame indices where `TVT_input` is NaN equals exactly the set of `row_idx` values parsed from `sample_submission.csv` for that well.

---

## A3 — Formation columns absent in test

**Assumption:** `ANCC`, `ASTNU`, `ASTNL`, `EGFDU`, `EGFDL`, `BUDA` are not available in the test horizontal well files.
**Basis:** Competition description: "These are marked as training-only."
**Risk:** If they are available in test, we leave signal on the table.
**Test:** Check test file column list on first data load; log to `context/DECISIONS.md`.

**Verification (Phase 1B):** **Passed.** Union of column names across **all** test horizontal-well CSV headers (**3** wells in this download) contains **none** of `{ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA}`. Logged under D002 follow-up in `context/DECISIONS.md`. If Kaggle adds more test wells, re-run the header union check.

---

## A4 — Typewell GR is a good vertical reference

**Assumption:** The typewell GR profile provides useful signal for correlating the horizontal well GR to TVT.
**Basis:** Standard oil & gas practice — typewells define the formation signature.
**Risk:** Poor typewell quality or mismatch between typewell and horizontal well geology.
**Test:** Plot horizontal GR vs. typewell GR at known TVT_input positions; check correlation.

---

## A5 — Sequential/spatial structure in TVT

**Assumption:** TVT varies smoothly along MD (measured depth) and can be modeled with lagged features or GP kernels.
**Basis:** Geology is continuous; TVT reflects stratigraphic dip which changes gradually.
**Risk:** Faults or large dip changes could cause jumps that require step-function models.
**Test:** Plot TVT vs. MD for training wells; look for discontinuities.

---

## A6 — Train/test wells are from the same geological area

**Assumption:** Training wells provide useful priors for test well TVT patterns.
**Basis:** Competition design — organizer provided training data to enable prediction.
**Risk:** Test wells could be from different fields/formations with different TVT behavior.
**Test:** Compare GR distributions between train and test wells.

---

## A7 — RMSE-optimal prediction is the posterior mean (for Bayesian models)

**Assumption:** For the Bayesian GP/BNN, the posterior mean minimizes expected squared error on the leaderboard.
**Basis:** Standard decision theory.
**Risk:** If the leaderboard uses a different metric or if the posterior is multimodal, median may be better.
**Test:** Compare mean vs. median predictions on validation set.

---

## A8 — No well-to-well data leakage in sample_submission

**Assumption:** Each test well is independent; predicting one well's TVT does not depend on another test well's predictions.
**Basis:** Geological wells are physically separate.
**Risk:** Low.
**Mitigation:** None needed.

---

## A9 — Bayesian models require explicit generative assumptions

**Assumption:** Bayesian GP/BNN results are only reportable after the generative story, priors, likelihood, and predictive-check expectations are written down outside the notebook output.
**Basis:** Bayesian model performance and uncertainty depend directly on prior and likelihood choices.
**Risk:** If assumptions live only in code cells, later results may be hard to compare or audit.
**Mitigation:** Record Bayesian assumptions in this file, `context/EXPERIMENTS.md`, and the run bundle notes before logging a Bayesian result as selected or rejected.
---

## Phase 2 EDA Addendum - 2026-06-21

**Notebook:** `notebooks/eda/02_phase2_geological_domain.ipynb`
**Run ID:** `2026-06-21__1506__eda_phase2_geological_domain__542b1c1`

### A4 update - Typewell GR is useful but noisy

Phase 2 measured horizontal/typewell GR correlation at known `TVT_input` positions. The median correlation is **0.8125**, which supports the typewell as a useful vertical reference, but the notebook also treats this as noisy enough to avoid raw one-to-one matching as a primary feature.

**Modeling implication:** Build robust typewell summaries, alignment scores, lagged/correlated windows, or smoothed typewell-derived features rather than trusting a single raw matched GR value.

### A5 update - Prediction is long trailing-zone extrapolation

Across train wells, the evaluation-zone fraction median is **0.740**. The local example test wells have evaluation-zone fractions from **0.674** to **0.796**. The known-zone median `dTVT/dMD` is **0.3200**, while the median last-50 known-row slope is **0.0000**.

**Modeling implication:** The core prediction problem is not random row interpolation. It is extrapolation through a long trailing block after the known boundary. Endpoint TVT, distance past boundary, normalized within-well position, recent-window slope, and smoothness priors should matter. Bayesian models should encode a boundary-conditioned generative story before fitting.

### A6 update - Local test files are diagnostic, not distribution evidence

The visible local test wells overlap train wells exactly: `000d7d20`, `00bbac68`, and `00e12e8b`. Their nearest train summary distances are zero because of that overlap.

**Modeling implication:** Use the local test files for submission-shape checks and notebook runtime validation only. Do not infer hidden-test distribution from their descriptor values, and do not tune modeling choices to local test similarity.

### A10 - GR missingness is part of the data-generating process

Mean GR missingness is **0.294** in train and **0.215** in the visible local test files. Phase 1 also showed GR missingness differs between known and evaluation zones.

**Assumption:** GR availability is not ignorable noise; it may reflect acquisition, processing, or interval-specific logging behavior.
**Modeling implication:** Add explicit GR missingness indicators, within-window availability summaries, and imputation choices before boosted-tree or Bayesian models. For Bayesian work, consider whether missing GR should be modeled as observed-with-missingness rather than silently imputed.
