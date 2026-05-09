# Modeling Assumptions

## A1 — Row index = submission id index

**Assumption:** The integer in `{WELLNAME}_{row_index}` matches the 0-based (or original) CSV row index of the horizontal well file.
**Basis:** sample_submission.csv id format; competition documentation.
**Risk:** If Kaggle reindexes files, id mapping breaks.
**Test:** Verify that all sample_submission ids resolve to valid rows in test horizontal_well files.

---

## A2 — TVT_input NaN marks evaluation zone

**Assumption:** Rows where `TVT_input` is NaN are exactly the rows to predict; rows where `TVT_input` is non-NaN are "known" and can be used as context.
**Basis:** Competition description: "TVT_input is a copy of TVT provided as a feature, but it contains NaN values for the evaluation zone."
**Risk:** Some NaN rows may be non-evaluation (data quality). Some non-NaN rows may be in the evaluation zone (partial reveal).
**Test:** Count NaN in TVT_input vs. ids in sample_submission per well; they should match.

---

## A3 — Formation columns absent in test

**Assumption:** `ANCC`, `ASTNU`, `ASTNL`, `EGFDU`, `EGFDL`, `BUDA` are not available in the test horizontal well files.
**Basis:** Competition description: "These are marked as training-only."
**Risk:** If they are available in test, we leave signal on the table.
**Test:** Check test file column list on first data load; log to `context/DECISIONS.md`.

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
