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

---

## Phase 3 EDA Addendum - 2026-06-21

**Notebook:** `notebooks/eda/03_phase3_feature_signal.ipynb`
**Run ID:** `2026-06-21__1847__eda_phase3_feature_signal__ebaa6c3`

### A11 - Endpoint hold is a strong diagnostic baseline

On sampled train target-zone rows, predicting every target row as the last known `TVT_input` gives diagnostic grouped RMSE **15.5815**, close to the best reportable local Ridge baseline.

**Modeling implication:** The target-zone delta from the boundary is often small enough that endpoint TVT must be treated as a first-class feature and baseline. Any richer model should prove it improves over endpoint hold, not only over interpolation.

### A12 - Naive slope extrapolation is risky

The last-50 known-zone slope projection gives diagnostic grouped RMSE **104.5322**, much worse than endpoint hold. Recent slopes are still signal-bearing in feature screening, but direct linear extrapolation from one recent slope is unstable.

**Modeling implication:** Use recent slopes as contextual features or priors, not as standalone deterministic extrapolators. Bayesian models should place conservative priors on boundary slope continuation.

### A13 - Candidate feature families carry additive target-zone signal

The diagnostic target-zone linear probe improves from endpoint hold RMSE **15.5815** to all-candidate RMSE **14.6236**. The strongest univariate Spearman signal for target-zone delta is `tvt_slope_last_10` (**0.199** absolute), while the strongest mutual-information feature is `z_slope_last_100` (**0.514**).

**Modeling implication:** Build the next production feature builder around boundary/endpoint, trajectory, recent-slope, GR-availability, and typewell features together. Do not rely on a single hand-picked raw feature.

### A14 - Typewell residuals are weak as raw row-level signals

`gr_minus_tw_proxy` has Spearman correlation **0.019** with target-zone delta in the Phase 3 sample.

**Modeling implication:** Typewell information may still help through robust summaries and alignment scores, but raw row-level typewell residuals should not be over-weighted without validation.

---

## Phase 4 EDA Addendum - 2026-06-21

**Notebook:** `notebooks/eda/04_phase4_validation.ipynb`
**Run ID:** `2026-06-21__1928__eda_phase4_validation__203566d`

### A15 - Endpoint hold is the validation floor

Full-row endpoint-hold RMSE is **15.9099**, essentially matching the earlier interpolation-style naive baseline. Direct last-50 slope projection is much worse at **112.4817** RMSE.

**Modeling implication:** Every future reportable model should be compared against endpoint hold. A direct recent-slope extrapolator should remain a diagnostic anti-baseline, not a candidate model.

### A16 - Fold balance must be reported, not assumed

The current sorted grouped folds have eval-row CV **0.003** and endpoint-RMSE range **1.466**. Descriptor-balanced folds reduce eval-row CV to **0.001**, but endpoint-RMSE range rises to **2.428**, showing that balancing one descriptor family does not automatically balance error difficulty.

**Modeling implication:** Validation reports should include descriptor balance and baseline difficulty by fold. No single local split should be treated as definitive.

### A17 - Public leaderboard disagreement is a validation warning

Random Forest beats Ridge on public LB by **0.032** RMSE, while local CV ranks Random Forest worse by **0.300** RMSE. ElasticNet has the largest public-minus-CV gap at **0.300**.

**Modeling implication:** Public LB should be a coarse external check only. Do not promote a modeling direction from small leaderboard gains unless local fold diagnostics and error analysis also support it.

---

## GAM-Style Modeling Addendum - 2026-06-21

**Notebook:** `notebooks/modeling/04_gam/01_gam_spline_baseline.ipynb`
**Run ID:** `2026-06-21__2007__gam_spline_baseline__8046b3f`

### A18 - Smooth additive structure has useful signal

The best GAM-style additive spline variant, `boundary_trajectory_slopes`, scored sampled grouped target-zone RMSE **14.8719**, compared with sampled endpoint-hold RMSE **15.5782**.

**Modeling implication:** Smooth additive structure is a worthwhile bridge between linear baselines and boosted trees. Boundary, trajectory, and recent-slope features should be included in the next model family.

### A19 - GAM results are target-zone diagnostic, not direct leaderboard proof

This GAM-style run trained and validated on sampled target-zone rows from train wells, while some earlier baselines trained on known-zone rows and scored target-zone rows.

**Modeling implication:** Treat the GAM result as strong evidence that target-zone feature construction matters, but create a self-contained Kaggle notebook before comparing it directly against public LB.

### A20 - GAM tuning mostly plateaus around slope-core splines

The tuned GAM sweep tested **45** candidates across target definition, spline knots, spline degree, Ridge alpha, and feature-family additions. The best candidate was `core_fine__delta_boundary__k7__d3__a60` with sampled grouped target-zone RMSE **14.8687**, only **0.0032** better than the first GAM baseline.

**Modeling implication:** More micro-tuning of sklearn additive splines is unlikely to be the next large gain. The strongest GAM evidence is that boundary, trajectory, and recent-slope features help; GR/typewell additions should be used cautiously unless a different model family can exploit them.

### A21 - Signed delta residuals favor robust continuous likelihoods

The GLM distribution diagnostic found `delta_tvt_from_boundary` ranges from **-103.78** to **98.83**, is only **53.3%** positive, and has excess kurtosis **4.09**. The best diagnostic model was `robust_delta_huber` with sampled grouped RMSE **14.7928**, beating Gaussian Ridge on the same slope-core delta features (**14.8398**) and the tuned GAM diagnostic (**14.8687**).

**Modeling implication:** For Bayesian work, prioritize Gaussian/Student-t/robust signed-continuous likelihoods for boundary deltas or endpoint residuals. Beta and Dirichlet are target-geometry mismatches, and positive raw-TVΤ Gamma/Tweedie/Poisson-style GLMs should not be the first probabilistic path.

### A22 - First Bayesian linear baseline improves uncertainty, not RMSE

The NumPyro/JAX Bayesian linear baseline found the best candidate was `bayes_student_t_heteroskedastic` with sampled grouped RMSE **14.9752**. Its posterior predictive intervals were reasonably calibrated for a first pass: **0.783** empirical coverage for 80% intervals and **0.933** for 95% intervals.

**Modeling implication:** Bayesian linear regression is useful as an uncertainty and likelihood-calibration baseline, but its posterior mean is not yet a Kaggle candidate because it trails the Huber GLM local diagnostic and the fine-tuned GAM submission. HSGP work should build on the Student-t/heteroskedastic calibration idea while adding a smooth residual function to improve the mean.

### A23 - Bayesian linear tuning favors robust MAP over extra scale complexity

The Bayesian linear tuning sweep found `student_t_fixed3_autodelta` as the best candidate with sampled grouped RMSE **14.8449**, improving the first Bayesian baseline by **0.1303** RMSE but still trailing the robust signed-delta Huber GLM diagnostic (**14.7928**). AutoNormal Student-t and Laplace variants clustered near **14.93**, while heteroskedastic extensions did not improve point RMSE in this sweep.

**Modeling implication:** Keep fixed-tail Student-t likelihoods as the Bayesian default, but do not spend much more time on linear-only Bayesian tuning. The next Bayesian model should add structured residual variation, likely HSGP over boundary/progress/trajectory coordinates, while using the Student-t likelihood and conservative uncertainty diagnostics from these linear runs.

### A24 - Targeted scale models help calibration more than point RMSE

The posterior-diagnostics and targeted-scale sweep found `hetero_boundary_progress_autodelta` as the best targeted scale candidate with sampled grouped RMSE **14.8626**, 80% coverage **0.780**, and mean Spearman correlation **0.315** between posterior predictive SD and absolute error. This improves the fixed Student-t reference inside the same notebook by **0.0503** RMSE, but remains **0.0176** worse than the prior tuned Bayesian linear best and **0.0697** worse than the Huber GLM diagnostic. The 90th percentile absolute well-level mean residual remains **18.60** TVT units.

**Modeling implication:** Boundary/progress-driven heteroskedasticity is useful for uncertainty diagnostics, but it is not enough as a standalone Bayesian-linear improvement. The next Bayesian model should add a smooth residual function with HSGP, and later hierarchical well effects should be considered because persistent well-level signed bias remains.
