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

### A25 - First HSGP residual model improves its linear reference but not the best Bayesian baseline

The first HSGP residual sweep found `hsgp_bp_m5_autodelta` as the best candidate with sampled grouped RMSE **14.8997**, improving the same-notebook linear Student-t reference by **0.0310** RMSE. However, it remains **0.0547** worse than the prior tuned Bayesian linear best (**14.8449**) and **0.1069** worse than the Huber GLM local diagnostic (**14.7928**). The best model used a small 2D boundary/progress basis with **25** functions; larger 2D basis (**49**) and a 3D boundary/progress/horizontal basis (**48**) did not improve RMSE.

**Modeling implication:** Smooth residual structure exists, but the first HSGP coordinate/basis design is not enough. Next GP work should try additive HSGP terms, coordinate-specific priors, or residual-only modeling around the strongest robust mean rather than simply increasing basis count.

### A26 - Additive HSGP residuals are useful but need a stronger mean model

The additive HSGP sweep found `additive_dps_m8_se_autodelta` as the best candidate with sampled grouped RMSE **14.8235**, improving the same-notebook linear Student-t reference by **0.0698** RMSE and the first joint boundary/progress HSGP by **0.0762** RMSE. The useful extra coordinate was `abs_projected_delta_10`, suggesting residual structure around local slope-instability in addition to boundary distance and evaluation progress. The best additive HSGP still trails the Huber GLM local diagnostic by **0.0307** RMSE. Early SVI attempts produced non-finite losses until the notebook used a lower learning rate and clipped Adam gradients.

**Modeling implication:** Additive one-dimensional HSGP terms are a better GP geometry than simply increasing joint basis count. The next HSGP should keep clipped/lower-step SVI and place the smooth residual around a stronger Huber or GAM mean rather than around the plain Bayesian linear mean.

### A27 - Residual HSGP works better on the robust Huber mean than on the reconstructed GAM mean

The residual-on-mean HSGP sweep found `huber_plus_hsgp_dps_m8` as the best candidate with sampled grouped RMSE **14.3586**, improving the same-notebook Huber mean-only reference by **0.0585** RMSE. The same additive residual geometry barely helped the reconstructed GAM mean and did not make the GAM direction locally competitive in this notebook, because one held-out fold was unstable for the spline mean.

**Modeling implication:** The next promotion candidate should be a Kaggle-safe Huber + additive Hilbert residual notebook, not a GAM-residual notebook. Keep the additive `distance_log1p`, `eval_progress`, and `abs_projected_delta_10` residual geometry, but treat the GAM residual result as a validation warning until the exact public-GAM implementation is mirrored and rechecked.

### A28 - Hierarchy must be contextual to transfer to hidden wells

The next hierarchical modeling track should not start with raw train-well random effects as the primary prediction mechanism. Hidden Kaggle wells may be unseen, so well-level effects must be inferred from test-safe descriptors such as known-zone summaries, boundary geometry, trajectory shape, GR availability, and residual diagnostics.

**Modeling implication:** Start hierarchy with residual bias audits and empirical Bayes/contextual offsets. Only move to full hierarchical HSGP after simpler partial-pooling models show grouped-CV gains that are explainable by test-safe well descriptors.

### A29 - Residual bias exists, but descriptor signal is modest

The hierarchical residual-bias audit found material well-level signed residual variation in the current Huber + additive Hilbert residual model. Across **773** held-out wells in the sampled residual artifact, the median absolute well mean residual is **6.890** TVT units, the 90th percentile is **18.929**, and the max is **56.454**. Fold-level mean residuals are small (**-0.052**, **0.408**, **0.450**), so the issue is more well-local than global fold bias.

The strongest descriptor association is modest: max boundary distance vs. MAE Spearman **0.175**, with related distance/log-distance and XY-span descriptors also near **0.16-0.17**.

**Modeling implication:** Proceed to empirical Bayes/contextual offsets, but keep corrections conservative and shrinkage-controlled. Boundary distance and trajectory span should enter first; raw well-ID effects remain unsafe.

### A30 - Simple contextual well offsets do not explain enough residual structure

The empirical Bayes offset screen tested global bias shrinkage, descriptor-predicted well intercepts, descriptor-predicted intercept plus progress slope, robust intercepts, and row-context Ridge offsets. The best candidate was `global_bias_shrink_050` with sampled RMSE **14.7378**, only **0.0008** better than the same-sample no-offset baseline (**14.7385**). Descriptor-based well offsets worsened sampled RMSE.

**Modeling implication:** Do not promote simple contextual well-intercept hierarchy. The residual bias is real, but the current known-zone descriptors do not predict it strongly enough. Future hierarchy should either target row/progress residual shape more directly or add stronger geology/trajectory descriptors before another contextual-offset attempt.

### A31 - Shallow residual-shape corrections also fail the hierarchy gate

The residual-shape screen tested empirical-Bayes progress bins, boundary-distance bins, slope-instability bins, additive bins, 2D progress-distance bins, Ridge row-shape corrections, polynomial Ridge, smooth progress splines, and robust Huber row-shape corrections. The best candidate was `row_linear_ridge_a100_s025` with sampled RMSE **14.7382**, only **0.0003** better than the same-sample no-shape baseline (**14.7385**).

**Modeling implication:** Do not promote shallow row/progress residual-shape hierarchy. Together with A30, this suggests the next gain is unlikely to come from post-hoc residual correction on the bounded residual sample. Prefer stronger feature construction, better validation alignment, or a model that changes the main mean structure rather than adding tiny residual offsets.

### A32 - Contextual descriptor interactions show weak signal but still fail promotion

The contextual hierarchical Huber screen tested descriptor main effects, descriptor-by-row interactions, quadratic contextual surfaces, and robust contextual row interactions. The best candidate, `context_poly2_ridge_a2000_s025`, scored sampled RMSE **14.7322**, improving the same-sample no-context baseline (**14.7385**) by **0.0064** RMSE. This is the strongest of the shallow hierarchy screens so far, but it remains far below the **0.05** RMSE promotion gate.

**Modeling implication:** Contextual interactions may contain weak transferable signal, but not enough as a post-hoc residual correction. Do not promote to Kaggle or full hierarchy. Revisit only after stronger feature construction or a mean-model redesign.

---

## Research-Guided Bayesian/Geologic Modeling Addendum - 2026-06-23

**Research sources:** Pisel et al. top-picking recommender system (https://arxiv.org/abs/2202.08869), Chaki et al. well-tops-guided reservoir property prediction (https://arxiv.org/abs/1509.07079), Herrera and van der Baan seismic-to-well ties / DTW alignment (https://arxiv.org/abs/1209.0201), Riutort-Mayol et al. Hilbert-space GP approximation (https://arxiv.org/abs/2004.11408), Rasmussen and Williams GPML (https://www.gaussianprocess.org/gpml/), PyMC GP covariance and HSGP docs (https://www.pymc.io/projects/docs/en/stable/api/gp.html), GPyTorch multitask GP example (https://docs.gpytorch.ai/en/stable/examples/03_Multitask_Exact_GPs/Multitask_GP_Regression.html), GPflow coregionalization example (https://gpflow.github.io/GPflow/develop/notebooks/advanced/coregionalisation.html), Snoek et al. input warping (https://arxiv.org/abs/1402.0929), Jylanki et al. Student-t likelihood GP (https://arxiv.org/abs/1106.4431), PyKrige docs (https://geostat-framework.readthedocs.io/projects/pykrige/en/stable/), GSTools docs (https://geostat-framework.readthedocs.io/projects/gstools/en/stable/), tslearn DTW docs (https://tslearn.readthedocs.io/en/stable/user_guide/dtw.html).

### A33 - Post-hoc residual hierarchy has plateaued

The first three hierarchy screens improved the bounded residual sample by **0.0008**, **0.0003**, and **0.0064** RMSE. These deltas are real enough to keep as diagnostics, but too small to justify a deeper hierarchical HSGP as the immediate next step.

**Modeling implication:** Stop stacking small residual corrections on the current residual sample until a stronger geologic feature or mean-structure hypothesis is introduced. Use the hierarchy results as evidence about what did not transfer, not as a reason to add raw complexity.

### A34 - Typewell and well-log alignment should be treated as a feature-construction problem

Well-top recommender and DTW alignment references both point toward comparing a target well against nearby or analogous wells through known intervals, trajectory geometry, and log-shape similarity. The current notebooks use only cautious typewell summaries, not alignment features that describe lag, stretch, local correlation, or nearest analog behavior.

**Modeling implication:** The next practical experiment should screen typewell/analog alignment features before further Bayesian hierarchy: best-lag correlation, rolling GR similarity, known-zone trajectory similarity, pseudo-zone labels, and typewell-transferred TVT deltas. All alignment features must be computed inside each validation fold using training wells only.

### A35 - Future GP work should prefer Matern, additive, warped, or multitask structure over larger squared-exponential bases

The HSGP papers and GP references support reduced-rank approximations, but the prior experiments show that simply increasing basis count did not improve RMSE. Geology is likely nonstationary across boundary distance, progress, and log-shape regimes, while heavy-tailed residuals remain important.

**Modeling implication:** If we return to GP modeling, keep Student-t or robust likelihoods, try Matern 3/2 or 5/2 kernels, consider input warping over progress/distance, and only use multitask/coregionalized GP ideas with descriptor-defined tasks that can exist for hidden wells.

### A36 - Regression kriging is plausible only after a spatial residual test

Geostatistical references suggest modeling deterministic predictions plus spatially correlated residuals. The current hierarchy audit found well-local residual bias, but not yet a formal spatial variogram or residual correlation over surface coordinates, TVT boundary coordinates, or trajectory descriptors.

**Modeling implication:** Before building kriging or spatial GP residuals, run a residual spatial-correlation screen on out-of-fold residuals. Promote that path only if residual semivariance or nearest-neighbor residual correlation is strong under grouped-well validation.

### A37 - Direct analog transfer is the first alignment signal worth extending

The first typewell/analog alignment screen found `analog_direct_s010` as the best candidate with sampled RMSE **14.6966**, improving the same-sample no-alignment baseline (**14.7385**) by **0.0419** RMSE. This is below the **0.05** feature-screen gate, but materially stronger than the prior shallow hierarchy gains. The strongest feature/residual association was `analog_delta_minus_baseline_delta` with correlation **0.0680**, and median absolute well-bias delta improved by **-0.0988**.

**Modeling implication:** Do not submit this direct analog residual correction yet, but continue the alignment track. The next attempt should make analog transfer less crude: multiple analogs, distance-weighted analog deltas, pseudo-zone-specific analogs, and robust blending into the Huber mean rather than a fixed 10 percent residual correction.

### A38 - Multi-analog averaging smooths useful nearest-analog signal

The multi-analog zone-blend screen found `weighted_k12_s150` as the best candidate with sampled mean-fold RMSE **14.7034** and same-notebook global-baseline delta **-0.0405** RMSE. This confirms analog transfer is still useful, but it did not beat the simpler nearest-analog `analog_direct_s010` result (**14.6966**). Pseudo-zone beta blending also trailed fixed-shrink analog transfer.

**Modeling implication:** Do not keep widening analog neighborhoods blindly. The next alignment move should improve analog selection quality rather than average more wells: stratigraphic/pseudo-zone matching, feature-specific neighbor constraints, robust local analog selection, or a Huber mean model that uses the nearest-analog delta as a primary feature instead of a post-hoc correction.

### A39 - Learned pseudo-zone analog heads help late zones but do not beat fixed analog shrink

The pseudo-zone Huber/Ridge head screen found `ridge_zone_a100_s050` as the best candidate with sampled RMSE **14.7004**, improving the same-sample baseline (**14.7385**) by **0.0381** RMSE. This remains worse than the simpler nearest-analog `analog_direct_s010` result (**14.6966**). Zone diagnostics show the analog head is most useful later in the target zone: `z2` improved by **0.0629** RMSE and `z3` by **0.0543**, while `z0` barely moved (**0.0032**).

**Modeling implication:** Do not promote learned pseudo-zone analog heads yet. The next improvement should gate analog correction by evaluation progress: small or zero correction near the boundary, stronger nearest-analog correction in late target-zone rows, and possibly a late-zone-specific HSGP or robust mean feature.

### A40 - Progress gating confirms the late-zone pattern but does not unlock a larger gain

The progress-gated analog sweep found `grid_025_025_075_100` as the best schedule with sampled RMSE **14.6966**, improving the same-sample baseline (**14.7385**) by **0.0420** RMSE. The schedule uses small shrinkage in early rows (`z0=0.025`, `z1=0.025`) and stronger shrinkage later (`z2=0.075`, `z3=0.100`). This is directionally consistent with A39 but only marginally better than fixed nearest-analog shrink.

**Modeling implication:** Gating by progress is sensible but not sufficient. The next alignment improvement should change analog selection or feature geometry: stratigraphic/log-shape constrained nearest analogs, residual spatial-correlation diagnostics, or using the analog delta inside a full Huber/Hilbert submission pipeline rather than more residual-sample schedule tuning.

### A41 - Spatial residual correlation is weak under current descriptors

The spatial residual-correlation screen found nearest-neighbor well-mean residual correlation **0.0422** using surface, known-zone endpoint, TVT-span, and trajectory descriptors. The best spatial KNN correction, `spatial_knn_k12_s200`, scored sampled RMSE **14.7144**, improving the same-sample baseline (**14.7385**) by only **0.0241** RMSE. Semivariance generally increased in the farthest descriptor-distance bin, but the signal is too weak for immediate regression kriging or spatial GP promotion.

**Modeling implication:** Do not build a kriging or spatial GP candidate yet. If spatial modeling returns, it should use stronger geology-aware coordinates or be combined with better analog/log-shape selection; the current descriptor-space residual field is not strong enough.

### A42 - Log-plus-trajectory analog selection clears the alignment promotion gate

The analog-selection refinement screen found `log_plus_trajectory_k1_progress_gate` as the best candidate with sampled RMSE **14.6383**, improving the same-sample baseline (**14.7385**) by **0.1003** RMSE. This is the first alignment result to clear the **0.05** gate. The best feature set combines recent GR shape, typewell lag/correlation, TVT span, XY span, and vertical drop; using one nearest analog with the progress-gated schedule beat multi-analog averaging.

**Modeling implication:** Promote the log-plus-trajectory nearest-analog selection idea into the next candidate pipeline. The next notebook should make this Kaggle-safe by recomputing analog descriptors inside the notebook and combining the analog delta with the Huber + additive Hilbert residual model, or by using it as the alignment feature for the next HSGP residual.

### A43 - Alignment signal is carried by direct analog transfer, not learned residual heads

The alignment-aware Hilbert residual screen found `direct_analog_gate` as the best candidate with sampled RMSE **14.6383**, matching the analog-selection refinement result and improving the same-sample Huber + Hilbert baseline (**14.7385**) by **0.1003** RMSE. Learned Ridge, Huber, polynomial, and RBF residual heads all trailed direct analog transfer.

**Modeling implication:** Do not add another smooth residual layer around the analog signal yet. The next useful step is a self-contained Kaggle-safe Huber + Hilbert + direct analog-gate submission candidate, with analog descriptors and selection recomputed inside the notebook.

### A44 - The analog-gate gain did not transfer publicly

The Kaggle-safe Huber + additive Hilbert plus direct analog-gate submission scored **14.603** public LB, worse than the ungated Huber + additive Hilbert submission at **14.514**. This reverses the bounded residual-sample diagnostic, where the analog gate improved same-sample RMSE by **0.1003**. The validation mismatch audit found mixed analog well-bias behavior, with mean distance-bin improvement share around **0.555**.

**Modeling implication:** Pause deeper hierarchy and particle/state-space modeling. The immediate bottleneck is validation transfer rather than model expressiveness. Next work should create stricter CV folds that stress far-analog wells, clustered/geologic separation, and high-extrapolation cases before LightGBM/XGBoost or additional Bayesian models.

### A45 - Strict hybrid folds are the next default validation protocol

The strict CV fold design produced four candidate well-level fold columns: `descriptor_cluster_fold`, `analog_distance_fold`, `extrapolation_fold`, and `strict_hybrid_fold`. The strictest diagnostic split was `analog_distance_fold`, but the recommended default is `strict_hybrid_fold` because it preserves practical fold balance while spreading high-stress wells. Its row imbalance is **1.000**, well imbalance is **1.024**, and mean nearest-training descriptor distance is **2.436**.

**Modeling implication:** Use `strict_hybrid_fold` for the first LightGBM/XGBoost strict-CV run. Treat `analog_distance_fold`, `descriptor_cluster_fold`, and `extrapolation_fold` as stress tests for any model that appears better than the Huber + Hilbert baseline under the default strict split.
