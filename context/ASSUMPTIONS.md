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

### A46 - LightGBM is the first strong strict-CV tree result

The LightGBM strict-CV screen found `lgbm_medium_l2` as the best candidate with sampled `strict_hybrid_fold` RMSE **13.2363** and last-known hold RMSE **15.5734** on the same rows. The same configuration scored **13.4453** RMSE on the stricter `analog_distance_fold`, with hold RMSE **15.5833**. The model leaned on extrapolation stress, last-known TVT, formation distances, GR summaries, typewell error features, and selected analog distance.

**Modeling implication:** Boosted trees are worth continuing. Do not push to Kaggle immediately; first run a reproducibility/tuning pass or XGBoost comparison under the same strict folds to confirm this is not a sampled-row artifact.

### A47 - XGBoost improves the strict-CV tree signal but depends partly on train-only formation proxies

The XGBoost strict-CV screen found `xgb_pseudohuber_shallow_robust` as the best candidate with sampled `strict_hybrid_fold` RMSE **12.7504** and analog-distance stress RMSE **12.9116**. This improves over the LightGBM strict-CV reference by **0.4859** RMSE and over the LightGBM analog-distance stress reference by **0.5337** RMSE. The best model's top features include robust row geometry (`md_from_start`, `distance_log1p`, `row_idx_original`), extrapolation stress, and formation-distance features such as `abs_dist_to_BUDA` and `dist_to_BUDA`. The inference-safe LightGBM Kaggle package scored **14.687**, worse than Huber + Hilbert **14.514**, while the inference-safe XGBoost package scored **14.472**, a small but real public improvement.

**Modeling implication:** Boosted trees are now the highest-priority modeling path toward the 10 RMSE goal. The next step should tune inference-safe XGBoost features and hyperparameters, especially hidden-test-safe proxies for formation distance and extrapolation stress, while guarding against leaderboard-only overfitting.

### A48 - First fold-safe formation proxies do not recover the direct formation signal

The inference-safe XGBoost tuning screen found `safe_typewell` as the best legal feature set with sampled `strict_hybrid_fold` RMSE **13.4178** and `analog_distance_fold` stress RMSE **13.7533**. The fold-safe proxy-formation feature set scored **13.5503**, worse than safe base (**13.4399**) by **0.1103** RMSE. The oracle direct-formation diagnostic still scored **12.9597**, which confirms the direct formation-distance signal is real but not captured by the current proxy model.

**Modeling implication:** Do not package the current proxy-formation recipe. The next tree-side improvement should use residual stacking, stronger typewell/stratigraphic structural proxies, or a submission-safe way to infer formation context that is more than a generic XGBoost proxy over safe descriptors.

### A49 - Nested residual stacking finds transferable post-XGBoost residual shape

The XGBoost residual-stacking screen found `ridge_residual_s025` as the best stacker. It improved `strict_hybrid_fold` RMSE from base-only **13.4178** to **13.2548** and improved `analog_distance_fold` stress RMSE from **13.7424** to **13.5632**. The residual stacker was trained on inner out-of-fold residuals inside each outer fold, and a reproducibility rerun matched the key metrics.

**Modeling implication:** Promote Ridge residual stacking into a self-contained Kaggle package next. Keep the residual correction shrunk and inference-safe; the first progress quartile slightly worsened while later rows improved, so the package should preserve conservative shrinkage and monitor early-zone behavior.

### A50 - Residual-stacking gains remain validation-fragile

The Ridge residual-stack Kaggle package scored public LB **14.583** (`ref=54157692`), worse than the current XGBoost strict inference-safe baseline **14.472**, despite improving both `strict_hybrid_fold` and `analog_distance_fold` locally. The package used only inference-safe row, descriptor, and typewell features, trained base OOF residuals over stable well folds, and produced a valid **14,151** row submission.

**Modeling implication:** Do not keep stacking generic residual heads without a stronger reason. Next improvements should focus on hidden-test-safe feature geometry, public-transfer diagnostics, and calibration of validation folds against leaderboard behavior rather than more post-hoc residual correction.

### A51 - TVT prediction is primarily a stratigraphic-state problem

Domain references on well geometry, formation-top picking, and well-log correlation support treating `TVT` as a formation-relative coordinate, not just a smooth row-level response over `MD`, `X`, `Y`, `Z`, and `GR`. The strongest local tree diagnostics also point this way: direct distances to named formations were among the best XGBoost features, while submission-safe proxies only recovered a small part of that signal. References reviewed: Pisel et al. on well-top recommendation (`https://arxiv.org/abs/2202.08869`), Chaki et al. on well-tops-guided reservoir property prediction (`https://arxiv.org/abs/1509.07079`), SLB Oilfield Glossary geometry terms (`https://glossary.slb.com/`).

**Modeling implication:** The next feature work should infer latent stratigraphic state from test-safe evidence: known-zone `TVT_input`, trajectory shape, typewell/log motif alignment, boundary behavior, and uncertainty in the implied top/zone position. Do not spend the next cycle on another generic residual learner unless it uses this state explicitly.

### A52 - Gamma-ray logs are useful alignment evidence but ambiguous on their own

Gamma ray helps distinguish lithology, but similar log motifs can recur across formations and local GR values lose phase/stretch information. Literature on log correlation and dynamic time warping supports using sequence alignment, lag/stretch, and alignment uncertainty rather than only scalar GR windows. References reviewed: Herrera and van der Baan on DTW for seismic-to-well ties (`https://arxiv.org/abs/1209.0201`) and tslearn DTW documentation (`https://tslearn.readthedocs.io/en/stable/user_guide/dtw.html`).

**Modeling implication:** Build features that summarize alignment quality over known-zone sequences: DTW distance to typewell/template windows, inferred lag, stretch/compression, segment correlation, recent motif mismatch, and ambiguity between the best and second-best alignments. These must be recomputed fold-safely and use only known-zone rows for held-out/test wells.

### A53 - Horizontal-well/geosteering context favors constrained trajectory corrections

Horizontal wells can move laterally through dipping or structurally complex stratigraphy. In that setting, row-independent prediction can fit average patterns but miss boundary crossings, local bed dip, and formation-relative progress. The repeated local/public mismatch suggests the hidden test may be stressing exactly those transitions.

**Modeling implication:** Prefer candidate models/features that preserve within-well continuity and relative position: monotone or softly monotone `TVT` paths, boundary-distance interactions, segment-level offsets/slopes, and postprocessing that removes unrealistic local jumps. Any such constraints must be tested against wells where true target-zone `TVT` sometimes bends or changes slope, not only smooth examples.

### A54 - Public leaderboard misses are evidence of validation-regime mismatch

The Huber + analog gate and XGBoost residual stack both improved local diagnostics but failed to beat the simpler public baselines. This is unlikely to be only random noise; it means the current strict folds are still not reproducing the hidden stratigraphic-transfer problem well enough.

**Modeling implication:** Add a validation audit before the next model package. Candidate folds/stress slices should condition on inferred stratigraphic novelty, typewell alignment uncertainty, known-zone length, GR missingness, target-zone extrapolation distance, and analog-quality scores. Public LB should remain an external check, but it should not be the first place these failure modes are discovered.

### A55 - Formation-top inference should be modeled as uncertain, not replaced by hard pseudo-labels

The direct formation-distance oracle gap is real, but hidden test wells do not expose the same train-only formation columns. A hard pseudo-formation classifier could overfit badly if it assigns confident wrong tops. Formation-top and log-correlation literature both point toward ranked candidates and uncertainty, not only one chosen top.

**Modeling implication:** If pseudo-top or pseudo-zone features are added, export uncertainty features too: top candidate distance, probability/margin, entropy, best-vs-second alignment score, and whether the target zone lies near a likely boundary. Trees can use these uncertainty summaries; Bayesian models can later turn them into priors or hierarchical shrinkage terms.

### A56 - First stratigraphic-state screen finds directional but sub-gate signal

The first `07_stratigraphic_state_feature_screen` notebook tested known-zone state summaries, typewell/log alignment uncertainty, and fold-safe nearest-analog descriptors. The best non-oracle strict-hybrid result was `huber_delta` with `strat_known_state` at **14.9922** RMSE, improving the same-notebook Huber safe-base result by **0.0391** RMSE but trailing the current local safe-XGBoost reference **13.4178**. The local XGBoost squared-error screen was not comparable to the packaged pseudo-Huber XGBoost path, but it showed directional same-model improvement from `safe_base` **22.9672** to `strat_known_state` **22.1933** and improved several high-stress slices.

**Modeling implication:** Keep known-zone stratigraphic-state summaries as candidate features, but do not package this screen. The next attempt should plug the state features into the exact current public-best XGBoost training recipe or a Bayesian stratigraphic-regime model, rather than judging them through the lightweight screen consumers alone. The first alignment-uncertainty and analog bundle added noise when combined naively.

### A57 - Raw known-state features do not improve the exact XGBoost recipe

The `10_xgboost_known_state_features` notebook injected known-zone state summaries into the current inference-safe XGBoost recipe. The best legal candidate remained `pseudo_shallow_more_trees__safe_typewell` at **13.4334** strict-hybrid RMSE and **13.8243** analog-distance stress RMSE, while the best state-augmented candidate was `square_medium_regularized__safe_typewell_known_state` at **13.5189** strict-hybrid RMSE. The direct-formation oracle diagnostic still scored **12.9353**, leaving a **0.4981** RMSE gap from the best legal model.

**Modeling implication:** The known-zone state summaries should not be added to the XGBoost Kaggle package as raw features. The direct-formation gap remains the right design target, but the next attempt should use state as a gated/shrunk latent quantity, a Bayesian prior, or a constrained stratigraphic-regime component rather than expecting trees to recover the signal from these scalar summaries alone.

### A58 - Bayesian gating makes known-zone state useful as a residual regime signal

The `07_gated_bayesian_regime` notebook fit a Huber signed-delta mean, residual Hilbert/state experts, and a Student-t MAP soft gate with shrinkage priors. The best candidate was `huber_plus_gate_conservative` at **14.1072** strict-hybrid RMSE, improving the same-notebook Huber mean-only result **14.5053** by **0.3981** RMSE. The conservative gate learned near-full residual shrinkage (**0.9940**) while assigning average weights of about **0.177** to zero correction, **0.202** to Hilbert residual, **0.584** to state residual, and **0.037** to blended residual.

**Modeling implication:** Known-zone state does not help XGBoost as raw columns, but it can help when it controls a residual regime. This is worth a public curiosity probe, but the validation/public mismatch history still applies: do not promote over XGBoost strict inference-safe unless the Kaggle score confirms transfer.

### A59 - Pseudo-top uncertainty is useful as latent state, not as raw XGBoost columns

The `08_pseudo_top_uncertainty` notebook confirmed that train formation columns are row-wise formation-surface elevations and hidden-test files do not expose them. Fold-safe squared-error top-elevation proxies reached mean top RMSE **35.0660** and nearest-top agreement **0.5804** on `strict_hybrid_fold`, so the pseudo-top inference is meaningful. However, raw pseudo-top features did not improve the current legal tree recipe: best legal remained `xgb_square_regularized__safe_typewell` at **13.4506** strict-hybrid RMSE and **13.8870** analog-distance stress RMSE, while best pseudo-top feature candidate `xgb_pseudohuber_known_state_recipe__safe_pseudo_top_uncertainty` scored **13.5891** strict-hybrid RMSE.

**Modeling implication:** Do not package the first pseudo-top XGBoost feature dump. Keep the pseudo-top columns, margin, entropy, and ensemble spread as latent stratigraphic-state evidence for a gated Bayesian/tree hybrid where uncertainty controls shrinkage or mixture weights.

### A60 - Pseudo-top gating helps stress behavior but not enough strict CV yet

The `09_pseudo_top_gated_regime` notebook used pseudo-top margin, confidence, entropy, ensemble spread, and state/typewell descriptors to shrink residual corrections. The best candidate was `xgb_pseudohuber_mean__ridge_pseudo_state_residual__uncertainty_gate_conservative` at **13.4694** strict-hybrid RMSE and **13.7665** analog-distance stress RMSE. It improved the same-notebook mean-only reference **13.4745** by **0.0051** RMSE and improved analog stress versus the prior safe/typewell reference **13.8870**, but it still trailed the strict local safe/typewell reference **13.4506**.

**Modeling implication:** Pseudo-top uncertainty is more useful as a stabilizer than as a point-RMSE breakthrough in the current residual-gate form. The next iteration should either move the gate into the primary mean model or use it as Bayesian prior/mixture evidence; do not package this residual gate as-is.

### A61 - Pseudo-top interaction features do not unlock the XGBoost mean

The `10_pseudo_top_interaction_xgboost` notebook moved pseudo-top uncertainty into the primary XGBoost mean through controlled interactions with progress, distance, analog stress, and typewell error. The best overall candidate remained `xgb_square_regularized__safe_typewell` at **13.4745** strict-hybrid RMSE and **13.8284** analog-distance stress RMSE. The best pseudo-top interaction candidate, `xgb_pseudohuber_reference__pseudo_interactions`, scored **13.4815** strict-hybrid RMSE and **13.8681** stress RMSE, slightly worse than the same-notebook safe/typewell baseline.

**Modeling implication:** Flat pseudo-top feature engineering has likely hit diminishing returns. The next pseudo-top/stratigraphy attempt should be a Bayesian or mixture-regime model where pseudo-top uncertainty changes formation-regime weights or priors, not another additive/interacted tree feature set.

### A62 - Soft pseudo-top mixture priors finally move strict RMSE, but only marginally

The `08_pseudo_top_mixture_regime` notebook converted fold-safe pseudo-top distances into soft formation-regime probabilities and fit Ridge MAP residual priors over formation probability, progress, distance, and near-top basis terms. The best candidate, `xgb_pseudohuber_mean__ridge_alpha650__mixture_conf_s050`, scored **13.4437** strict-hybrid RMSE and **13.7593** analog-distance stress RMSE. This beats the prior strict safe/typewell reference **13.4506** by **0.0069** RMSE and improves stress versus **13.8870**, but the gain is far below the normal **0.05** packaging margin.

**Modeling implication:** The mixture-regime framing is the most promising pseudo-top use so far, but should be iterated before Kaggle packaging. Next work should model the TVT mean directly by stratigraphic regime or calibrate the mixture against public-transfer slices, rather than only adding residual MAP corrections.

### A63 - Direct pseudo-top regime means unlock a larger local gain

The `09_pseudo_top_regime_mean` notebook moved pseudo-top probabilities into the mean function itself. Each formation regime received a weighted Ridge MAP mean over legal row/typewell/state features, and predictions blended those regime means with a global safe/typewell mean according to pseudo-top confidence. The best candidate, `xgb_square_global__regime_ridge_alpha80__confidence_s100`, scored **13.0928** strict-hybrid RMSE and **13.4006** analog-distance stress RMSE. This improves the previous pseudo-top mixture prior **13.4437** by **0.3510** RMSE and the strict safe/typewell reference **13.4506** by **0.3579** RMSE.

**Modeling implication:** Pseudo-top structure is most valuable when it defines the primary stratigraphic-regime mean, not as a raw feature or post-hoc residual correction. This clears the local packaging gate, but public-transfer caution remains because earlier local gains did not always transfer.

### A64 - Simple regime-mean tuning is nearly saturated

The `10_pseudo_top_regime_mean_tuning` notebook narrowed the search around the direct regime-mean winner with Ridge alphas **40**, **80**, **140**, and **220**, two XGBoost global means, and fixed/confidence/margin-heavy blend schedules. The best candidate, `xgb_square_global__regime_ridge_alpha40__confidence_s100`, scored **13.0910** strict-hybrid RMSE and **13.4065** analog-distance stress RMSE. This improves strict RMSE by only **0.0018** versus the first regime-mean baseline **13.0928** and worsens stress by **0.0059** versus **13.4006**.

**Modeling implication:** The direct regime-mean family is useful, but small alpha/blend tuning is saturated. Keep the original `alpha80` package as the cleaner public probe, and move the next RMSE push to boosted stacking, better pseudo-top inference, or a structurally different Bayesian/partial-pooling regime model.

### A65 - Regime-mean residual stacking improves strict RMSE but hurts stress

The `11_xgboost_regime_mean_stack` notebook used fold-safe regime-mean predictions as meta-features and residual baselines for direct XGBoost, XGBoost residual, and Ridge residual heads. The best candidate, `blend_plus_ridge_residual_s15`, scored **13.0769** strict-hybrid RMSE, improving the direct regime-mean reference **13.0928** by **0.0159** RMSE. However, analog-distance stress worsened from **13.4006** to **13.4468**.

**Modeling implication:** There is residual signal around the regime mean, but the first stack correction is not stress-stable. Continue only with stress-gated or uncertainty-gated residual corrections; do not package the raw stack over the cleaner regime-mean notebook.

### A66 - Stress-gated residual stacks trade strict gain for stress but do not beat the regime mean

The `12_xgboost_regime_stress_gated_stack` notebook tested uniform, confidence, margin, low-stress, hybrid, and stress-floor gates on top of Ridge/XGBoost residual heads. The best strict candidate remained the uniform `blend_plus_ridge_residual_s15_uniform` at **13.0769** strict-hybrid RMSE and **13.4468** analog-distance stress RMSE. The best lower-stress gated compromise, `blend_plus_ridge_residual_s15_margin`, scored **13.0853** strict RMSE and **13.4079** stress RMSE, still slightly worse than the direct regime-mean stress **13.4006**.

**Modeling implication:** Gating confirms that residual corrections contain fragile, transfer-risk-sensitive signal. Since none of the gates improve both strict and stress against the original regime mean, the next major improvement should target pseudo-top inference quality or a different partial-pooling structure rather than residual shrinkage schedules.

### A67 - Pseudo-top probability scale 50 is already locally optimal

The `11_pseudo_top_probability_scale_tuning` notebook tested softmax probability scales **25**, **35**, **50**, **75**, **100**, and **150** inside the same direct regime-mean recipe. The original scale **50** remained best on both strict RMSE (**13.0928**) and analog-distance stress RMSE (**13.4006**). Sharper scales over-committed to noisy pseudo-tops, while smoother scales diluted useful stratigraphic separation.

**Modeling implication:** The next step should not be another probability-scale or confidence-calibration sweep. To reach a materially lower RMSE, the pseudo-top inference model itself must improve, or the regime means need a structurally different pooling/prior design.

### A68 - Denser target-row sampling does not improve the regime mean

The `11_pseudo_top_regime_mean_more_rows` notebook increased the per-well target-row cap from **120** to **220**, raising evaluated rows from **92,760** to **170,060**. The best candidate shifted to `xgb_pseudohuber_global__regime_ridge_alpha80__confidence_s100` with strict RMSE **13.1185** and analog-distance stress RMSE **13.4745**, worse than the original 120-row regime mean **13.0928** strict and **13.4006** stress.

**Modeling implication:** The regime-mean plateau is not caused by too little target-zone row density in the local notebook. Keep the lighter **120** row cap for fast iteration and package stability unless a future model specifically benefits from more rows.

### A69 - Local GR/typewell alignment features help strict RMSE slightly but do not improve pseudo-top state

The `12_alignment_aware_pseudo_top_regime_mean` notebook added hidden-test-safe local GR/typewell alignment features: local GR window statistics, local DTW distance to typewell GR windows, local correlation to linear/best-lag typewell windows, and alignment margins. The best candidate, `xgb_square_global__regime_ridge_alpha80__confidence_s100`, scored **13.0838** strict-hybrid RMSE, a **0.0089** improvement over the original regime mean **13.0928**. However, analog-distance stress worsened to **13.4303** versus **13.4006**, and pseudo-top proxy RMSE did not improve.

**Modeling implication:** Local alignment descriptors carry some row-level mean signal, but they are not a better pseudo-top state estimator in this form. Do not package over the original regime mean. If alignment is revisited, separate it into a stress-aware mean feature path or a different top-inference architecture rather than simply adding DTW/correlation columns to the pseudo-top XGBoost.

### A70 - Shared partial pooling does not rescue the current pseudo-top regime signal

The `12_bayesian_regime_partial_pooling` notebook tested an empirical-Bayes style shared Ridge structure around the direct pseudo-top regime mean. The model used fold-safe pseudo-top probabilities, probability-by-feature interactions, and Ridge penalties as Gaussian shrinkage analogues for residual and direct partial-pooling corrections. The best overall candidate was the independent regime-mean reference re-fit, `xgb_square_global__reference_regime_ridge_alpha80__reference_confidence_s100`, at **13.0953** strict-hybrid RMSE and **13.4038** analog-distance stress RMSE. The best new residual partial-pooling layer worsened to **13.4272** strict RMSE, and the best new direct partial-pooling layer worsened to **13.2049** strict RMSE.

**Modeling implication:** With the current pseudo-top probabilities, sharing all regime effects through one interaction-heavy Ridge layer washes out useful formation-specific mean structure. More Bayesian shrinkage on the same pseudo-top signal is unlikely to create a step-change toward **10 RMSE**. The next improvement should target better pseudo-top/state inference, state revision, or a different high-capacity model family rather than additional partial-pooling schedules.

### A71 - Plateau errors concentrate in the same wells across model variants

The `02_plateau_failure_slice_audit` notebook compared eight recent pseudo-top/regime experiments. The best strict variants were the residual stacks at **13.0769** RMSE, but both worsened analog-distance stress to **13.4468**. More importantly, the same hard wells appeared in the top-15 error list across all eight recent runs, including `86454a6f`, `1b1eba53`, `a959858c`, `5f4d2a52`, `ba48188d`, `2fd68f7b`, `f88ddb26`, and `fef8af96`.

**Modeling implication:** The plateau is not mostly random variance from candidate schedules. It is concentrated in persistent well/state failures that the current feature families cannot explain. The next major experiment should either revise the pseudo-top/state labels for these wells or use a local-log sequence/window model that can learn motifs missed by scalar regime probabilities.

### A72 - Local sequence summaries worsen when used as flat stack features

The `13_xgboost_local_sequence_stack` notebook tested local GR/typewell window summaries and probability-by-sequence interactions as direct meta-features around the pseudo-top regime mean. The best candidate was only the alignment-aware regime blend reference at **13.1613** strict RMSE and **13.4561** stress RMSE. The best residual correction, `blend_plus_sequence_ridge_residual_s08`, scored **13.1626** strict RMSE, and the direct XGBoost sequence meta-head collapsed to **13.8082**.

**Modeling implication:** The local-window signal is not useful as a flat scalar stack layer in the current recipe. If sequence context matters, it likely needs either direct state/top-error correction or a true window/sequence architecture that preserves motif shape, not just window means, slopes, DTW distances, and interaction columns.

### A73 - Top-state correction can improve stress while hurting strict RMSE

The `13_pseudo_top_state_revision` notebook trained second-stage residual correctors for fold-safe pseudo-top elevation predictions. The best TVT candidate scored **13.1194** strict RMSE and **13.3883** analog-distance stress RMSE. This worsened strict RMSE versus the original regime mean **13.0928**, but improved stress by **0.0123** versus **13.4006**. Revision diagnostics showed that **0.25** shrink slightly improved strict top RMSE (**-0.0064** average delta), while **0.50** and **0.75** worsened top RMSE.

**Modeling implication:** Pseudo-top residual correction has real stress-control signal, but broad correction over-moves the local state. The next revision should be lighter or gated to high-stress/high-uncertainty slices rather than applied globally.

### A74 - Gated state correction preserves the base strict winner but only improves stress in worse strict rows

The `14_pseudo_top_state_revision_gated` notebook kept separate probability columns for base, global, uncertainty-gated, stress-gated, and hybrid pseudo-top corrections. The best strict candidate was the unchanged base regime mean at **13.0928** strict RMSE and **13.4006** stress RMSE. The best stress candidate, `hybrid_s75__xgb_square_global__regime_ridge_alpha220__s050`, reached **13.3710** stress RMSE but worsened strict RMSE to **13.1309**.

**Modeling implication:** Top residual correction is a stress-calibration tool, not a strict-RMSE breakthrough in this form. Since gating did not produce a joint strict/stress win, stop correction-schedule tuning and move to a different state representation or a public-transfer calibration decision.

### A75 - Raw local motif bins do not help as flat tabular features

The `14_xgboost_raw_window_sequence` notebook added raw local GR and typewell window bins to preserve motif shape around each target row. The best candidate, `xgb_square_global__raw_window_regime_ridge_alpha80__confidence_s100`, scored **13.1664** strict RMSE and **13.4758** analog-distance stress RMSE, both worse than the original pseudo-top regime mean.

**Modeling implication:** The current plateau is not solved by adding more flat local motif columns. If log-shape context is important, it likely needs a true sequence/representation model or a different state target, not tabular bins layered into the existing regime-mean recipe.

### A76 - Current residual candidates are too correlated for blending to break the plateau

The `03_oof_candidate_blend_audit` notebook preserved row-level OOF predictions for the regime-mean reference and residual-stack heads, then tested fixed averages and leave-one-fold-out positive linear meta-blends. The previous best `blend_plus_ridge_residual_s15` remained best at **13.0769** strict RMSE and **13.4468** analog-distance stress RMSE. The best fixed average, `oof_avg__base_ridge25`, was close at **13.0779** strict / **13.4371** stress, but did not beat strict RMSE. Positive-linear meta-blends worsened to at least **13.1001** strict and **13.5604** stress.

**Modeling implication:** The current pseudo-top regime and residual-stack candidates mostly share the same errors. More candidate weighting is unlikely to create a path toward **10 RMSE**. A useful next experiment needs a new state target, formation-transition representation, or sequence model that changes which wells/rows are hard.

### A77 - Transition-distance geometry does not improve the tabular regime mean

The `15_pseudo_top_transition_distance` notebook represented fold-safe pseudo tops as signed distances to adjacent inferred formation boundaries, nearest-top order, normalized spacing, probability-weighted offsets, boundary pressure, and adjacent-pair transition features. The unchanged reference remained best at **13.0928** strict RMSE and **13.4006** analog-distance stress RMSE. The best transition variant reached **13.0950** strict RMSE and worsened stress to **13.5710**.

**Modeling implication:** Pseudo-top geometry is not enough as additional scalar columns in the current Ridge/XGBoost setup. The plateau likely requires changing the target or architecture, especially for persistent hard wells, rather than adding more tabular pseudo-top derivatives.

### A78 - A direct local-window CNN changes some hard-well errors but is not a standalone solution

The `05_torch_local_window_cnn` notebook trained a compact three-channel Conv1D model over target GR, typewell GR, and difference windows plus safe scalar context. It scored **13.6857** strict RMSE and **14.5202** analog-distance stress RMSE, far worse than the direct regime mean. However, the hard-well ordering changed: `86454a6f` improved to about **49.9** RMSE from roughly **60.0** in the residual-stack diagnostics, while other hard wells worsened.

**Modeling implication:** Sequence learning has different errors but poor calibration and transfer as a direct model. Future neural work should be gated or residualized against the regime-mean anchor, with explicit hard-well/slice safeguards, instead of replacing the anchor outright.

### A79 - CNN residual blending is the first strong post-plateau signal

The `04_cnn_regime_gated_specialist` audit recomputed a fold-safe regime-mean anchor and local-window CNN specialist in the same validation frame. A uniform **20%** blend toward the CNN reached **12.7816** strict RMSE and **13.1223** analog-distance stress RMSE, improving both the direct regime mean and the residual-stack best. The leakage-only oracle row switch reached **10.7716** strict RMSE, while simple learned gates did not beat the uniform blend.

**Modeling implication:** The CNN is useful as a residual direction, not as a replacement. The next bottleneck is calibration: tune blend weights, stabilize CNN predictions across seeds, and build gates only if they improve on the uniform blend without stress regression.

### A80 - CNN blend optimum is beyond the first tuning grid

The `05_cnn_regime_blend_tuning` notebook averaged two lightweight CNN seeds and swept uniform blend weights from **0.00** to **0.50**. Strict RMSE kept improving through the largest tested weight, reaching **12.6514** at `blend_cnn_50`. Stress RMSE was best near `blend_cnn_35` at **12.9705**, while `blend_cnn_50` still improved stress to **13.0331** versus the direct regime-mean **13.4006**.

**Modeling implication:** The CNN correction is stronger than the first 20% audit suggested. The next sweep must bracket the optimum with weights above **0.50** and choose using both strict and analog-distance stress, rather than assuming a small residual blend is enough.

### A81 - Uniform CNN blend optimum is bracketed near 0.45

The `06_cnn_regime_blend_wide_grid` notebook widened the two-seed blend grid above **0.50**. Strict RMSE peaked at `blend_cnn_45` with **12.6497**, while `blend_cnn_50` was nearly tied at **12.6514**. Higher weights degraded stress, and the pure CNN blend (`blend_cnn_100`) worsened strict RMSE to **13.2216**.

**Modeling implication:** The deployable uniform blend is now bracketed around **0.40-0.50**. Further gains are more likely from better CNN seed averaging, architecture/target changes, or gate learning than from larger uniform CNN weights.

### A82 - Learned row weights improve stress but barely improve strict RMSE

The `07_cnn_regime_learned_weight_gate` notebook trained fold-heldout XGBoost regressors to predict clipped row-wise CNN blend weights. The best deployable model, `xgb_weight_soft_s100`, scored **12.6471** strict RMSE and **12.9475** analog-distance stress RMSE. This slightly improves uniform `blend_cnn_45` strict RMSE (**12.6497**) and stress (**12.9956**). The leakage-only continuous oracle is **10.8284**.

**Modeling implication:** Row-wise blend weights are learnable enough to help stress, but current gate features only recover a tiny part of the oracle gap. Larger gains likely require improving CNN specialist predictions or adding richer gate features that explain when the CNN correction is trustworthy.

### A83 - Richer calibrated gates improve strict RMSE but lose stress robustness

The `08_cnn_regime_calibrated_gate` notebook added per-well prediction-context features and fold-heldout CNN-win switch gates. The best deployable candidate, `xgb_switch_p100`, scored **12.6364** strict RMSE, improving `xgb_weight_soft_s100` by **0.0107**. However, analog-distance stress RMSE worsened from **12.9475** to **13.0089**. The leakage-only continuous oracle improved slightly to **10.8172**.

**Modeling implication:** More expressive row gates can find a small strict-CV gain but are not yet robust across analog-distance stress splits. The path to a larger RMSE drop is unlikely to be gate feature expansion alone; the CNN specialist itself or a stress-aware selection rule needs to improve.

### A84 - Residual-targeted CNN is worse than direct-delta CNN blending

The `09_cnn_regime_residual_specialist` notebook trained the local-window CNN on residuals after the pseudo-top regime anchor. The best deployable row, `residual_cnn_s085`, scored **12.7353** strict RMSE and **13.1028** analog-distance stress RMSE, trailing both `xgb_switch_p100` (**12.6364** strict) and `xgb_weight_soft_s100` (**12.9475** stress). Its leakage-only continuous shrink oracle was **11.8132**, worse than the direct-delta CNN/regime oracle around **10.817-10.828**.

**Modeling implication:** The CNN is more useful as an alternate direct-delta specialist than as a residual learner after the current regime anchor. Do not package the residual-targeted branch; return to direct-delta CNN quality, seed stability, or stress-aware gate selection.

### A85 - Stronger direct-delta CNN seed stability creates a real strict-CV gain

The `10_cnn_regime_direct_stability` notebook upgraded the direct-delta CNN specialist to a slightly wider three-convolution architecture and averaged three seed offsets. The strict-best candidate, `logit_switch_floor20`, scored **12.5179** strict RMSE, beating the prior strict-best **12.6364** by **0.1185**. Its stress RMSE was **12.9574**, slightly worse than the prior stress-best **12.9475**. The compromise candidate `xgb_weight_soft_s075` scored **12.5788** strict and **12.9485** stress.

**Modeling implication:** Improving the direct-delta CNN correction vector is more valuable than residual-targeting or shallow gate-only expansion. The next step should tune stress-aware selection around the stronger CNN candidates or test Kaggle transfer, while preserving the stronger architecture/seed recipe.

### A86 - Dilated CNN windows improve both strict and stress RMSE

The `11_cnn_regime_dilated_direct` notebook replaced the stronger direct CNN with a dilated local-window CNN while keeping the three-seed setup and comparable gates. The best deployable candidate, `logit_switch_floor20`, scored **12.4888** strict RMSE and **12.8465** analog-distance stress RMSE. This improves the prior strict-best **12.5179** and the prior stress-best **12.9475**. The leakage-only continuous oracle improved to **10.6114**.

**Modeling implication:** Local motif representation quality is still yielding measurable gains. The dilated direct-delta CNN should become the active package candidate, and the next modeling step should either tune this architecture further or use Kaggle public transfer to validate whether the local strict/stress gains carry over.

### A87 - Max pooling does not improve the dilated CNN branch

The `12_cnn_regime_dilated_pooling` notebook added adaptive max pooling alongside average pooling in the dilated CNN sequence encoder. The best deployable candidate, `xgb_weight_medium_s100`, scored **12.5051** strict RMSE and **12.9361** analog-distance stress RMSE, both worse than the active average-pooled dilated run (**12.4888** strict / **12.8465** stress). Its oracle also worsened from **10.6114** to **10.6897**.

**Modeling implication:** Peak-response pooling is not helping this small local-window representation. Keep the average-pooled dilated CNN as the active architecture and avoid further avg+max pooling variants unless the window design changes.

### A88 - Learned attention pooling also worsens the dilated CNN branch

The `13_cnn_regime_dilated_attention` notebook added a learned attention-weighted sequence summary alongside average pooling. The best deployable candidate, `logit_switch_floor20`, scored **12.5366** strict RMSE and **12.9839** analog-distance stress RMSE, both worse than the active average-pooled dilated candidate (**12.4888** strict / **12.8465** stress). Its oracle worsened to **10.7206**.

**Modeling implication:** The failure of both max pooling and attention pooling suggests the current 17-bin local-window representation benefits from smoothed dilated context rather than sharper pooling summaries. Further sequence work should change the window representation itself, not just the pooling head.

### A89 - Hard-well signed residual bias is partially predictable

The `14_cnn_regime_dilated_well_bias` notebook trained nested fold-heldout well-level Huber bias models on hidden-test-safe well and prediction summaries. The best candidate, `logit_switch_floor20__well_bias_s25`, scored **12.4568** strict RMSE and **12.7684** analog-distance stress RMSE, improving the active dilated-direct branch on both metrics. Larger shrink values were not best, so the useful signal is real but should be applied conservatively.

**Modeling implication:** Some of the persistent hard-well error is a signed offset that can be estimated from well/prediction summaries, not only local row-level sequence features. Conservative well-level calibration is now part of the active candidate recipe.

### A90 - Fine-tuning well-bias shrinkage gives only a marginal gain

The `15_cnn_regime_dilated_well_bias_tuning` notebook tuned the nested well-bias layer using a restricted deployable source grid, Huber alphas **0.15** and **0.40**, and shrink values **0.10-0.30**. The best candidate, `logit_switch_floor20__well_bias_h015_s30`, scored **12.4549** strict RMSE and **12.7649** analog-distance stress RMSE. This beats the original **0.25** shrink candidate by only **0.0019** strict RMSE and **0.0034** stress RMSE.

**Modeling implication:** The well-bias signal is real but nearly saturated under the current features. Keep the tuned conservative correction as the active recipe, but the next large gain should come from improving the CNN/state representation or deployable gate rather than another narrow well-bias shrink grid.
