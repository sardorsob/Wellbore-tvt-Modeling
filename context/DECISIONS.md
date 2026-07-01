# Decisions Log

## D001 — Use NumpPyro (JAX) as primary Bayesian backend

**Date:** 2026-05-08
**Decision:** Use NumpPyro for Gaussian Process and BNN implementation rather than PyMC.
**Alternatives considered:**
- PyMC (PyTensor backend) — mature, easier API, but slower on large datasets
- TensorFlow Probability — good GP support, but less flexible for custom BNN architectures
- GPyTorch — excellent GP, but PyTorch-centric; JAX not available
**Rationale:** JAX offers JIT compilation and GPU/TPU acceleration critical for iterating on GP kernels with large well datasets. NumpPyro provides HMC/NUTS + SVI in the same framework. Kaggle GPU notebooks run JAX efficiently.
**Consequences:** Steeper learning curve; requires careful JAX RNG key management.
**Follow-up:** If NumpPyro compilation is too slow for Kaggle 9h limit, fall back to GPyTorch GP.

---

## D002 — Do not use formation columns (ANCC, ASTNU, etc.) in baseline

**Date:** 2026-05-08
**Decision:** Exclude `ANCC`, `ASTNU`, `ASTNL`, `EGFDU`, `EGFDL`, `BUDA` from all models until confirmed present in test files.
**Alternatives considered:** Use them anyway and handle KeyError at predict time.
**Rationale:** Competition explicitly marks them as "training-only." Using them in training would cause a feature mismatch at test time.
**Consequences:** May leave signal on the table if test files do include these columns.
**Follow-up:** On first data load, log test file columns to `context/DECISIONS.md` and update this decision.

**Follow-up verification (notebook `01_phase0_phase1_data_and_schema.ipynb`, Phase 1B):** Downloaded **Kaggle** test split (**3** wells in bundle checked). `get_test_column_names(data/raw/test/)` confirms **no** formation-depth columns in test headers. **Decision unchanged:** keep these columns **out** of any train/test-parity feature list; they may still be used in **train-only** analysis or models that explicitly drop them at inference. Re-verify if competition releases additional test files with different schemas.

---

## D003 — Validate by masking TVT_input in evaluation zone

**Date:** 2026-05-08
**Decision:** For local validation, simulate test conditions by zeroing TVT_input in the evaluation zone of training wells, then measuring RMSE of predictions vs. true TVT.
**Alternatives considered:** Standard k-fold (but wells have sequential structure; random splits would leak context).
**Rationale:** Preserves the sequential context structure. Matches the exact test scenario.
**Consequences:** Validation RMSE directly comparable to leaderboard RMSE.
**Follow-up:** Consider leave-one-well-out CV for more stable estimates once baseline is running.

---

## D004 — JAX/Flax for all neural networks

**Date:** 2026-05-08
**Decision:** Use Flax (on JAX) for both BNN and standard NN architectures.
**Alternatives considered:** PyTorch + Pyro (BNN), Keras/TensorFlow.
**Rationale:** Consistent JAX ecosystem — single backend for GP, BNN, and deterministic NN. Optax provides all needed optimizers.
**Consequences:** Less online documentation than PyTorch. Functional programming style requires explicit state management.
**Follow-up:** None.

---

## D005 — Submission output path auto-detection

**Date:** 2026-05-08
**Decision:** `src/utils/paths.py` auto-detects Kaggle vs. local environment and sets data/output paths accordingly.
**Alternatives considered:** Environment variable override only.
**Rationale:** Kaggle notebooks should work identically to local code without any manual path changes.
**Consequences:** Slightly magic behavior; documented in `context/INTERFACES.md`.
**Follow-up:** None.

---

## D006 — Do not keep one-off notebook generator scripts

**Date:** 2026-06-21
**Decision:** Delete ad hoc Python scripts that only generate or rewrite a specific notebook once their output has been produced and verified. Keep only scripts that are reusable project tooling with a clear future role.
**Alternatives considered:**
- Keep every notebook generator for provenance.
- Move one-off generators into an archive folder.
**Rationale:** The notebooks, artifacts, run bundles, and context logs are the project deliverables. Keeping one-off generator scripts adds clutter and can confuse the source of truth.
**Consequences:** If a generated notebook needs broad structural changes later, recreate or refactor reusable tooling intentionally rather than relying on stale one-off scaffolding.
**Follow-up:** When a script is kept, document why it is reusable in its docstring or the relevant context file.

---

## D007 — Log Bayesian generative assumptions before reportable fitting

**Date:** 2026-06-21
**Decision:** For Bayesian GP, BNN, or other probabilistic models, write down the generative story, likelihood, prior rationale, and predictive-check expectations in `context/ASSUMPTIONS.md` or the relevant experiment notes before treating results as reportable.
**Alternatives considered:** Capture priors only inside notebooks.
**Rationale:** Bayesian modeling choices are part of the model definition, not incidental implementation details. Externalizing them keeps the reasoning auditable across long-running experiments.
**Consequences:** Bayesian notebooks may move slightly slower up front, but comparisons and debugging should be cleaner.
**Follow-up:** Add model-specific assumptions when starting the Bayesian GP notebook.

---

## D008 - Use Phase 2 EDA to guide the next feature pass

**Date:** 2026-06-21
**Decision:** Before serious LightGBM/XGBoost tuning or Bayesian modeling, prioritize feature engineering and validation checks suggested by Phase 2 EDA: boundary-distance features, endpoint TVT summaries, normalized evaluation-zone position, recent known-zone slope/window summaries, GR missingness flags, and robust typewell-alignment summaries.
**Alternatives considered:**
- Move directly to boosted-tree hyperparameter tuning on the current simple tabular features.
- Start Bayesian GP modeling immediately.
**Rationale:** Phase 2 confirms the task is dominated by long trailing-zone extrapolation. The current simple features are useful baselines, but the next lift is more likely to come from well-relative and boundary-aware signal than from tuning alone.
**Consequences:** Boosted trees will start from a richer, more geologically informed feature set. Bayesian modeling is deferred until the generative story around boundary extrapolation, GR missingness, and typewell alignment is explicit.
**Follow-up:** Run Phase 3 feature-signal EDA, then update `src/preprocessing/tabular.py` or add a richer feature builder before the first tuned LightGBM/XGBoost notebook.

---

## D009 - Build a richer well-relative feature builder before boosted-tree tuning

**Date:** 2026-06-21
**Decision:** Implement the next modeling feature pass around endpoint/boundary, trajectory, recent-slope, GR-availability, and cautious typewell summary features before serious LightGBM/XGBoost tuning.
**Alternatives considered:**
- Tune XGBoost/LightGBM on the existing 15 simple tabular features.
- Use last-known slope projection as the next baseline.
**Rationale:** Phase 3 shows all-candidate diagnostic probes beat endpoint hold (**14.6236** vs **15.5815** RMSE), while direct last-50 slope projection is unstable (**104.5322** RMSE). Feature-family combination looks more promising than one-feature extrapolation.
**Consequences:** The next modeling notebook should include a reusable feature builder or carefully documented notebook-local feature construction before hyperparameter sweeps.
**Follow-up:** Start with a richer deterministic feature builder and compare Ridge/Random Forest/LightGBM using the same grouped validation before Kaggle submission.

---

## D010 - Validation reports must include baseline and fold diagnostics

**Date:** 2026-06-21
**Decision:** Future reportable modeling notebooks should include endpoint-hold baseline comparison, grouped fold-level metrics, descriptor-balance summaries, and public-LB gap interpretation before a model is promoted or submitted repeatedly.
**Alternatives considered:**
- Continue using only mean grouped CV RMSE.
- Treat public leaderboard ordering as the tie-breaker.
**Rationale:** Phase 4 shows local CV and public LB can disagree on model ordering, and fold descriptor balance is not captured by one mean RMSE.
**Consequences:** Modeling notebooks will be slightly more verbose, but model decisions should be more robust and less leaderboard-driven.
**Follow-up:** Apply this protocol to the richer feature builder and the first LightGBM/XGBoost pass.

---

## D011 - Use notebook-local feature construction for submission-candidate models

**Date:** 2026-06-21
**Decision:** For models that may become Kaggle submissions, keep feature construction inline in the modeling notebook or duplicate it into the Kaggle notebook rather than depending on precomputed local artifacts or repo-only helpers.
**Alternatives considered:**
- Create a reusable production feature builder in `src/` and import it from every notebook.
- Save processed feature matrices locally and submit predictions from those.
**Rationale:** This is a Kaggle code competition. The notebook is rerun on hidden test data, so all inference-time feature logic must travel with the notebook.
**Consequences:** Some feature code may be duplicated across submission notebooks, but submissions will be self-contained and reproducible.
**Follow-up:** If a model is selected for Kaggle, create a self-contained Kaggle notebook with the same inline feature logic and no local artifact dependency.

---

## D012 - Pause GAM micro-tuning after the slope-core sweep

**Date:** 2026-06-21
**Decision:** Treat `core_fine__delta_boundary__k7__d3__a60` as the current best sklearn GAM-style candidate and avoid additional small spline/Ridge sweeps unless new feature families are introduced.
**Alternatives considered:**
- Continue fine-tuning knots, degree, alpha, and target definitions.
- Switch immediately to a true GAM package such as `pygam`.
- Submit the first GAM baseline without tuning.
**Rationale:** The tuned sweep improved the first GAM baseline by only **0.0032** RMSE, while GR and typewell additions did not beat the slope-core feature family. More micro-tuning is unlikely to yield the next meaningful gain.
**Consequences:** The GAM result remains useful as evidence for boundary/trajectory/recent-slope features, but the next larger modeling step should be either a self-contained GAM Kaggle read or boosted-tree modeling on the same feature logic.
**Follow-up:** If submitting, create a Kaggle-safe notebook for `core_fine__delta_boundary__k7__d3__a60`; otherwise proceed to LightGBM/XGBoost.

---

## D013 - Prefer robust signed-delta GLMs over proportion or count GLMs

**Date:** 2026-06-22
**Decision:** For the pre-Bayesian GLM path, model signed boundary deltas or endpoint residuals with Gaussian/robust/quantile-style continuous-response models before trying count or proportion families.
**Alternatives considered:**
- Fit Poisson models to raw `TVT`.
- Fit Gamma/Tweedie models to raw `TVT` or shifted deltas.
- Force Beta or Dirichlet models through normalization.
**Rationale:** The target-zone delta is signed and heavy-tailed, while `TVT` is a scalar depth response. Huber on signed deltas scored **14.7928** sampled grouped RMSE, while raw positive-response GLMs were much worse and Beta/Dirichlet do not match the target geometry.
**Consequences:** A Huber/robust signed-delta Kaggle notebook is a reasonable next external read. Bayesian models should start with Student-t or robust Gaussian residual assumptions rather than Beta, Dirichlet, or Poisson likelihoods.
**Follow-up:** If promoting this result, create a self-contained Huber slope-core submission notebook and compare against the fine-tuned GAM public LB **15.055**.

---

## D014 - Build hierarchy through contextual partial pooling before full HSGP

**Date:** 2026-06-22
**Decision:** Start the hierarchical modeling track with residual-bias diagnostics and contextual partial-pooling models before attempting a full hierarchical Hilbert-space Gaussian process.
**Alternatives considered:**
- Build a full hierarchical HSGP immediately with well-level random effects, additive HSGP terms, and hierarchical priors over length scales/amplitudes.
- Detour to LightGBM/XGBoost on the current slope-core and residual features before doing more Bayesian hierarchy work.
- Keep micro-tuning the existing Huber + additive Hilbert residual model.
**Rationale:** The current best public model is Huber + additive Hilbert residual with public LB **14.514**, and the best local diagnostic is `huber_plus_hsgp_dps_m8` with sampled grouped RMSE **14.3586**. The useful gain came from a robust signed-delta mean plus additive residual geometry, not from raw GP complexity. Direct train-well random effects can overfit grouped validation and fail on hidden Kaggle wells, so hierarchy should be inferred from test-safe well descriptors: known-zone summaries, boundary geometry, trajectory shape, GR availability, and residual diagnostics.
**Research basis:** HSGP references emphasize basis/boundary diagnostics and reduced-rank approximation checks; GP multitask/coregionalization references are useful later but require care because hidden wells may be unseen tasks. Stan-style hierarchical regression and posterior predictive checks support building from simpler partial-pooling pieces before larger Bayesian structures.
**Consequences:** Create `notebooks/modeling/07_hierarchical/` as the hierarchical track, but start with `00_residual_bias_audit.ipynb` and `01_empirical_bayes_offsets.ipynb`. Defer full hierarchical HSGP until contextual hierarchy proves transferable under grouped-well validation.
**Follow-up:** Use the notebook ladder recorded in `context/STATUS.md`; update `context/ASSUMPTIONS.md`, `context/EXPERIMENTS.md`, and `context/RISKS.md` after each hierarchical run.

---

## D015 - Pivot from shallow residual hierarchy to geologic alignment features

**Date:** 2026-06-23
**Decision:** Pause the next full hierarchical HSGP attempt and run a research-guided alignment/segmentation feature screen first. Prioritize typewell/analog similarity, pseudo-stratigraphic zone features, and fold-safe GR/trajectory alignment summaries before adding more Bayesian hierarchy.
**Alternatives considered:**
- Continue directly to `04_contextual_hierarchy_hilbert_residual.ipynb`.
- Build a full multitask/coregionalized GP keyed by well identity.
- Tune the current Huber + additive Hilbert residual basis and shrinkage parameters.
**Rationale:** Three shallow hierarchy screens produced gains below the promotion gate, while well-log and GP references suggest the missing structure is more likely in alignment, segmentation, nonstationarity, and geologic analogs than in another small post-hoc residual correction.
**Research basis:** Top-picking recommender systems use known tops and nearby/analog wells; well-tops-guided modeling supports stratigraphic segmentation; DTW references support explicit log-shape alignment; HSGP/GP references point toward basis diagnostics, robust likelihoods, and nonstationary or multitask structure only after the feature geometry is clear.
**Consequences:** Create an alignment-focused modeling track before deeper hierarchy. Any future hierarchical GP should consume fold-safe alignment and pseudo-zone descriptors rather than raw well IDs or weak residual descriptors alone.
**Follow-up:** Add `notebooks/modeling/08_alignment/00_typewell_alignment_feature_screen.ipynb` or equivalent, export artifacts, and record whether alignment features beat the current Huber/Hilbert local reference or at least improve a same-sample diagnostic by **0.05 RMSE**.

---

## D016 - Audit validation mismatch before adding deeper hierarchy or particle models

**Date:** 2026-06-26
**Decision:** Pause full hierarchical HSGP and particle/state-space modeling until the local validation protocol better explains public leaderboard behavior.
**Alternatives considered:**
- Continue to full contextual hierarchy plus Hilbert residuals.
- Build particle or latent trajectory models immediately.
- Keep tuning analog-gated residual corrections despite the first public miss.
**Rationale:** The analog-gated Huber + Hilbert candidate improved the bounded local residual sample by **0.1003 RMSE**, but the public score was **14.603**, worse than the ungated Huber + Hilbert public baseline **14.514**. Earlier shallow hierarchy screens improved local diagnostics by only **0.0008**, **0.0003**, and **0.0064 RMSE**. This pattern suggests validation mismatch and fragile feature transfer are the current bottlenecks, not insufficient model complexity.
**Consequences:** Create a validation-focused modeling track before deeper Bayesian work. The next notebook should compare local diagnostics, public leaderboard scores, fold/well descriptors, analog distance/quality, and promotion gates. Future LightGBM/XGBoost work should use stricter folds informed by this audit.
**Follow-up:** Add `notebooks/modeling/09_validation/00_local_public_mismatch_audit.ipynb`, export mismatch tables and a report, then define stricter CV folds before boosted-tree tuning.

---

## D017 - Use strict hybrid folds as the default boosted-tree validation split

**Date:** 2026-06-26
**Decision:** Use `strict_hybrid_fold` from `validation__strict_cv_fold_assignments__2026-06-26.csv` as the default validation split for the next LightGBM/XGBoost notebooks.
**Alternatives considered:**
- Use `analog_distance_fold` as the default because it is the strictest diagnostic stress test.
- Use `descriptor_cluster_fold` as the default because it maximizes descriptor separation.
- Continue with the earlier grouped-well folds.
**Rationale:** `strict_hybrid_fold` keeps the fold design practical for model tuning with row imbalance **1.000** and well imbalance **1.024**, while still spreading combined analog/extrapolation stress. `analog_distance_fold` has the highest strictness score and should be used as a follow-up stress test for promising candidates, but it is less appropriate as the first tuning split because it isolates one failure mode too strongly.
**Consequences:** Boosted-tree notebooks should read the strict fold assignment table and report results for `strict_hybrid_fold` first. Any candidate that looks strong should also be checked on `analog_distance_fold`, `descriptor_cluster_fold`, and `extrapolation_fold` before Kaggle packaging.
**Follow-up:** Start `notebooks/modeling/02_trees/06_lightgbm_strict_cv.ipynb` with the strict fold table and compare against current public baseline logic.

---

## D018 - Continue boosted trees but require a confirmation pass before Kaggle

**Date:** 2026-06-26
**Decision:** Continue the boosted-tree track after the first strict-CV LightGBM result, but do not package or submit it to Kaggle until one confirmation pass is complete.
**Alternatives considered:**
- Immediately package the LightGBM strict-CV winner for Kaggle.
- Abandon trees and return to Bayesian hierarchy.
- Move directly to XGBoost without recording the LightGBM promotion gate.
**Rationale:** `lgbm_medium_l2` scored **13.2363** on `strict_hybrid_fold` and **13.4453** on `analog_distance_fold`, both much better than the same-row last-known hold baseline. However, the analog-gate result already showed that local diagnostics can overstate public transfer, and this LightGBM screen uses sampled target rows rather than a full hidden-test rerun.
**Consequences:** The next step should be either a LightGBM reproducibility/tuning pass with varied row caps/seeds or an XGBoost strict-CV comparison using the same fold assignment table. Kaggle packaging should wait until the result is stable across at least one confirmation axis.
**Follow-up:** Run `notebooks/modeling/02_trees/07_xgboost_strict_cv.ipynb` or a LightGBM tuning notebook that varies row cap, learning rate, leaf count, and regularization under `strict_hybrid_fold` plus `analog_distance_fold`.

---

## D019 - Use one LightGBM Kaggle submission as an external transfer check

**Date:** 2026-06-27
**Decision:** Submit a self-contained, inference-safe LightGBM strict-CV notebook to Kaggle as an external transfer check, while keeping Huber + additive Hilbert residual public LB **14.514** as the active baseline until the LightGBM public score completes.
**Alternatives considered:**
- Wait for XGBoost or a LightGBM reproducibility pass before any Kaggle submission.
- Submit the local strict-CV notebook directly with formation-distance features.
- Return to Bayesian hierarchy without testing whether the strict-CV boosted-tree gain transfers externally.
**Rationale:** The strict-CV LightGBM result is large enough locally (`strict_hybrid_fold` RMSE **13.2363**, `analog_distance_fold` RMSE **13.4453**) to justify one leaderboard read, but the Kaggle package must exclude local-only formation columns that are absent from hidden test files. This means the submission tests an inference-safe approximation of the local LightGBM idea, not the exact local feature set.
**Consequences:** Submission `54097453` completed with public LB **14.687**, worse than Huber + additive Hilbert **14.514**. Treat the gap as evidence that full-feature strict-CV tree gains need stronger hidden-test-safe feature transfer before Kaggle promotion.
**Follow-up:** Build inference-safe boosted-tree tuning that removes train-only formation-distance columns and replaces them with typewell, analog, pseudo-zone, or trajectory-derived proxies.

---

## D020 - Promote XGBoost to inference-safe boosted-tree tuning

**Date:** 2026-06-27
**Decision:** Promote XGBoost as the next boosted-tree direction, but do not package the local strict-CV notebook directly for Kaggle because it uses formation-distance features that are absent from hidden test files.
**Alternatives considered:**
- Package the local XGBoost strict-CV winner immediately.
- Continue tuning LightGBM only.
- Return to Bayesian hierarchy after one boosted-tree confirmation.
**Rationale:** `xgb_pseudohuber_shallow_robust` scored **12.7504** on `strict_hybrid_fold` and **12.9116** on `analog_distance_fold`, beating the LightGBM strict and stress references by about **0.49** and **0.53** RMSE. This is the largest local diagnostic improvement since Huber + Hilbert. However, top features include `abs_dist_to_BUDA` and `dist_to_BUDA`, so direct Kaggle packaging would leak reliance on train-only columns that hidden test wells do not expose.
**Consequences:** The first inference-safe XGBoost package scored public LB **14.472**, beating Huber + Hilbert **14.514** and inference-safe LightGBM **14.687**. Treat XGBoost as the active external baseline, but do not assume the full-feature local RMSE will transfer without hidden-test-safe formation proxies.
**Follow-up:** Build `notebooks/modeling/02_trees/08_xgboost_inference_safe_tuning.ipynb` or an equivalent boosted-tree feature-ablation notebook using `strict_hybrid_fold` plus `analog_distance_fold`, then package only candidates with a clear local rationale beyond single-leaderboard movement.

---

## D021 - Do not package the first proxy-formation XGBoost tuning pass

**Date:** 2026-06-28
**Decision:** Do not submit `08_xgboost_inference_safe_tuning.ipynb` to Kaggle yet.
**Alternatives considered:**
- Package the best legal `safe_typewell` candidate because it slightly beat safe base locally.
- Package the proxy-formation candidate because it was designed to mimic the full-feature formation-distance signal.
- Package the oracle direct-formation diagnostic.
**Rationale:** The best legal model, `pseudo_shallow_more_trees__safe_typewell`, scored **13.4178** on `strict_hybrid_fold` and **13.7533** on `analog_distance_fold`, only **0.0222** better than safe base locally. The proxy-formation feature set scored **13.5503**, worse than safe base by **0.1103** RMSE. The oracle direct-formation diagnostic scored **12.9597**, but those direct formation features are not hidden-test-safe.
**Consequences:** Keep XGBoost as the active boosted-tree direction, but move away from this exact formation-proxy recipe. Treat the oracle gap as evidence that stronger structural/geologic proxy features are needed rather than as a Kaggle packaging candidate.
**Follow-up:** Try residual stacking around the public-best inference-safe XGBoost path or build a stronger submission-safe stratigraphic proxy before the next Kaggle push.

---

## D022 - Promote Ridge residual stacking to the next Kaggle package

**Date:** 2026-06-28
**Decision:** Package the Ridge residual-stacked inference-safe XGBoost candidate next.
**Alternatives considered:**
- Continue tuning proxy-formation features despite their local miss.
- Submit the base safe/typewell XGBoost tuning pass.
- Use an XGBoost residual head instead of a linear Ridge residual correction.
**Rationale:** `ridge_residual_s025` improved strict-CV RMSE from **13.4178** to **13.2548** and analog-distance stress RMSE from **13.7424** to **13.5632**. Both improvements clear the **0.05 RMSE** gate. The XGBoost residual head worsened strict CV, while Huber was close but slightly worse than Ridge. The result was reproduced with a clean rerun.
**Consequences:** Build a self-contained Kaggle notebook that trains the base XGBoost model and a nested/residual Ridge correction using only hidden-test-safe row, descriptor, and typewell features. Preserve the **25%** residual shrinkage unless a packaging validation run proves a safer schedule.
**Follow-up:** Create `kaggle/xgboost_residual_stack_submission/`, validate local output shape, push to Kaggle, and submit only if the notebook writes `/kaggle/working/submission.csv` successfully.

---

## D023 - Reject residual stacking after public transfer miss

**Date:** 2026-06-29
**Decision:** Do not promote the XGBoost Ridge residual-stack package over the current XGBoost strict inference-safe baseline.
**Alternatives considered:**
- Keep tuning residual shrinkage after the public miss.
- Treat the local strict/stress gains as stronger evidence than the public score.
- Revert to the current XGBoost strict inference-safe package as the active public baseline.
**Rationale:** The residual-stack package completed on Kaggle version 2 and submission `54157692` scored **14.583**, worse than the current XGBoost strict inference-safe public LB **14.472**. This happened despite local strict-CV improvement of **0.1630** RMSE and analog-distance stress improvement of **0.1792** RMSE. The mismatch indicates that the residual head learned local residual shape that did not transfer to the public rerun distribution.
**Consequences:** Keep `54129475` / **14.472** as the active public benchmark. Treat generic post-hoc residual stacking as fragile unless a future residual feature has a stronger geologic mechanism or passes a validation design calibrated against public transfer.
**Follow-up:** Focus next on hidden-test-safe feature geometry, stratigraphic/typewell structure, or validation recalibration rather than another generic residual stacker.

---

## D024 - Pivot the next modeling cycle to stratigraphic transfer features

**Date:** 2026-06-30
**Decision:** Pause generic residual stacking and model-class hopping until we run a domain-driven stratigraphic-transfer audit and feature screen.
**Alternatives considered:**
- Keep tuning XGBoost residual heads because local strict-CV RMSE moved.
- Jump to a hierarchical GP/HSGP immediately.
- Submit another tree package with small hyperparameter changes.
**Rationale:** Deep domain review and our own results agree that the missing signal is formation-relative state. Direct formation-distance features helped local XGBoost strongly, but legal proxies did not transfer enough, and two locally promising residual/analog corrections missed publicly. Formation-top and well-log correlation literature support sequence alignment, top/proximity uncertainty, and log motif matching as the next source of hidden-test-safe signal.
**Consequences:** The next notebook should measure and build fold-safe stratigraphic-state features before another Kaggle submission. The notebook should record whether typewell/known-zone alignment uncertainty explains public-miss-like wells, and only then feed those features into XGBoost/LightGBM/Huber/HSGP candidates.
**Follow-up:** Create the next alignment/validation notebook around DTW/log-shape alignment, pseudo-top uncertainty, known-zone boundary behavior, and stress-slice validation. Require a clear strict-fold and stress-slice gain before packaging.

---

## D025 - Do not package the first stratigraphic-state feature screen

**Date:** 2026-06-30
**Decision:** Do not promote `notebooks/modeling/08_alignment/07_stratigraphic_state_feature_screen.ipynb` to Kaggle packaging.
**Alternatives considered:**
- Package the known-zone state features because they improved same-notebook Huber safe-base RMSE by **0.0391**.
- Continue tuning the lightweight XGBoost squared-error consumer used in the screen.
- Move directly to a Bayesian stratigraphic-regime model using the same features.
**Rationale:** The screen found directional signal, especially for known-zone state summaries, but it did not clear the promotion gate. Best non-oracle strict-hybrid RMSE was **14.9922**, far worse than the current safe-XGBoost local reference **13.4178** and public baseline **14.472**. The alignment-uncertainty and analog additions worsened the Huber screen when combined naively, and the XGBoost squared-error consumer in this notebook is not comparable to the current packaged pseudo-Huber XGBoost recipe.
**Consequences:** Keep the reusable `src/preprocessing/stratigraphic.py` utilities and the artifacts as evidence. Treat known-zone state features as a promising component, not a submission candidate.
**Follow-up:** Run a tighter second-stage screen that injects the known-zone state features into the exact current XGBoost inference-safe package recipe, and separately consider a Bayesian stratigraphic-regime model where alignment uncertainty controls shrinkage rather than entering as raw tree features.

---

## D026 - Reject raw known-state XGBoost injection

**Date:** 2026-06-30
**Decision:** Do not package `notebooks/modeling/02_trees/10_xgboost_known_state_features.ipynb` for Kaggle.
**Alternatives considered:**
- Promote the notebook because it uses the current inference-safe XGBoost recipe.
- Keep tuning raw known-zone state features inside XGBoost.
- Move the state signal into a gated or Bayesian stratigraphic-regime model.
**Rationale:** Exact-recipe injection did not improve the current local XGBoost benchmark. The best legal result stayed with `safe_typewell` at **13.4334** strict-hybrid RMSE, while known-state feature sets were worse; the best state-augmented result was **13.5189**. Stress RMSE also trailed the current local reference, and the notebook did not beat the active public benchmark submission path.
**Consequences:** Keep the artifacts as a negative screen and do not create a Kaggle package from this notebook. Known-zone state remains useful evidence, but raw scalar injection is not the right expression for XGBoost.
**Follow-up:** Try a stratigraphic-regime or Bayesian gating design where state features control shrinkage, mixture weights, or priors, then compare against the current XGBoost public baseline before any submission.

---

## D027 - Package gated Bayesian regime as a curiosity probe

**Date:** 2026-06-30
**Decision:** Package `huber_plus_gate_conservative` for one Kaggle public-score read, but do not promote it before the score returns.
**Alternatives considered:**
- Stop after the local notebook because the model remains worse than local XGBoost.
- Tune wider gates or weaker priors before any external read.
- Submit the conservative gate immediately as an informative Bayesian-regime probe.
**Rationale:** The conservative gate improved same-notebook Huber mean-only by **0.3981** RMSE and gave interpretable mixture weights dominated by the state residual expert. The result is still worse than the current local XGBoost reference, but it directly tests the hypothesis that state should control shrinkage/mixture weights rather than enter as raw features.
**Consequences:** Create and push a self-contained Kaggle notebook under `kaggle/gated_bayesian_regime_submission/`. Treat the public score as evidence about state-regime transfer, not as an automatic new baseline.
**Follow-up:** If the public score beats **14.472**, promote and tune this family. If it misses, keep the gating evidence locally and combine the idea with the stronger XGBoost mean or stricter validation calibration.

---

## D028 - Iterate on pseudo-top uncertainty through gating, not raw tree features

**Date:** 2026-06-30
**Decision:** Do not package the first pseudo-top uncertainty feature screen for Kaggle, but keep the feature family for gated/shrinkage models.
**Alternatives considered:**
- Package the best pseudo-top XGBoost candidate because it remains below the active public LB benchmark locally.
- Keep tuning raw pseudo-top feature columns inside XGBoost.
- Move pseudo-top margin, entropy, and ensemble spread into a regime gate or Bayesian prior.
**Rationale:** The pseudo-top proxy itself is plausible, with mean top-elevation RMSE **35.0660** and nearest-top agreement **0.5804** under `strict_hybrid_fold`. But downstream raw feature injection worsened the legal XGBoost path: best legal overall was `xgb_square_regularized__safe_typewell` at **13.4506** strict-hybrid RMSE, while best pseudo-top candidate `xgb_pseudohuber_known_state_recipe__safe_pseudo_top_uncertainty` scored **13.5891**. The stress RMSE for the best legal candidate was **13.8870**, so the package gate is not cleared.
**Consequences:** Keep the artifacts as a clean fold-safe pseudo-top audit. Treat pseudo-top uncertainty as state evidence for gates, priors, or mixture weights rather than a broad additive tree feature set.
**Follow-up:** Build a gated XGBoost/Bayesian hybrid where pseudo-top uncertainty controls residual shrinkage and formation-regime weights, then compare against the current public-best XGBoost strict inference-safe path.

---

## D029 - Do not package first pseudo-top gated residual regime

**Date:** 2026-06-30
**Decision:** Keep the pseudo-top gated-regime notebook as an iteration artifact, but do not package it for Kaggle yet.
**Alternatives considered:**
- Package because analog-distance stress RMSE improved versus the prior safe/typewell stress reference.
- Tune wider residual gates and larger corrections immediately.
- Move pseudo-top uncertainty into the primary mean model or Bayesian prior before packaging.
**Rationale:** The best gated candidate `xgb_pseudohuber_mean__ridge_pseudo_state_residual__uncertainty_gate_conservative` scored **13.4694** strict-hybrid RMSE and **13.7665** analog-distance stress RMSE. This improved the same-notebook mean-only reference **13.4745** by only **0.0051** RMSE and did not beat the strict local reference **13.4506**, even though stress behavior improved versus **13.8870**.
**Consequences:** Do not create a Kaggle package from this residual gate. Preserve the gate diagnostics because they suggest pseudo-top uncertainty can stabilize stress folds.
**Follow-up:** Try incorporating pseudo-top uncertainty into the mean learner itself, or into a Bayesian mixture/prior design where it changes regime weights before residuals are estimated.
