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

---

## D030 - Stop flat pseudo-top XGBoost feature engineering

**Date:** 2026-06-30
**Decision:** Do not package or continue tuning the first pseudo-top interaction XGBoost feature set.
**Alternatives considered:**
- Package because the same-notebook safe/typewell baseline remains below the public-best benchmark locally.
- Keep expanding pseudo-top interactions with deeper trees or more crosses.
- Pivot to a Bayesian or mixture-regime formulation.
**Rationale:** The interaction notebook did not beat its own safe/typewell baseline. Best overall was `xgb_square_regularized__safe_typewell` at **13.4745** strict-hybrid RMSE and **13.8284** analog-distance stress RMSE. Best pseudo-top interaction candidate `xgb_pseudohuber_reference__pseudo_interactions` scored **13.4815** strict RMSE and **13.8681** stress RMSE.
**Consequences:** Treat raw/interacted pseudo-top XGBoost features as a completed negative screen. Keep the artifacts and feature importances as evidence.
**Follow-up:** Move pseudo-top uncertainty into a Bayesian mixture/prior model over stratigraphic regimes, or revisit validation design before more tree feature engineering.

---

## D031 - Iterate pseudo-top Bayesian mixture before packaging

**Date:** 2026-07-01
**Decision:** Keep the pseudo-top Bayesian mixture-regime model as the next promising path, but do not package it yet.
**Alternatives considered:**
- Package immediately because it beats the prior strict safe/typewell reference.
- Reject because the strict gain is too small.
- Iterate the mixture-regime formulation before a public-score probe.
**Rationale:** `xgb_pseudohuber_mean__ridge_alpha650__mixture_conf_s050` scored **13.4437** strict-hybrid RMSE and **13.7593** analog-distance stress RMSE. This is the first pseudo-top family result to beat the strict safe/typewell reference **13.4506**, but the margin is only **0.0069** RMSE and does not clear the **0.05** promotion gate.
**Consequences:** Do not push to Kaggle yet. Preserve the notebook as evidence that soft formation-regime priors are more promising than raw/interacted pseudo-top features.
**Follow-up:** Try a direct stratigraphic-regime mean model or calibrate the mixture prior against stress slices before packaging.

---

## D032 - Package pseudo-top regime mean for Kaggle probe

**Date:** 2026-07-01
**Decision:** Promote `xgb_square_global__regime_ridge_alpha80__confidence_s100` to a self-contained Kaggle package next.
**Alternatives considered:**
- Keep iterating locally because previous local gains have sometimes failed publicly.
- Package the smaller residual mixture prior instead.
- Move immediately to a heavier hierarchical Bayesian regime mean.
**Rationale:** The direct regime-mean model scored **13.0928** strict-hybrid RMSE and **13.4006** analog-distance stress RMSE. This clears the **0.05** local promotion margin versus both the pseudo-top mixture prior **13.4437** and the strict safe/typewell reference **13.4506**. It also improves analog stress by **0.3588** versus the previous pseudo-top mixture-prior stress score **13.7593**.
**Consequences:** Build a Kaggle code-competition notebook that recomputes pseudo-top proxy models on all train wells, predicts hidden test pseudo-top probabilities, fits the global XGBoost mean and regime-specific Ridge means, blends by confidence, and writes `/kaggle/working/submission.csv`.
**Follow-up:** Package and run locally first; submit to Kaggle only after the package writes finite predictions with the exact sample submission row count.

---

## D033 - Stop small-grid pseudo-top regime-mean tuning

**Date:** 2026-07-01
**Decision:** Do not replace the packaged `alpha80` regime-mean candidate with the tuned `alpha40` near-tie.
**Alternatives considered:**
- Package `xgb_square_global__regime_ridge_alpha40__confidence_s100` because it has the lowest strict RMSE.
- Continue widening Ridge alpha and confidence schedule sweeps.
- Move to boosted stacking or pseudo-top inference improvements.
**Rationale:** The tuned candidate scored **13.0910** strict-hybrid RMSE, only **0.0018** better than the packaged `alpha80` baseline **13.0928**, while analog-distance stress worsened from **13.4006** to **13.4065**. The improvement is below any meaningful promotion margin and adds no new structure.
**Consequences:** Keep the `alpha80` pseudo-top regime-mean package as the next Kaggle public-transfer probe. Treat the tuning notebook as evidence that this small deterministic family is saturated.
**Follow-up:** Move the next local modeling notebook to regime-mean stacking or pseudo-top inference quality rather than another narrow alpha/blend sweep.

---

## D034 - Do not package ungated regime-mean residual stack

**Date:** 2026-07-01
**Decision:** Keep `blend_plus_ridge_residual_s15` as a local diagnostic improvement, but do not package it over the pseudo-top regime-mean notebook.
**Alternatives considered:**
- Package the stack because it is the new best strict-CV result.
- Increase residual shrinkage or use direct XGBoost meta-features.
- Add stress-gated residual shrinkage before reconsidering packaging.
**Rationale:** `blend_plus_ridge_residual_s15` improved strict RMSE from **13.0928** to **13.0769**, but analog-distance stress worsened from **13.4006** to **13.4468**. The local strict gain is only **0.0159**, below the **0.05** promotion gate, and it moves the known transfer-risk metric in the wrong direction.
**Consequences:** Do not push this stack to Kaggle as-is. Preserve it as evidence that residual structure exists around the regime mean.
**Follow-up:** Try stress-gated or pseudo-top-uncertainty-gated residual shrinkage, or improve the pseudo-top inference stage itself.

---

## D035 - Move beyond residual shrinkage gates

**Date:** 2026-07-01
**Decision:** Do not continue local tuning of simple stress/confidence gates on the regime-mean residual stack.
**Alternatives considered:**
- Package the best strict residual stack despite stress degradation.
- Use the lower-stress margin-gated residual stack as a safer local candidate.
- Move to pseudo-top inference tuning or partial-pooling regime means.
**Rationale:** Stress gates produced a tradeoff but no promotion candidate. The best strict model remained `blend_plus_ridge_residual_s15_uniform` at **13.0769** strict RMSE and **13.4468** stress RMSE. The margin-gated compromise reduced stress to **13.4079**, but strict RMSE relaxed to **13.0853** and stress still trailed the original regime-mean reference **13.4006**.
**Consequences:** Keep the original pseudo-top regime-mean Kaggle package as the next public probe. Treat residual stacks as local diagnostics until a new gate or feature family improves both strict and stress.
**Follow-up:** Rebuild or calibrate pseudo-top inference, then revisit residual stacking only if the pseudo-top state estimate improves.

---

## D036 - Stop probability-scale calibration sweeps

**Date:** 2026-07-01
**Decision:** Keep pseudo-top probability scale **50** as the default and stop tuning softmax scale in isolation.
**Alternatives considered:**
- Use a sharper pseudo-top probability scale to emphasize nearest formation state.
- Use a smoother scale to hedge pseudo-top uncertainty.
- Continue searching intermediate scales.
**Rationale:** Scale **50** was best on strict RMSE (**13.0928**) and stress RMSE (**13.4006**). Every tested alternative worsened strict RMSE by at least **0.0717** and stress by at least **0.0849**.
**Consequences:** The current regime-mean package already uses the best tested probability scale. Further gains require better pseudo-top predictions or a different regime-mean modeling structure.
**Follow-up:** Trigger the stop-and-brainstorm gate before building the next notebook because several local iterations have not produced a package-worthy improvement.

---

## D037 - Keep the 120-row regime-mean sample cap

**Date:** 2026-07-01
**Decision:** Do not rebuild the pseudo-top regime-mean Kaggle package with the **220** target-row cap.
**Alternatives considered:**
- Increase row cap to train on a denser target-zone sample.
- Use pseudo-Huber global mean because it won the denser sample.
- Keep the original **120** cap and faster package.
**Rationale:** The **220** cap produced **13.1185** strict RMSE and **13.4745** stress RMSE, both worse than the original **120** cap result of **13.0928** strict and **13.4006** stress.
**Consequences:** Keep the existing pseudo-top regime-mean package unchanged. Do not spend more overnight time on row-cap sweeps.
**Follow-up:** Move to a genuinely different model family or revisit pseudo-top inference labels/features.

---

## D038 - Do not package alignment-aware pseudo-top regime mean

**Date:** 2026-07-01
**Decision:** Keep `12_alignment_aware_pseudo_top_regime_mean` as a diagnostic improvement, but do not replace the original pseudo-top regime-mean package.
**Alternatives considered:**
- Package because strict RMSE improved from **13.0928** to **13.0838**.
- Continue adding more local alignment descriptors.
- Treat alignment features as separate stress-aware mean features instead of pseudo-top predictors.
**Rationale:** Alignment-aware features improved strict RMSE by only **0.0089** and worsened stress from **13.4006** to **13.4303**. The proxy-top quality table also does not show a better pseudo-top estimator, so the gain is likely a fragile row-level mean effect rather than improved stratigraphic state.
**Consequences:** Do not push this candidate to Kaggle. Preserve artifacts because they show local alignment features have signal but need stress control.
**Follow-up:** Either build a different pseudo-top inference architecture or move to Bayesian/partial-pooling regime means where alignment uncertainty affects priors rather than raw features.

---

## D039 - Pause current partial-pooling route after negative result

**Date:** 2026-07-01
**Decision:** Do not package or continue tuning `12_bayesian_regime_partial_pooling` in its current shared Ridge form.
**Alternatives considered:**
- Promote the reference regime-mean re-fit because it nearly matches the package-ready regime mean.
- Continue sweeping Ridge alphas and confidence schedules in the partial-pooling residual/direct layers.
- Move to better pseudo-top/state inference or a different model class before returning to Bayesian hierarchy.
**Rationale:** The best overall candidate was the independent regime-mean reference at **13.0953** strict RMSE and **13.4038** stress RMSE, slightly worse than the package-ready regime mean **13.0928** strict / **13.4006** stress. The new shared residual and direct partial-pooling layers were materially worse, with best strict RMSE **13.4272** and **13.2049** respectively.
**Consequences:** Keep this notebook as a documented negative Bayesian shrinkage screen. Do not spend more time on shared Ridge partial-pooling schedules until the pseudo-top/state signal improves.
**Follow-up:** Trigger the stop-and-brainstorm gate and prioritize state revision, pseudo-top inference architecture, or a neural/sequence model that can learn local log motifs more directly.

---

## D040 - Pivot from schedule tuning to state-error modeling

**Date:** 2026-07-01
**Decision:** Stop simple shrinkage, confidence-schedule, row-density, and direct alignment-feature sweeps for the current pseudo-top regime-mean family.
**Alternatives considered:**
- Continue residual-stack tuning because it owns the best strict RMSE **13.0769**.
- Package alignment-aware or tuned regime means because they slightly improve strict RMSE.
- Build a failure-slice audit and change the next modeling target.
**Rationale:** The stop-gate audit shows the same hard wells recur across all eight recent runs. Strict-only improvements worsen analog stress, while the original regime mean remains the cleanest joint strict/stress anchor at **13.0928** / **13.4006**.
**Consequences:** The next notebook should focus on pseudo-top/state revision or a local-log sequence architecture. Do not create another notebook whose main novelty is a different blend weight, Ridge alpha, probability scale, or target-row cap.
**Follow-up:** Start with a state-revision/top-error audit that uses fold-safe proxy-top residuals, hard-well membership, progress zones, analog distance, GR motifs, and typewell lag diagnostics.

---

## D041 - Do not continue scalar local-sequence stack tuning

**Date:** 2026-07-01
**Decision:** Keep `13_xgboost_local_sequence_stack` as a negative diagnostic and do not tune its flat local-window stack further.
**Alternatives considered:**
- Continue shrinking/gating local-sequence residual corrections.
- Package the local-sequence stack as a public probe.
- Move to top-error/state revision.
**Rationale:** The best local-sequence stack candidate was the reference itself at **13.1613** strict RMSE and **13.4561** stress RMSE, both worse than the original regime mean **13.0928** / **13.4006**. Residual and direct meta heads worsened further.
**Consequences:** Local GR/typewell window summaries should not be added to the Kaggle package or residual stack as flat scalar features.
**Follow-up:** Build a pseudo-top state-revision notebook that learns systematic top residual corrections before regime probabilities are formed.

---

## D042 - Keep state revision as stress-control, not package candidate

**Date:** 2026-07-01
**Decision:** Do not package `13_pseudo_top_state_revision`, but continue with lighter or gated state revision.
**Alternatives considered:**
- Package because analog-distance stress improved from **13.4006** to **13.3883**.
- Abandon state revision because strict RMSE worsened to **13.1194**.
- Continue with a lighter or stress-gated correction schedule.
**Rationale:** The broad corrected-state path is a strict/stress tradeoff. It fails the strict promotion gate, but it is the first post-plateau experiment to improve analog stress. The top-correction diagnostics show **0.25** shrink is safer for top RMSE than the heavier correction used downstream.
**Consequences:** Do not push this notebook to Kaggle. Use it to design a gated state-revision model that applies correction only when pseudo-top uncertainty or analog stress justifies it.
**Follow-up:** Build `pseudo_top_state_revision_gated`: compare base, 0.25 global correction, confidence-gated correction, stress-gated correction, and hybrid stress-confidence correction.

---

## D043 - Stop pseudo-top residual correction schedules

**Date:** 2026-07-01
**Decision:** Do not continue tuning global, uncertainty-gated, stress-gated, or hybrid pseudo-top residual correction schedules.
**Alternatives considered:**
- Promote the best stress variant because it reached **13.3710** stress RMSE.
- Continue searching smaller gates or alternative shrinkage weights.
- Return to the unchanged regime mean and pursue a different state/model direction.
**Rationale:** The best strict row in `14_pseudo_top_state_revision_gated` is the unchanged base regime mean (**13.0928** strict / **13.4006** stress). The best stress row improves stress to **13.3710** but worsens strict to **13.1309**. This is a calibration tradeoff, not movement toward **10 RMSE**.
**Consequences:** Keep the original pseudo-top regime mean as the clean package anchor. Do not create another notebook whose main novelty is a top residual shrinkage schedule.
**Follow-up:** Next work should either calibrate public-transfer risk for the existing package anchor or test a genuinely different state model / higher-capacity architecture.

---

## D044 - Stop flat local-window tabular feature additions

**Date:** 2026-07-01
**Decision:** Do not continue adding raw local GR/typewell window bins to the current tabular regime-mean recipe.
**Alternatives considered:**
- Use more raw window bins or tune XGBoost regularization around them.
- Keep only the target GR bins and drop typewell/difference bins.
- Move to a genuinely different architecture or public-transfer calibration.
**Rationale:** Raw-window bins worsened strict RMSE to **13.1664** and stress RMSE to **13.4758**. This is weaker than both the original pseudo-top regime mean and the earlier scalar alignment-aware variant.
**Consequences:** Do not package this branch and do not spend more iteration budget on flat motif bins.
**Follow-up:** Either submit the clean pseudo-top regime-mean package as the current stable probe or test a non-tabular sequence/state architecture.

---

## D045 - Stop same-family OOF candidate blending

**Date:** 2026-07-01
**Decision:** Do not continue fixed averaging or positive-linear meta-blending among the current pseudo-top regime-mean, Ridge residual, and XGBoost residual stack candidates.
**Alternatives considered:**
- Use simple averages to reduce analog-distance stress while preserving most of the residual-stack strict gain.
- Use leave-one-fold-out positive linear meta-blends to learn fold-safe candidate weights.
- Move to a structurally different state representation or target.
**Rationale:** `03_oof_candidate_blend_audit` found the previous `blend_plus_ridge_residual_s15` candidate still best at **13.0769** strict RMSE / **13.4468** stress RMSE. The best simple blend, `oof_avg__base_ridge25`, reached **13.0779** strict / **13.4371** stress, close but not better. Positive-linear meta-blends worsened to **13.1001+** strict and **13.5604** stress, indicating poor transfer from candidate weights.
**Consequences:** Treat the residual-stack family as saturated unless a new candidate source changes the error geometry. Do not spend overnight iteration budget on more candidate weights, foldwise average schedules, or same-family stack heads.
**Follow-up:** Build the next notebook around a different target or representation: formation-boundary/state-transition modeling, hard-well pseudo-top relabeling diagnostics, or a true sequence representation that is evaluated against the same strict/stress gates.

---

## D046 - Stop tabular transition-distance features

**Date:** 2026-07-01
**Decision:** Do not continue adding pseudo-top transition-distance geometry as tabular columns to the current regime-mean model.
**Alternatives considered:**
- Add signed distances to adjacent pseudo tops only inside regime-specific Ridge means.
- Add transition features to both the global XGBoost mean and regime-specific Ridge means.
- Keep the unchanged direct regime-mean reference.
**Rationale:** `15_pseudo_top_transition_distance` found the unchanged reference still best at **13.0928** strict / **13.4006** stress. The best non-reference transition candidate scored **13.0950** strict and worsened stress to **13.5710**. Adding transition geometry to the global XGBoost path was worse.
**Consequences:** Treat pseudo-top-derived scalar feature additions as saturated. They can support diagnostics, but they should not consume more main modeling iterations unless paired with a different supervised target.
**Follow-up:** Move to hard-well/state-target diagnostics, sequence learning, or a direct well-trajectory architecture rather than more pseudo-top column engineering.
