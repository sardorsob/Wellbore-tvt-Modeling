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
