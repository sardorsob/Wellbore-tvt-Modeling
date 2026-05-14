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
