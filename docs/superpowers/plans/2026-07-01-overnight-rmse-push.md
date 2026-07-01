# Overnight RMSE Push Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Push the Wellbore TVT modeling pipeline below **10 RMSE** if feasible, while preserving clean experiment records and commit checkpoints.

**Architecture:** Work in experiment-sized notebooks, starting from the current pseudo-top regime-mean winner. Each notebook recomputes all features inside the notebook, exports tables/reports/run bundles, updates context files, validates, and commits before moving to the next idea.

**Tech Stack:** Python notebooks, pandas, NumPy, scikit-learn, XGBoost, existing repo artifact conventions, Kaggle code-competition packaging.

---

### Task 1: Package Current Pseudo-Top Regime Mean

**Files:**
- Create: `kaggle/pseudo_top_regime_mean_submission/`
- Create: `kaggle/pseudo_top_regime_mean_submission/kernel-metadata.json`
- Create: `kaggle/pseudo_top_regime_mean_submission/pseudo_top_regime_mean_submission.ipynb`
- Modify: `kaggle/SUBMISSIONS.md`
- Modify: `context/STATUS.md`

- [x] Build a self-contained Kaggle notebook from `notebooks/modeling/06_bayesian/09_pseudo_top_regime_mean.ipynb`.
- [x] Verify it discovers `/kaggle/input/...` files dynamically and writes `/kaggle/working/submission.csv`.
- [x] Run a local validation path against the repository data.
- [x] Record the package status and commit.

### Task 2: Tune Regime-Mean Shrinkage And Confidence

**Files:**
- Create: `notebooks/modeling/06_bayesian/10_pseudo_top_regime_mean_tuning.ipynb`
- Create: `artifacts/reports/model__pseudo_top_regime_mean_tuning__2026-07-01.md`
- Create: `artifacts/tables/model__pseudo_top_regime_mean_tuning_*__2026-07-01.csv`
- Modify: `context/ASSUMPTIONS.md`
- Modify: `context/DECISIONS.md`
- Modify: `context/EXPERIMENTS.md`
- Modify: `context/STATUS.md`
- Modify: `kaggle/FEATURES.md`

- [x] Expand the candidate grid around the best regime-mean setup.
- [x] Test confidence schedules, regime shrinkage, stress-aware blending, and robust fallback.
- [x] Export strict/stress summaries and failure slices.
- [x] Commit if the notebook produces reportable artifacts.

### Task 3: Stack Regime Means With Boosted Global Models

**Files:**
- Create: `notebooks/modeling/02_trees/11_xgboost_regime_mean_stack.ipynb`
- Create: `artifacts/reports/model__xgboost_regime_mean_stack__2026-07-01.md`
- Create: `artifacts/tables/model__xgboost_regime_mean_stack_*__2026-07-01.csv`
- Modify: context and feature-log files as in Task 2.

- [x] Use the regime-mean prediction as a fold-safe meta-feature or blend component.
- [x] Compare XGBoost square, pseudo-Huber, and conservative regularized recipes.
- [x] Reject candidates that improve strict RMSE but worsen analog-distance stress materially.
- [x] Commit if the notebook produces reportable artifacts.

### Task 4: Rebuild Pseudo-Top Inference If Plateaued

**Files:**
- Create: `notebooks/modeling/08_alignment/11_pseudo_top_inference_tuning.ipynb`
- Create: `artifacts/reports/model__pseudo_top_inference_tuning__2026-07-01.md`
- Create: `artifacts/tables/model__pseudo_top_inference_tuning_*__2026-07-01.csv`
- Modify: context and feature-log files as in Task 2.

- [ ] Improve the pseudo-top probability engine itself.
- [x] Test calibration, top-label smoothing, uncertainty margins, and alternative distance transforms.
- [x] Promote only if downstream TVT RMSE improves, not merely top-label proxy accuracy.
- [x] Commit if the notebook produces reportable artifacts.

### Extra Task: Stress-Gated Regime Residual Stack

**Files:**
- Create: `notebooks/modeling/02_trees/12_xgboost_regime_stress_gated_stack.ipynb`
- Create: `artifacts/reports/model__xgboost_regime_stress_gated_stack__2026-07-01.md`
- Create: `artifacts/tables/model__xgboost_regime_stress_gated_stack_*__2026-07-01.csv`
- Modify: context and feature-log files as in Task 2.

- [x] Test whether confidence, margin, low-stress, hybrid, or stress-floor gates rescue the residual-stack stress penalty.
- [x] Export strict/stress summaries and failure slices.
- [x] Reject candidates that do not beat the original regime mean on both strict and stress.
- [x] Commit if the notebook produces reportable artifacts.

### Task 5: Bayesian Partial-Pooling Regime Means

**Files:**
- Create: `notebooks/modeling/06_bayesian/11_bayesian_regime_partial_pooling.ipynb`
- Actual: `notebooks/modeling/06_bayesian/12_bayesian_regime_partial_pooling.ipynb`
- Create: `artifacts/reports/model__bayesian_regime_partial_pooling__2026-07-01.md`
- Create: `artifacts/tables/model__bayesian_regime_partial_pooling_*__2026-07-01.csv`
- Modify: context and feature-log files as in Task 2.

- [x] Add partial pooling only after deterministic regime means plateau.
- [x] Use conservative Student-t or Ridge-MAP analogues first, then heavier Bayesian tooling only if warranted.
- [x] Record priors and generative assumptions in `context/ASSUMPTIONS.md`.
- [x] Commit if the notebook produces reportable artifacts.

### Task 6: Stop-And-Brainstorm Gate

**Files:**
- Modify: `context/STATUS.md`
- Modify: `context/EXPERIMENTS.md`

- [x] If three consecutive completed notebooks fail to improve strict RMSE by at least **0.05**, stop coding.
- [x] Compare best/worst wells, high-error slices, pseudo-top errors, and public-transfer assumptions.
- [x] Write the next concrete plan before resuming model generation.

### Extra Task: Target-Row Density Check

**Files:**
- Create: `notebooks/modeling/06_bayesian/11_pseudo_top_regime_mean_more_rows.ipynb`
- Create: `artifacts/reports/model__pseudo_top_regime_mean_more_rows__2026-07-01.md`
- Create: `artifacts/tables/model__pseudo_top_regime_mean_more_rows_*__2026-07-01.csv`
- Modify: context and feature-log files as in Task 2.

- [x] Increase target-row cap from **120** to **220** while holding the direct regime-mean recipe fixed.
- [x] Export strict/stress summaries and failure slices.
- [x] Reject if denser sampling does not beat the original 120-row regime mean.
- [x] Commit if the notebook produces reportable artifacts.

### Extra Task: Alignment-Aware Pseudo-Top Regime Mean

**Files:**
- Create: `notebooks/modeling/08_alignment/12_alignment_aware_pseudo_top_regime_mean.ipynb`
- Create: `artifacts/reports/model__alignment_aware_pseudo_top_regime_mean__2026-07-01.md`
- Create: `artifacts/tables/model__alignment_aware_pseudo_top_regime_mean_*__2026-07-01.csv`
- Modify: context and feature-log files as in Task 2.

- [x] Add hidden-test-safe local GR/typewell alignment features to the pseudo-top proxy and regime mean.
- [x] Export strict/stress summaries, pseudo-top proxy quality, and failure slices.
- [x] Reject if alignment features do not improve both strict and stress against the original regime mean.
- [x] Commit if the notebook produces reportable artifacts.

## Verification

For every completed notebook:

- Run the notebook end-to-end.
- Run `git diff --check`.
- Validate notebook schema with `nbformat`.
- Run focused tests when shared helpers changed.
- Check `git status --short` and leave unrelated EDA notebook changes unstaged.
- Commit with no co-author trailer.
