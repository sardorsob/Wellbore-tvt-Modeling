# Pseudo-Top Bayesian Mixture Regime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Test a Bayesian/MAP mixture-regime model where pseudo-top uncertainty controls soft stratigraphic-regime residual priors.

**Architecture:** Reuse the fold-safe pseudo-top builder, convert pseudo-top distances into soft formation probabilities, build formation-regime basis features, and fit Ridge MAP residual corrections with explicit shrinkage schedules. Compare against mean-only safe/typewell references under strict and analog-distance folds.

**Tech Stack:** Python, pandas, numpy, scikit-learn Ridge/Huber, XGBoost, nbformat/Jupyter, existing artifact tables/reports/run bundles.

---

### Task 1: Build Notebook And Run Model

**Files:**
- Create: `notebooks/modeling/06_bayesian/08_pseudo_top_mixture_regime.ipynb`
- Create: `artifacts/reports/model__pseudo_top_mixture_regime__2026-07-01.md`
- Create: `artifacts/tables/model__pseudo_top_mixture_regime_*__2026-07-01.csv`

- [x] Build fold-safe pseudo-top feature frame.
- [x] Convert pseudo-top distances to soft formation probabilities.
- [x] Fit safe/typewell mean models and Ridge MAP mixture residual priors.
- [x] Apply fixed and confidence-weighted shrinkage schedules.
- [x] Export strict/stress diagnostics, report, model comparison row, and run bundle.

### Task 2: Validate And Record

**Files:**
- Modify: `context/ASSUMPTIONS.md`
- Modify: `context/DECISIONS.md`
- Modify: `context/EXPERIMENTS.md`
- Modify: `context/STATUS.md`
- Modify: `context/CHANGELOG.md`
- Modify: `kaggle/FEATURES.md`

- [x] Execute notebook successfully.
- [x] Record exact metrics and decision.
- [x] Validate notebook schema, whitespace, and stratigraphic helper tests.

**Execution status:** Completed on 2026-07-01. Best candidate `xgb_pseudohuber_mean__ridge_alpha650__mixture_conf_s050` scored strict RMSE **13.4437** and analog-distance stress RMSE **13.7593**, improving the strict safe/typewell reference **13.4506** by **0.0069** but not clearing the packaging margin.
