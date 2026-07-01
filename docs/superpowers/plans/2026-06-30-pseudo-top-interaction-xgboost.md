# Pseudo-Top Interaction XGBoost Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Test whether pseudo-top uncertainty improves the primary XGBoost mean model when expressed as targeted interaction features.

**Architecture:** Reuse the fold-safe pseudo-top builder from `08_pseudo_top_uncertainty.ipynb`, then add controlled interactions between pseudo-top uncertainty and progress, distance, analog stress, and typewell alignment. Compare safe/typewell baselines against pseudo-top raw, uncertainty, interaction-only, and raw-plus-interaction feature sets under strict and analog-distance folds.

**Tech Stack:** Python, pandas, numpy, scikit-learn, xgboost, nbformat/Jupyter, existing artifact tables/reports/run bundle conventions.

---

### Task 1: Build Notebook And Artifacts

**Files:**
- Create: `notebooks/modeling/08_alignment/10_pseudo_top_interaction_xgboost.ipynb`
- Create: `artifacts/reports/model__pseudo_top_interaction_xgboost__2026-06-30.md`
- Create: `artifacts/tables/model__pseudo_top_interaction_xgboost_*__2026-06-30.csv`

- [x] Build fold-safe pseudo-top feature frame.
- [x] Add interaction columns for strict and stress folds.
- [x] Run XGBoost comparison over safe, raw pseudo-top, uncertainty, interaction, and raw-plus-interaction feature sets.
- [x] Export summary, fold metrics, zone/analog-bin diagnostics, well diagnostics, feature importance, findings, report, and model comparison row.

### Task 2: Validate And Record

**Files:**
- Modify: `context/ASSUMPTIONS.md`
- Modify: `context/DECISIONS.md`
- Modify: `context/EXPERIMENTS.md`
- Modify: `context/STATUS.md`
- Modify: `context/CHANGELOG.md`
- Modify: `kaggle/FEATURES.md`

- [x] Execute notebook successfully top-to-bottom.
- [x] Record exact metrics and decision.
- [x] Validate notebook schema, whitespace, and stratigraphic helper tests.

**Execution status:** Completed on 2026-06-30. Best overall remained `xgb_square_regularized__safe_typewell` at strict RMSE **13.4745** and stress RMSE **13.8284**. Best pseudo-top interaction candidate `xgb_pseudohuber_reference__pseudo_interactions` scored **13.4815**, so interaction features did not beat the same-notebook safe/typewell baseline.
