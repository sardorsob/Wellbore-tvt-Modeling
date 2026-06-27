# XGBoost Strict-CV Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run an XGBoost strict-CV notebook that tests whether boosted-tree gains transfer beyond LightGBM under the same validation design.

**Architecture:** Create one reportable notebook in `notebooks/modeling/02_trees/07_xgboost_strict_cv.ipynb`. It will reuse the strict fold table, target-row sampling, signed-delta target, local formation features, and descriptor features from the LightGBM strict-CV notebook, then compare several XGBoost configurations on `strict_hybrid_fold` and stress-test the best candidate on `analog_distance_fold`.

**Tech Stack:** Python, pandas, numpy, scikit-learn metrics/preprocessing, XGBoost `XGBRegressor`, nbformat/nbconvert, markdown context trackers.

---

### Task 1: Scaffold The Notebook

**Files:**
- Create: `notebooks/modeling/02_trees/07_xgboost_strict_cv.ipynb`

- [ ] **Step 1: Read the LightGBM strict-CV notebook structure**

Run: `python -c "import nbformat; nb=nbformat.read('notebooks/modeling/02_trees/06_lightgbm_strict_cv.ipynb', as_version=4); print(len(nb.cells))"`
Expected: prints the number of cells without modifying files.

- [ ] **Step 2: Create the XGBoost notebook with reader-facing sections**

Use `nbformat` from an inline command to write markdown/code cells with these sections: tl;dr, Context & Methods, Setup, Data & Feature Helpers, Strict-CV Fit, Analog-Distance Stress Test, Artifacts, Summary.

- [ ] **Step 3: Keep code comments notebook-style**

Use string-literal comments such as `"Load strict fold assignments."` inside code cells and avoid leading `#` comments.

### Task 2: Implement Strict-CV XGBoost

**Files:**
- Modify: `notebooks/modeling/02_trees/07_xgboost_strict_cv.ipynb`

- [ ] **Step 1: Build the same feature table as LightGBM**

Use local raw train files, `artifacts/tables/validation__strict_cv_fold_assignments__2026-06-26.csv`, deterministic target-row sampling, well descriptors, typewell alignment summaries, formation-distance features, and signed target delta from last known TVT.

- [ ] **Step 2: Fit candidate XGBoost models**

Evaluate conservative, medium, regularized, and pseudo-Huber candidates when supported by the installed XGBoost version. Use fold-level RMSE, MAE, R2, and same-row last-known hold RMSE.

- [ ] **Step 3: Stress-test the best candidate**

Refit the best candidate on `analog_distance_fold` and export fold metrics so the result is comparable to the LightGBM stress test.

### Task 3: Export Artifacts And Context

**Files:**
- Create: `artifacts/tables/model__xgboost_strict_cv_*__2026-06-27.csv`
- Create: `artifacts/reports/model__xgboost_strict_cv_report__2026-06-27.md`
- Modify: `context/EXPERIMENTS.md`
- Modify: `context/STATUS.md`
- Modify: `context/CHANGELOG.md`

- [ ] **Step 1: Export reportable tables**

Write candidate summary, fold metrics, analog-distance stress metrics, feature importance, residual sample, well diagnostics, findings, and model-comparison row.

- [ ] **Step 2: Update context trackers**

Record the primary RMSE, stress RMSE, comparison to LightGBM, and the decision on whether XGBoost is worth packaging or tuning.

### Task 4: Validate And Commit

**Files:**
- Validate: `notebooks/modeling/02_trees/07_xgboost_strict_cv.ipynb`
- Commit: notebook, artifacts, context updates, and plan file

- [ ] **Step 1: Execute notebook top-to-bottom**

Run: `python -m jupyter nbconvert --to notebook --execute notebooks/modeling/02_trees/07_xgboost_strict_cv.ipynb --inplace --ExecutePreprocessor.timeout=1800`
Expected: notebook executes without runtime errors.

- [ ] **Step 2: Verify comment style and git scope**

Run: `Select-String -Path notebooks/modeling/02_trees/07_xgboost_strict_cv.ipynb -Pattern '^\s*#'`
Expected: no matches.

- [ ] **Step 3: Commit without co-author trailer**

Run: `git add ...` and `git commit -m "feat(model): run xgboost strict cv" -m "Compare XGBoost against LightGBM under strict boosted-tree folds."`
Expected: commit succeeds and does not include AI co-author metadata.
