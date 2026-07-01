# Pseudo-Top Gated Regime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the next modeling notebook that applies pseudo-top uncertainty as a residual gate/shrinkage signal instead of raw downstream XGBoost predictors.

**Architecture:** Create one notebook under `notebooks/modeling/08_alignment/09_pseudo_top_gated_regime.ipynb`. It rebuilds fold-safe pseudo-top uncertainty features, fits safe/typewell mean models, learns conservative residual experts, gates their corrections by pseudo-top uncertainty and state descriptors, and exports strict/stress diagnostics plus context updates.

**Tech Stack:** Python, pandas, numpy, scikit-learn, xgboost, nbformat/Jupyter, existing artifact tables/reports/run bundles.

**Execution status:** Completed on 2026-06-30. Best candidate `xgb_pseudohuber_mean__ridge_pseudo_state_residual__uncertainty_gate_conservative` scored strict RMSE **13.4694** and analog-distance stress RMSE **13.7665**. This improves the same-notebook mean-only reference by **0.0051** RMSE and improves analog stress versus the prior safe/typewell reference, but does not beat the strict local reference **13.4506**.

---

## File Structure

- Create: `notebooks/modeling/08_alignment/09_pseudo_top_gated_regime.ipynb`
  - Reader-facing experiment notebook for pseudo-top gated residual modeling.
- Modify: `context/ASSUMPTIONS.md`
  - Add the run conclusion after execution.
- Modify: `context/DECISIONS.md`
  - Record package/iterate/reject decision.
- Modify: `context/EXPERIMENTS.md`
  - Add run entry.
- Modify: `context/STATUS.md`
  - Update current next steps and model history.
- Modify: `context/CHANGELOG.md`
  - Add artifact/result summary.
- Modify: `kaggle/FEATURES.md`
  - Add gated pseudo-top feature family.
- Modify: `artifacts/tables/model_comparison.csv`
  - Append/replace run row.
- Create outputs:
  - `artifacts/tables/model__pseudo_top_gated_regime_summary__2026-06-30.csv`
  - `artifacts/tables/model__pseudo_top_gated_regime_fold_metrics__2026-06-30.csv`
  - `artifacts/tables/model__pseudo_top_gated_regime_gate_diagnostics__2026-06-30.csv`
  - `artifacts/tables/model__pseudo_top_gated_regime_zone_summary__2026-06-30.csv`
  - `artifacts/tables/model__pseudo_top_gated_regime_analog_bin_summary__2026-06-30.csv`
  - `artifacts/tables/model__pseudo_top_gated_regime_well_diagnostics__2026-06-30.csv`
  - `artifacts/tables/model__pseudo_top_gated_regime_feature_preview__2026-06-30.csv`
  - `artifacts/tables/model__pseudo_top_gated_regime_findings__2026-06-30.csv`
  - `artifacts/reports/model__pseudo_top_gated_regime__2026-06-30.md`
  - `artifacts/logs/runs/2026-06-30__pseudo_top_gated_regime/`

### Task 1: Build And Run Notebook

**Files:**
- Create: `notebooks/modeling/08_alignment/09_pseudo_top_gated_regime.ipynb`

- [ ] **Step 1: Generate notebook skeleton**

Use a temporary `scripts/_tmp_generate_pseudo_top_gated_regime.py` that writes the notebook with these sections:

```markdown
# Pseudo-Top Gated Regime Model
## 1. Purpose And Leakage Rules
## 2. Setup
## 3. Fold-Safe Pseudo-Top Frame
## 4. Mean And Residual Experts
## 5. Pseudo-Top Gate
## 6. Strict-CV Results
## 7. Stress Diagnostics
## 8. Artifact Export
## 9. Findings
```

- [ ] **Step 2: Implement fold-safe pseudo-top features**

Reuse the signed convention from `08_pseudo_top_uncertainty.ipynb`: train top-elevation proxy models only on rows with labels inside each fold; derive `pseudo_signed_offset = Z - pseudo_top_elev`; use squared-error proxy top models.

- [ ] **Step 3: Implement mean models and residual experts**

Use legal feature sets:

```python
safe_typewell = SAFE_FEATURE_COLS + TYPEWELL_FEATURE_COLS
gate_features = PSEUDO_UNCERTAINTY_COLS + STATE_FEATURE_COLS + SELECTED_DESCRIPTOR_COLS
```

Mean models:

```python
xgb_pseudohuber_mean
xgb_square_mean
huber_mean
```

Residual experts:

```python
ridge_pseudo_state_residual
huber_pseudo_state_residual
```

- [ ] **Step 4: Implement conservative gates**

For each fold, train residual experts on fold-train residuals, then apply shrinkage schedules:

```python
fixed_s010, fixed_s020, fixed_s030
uncertainty_gate_conservative
uncertainty_gate_open
```

The uncertainty gate should increase correction weight when pseudo-top margin/confidence is high and entropy/std are low. Clip correction weights to `[0.0, 0.35]` for conservative candidates and `[0.0, 0.50]` for open candidates.

- [ ] **Step 5: Execute notebook**

Run:

```powershell
C:\Users\sardo\AppData\Local\Programs\Python\Python313\python.exe -m jupyter nbconvert --to notebook --execute notebooks\modeling\08_alignment\09_pseudo_top_gated_regime.ipynb --output 09_pseudo_top_gated_regime.ipynb --output-dir notebooks\modeling\08_alignment
```

Expected: notebook executes successfully and writes all listed artifacts.

### Task 2: Update Context And Validate

**Files:**
- Modify: `context/ASSUMPTIONS.md`
- Modify: `context/DECISIONS.md`
- Modify: `context/EXPERIMENTS.md`
- Modify: `context/STATUS.md`
- Modify: `context/CHANGELOG.md`
- Modify: `kaggle/FEATURES.md`

- [ ] **Step 1: Record exact metrics**

Add the best legal strict RMSE, stress RMSE, best baseline, best gated candidate, and decision.

- [ ] **Step 2: Validate**

Run:

```powershell
git diff --check
C:\Users\sardo\AppData\Local\Programs\Python\Python313\python.exe -c "import nbformat; nb=nbformat.read('notebooks/modeling/08_alignment/09_pseudo_top_gated_regime.ipynb', as_version=4); nbformat.validate(nb); print('notebook ok')"
C:\Users\sardo\AppData\Local\Programs\Python\Python313\python.exe -m pytest tests\test_stratigraphic_features.py -q
```

Expected: all checks pass.

### Task 3: Commit

**Files:**
- Stage only the new notebook, new artifacts, context docs, feature log, plan file, and model comparison.

- [ ] **Step 1: Stage scoped files**

Run:

```powershell
git add notebooks\modeling\08_alignment\09_pseudo_top_gated_regime.ipynb artifacts\tables\model__pseudo_top_gated_regime_*__2026-06-30.csv artifacts\reports\model__pseudo_top_gated_regime__2026-06-30.md artifacts\logs\runs\2026-06-30__pseudo_top_gated_regime artifacts\tables\model_comparison.csv context\ASSUMPTIONS.md context\DECISIONS.md context\EXPERIMENTS.md context\STATUS.md context\CHANGELOG.md kaggle\FEATURES.md docs\superpowers\plans\2026-06-30-pseudo-top-gated-regime.md
```

- [ ] **Step 2: Commit**

Run:

```powershell
git commit -m "feat(model): add pseudo top gated regime" -m "Use fold-safe pseudo-top uncertainty as residual-gate evidence and record strict/stress diagnostics."
```

Do not stage unrelated dirty EDA notebooks.

## Self-Review

- Spec coverage: Covers notebook, artifacts, context docs, validation, and commit.
- Placeholder scan: No TBD/TODO placeholders remain.
- Type consistency: Run ID, artifact stem, and notebook path use `pseudo_top_gated_regime` consistently.
