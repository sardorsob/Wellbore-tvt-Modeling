# Pseudo-Top Uncertainty Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a fold-safe pseudo-formation/top uncertainty notebook that turns train-only formation columns into hidden-test-safe latent stratigraphic features and tests whether those features improve downstream RMSE.

**Architecture:** Create one reportable notebook under `notebooks/modeling/08_alignment/08_pseudo_top_uncertainty.ipynb`. The notebook uses direct formation columns only as local training labels and oracle diagnostics, learns fold-safe pseudo-top regressors from known-zone/test-safe evidence, exports uncertainty features, and tests those features first in an XGBoost-style signed-delta consumer plus a robust Huber consumer. Context docs record whether the pseudo-top route is promising enough for a Kaggle package.

**Tech Stack:** Python, pandas, numpy, scikit-learn, xgboost, nbformat/Jupyter, existing `artifacts/tables`, `artifacts/reports`, and `artifacts/logs/runs` conventions.

---

## File Structure

- Create: `notebooks/modeling/08_alignment/08_pseudo_top_uncertainty.ipynb`
  - Reader-facing experiment notebook.
  - Builds target-zone row frame, pseudo-top labels, fold-safe pseudo-top predictions, uncertainty features, downstream model comparisons, artifacts, and report.
- Modify: `context/ASSUMPTIONS.md`
  - Add a new assumption describing whether pseudo-top uncertainty recovers formation-distance signal.
- Modify: `context/DECISIONS.md`
  - Add a decision to package, iterate, or reject this first pseudo-top feature pass.
- Modify: `context/EXPERIMENTS.md`
  - Add the run entry `2026-06-30__pseudo_top_uncertainty`.
- Modify: `context/STATUS.md`
  - Update task ladders and next-up items.
- Modify: `context/CHANGELOG.md`
  - Add the new experiment and artifact summary.
- Modify: `kaggle/FEATURES.md`
  - Add the pseudo-top uncertainty feature family and leakage notes.
- Modify: `artifacts/tables/model_comparison.csv`
  - Append or replace the row for the pseudo-top uncertainty run.
- Create outputs:
  - `artifacts/tables/model__pseudo_top_uncertainty_feature_preview__2026-06-30.csv`
  - `artifacts/tables/model__pseudo_top_uncertainty_top_label_summary__2026-06-30.csv`
  - `artifacts/tables/model__pseudo_top_uncertainty_proxy_quality__2026-06-30.csv`
  - `artifacts/tables/model__pseudo_top_uncertainty_summary__2026-06-30.csv`
  - `artifacts/tables/model__pseudo_top_uncertainty_fold_metrics__2026-06-30.csv`
  - `artifacts/tables/model__pseudo_top_uncertainty_zone_summary__2026-06-30.csv`
  - `artifacts/tables/model__pseudo_top_uncertainty_feature_importance__2026-06-30.csv`
  - `artifacts/tables/model__pseudo_top_uncertainty_findings__2026-06-30.csv`
  - `artifacts/reports/model__pseudo_top_uncertainty__2026-06-30.md`
  - `artifacts/logs/runs/2026-06-30__pseudo_top_uncertainty/`

---

### Task 1: Inspect Data Contracts Before Writing Notebook

**Files:**
- Read: `data/raw/train/*__horizontal_well.csv`
- Read: `artifacts/tables/validation__strict_cv_fold_assignments__2026-06-26.csv`
- Read: `context/ASSUMPTIONS.md`
- Read: `notebooks/modeling/02_trees/10_xgboost_known_state_features.ipynb`

- [ ] **Step 1: Verify formation column semantics**

Run:

```powershell
@'
import pandas as pd
from pathlib import Path
root = Path.cwd()
train_dir = root / "data" / "raw" / "train"
formation_cols = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]
rows = []
for path in sorted(train_dir.glob("*__horizontal_well.csv"))[:25]:
    df = pd.read_csv(path)
    row = {"wellname": path.name.replace("__horizontal_well.csv", ""), "n_rows": len(df)}
    for col in formation_cols:
        values = pd.to_numeric(df[col], errors="coerce")
        row[f"{col}_nunique"] = int(values.nunique(dropna=True))
        row[f"{col}_min"] = float(values.min())
        row[f"{col}_max"] = float(values.max())
    rows.append(row)
summary = pd.DataFrame(rows)
print(summary[[c for c in summary.columns if c.endswith("_nunique")]].describe().to_string())
print(summary.head().to_string())
'@ | C:\Users\sardo\AppData\Local\Programs\Python\Python313\python.exe -
```

Expected: Each formation column should behave like a top/depth reference varying by well or row. If columns are constants per well, train pseudo-top regressors at well/row level but label semantics become distance-to-top. If columns vary row-wise, train row-level signed-distance labels.

- [ ] **Step 2: Verify submission-side absence**

Run:

```powershell
@'
import pandas as pd
from pathlib import Path
test_dir = Path("data/raw/test")
formation_cols = {"ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"}
for path in sorted(test_dir.glob("*__horizontal_well.csv")):
    cols = set(pd.read_csv(path, nrows=1).columns)
    print(path.name, sorted(formation_cols & cols))
'@ | C:\Users\sardo\AppData\Local\Programs\Python\Python313\python.exe -
```

Expected: The printed intersection is empty or not usable for hidden-test submission. The notebook must treat formation columns as labels/oracle diagnostics only.

- [ ] **Step 3: Record label decision in the notebook design notes**

Use this rule:

```text
Pseudo-top labels are direct signed distances from each candidate formation top to the row position when available locally.
Direct formation columns are never downstream model features in legal candidates.
Oracle feature sets may include direct distances only to quantify the recoverable signal ceiling.
```

### Task 2: Build Notebook Skeleton And Feature Frame

**Files:**
- Create: `notebooks/modeling/08_alignment/08_pseudo_top_uncertainty.ipynb`

- [ ] **Step 1: Create notebook sections**

Use these sections:

```markdown
# Pseudo-Top Uncertainty Feature Screen
## 1. Purpose And Leakage Rules
## 2. Setup
## 3. Data Loading And Label Audit
## 4. Fold-Safe Pseudo-Top Feature Builders
## 5. Downstream Model Consumers
## 6. Strict-CV Results
## 7. Stress Diagnostics
## 8. Artifact Export
## 9. Findings
```

Code comments inside code cells should use string-literal comments like:

```python
"Build row-level labels and legal predictor features."
```

- [ ] **Step 2: Add setup constants**

Use:

```python
RUN_DATE = "2026-06-30"
RUN_ID = f"{RUN_DATE}__pseudo_top_uncertainty"
FORMATION_COLS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]
MAX_TARGET_ROWS_PER_WELL = 120
SEED = 20260630
PRIMARY_FOLD = "strict_hybrid_fold"
STRESS_FOLD = "analog_distance_fold"
```

- [ ] **Step 3: Build target-row frame**

Reuse and adapt the row construction pattern from `10_xgboost_known_state_features.ipynb`:

```python
def build_target_rows(csv_path, meta_row):
    df = pd.read_csv(csv_path).sort_values("MD").reset_index(drop=True)
    known_mask = df["TVT_input"].notna()
    if known_mask.sum() == 0:
        return pd.DataFrame()
    last_known_idx = int(np.flatnonzero(known_mask.to_numpy())[-1])
    known = df.loc[:last_known_idx].copy()
    target = df.loc[last_known_idx + 1:].copy()
    target = target.loc[target["TVT"].notna()].copy()
    if len(target) > MAX_TARGET_ROWS_PER_WELL:
        idx = np.linspace(0, len(target) - 1, MAX_TARGET_ROWS_PER_WELL).round().astype(int)
        target = target.iloc[np.unique(idx)].copy()
    # add safe row, known-state, typewell descriptor, and fold columns here
```

Expected output: `data` has one row per sampled target row and includes legal row descriptors, known-zone summaries, fold columns, `TVT`, `target_delta`, and oracle label columns.

### Task 3: Add Pseudo-Top Label And Feature Builders

**Files:**
- Modify: `notebooks/modeling/08_alignment/08_pseudo_top_uncertainty.ipynb`

- [ ] **Step 1: Define direct labels**

For each formation:

```python
for formation in FORMATION_COLS:
    data[f"label_dist_to_{formation}"] = data[f"formation_top_{formation}"] - data["Z"]
    data[f"label_abs_dist_to_{formation}"] = np.abs(data[f"label_dist_to_{formation}"])
data["label_nearest_formation"] = data[[f"label_abs_dist_to_{f}" for f in FORMATION_COLS]].idxmin(axis=1).str.replace("label_abs_dist_to_", "")
data["label_nearest_abs_dist"] = data[[f"label_abs_dist_to_{f}" for f in FORMATION_COLS]].min(axis=1)
```

- [ ] **Step 2: Define legal predictors for pseudo-top models**

Start with:

```python
PSEUDO_TOP_PREDICTOR_COLS = [
    "row_idx_original", "MD", "X", "Y", "Z", "GR",
    "last_known_tvt_feature", "last_known_md",
    "md_from_last_known", "md_from_start", "eval_progress",
    "dx_from_last", "dy_from_last", "dz_from_last",
    "horizontal_from_last", "distance_from_last", "distance_log1p",
    "known_rows", "known_tvt_span", "known_md_span",
    "known_gr_mean_last300", "known_gr_std_last300", "known_gr_slope_last300",
    "xy_span_known", "z_drop_known",
    "typewell_corr_last300", "typewell_mae_last300",
    "typewell_best_lag_corr", "typewell_best_lag_tvt", "typewell_best_lag_mae",
    "selected_analog_distance", "extrapolation_stress_score",
    "analog_stress_score", "combined_stress_score",
    "state_known_tvt_slope", "state_known_gr_coverage",
    "state_known_gr_last", "state_known_gr_iqr",
    "state_known_xy_span", "state_known_z_drop", "state_known_md_span",
]
```

- [ ] **Step 3: Fit fold-safe pseudo-top regressors**

For each validation fold:

```python
models = {}
for formation in FORMATION_COLS:
    model = XGBRegressor(
        objective="reg:pseudohubererror",
        n_estimators=450,
        learning_rate=0.035,
        max_depth=3,
        min_child_weight=80,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.2,
        reg_lambda=10.0,
        tree_method="hist",
        random_state=SEED,
        n_jobs=4,
        verbosity=0,
    )
    model.fit(X_train, y_train[f"label_dist_to_{formation}"])
    pred[f"pseudo_dist_to_{formation}"] = model.predict(X_valid)
```

Expected: Pseudo predictions for held-out rows are generated only from models trained on other folds.

- [ ] **Step 4: Export uncertainty features**

Compute:

```python
pseudo_dist_cols = [f"pseudo_dist_to_{f}" for f in FORMATION_COLS]
pseudo_abs = np.abs(pred[pseudo_dist_cols])
pred["pseudo_nearest_abs_dist"] = pseudo_abs.min(axis=1)
pred["pseudo_second_abs_dist"] = np.sort(pseudo_abs.to_numpy(), axis=1)[:, 1]
pred["pseudo_top_margin"] = pred["pseudo_second_abs_dist"] - pred["pseudo_nearest_abs_dist"]
pred["pseudo_top_confidence"] = 1.0 / (1.0 + np.exp(-pred["pseudo_top_margin"].clip(-100, 100) / 25.0))
pred["pseudo_near_boundary"] = (pred["pseudo_nearest_abs_dist"] <= 25.0).astype(float)
pred["pseudo_uncertain_boundary"] = ((pred["pseudo_nearest_abs_dist"] <= 50.0) & (pred["pseudo_top_margin"] <= 25.0)).astype(float)
pred["pseudo_mean_abs_dist"] = pseudo_abs.mean(axis=1)
```

Expected: Downstream features contain both predicted distances and uncertainty/margin signals.

### Task 4: Test Downstream Consumers

**Files:**
- Modify: `notebooks/modeling/08_alignment/08_pseudo_top_uncertainty.ipynb`

- [ ] **Step 1: Define feature sets**

Use:

```python
feature_sets = {
    "safe_base": safe_feature_cols,
    "safe_typewell": safe_feature_cols + typewell_feature_cols,
    "safe_pseudo_top": safe_feature_cols + typewell_feature_cols + pseudo_top_feature_cols,
    "safe_pseudo_top_uncertainty": safe_feature_cols + typewell_feature_cols + pseudo_top_feature_cols + pseudo_top_uncertainty_cols,
    "oracle_direct_formations": safe_feature_cols + typewell_feature_cols + direct_formation_feature_cols,
}
```

- [ ] **Step 2: Train XGBoost signed-delta consumers**

Use the current inference-safe XGBoost recipe as the primary consumer:

```python
candidate = XGBRegressor(
    objective="reg:pseudohubererror",
    n_estimators=700,
    learning_rate=0.030,
    max_depth=4,
    min_child_weight=100,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.15,
    reg_lambda=7.0,
    gamma=0.05,
    tree_method="hist",
    eval_metric="rmse",
    random_state=SEED,
    n_jobs=4,
    verbosity=0,
)
```

Primary target: `target_delta = TVT - last_known_tvt_feature`.

- [ ] **Step 3: Train robust Huber sanity consumer**

Use:

```python
HuberRegressor(alpha=0.0001, epsilon=1.35, max_iter=600)
```

Expected: Huber should reveal whether pseudo-top features improve a non-tree model or only interact with trees.

- [ ] **Step 4: Score strict CV and stress CV**

Report:

```python
rmse_mean, rmse_std, mae_mean, r2_mean, last_known_hold_rmse_mean
```

For the best legal strict-CV candidate, rerun under `analog_distance_fold`.

### Task 5: Export Diagnostics And Decide Next Step

**Files:**
- Modify: `notebooks/modeling/08_alignment/08_pseudo_top_uncertainty.ipynb`
- Create artifacts listed in file structure.

- [ ] **Step 1: Export proxy quality diagnostics**

Include:

```python
proxy_quality = pd.DataFrame({
    "formation": formation,
    "rmse": rmse(true_distance, predicted_distance),
    "mae": mean_absolute_error(true_distance, predicted_distance),
    "corr": np.corrcoef(true_distance, predicted_distance)[0, 1],
})
```

Also export nearest-formation accuracy and boundary-confidence slices.

- [ ] **Step 2: Export downstream model summaries**

Write all table artifacts with `model__pseudo_top_uncertainty_*__2026-06-30.csv` names and copy them into `artifacts/logs/runs/2026-06-30__pseudo_top_uncertainty/`.

- [ ] **Step 3: Write findings**

Use these decision rules:

```python
if best_legal_strict_rmse < 13.4178 - 0.05 and best_legal_stress_rmse <= 13.7533:
    recommendation = "package_pseudo_top_xgboost"
elif best_legal_strict_rmse < 14.472:
    recommendation = "iterate_before_package"
else:
    recommendation = "diagnostic_only"
```

- [ ] **Step 4: Write markdown report**

The report must include:

```markdown
# Pseudo-Top Uncertainty Feature Screen
## Result
## Pseudo-Top Quality
## Downstream RMSE
## Leakage Notes
## Decision
## Next Move
```

### Task 6: Update Context And Commit

**Files:**
- Modify: `context/ASSUMPTIONS.md`
- Modify: `context/DECISIONS.md`
- Modify: `context/EXPERIMENTS.md`
- Modify: `context/STATUS.md`
- Modify: `context/CHANGELOG.md`
- Modify: `kaggle/FEATURES.md`

- [ ] **Step 1: Add assumption**

Add:

```markdown
### A59 - Pseudo-top uncertainty [summary after results]
```

Fill with exact RMSE values from executed artifacts.

- [ ] **Step 2: Add decision**

Add:

```markdown
## D028 - [Package/iterate/reject] first pseudo-top uncertainty features
```

Fill with exact recommendation from findings.

- [ ] **Step 3: Verify**

Run:

```powershell
C:\Users\sardo\AppData\Local\Programs\Python\Python313\python.exe -m jupyter nbconvert --to notebook --execute notebooks\modeling\08_alignment\08_pseudo_top_uncertainty.ipynb --output 08_pseudo_top_uncertainty.ipynb --output-dir notebooks\modeling\08_alignment
git diff --check
```

Expected: Notebook executes successfully and `git diff --check` reports no whitespace errors.

- [ ] **Step 4: Commit scoped files**

Stage only:

```powershell
git add notebooks\modeling\08_alignment\08_pseudo_top_uncertainty.ipynb artifacts\tables\model__pseudo_top_uncertainty_*__2026-06-30.csv artifacts\reports\model__pseudo_top_uncertainty__2026-06-30.md artifacts\logs\runs\2026-06-30__pseudo_top_uncertainty context\ASSUMPTIONS.md context\DECISIONS.md context\EXPERIMENTS.md context\STATUS.md context\CHANGELOG.md kaggle\FEATURES.md artifacts\tables\model_comparison.csv
git commit -m "feat(model): screen pseudo top uncertainty" -m "Build fold-safe pseudo-formation/top uncertainty features and test them in downstream RMSE consumers."
```

Do not stage unrelated dirty EDA notebooks.

---

## Self-Review

- Spec coverage: The plan covers leakage audit, pseudo-top label construction, uncertainty feature export, downstream consumers, diagnostics, context docs, and commit.
- Placeholder scan: No placeholder-only implementation tasks remain; result-specific docs are explicitly filled after execution with artifact values.
- Type consistency: The run id, artifact stem, feature column names, and decision thresholds are consistent across tasks.
