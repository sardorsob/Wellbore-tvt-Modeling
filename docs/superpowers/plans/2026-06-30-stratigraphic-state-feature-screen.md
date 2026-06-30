# Stratigraphic State Feature Screen Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and evaluate hidden-test-safe stratigraphic-state features that approximate formation-relative position from known-zone trajectory, TVT, GR, typewell, and alignment evidence.

**Architecture:** Add small reusable preprocessing utilities for sequence/alignment descriptors, then use a self-contained modeling notebook to screen feature families under strict well-level validation. The notebook must export tables, reports, and run metadata, and the context docs must record what improved, what failed, and whether any candidate is Kaggle-package-worthy.

**Tech Stack:** Python, pandas, NumPy, scikit-learn, XGBoost if available, existing repo artifact conventions, Jupyter notebooks.

---

## External Workflow Notes

- Borrow from Cookiecutter Data Science: keep raw data immutable, put reusable code in `src/`, and write generated evidence to `artifacts/`.
- Borrow from DVC-style experiment discipline without adding DVC: every reportable run needs metrics, provenance, and a reproducible config.
- Borrow from mature ML example repos such as Microsoft Recommenders: keep notebooks readable, but push reusable feature logic into tested utilities when it may be reused.

## File Map

- Create `src/preprocessing/stratigraphic.py`: reusable known-zone feature builders for sequence resampling, normalized cross-correlation, DTW-style distance, alignment candidates, and uncertainty summaries.
- Create `tests/test_stratigraphic_features.py`: focused unit tests on tiny synthetic sequences and well frames.
- Create `notebooks/modeling/08_alignment/07_stratigraphic_state_feature_screen.ipynb`: executable screening notebook with clear markdown sections and string-literal code comments.
- Modify `kaggle/FEATURES.md`: record the feature family if it becomes reportable.
- Modify `context/ASSUMPTIONS.md`: record one or more assumptions based on results.
- Modify `context/DECISIONS.md`: record promotion/rejection decision.
- Modify `context/STATUS.md`: update next-step state after results.
- Modify `context/EXPERIMENTS.md`: add run entry and artifact list.

---

### Task 1: Add Reusable Stratigraphic Feature Utilities

**Files:**
- Create: `src/preprocessing/stratigraphic.py`
- Test: `tests/test_stratigraphic_features.py`

- [ ] **Step 1: Write tests for sequence resampling and safe summary behavior**

Add this test skeleton first:

```python
import numpy as np
import pandas as pd

from src.preprocessing.stratigraphic import (
    alignment_uncertainty,
    best_lag_correlation,
    dtw_distance,
    resample_sequence,
    summarize_known_zone_state,
)


def test_resample_sequence_interpolates_to_fixed_length():
    values = np.array([0.0, 10.0, 20.0])
    result = resample_sequence(values, length=5)
    assert result.shape == (5,)
    assert np.allclose(result[[0, -1]], [0.0, 20.0])
    assert np.isfinite(result).all()


def test_resample_sequence_handles_empty_or_nan_input():
    assert np.allclose(resample_sequence([], length=4), np.zeros(4))
    assert np.allclose(resample_sequence([np.nan, np.nan], length=3), np.zeros(3))


def test_dtw_distance_is_zero_for_identical_sequences():
    seq = np.array([1.0, 2.0, 3.0, 2.0])
    assert dtw_distance(seq, seq) == 0.0


def test_best_lag_correlation_finds_shifted_match():
    target = np.array([0.0, 1.0, 2.0, 3.0, 0.0])
    template = np.array([1.0, 2.0, 3.0])
    result = best_lag_correlation(target, template, max_lag=2)
    assert result["best_lag"] == 1
    assert result["best_corr"] > 0.99


def test_alignment_uncertainty_reports_margin_and_entropy():
    scores = np.array([0.9, 0.7, 0.1])
    result = alignment_uncertainty(scores)
    assert result["alignment_margin"] > 0
    assert result["alignment_entropy"] > 0
    assert result["alignment_score_best"] == 0.9


def test_summarize_known_zone_state_uses_only_known_rows():
    frame = pd.DataFrame(
        {
            "MD": [100.0, 110.0, 120.0, 130.0],
            "X": [0.0, 1.0, 2.0, 3.0],
            "Y": [0.0, 0.0, 0.0, 0.0],
            "Z": [-10.0, -11.0, -12.0, -13.0],
            "GR": [50.0, 55.0, np.nan, 80.0],
            "TVT_input": [5.0, 6.0, np.nan, np.nan],
        }
    )
    result = summarize_known_zone_state(frame)
    assert result["known_row_count"] == 2
    assert result["known_tvt_last"] == 6.0
    assert result["known_gr_coverage"] == 1.0
```

- [ ] **Step 2: Run tests and confirm they fail because the module does not exist**

Run:

```powershell
python -m pytest tests/test_stratigraphic_features.py -q
```

Expected: import failure for `src.preprocessing.stratigraphic`.

- [ ] **Step 3: Implement the utility module**

Implement:

```python
from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd


def _finite_array(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    return arr[np.isfinite(arr)]


def resample_sequence(values: Iterable[float], length: int = 32) -> np.ndarray:
    arr = _finite_array(values)
    if length <= 0:
        raise ValueError("length must be positive")
    if arr.size == 0:
        return np.zeros(length, dtype=float)
    if arr.size == 1:
        return np.full(length, float(arr[0]), dtype=float)
    x_old = np.linspace(0.0, 1.0, arr.size)
    x_new = np.linspace(0.0, 1.0, length)
    return np.interp(x_new, x_old, arr)


def robust_standardize(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr, dtype=float)
    med = float(np.nanmedian(arr[finite]))
    mad = float(np.nanmedian(np.abs(arr[finite] - med)))
    scale = 1.4826 * mad if mad > 1e-9 else float(np.nanstd(arr[finite]))
    if not np.isfinite(scale) or scale <= 1e-9:
        scale = 1.0
    out = (np.where(finite, arr, med) - med) / scale
    return np.clip(out, -8.0, 8.0)


def dtw_distance(a: Iterable[float], b: Iterable[float]) -> float:
    x = robust_standardize(a)
    y = robust_standardize(b)
    if x.size == 0 or y.size == 0:
        return 0.0
    cost = np.full((x.size + 1, y.size + 1), np.inf)
    cost[0, 0] = 0.0
    for i in range(1, x.size + 1):
        for j in range(1, y.size + 1):
            step = abs(x[i - 1] - y[j - 1])
            cost[i, j] = step + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    return float(cost[x.size, y.size] / max(x.size, y.size))


def best_lag_correlation(target: Iterable[float], template: Iterable[float], max_lag: int = 8) -> dict[str, float]:
    x = robust_standardize(target)
    y = robust_standardize(template)
    if x.size == 0 or y.size == 0:
        return {"best_lag": 0.0, "best_corr": 0.0, "second_corr": 0.0}
    rows = []
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            x_part = x[lag : lag + y.size]
            y_part = y[: x_part.size]
        else:
            y_part = y[-lag : -lag + x.size]
            x_part = x[: y_part.size]
        if x_part.size < 3 or y_part.size < 3:
            corr = 0.0
        else:
            corr = float(np.corrcoef(x_part, y_part)[0, 1])
            if not np.isfinite(corr):
                corr = 0.0
        rows.append((lag, corr))
    rows.sort(key=lambda item: item[1], reverse=True)
    second = rows[1][1] if len(rows) > 1 else rows[0][1]
    return {"best_lag": float(rows[0][0]), "best_corr": float(rows[0][1]), "second_corr": float(second)}


def alignment_uncertainty(scores: Iterable[float]) -> dict[str, float]:
    arr = _finite_array(scores)
    if arr.size == 0:
        return {"alignment_score_best": 0.0, "alignment_score_second": 0.0, "alignment_margin": 0.0, "alignment_entropy": 0.0}
    ordered = np.sort(arr)[::-1]
    best = float(ordered[0])
    second = float(ordered[1]) if ordered.size > 1 else 0.0
    shifted = ordered - np.max(ordered)
    probs = np.exp(shifted)
    probs = probs / probs.sum()
    entropy = float(-(probs * np.log(probs + 1e-12)).sum())
    return {
        "alignment_score_best": best,
        "alignment_score_second": second,
        "alignment_margin": best - second,
        "alignment_entropy": entropy,
    }


def summarize_known_zone_state(frame: pd.DataFrame) -> dict[str, float]:
    known = frame[frame["TVT_input"].notna()].copy()
    if known.empty:
        return {
            "known_row_count": 0.0,
            "known_tvt_last": 0.0,
            "known_tvt_slope": 0.0,
            "known_gr_coverage": 0.0,
            "known_md_span": 0.0,
            "known_xy_span": 0.0,
            "known_z_drop": 0.0,
        }
    md_span = float(known["MD"].iloc[-1] - known["MD"].iloc[0]) if "MD" in known else 0.0
    tvt_delta = float(known["TVT_input"].iloc[-1] - known["TVT_input"].iloc[0])
    xy_span = float(np.hypot(known["X"].iloc[-1] - known["X"].iloc[0], known["Y"].iloc[-1] - known["Y"].iloc[0]))
    z_drop = float(known["Z"].iloc[-1] - known["Z"].iloc[0])
    return {
        "known_row_count": float(len(known)),
        "known_tvt_last": float(known["TVT_input"].iloc[-1]),
        "known_tvt_slope": tvt_delta / max(abs(md_span), 1.0),
        "known_gr_coverage": float(known["GR"].notna().mean()) if "GR" in known else 0.0,
        "known_md_span": md_span,
        "known_xy_span": xy_span,
        "known_z_drop": z_drop,
    }
```

- [ ] **Step 4: Run tests and confirm they pass**

Run:

```powershell
python -m pytest tests/test_stratigraphic_features.py -q
```

Expected: all tests pass.

---

### Task 2: Build The Stratigraphic-State Feature Screen Notebook

**Files:**
- Create: `notebooks/modeling/08_alignment/07_stratigraphic_state_feature_screen.ipynb`

- [ ] **Step 1: Create notebook sections**

Notebook sections:

```text
# Stratigraphic State Feature Screen
## 1. Purpose And Promotion Gates
## 2. Imports And Configuration
## 3. Data Loading
## 4. Fold-Safe Stratigraphic Feature Builders
## 5. Feature Family Definitions
## 6. Strict-CV Model Consumers
## 7. Stress-Slice Diagnostics
## 8. Artifact Export
## 9. Findings
```

- [ ] **Step 2: Implement fold-safe feature families**

Use only training wells to build fold references. For validation wells, compute well descriptors from known-zone rows only. Feature families:

```text
safe_base
safe_typewell_existing
strat_known_state
strat_alignment_uncertainty
strat_alignment_plus_boundary
strat_full
oracle_direct_formations
```

`oracle_direct_formations` is diagnostic only and must be excluded from Kaggle packaging.

- [ ] **Step 3: Implement model consumers**

Evaluate at minimum:

```text
xgb_pseudohuber_public_params
huber_linear_delta
huber_plus_hilbert_residual_light
blend_xgb_huber_hilbert
```

If XGBoost is unavailable locally, skip the XGBoost consumer with a clear report row and still run Huber consumers.

- [ ] **Step 4: Export required artifacts**

Write:

```text
artifacts/tables/model__stratigraphic_state_feature_screen_summary__2026-06-30.csv
artifacts/tables/model__stratigraphic_state_feature_screen_fold_metrics__2026-06-30.csv
artifacts/tables/model__stratigraphic_state_feature_screen_stress_slices__2026-06-30.csv
artifacts/tables/model__stratigraphic_state_feature_screen_feature_importance__2026-06-30.csv
artifacts/reports/model__stratigraphic_state_feature_screen__2026-06-30.md
artifacts/logs/runs/2026-06-30__stratigraphic_state_feature_screen/config.json
```

---

### Task 3: Run The Notebook And Interpret Results

**Files:**
- Modify: generated notebook outputs and artifacts

- [ ] **Step 1: Execute the notebook**

Run:

```powershell
jupyter nbconvert --to notebook --execute notebooks\modeling\08_alignment\07_stratigraphic_state_feature_screen.ipynb --output 07_stratigraphic_state_feature_screen.ipynb --output-dir notebooks\modeling\08_alignment
```

Expected: notebook completes and all required artifacts exist.

- [ ] **Step 2: Check promotion gates**

Promote only if the best non-oracle candidate:

```text
beats current safe XGBoost local reference on strict_hybrid_fold,
does not regress on analog_distance_fold,
improves at least two high-stress slices,
has a clear geologic mechanism in the findings report.
```

- [ ] **Step 3: Decide next path**

Possible decisions:

```text
Promote to Kaggle package if feature gains are robust.
Use the features in a Bayesian stratigraphic-regime model if gains are strongest in uncertainty/alignment slices.
Reject the feature family if gains only appear in oracle or unstable slices.
```

---

### Task 4: Update Project Memory And Commit

**Files:**
- Modify: `kaggle/FEATURES.md`
- Modify: `context/ASSUMPTIONS.md`
- Modify: `context/DECISIONS.md`
- Modify: `context/STATUS.md`
- Modify: `context/EXPERIMENTS.md`

- [ ] **Step 1: Record feature result**

Add one row to `kaggle/FEATURES.md`:

```markdown
| 2026-06-30 | Stratigraphic-state screen features | `notebooks/modeling/08_alignment/07_stratigraphic_state_feature_screen.ipynb`, `artifacts/reports/model__stratigraphic_state_feature_screen__2026-06-30.md` | Test hidden-safe formation-state proxies from known-zone state, typewell/log alignment uncertainty, and boundary continuity | Fold-safe known-zone evidence only; direct formation columns remain oracle diagnostics. |
```

- [ ] **Step 2: Record assumption and decision**

Add an assumption that states whether stratigraphic-state features recovered any part of the oracle formation-distance gap. Add a decision that either promotes, revises, or rejects the feature family.

- [ ] **Step 3: Run verification**

Run:

```powershell
python -m pytest tests/test_stratigraphic_features.py -q
git diff --check
```

Expected: tests pass and diff check has no whitespace errors.

- [ ] **Step 4: Commit with no co-author trailer**

Run:

```powershell
git add src/preprocessing/stratigraphic.py tests/test_stratigraphic_features.py notebooks/modeling/08_alignment/07_stratigraphic_state_feature_screen.ipynb artifacts/tables artifacts/reports artifacts/logs/runs/2026-06-30__stratigraphic_state_feature_screen kaggle/FEATURES.md context/ASSUMPTIONS.md context/DECISIONS.md context/STATUS.md context/EXPERIMENTS.md
git commit -m "feat(model): screen stratigraphic state features" -m "Add fold-safe stratigraphic-state feature utilities, notebook screening, artifacts, and context updates."
```

Expected: commit succeeds and contains no `Co-authored-by` trailer.

---

## Self-Review

- Spec coverage: covers reusable features, notebook screening, artifacts, validation gates, and context updates.
- Placeholder scan: no unresolved placeholders remain; all paths and artifact names are concrete.
- Scope check: focused on one feature-screen notebook, not Kaggle packaging or full Bayesian regime modeling.
- Type consistency: utility names in tests match utility names in implementation steps.
