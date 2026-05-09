# Interfaces — I/O Contracts

## Data loader outputs

### `src/io/loaders.load_horizontal_well(path) → pd.DataFrame`
- Index: original CSV row index (preserved)
- Columns: all columns from the CSV
- Row with `TVT_input == NaN`: evaluation zone rows to predict

### `src/io/loaders.load_typewell(path) → pd.DataFrame`
- Columns: `TVT`, `GR`, `Geology`
- Index: default integer

### `src/io/loaders.discover_wells(directory) → list[str]`
- Returns sorted list of well names (8-char hex strings)
- Discovers by globbing `*__horizontal_well.csv`

### `src/io/loaders.load_all_wells(directory) → dict[str, dict]`
- Returns `{wellname: {"horizontal": df, "typewell": df}}`

---

## Feature matrix contract

### `src/preprocessing/features.build_features(df_horiz, df_typewell) → pd.DataFrame`
- Input: one horizontal well DataFrame + matching typewell DataFrame
- Output: feature matrix aligned to the same row index
- Columns guaranteed to exist in both train and test:
  - `MD`, `X`, `Y`, `Z`, `GR`, `TVT_input`
  - `row_idx` (original index)
  - `tvt_input_lag_*`, `tvt_input_rolling_*` (derived)
  - `typewell_gr_at_tvt_input`, `typewell_formation_*` (typewell join)
- Never includes `TVT` (target) — that must be added separately at training time

---

## Model interface

All model classes in `src/modeling/` must implement:

```python
class BaseModel:
    def fit(self, X: np.ndarray, y: np.ndarray, config: dict) -> None: ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...
    def save(self, path: str) -> None: ...
    def load(self, path: str) -> None: ...
```

Bayesian models additionally expose:
```python
    def predict_with_uncertainty(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (mean, std) arrays."""
```

---

## Submission contract

```python
# Final output shape
submission: pd.DataFrame
# Required:
assert list(submission.columns) == ["id", "tvt"]
assert len(submission) == len(sample_submission)
assert submission["id"].equals(sample_submission["id"])
assert submission["tvt"].notna().all()
assert np.isfinite(submission["tvt"].values).all()
```

---

## Script CLI contracts

### `scripts/train.py`
```
python scripts/train.py --config configs/<name>.yaml [--run-id <id>]
```
- Writes run bundle to `artifacts/logs/runs/<run_id>/`
- Saves model to `artifacts/models/<model>__<run_id>.pkl`

### `scripts/predict.py`
```
python scripts/predict.py --config configs/<name>.yaml --output submission.csv
```
- Loads model from `artifacts/models/`
- Writes submission CSV

### `scripts/eval.py`
```
python scripts/eval.py --config configs/<name>.yaml [--fold <n>]
```
- Runs cross-validation
- Writes metrics to run bundle `metrics.json`

### `scripts/download_data.py`
```
python scripts/download_data.py [--dest data/raw]
```
- Uses Kaggle API; requires `~/.kaggle/kaggle.json`
