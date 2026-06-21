# Datasets

## Competition data

**Source:** Kaggle — `rogii-wellbore-geology-prediction`
**Download:** `python scripts/download_data.py`
**Local path:** `data/raw/`
**Kaggle path:** `/kaggle/input/rogii-wellbore-geology-prediction/`
**License:** Kaggle competition rules apply; not redistributable

### File inventory

```
data/raw/
├── train/
│   ├── {WELLNAME}__horizontal_well.csv   # trajectory + logs + TVT target
│   ├── {WELLNAME}__typewell.csv          # vertical reference log
│   └── {WELLNAME}.png                   # geological cross-section image
├── test/
│   ├── {WELLNAME}__horizontal_well.csv   # trajectory + logs (TVT masked in eval zone)
│   └── {WELLNAME}__typewell.csv          # vertical reference log
└── sample_submission.csv                 # id → tvt placeholder
```

### Well name format
8-character hex hash, e.g. `015fe0d2`. Never hardcode well names — discover dynamically.

---

## Horizontal well CSV schema

**Observed locally:** Horizontal files use the **filename** `{WELLNAME}__horizontal_well.csv` for the well id; the CSV body may **not** include a `WELLNAME` column (confirmed on at least one train file in EDA). Treat well identity as from discovery/`load_all_wells` paths, not from a required column.

| Column | Type | Description |
|---|---|---|
| `WELLNAME` | str | Well identifier *(may be absent in CSV; use filename if needed)* |
| `MD` | float | Measured Depth (ft) |
| `X` | float | Easting (ft) |
| `Y` | float | Northing (ft) |
| `Z` | float | True Vertical Depth (ft) |
| `GR` | float | Gamma Ray log (API units) |
| `TVT_input` | float | Known TVT (NaN in evaluation zone) |
| `TVT` | float | **Target** — only in train files |
| `ANCC` | float | Formation top depth — **train only** |
| `ASTNU` | float | Formation top depth — **train only** |
| `ASTNL` | float | Formation top depth — **train only** |
| `EGFDU` | float | Formation top depth — **train only** |
| `EGFDL` | float | Formation top depth — **train only** |
| `BUDA` | float | Formation top depth — **train only** |

**Important:** `ANCC`, `ASTNU`, `ASTNL`, `EGFDU`, `EGFDL`, `BUDA` are absent in test — do not use as features unless confirmed present in test files.

---

## Typewell CSV schema

| Column | Type | Description |
|---|---|---|
| `TVT` | float | Vertical depth index (ft) |
| `GR` | float | Gamma Ray (API units) |
| `Geology` | str | Formation label (e.g., `EGFDL`, `BUDA`) |

---

## sample_submission.csv schema

| Column | Type | Description |
|---|---|---|
| `id` | str | `{WELLNAME}_{row_index}` |
| `tvt` | float | Predicted TVT (placeholder = 0.0) |

### Local visible test caveat

Phase 2 EDA found that the three visible local test wells also appear in the train split: `000d7d20`, `00bbac68`, and `00e12e8b`. Treat these files as code-competition examples for schema, row-id mapping, and submission-shape validation. Do not use their exact descriptor values as evidence about the hidden scoring distribution.

---

## Data governance notes

- `data/raw/` is immutable and gitignored. Never modify in place.
- `data/interim/` holds cached feature matrices — gitignored.
- `data/processed/` holds final merged feature datasets — gitignored (large).
- Document schema changes in this file and in `context/ASSUMPTIONS.md`.
