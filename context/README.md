# ROGII Wellbore Geology Prediction — Project Documentation

## What this is

A Kaggle competition (May–August 2026) to predict **True Vertical Thickness (TVT)** along horizontal wellbores using well trajectory data, gamma ray logs, and vertical reference wells (typewells). Scored by RMSE.

## How to run

### Environment setup
```bash
pip install -r requirements.txt
```

### Download data
```bash
# Requires ~/.kaggle/kaggle.json with valid credentials
python scripts/download_data.py
# Data lands in data/raw/
```

### EDA
```bash
jupyter lab notebooks/eda/01_phase0_phase1_data_and_schema.ipynb
```

### Train a model
```bash
python scripts/train.py --config configs/xgboost.yaml
# Run bundle written to artifacts/logs/runs/<run_id>/
```

### Generate submission
```bash
python scripts/predict.py --config configs/xgboost.yaml --output submission.csv
```

### Run tests
```bash
pytest tests/ -v
```

## Key paths

| What | Where |
|---|---|
| Repo structure & results | `context/structure.md` |
| Project decisions | `context/DECISIONS.md` |
| Modeling assumptions | `context/ASSUMPTIONS.md` |
| Dataset documentation | `context/DATASETS.md` |
| Domain glossary | `context/GLOSSARY.md` |
| Current status | `context/STATUS.md` |
| Known risks | `context/RISKS.md` |
| Operational rules | `context/development_rules.md` |

## Kaggle submission workflow

1. Develop + validate locally
2. Push code to GitHub
3. Create/update Kaggle Notebook
4. Attach competition dataset (`/kaggle/input/rogii-wellbore-geology-prediction/`)
5. Run notebook → writes `/kaggle/working/submission.csv`
6. Commit notebook and submit output

## Environment detection

All scripts automatically detect Kaggle vs local:
```python
from src.utils.paths import get_base_dir
base = get_base_dir()  # returns Kaggle or local path
```
