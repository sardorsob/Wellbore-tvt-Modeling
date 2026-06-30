# Wellbore TVT Modeling

Modeling workspace for the Kaggle **ROGII Wellbore Geology Prediction** competition.

The goal is to predict missing **TVT** values, or True Vertical Thickness, along horizontal wellbores. Each well contains trajectory measurements, gamma ray logs, partially observed `TVT_input`, and a paired vertical typewell reference. The competition scores predictions with **RMSE**, so lower error is better.

This repository is organized as an experiment lab: notebooks explore modeling ideas, reusable code lives under `src/`, generated reports and tables are written under `artifacts/`, and Kaggle-ready submission notebooks live under `kaggle/`.

## Problem Context

Horizontal wells are drilled through layered geologic formations. In the competition data, `TVT_input` is known for part of each horizontal well and masked in the evaluation zone. The model must infer the missing `TVT` values from:

- well trajectory: `MD`, `X`, `Y`, `Z`
- gamma ray log: `GR`
- known `TVT_input` context before the masked zone
- vertical typewell logs with `TVT`, `GR`, and geology labels

The hard part is not only fitting a smooth curve. The target is tied to geologic position, formation boundaries, log motifs, and transfer across wells. Many notebooks therefore test validation design, alignment features, robust losses, tree models, GAM-like splines, Bayesian models, Hilbert-space Gaussian process residuals, and Kaggle-safe submission packaging.

## Data

Competition data is expected under `data/raw/` locally or under `/kaggle/input/rogii-wellbore-geology-prediction/` inside Kaggle notebooks.

Typical layout:

```text
data/raw/
  train/
    {WELLNAME}__horizontal_well.csv
    {WELLNAME}__typewell.csv
    {WELLNAME}.png
  test/
    {WELLNAME}__horizontal_well.csv
    {WELLNAME}__typewell.csv
  sample_submission.csv
```

Important data notes:

- `TVT` is the target and is available in training horizontal wells.
- `TVT_input` is a partially observed version of `TVT`; missing values mark the prediction region.
- Train-only formation columns such as `ANCC`, `ASTNU`, `ASTNL`, `EGFDU`, `EGFDL`, and `BUDA` are useful for diagnostics but must not be used as submission-time features unless they are also available at inference.
- Well names should be discovered dynamically from files and `sample_submission.csv`; do not hardcode visible example wells.
- Raw competition data is not committed. `data/raw/` is intentionally gitignored except for documentation placeholders.

## Repository Map

```text
configs/        Experiment and model configuration files
context/        Project memory: assumptions, decisions, risks, status, glossary
data/           Local data mount points; raw/interim/processed data are mostly ignored
docs/           Planning notes and workflow documents
kaggle/         Self-contained Kaggle submission notebooks and submission logs
notebooks/      EDA, modeling, validation, and reporting notebooks
scripts/        Small CLI utilities such as data download helpers
src/            Reusable Python package for loading, preprocessing, modeling, evaluation
tests/          Unit and smoke tests for reusable code
artifacts/      Generated reports, tables, logs, figures, and local submission CSVs
```

The most useful orientation files are:

- `context/PROJECT_BRIEF.md` for competition framing
- `context/DATASETS.md` for schema and data governance
- `context/GLOSSARY.md` for domain terms
- `context/ASSUMPTIONS.md` for modeling assumptions and lessons learned
- `context/DECISIONS.md` for durable modeling decisions
- `context/STATUS.md` for current experiment state
- `kaggle/CV_STRATEGY.md` for validation rules
- `kaggle/FEATURES.md` for feature-family notes
- `kaggle/SUBMISSIONS.md` for Kaggle submission history

## Setup

Create an environment with Python 3.10+ and install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Download the competition data with Kaggle credentials configured outside the repo:

```powershell
python scripts\download_data.py
```

Never commit `kaggle.json`, `.env`, API keys, or local credential files. Keep credentials in the standard Kaggle config location or local environment variables outside Git.

## Working With Experiments

Most experimentation happens in notebooks:

- `notebooks/eda/` investigates schema, geologic context, feature signal, validation behavior, and leaderboard mismatch.
- `notebooks/modeling/00_baseline/` contains naive baselines.
- `notebooks/modeling/01_linear/` contains OLS, Ridge, Lasso, robust linear, GLM, and related baselines.
- `notebooks/modeling/02_trees/` contains Random Forest, Extra Trees, LightGBM, XGBoost, and related tree experiments.
- `notebooks/modeling/04_gam/` contains additive spline / GAM-style experiments.
- `notebooks/modeling/06_bayesian/` contains Bayesian linear, posterior diagnostics, and Hilbert-space GP residual work.
- `notebooks/modeling/07_hierarchical/` contains hierarchical/residual-bias screens.
- `notebooks/modeling/08_alignment/` contains typewell, analog, and stratigraphic alignment experiments.
- `notebooks/modeling/09_validation/` contains validation and leaderboard-transfer diagnostics.

For a reportable experiment:

1. Keep feature construction fold-safe.
2. Fit preprocessing only on the training fold.
3. Report grouped validation metrics, not only global RMSE.
4. Export tables, reports, and run metadata under `artifacts/`.
5. Record durable lessons in `context/ASSUMPTIONS.md`, `context/DECISIONS.md`, or the relevant Kaggle docs.
6. Package Kaggle submissions as self-contained notebooks that can run without internet and write `/kaggle/working/submission.csv`.

## Kaggle Submission Contract

This is a notebook-based Kaggle code competition. A valid submission notebook must:

- attach the competition dataset as input
- run with internet disabled
- discover train/test wells dynamically
- write exactly `/kaggle/working/submission.csv`
- preserve the row order from `sample_submission.csv`
- output finite predictions in a two-column file: `id`, `tvt`

Submission notebooks are kept in `kaggle/*_submission/` with `kernel-metadata.json` where needed.

## Validation Philosophy

Local validation masks target regions in training wells to simulate hidden test prediction. Because public leaderboard results can disagree with local CV, models should be checked against multiple views:

- grouped well-level folds
- endpoint/last-known baselines
- analog-distance and extrapolation stress tests
- feature leakage checks for train-only formation columns
- alignment and stratigraphic-transfer diagnostics

The active validation notes live in `kaggle/CV_STRATEGY.md`.

## Public Repo Hygiene

The repo is intended to be readable and reproducible without local secrets.

- Do not commit credentials, `.env`, `kaggle.json`, private keys, or API tokens.
- Keep raw Kaggle data and large generated caches out of Git.
- Keep generated artifacts only when they are useful for reviewing experiments.
- Do not add assistant or AI co-author trailers to commits.
- Use clear commit messages tied to a notebook, model, documentation update, or Kaggle submission package.

## License And Data Terms

Code and notes in this repository are separate from Kaggle competition data. The competition dataset and submission rules are governed by the Kaggle competition terms. If you fork or reuse this repository, download the data from Kaggle and follow the competition rules for sharing notebooks, outputs, and external data.
