# CURSOR_RULES.md
# Single operational contract for AI-assisted development in this repository.
# Full rules: context/development_rules.md (copied from rules_templet.md)

This repository follows a strict **artifact-first + documented-results** workflow.

The single source of truth for project structure and results is: `context/structure.md`.

---

## Required workflow for every notebook change

1. Save figures → `artifacts/figures/<stage>__<desc>__<YYYY-MM-DD>.png`
2. Save derived datasets → `artifacts/data/<stage>__<name>__<YYYY-MM-DD>.csv`
3. Update `context/structure.md` with findings + artifact references
4. Notebook must run top-to-bottom without manual state
5. No noisy prints; no excessive comments

If `context/structure.md` is NOT updated, the change is NOT done.

---

## Project layout

```
configs/             yaml configs for all experiments
context/             all project docs + memory (structure.md is the oracle)
data/
  raw/               immutable source data (gitignored)
  interim/           cached transforms (gitignored)
  processed/         reusable derived datasets (selective)
  external/          reference files
notebooks/
  eda/               01_, 02_, ... exploratory
  modeling/          03_, 04_, ... training & evaluation
  reporting/         08_, ... stakeholder summaries
src/                 reusable Python package
  io/                data loaders
  preprocessing/     feature engineering, transforms
  modeling/          model definitions (baseline, classical, bayesian, BNN, ensemble)
  evaluation/        metrics, cross-validation
  viz/               plotting helpers
  utils/             paths, seeds, logging, run bundles
scripts/             CLI entrypoints (train.py, predict.py, eval.py, download_data.py)
tests/               unit + smoke tests
artifacts/
  figures/           exported plots
  tables/            exported tables
  models/            serialized models
  reports/           auto-generated markdown/HTML
  logs/
    runs/            per-run metadata bundles
    prompts/         LLM prompt/response snapshots
    provenance/      file hashes / manifests
```

---

## Artifact naming convention

```
<stage>__<short_description>__<YYYY-MM-DD>.<ext>
```

Examples:
- `eda__gr_distribution_by_formation__2026-05-08.png`
- `eval__rmse_comparison__2026-05-08.csv`
- `model__xgboost_final__2026-05-08.pkl`

---

## Run bundle convention

Every reportable run writes to `artifacts/logs/runs/<run_id>/`:
- `meta.json` — run_id, timestamp, git hash
- `config.yaml` — exact config used
- `seeds.json` — RNG seeds
- `inputs.json` — dataset paths + sizes
- `outputs.json` — produced artifact paths
- `metrics.json` — key evaluation metrics
- `notes.md` — short narrative (what / why / results / next steps)

---

## Code standards

- Small functions with docstrings (purpose + inputs/outputs)
- No magic numbers → `configs/*.yaml`
- No absolute paths → use `src/utils/paths.py` helpers
- Set + log random seeds via `src/utils/seeds.py`
- Prefer returning objects over printing

---

## Modeling approach priority (this project)

1. Interpolation baseline (TVT_input forward-fill)
2. Classical ML: XGBoost / LightGBM
3. JAX Bayesian: Gaussian Process via NumpPyro
4. Bayesian Neural Network: Flax + NumpPyro SVI/HMC
5. Standard NN: Flax deterministic
6. Ensemble: weighted average of best approaches

Evaluation: RMSE on held-out evaluation zone (simulate by masking TVT_input).
