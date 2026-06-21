# Kaggle Competition

## Competition

- Name: ROGII Wellbore Geology Prediction
- Slug: `rogii-wellbore-geology-prediction`
- Timeline: May 5, 2026 to August 5, 2026
- Entry deadline: July 29, 2026
- Team merger deadline: July 29, 2026
- Final submission deadline: August 5, 2026

## Objective

Predict missing `TVT` values along horizontal wellbores using trajectory data, gamma ray logs, known `TVT_input` context, and typewell references.

## Metric

Primary metric: RMSE. Lower is better.

## Submission Shape

Submission file must match `data/raw/sample_submission.csv`:

- columns: `id`, `tvt`
- `id` format: `{WELLNAME}_{row_index}`
- rows must remain in sample submission order
- all `tvt` values must be finite

## Local Source Of Truth

- Project brief: `context/PROJECT_BRIEF.md`
- Dataset details: `context/DATASETS.md`
- Submission contract: `context/INTERFACES.md`
- Status and next steps: `context/STATUS.md`
- Artifact and result index: `context/structure.md`

