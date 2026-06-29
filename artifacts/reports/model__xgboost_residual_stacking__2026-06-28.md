# XGBoost Residual Stacking Report

Run id: `2026-06-28__xgboost_residual_stacking`

## Result

- Best stacker: `ridge_residual_s025`
- Best strict-CV RMSE: `13.2548`
- Base-only strict-CV RMSE: `13.4178`
- Strict best minus base RMSE: `-0.1630`
- Selected stacker analog-distance stress RMSE: `13.5632`
- Stress base-only RMSE: `13.7424`
- Stress selected minus base RMSE: `-0.1792`
- Best minus previous legal XGBoost RMSE: `-0.1630`

## Decision

Recommendation: `do_not_promote_after_public_score`.

Nested residual stacking cleared both strict-CV and analog-distance stress promotion gates, and the reproducibility rerun matched the key metrics.

## Interpretation

The stackers are trained on inner out-of-fold residuals inside each held-out fold, so the result is a leakage-resistant test of whether residual shape remains after the base inference-safe XGBoost model. The analog-distance stress test checks whether the correction survives less similar wells.

## Next Move

Package the Ridge residual stacker for Kaggle as the next external validation probe.
## External Validation

Kaggle package version 2 completed and submission `54157692` scored public LB `14.583`, worse than the current XGBoost strict inference-safe baseline `14.472`. Do not promote this residual stacker as the active best.

