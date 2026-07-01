# Pseudo-Top Interaction XGBoost

## Result

Best overall candidate: `xgb_square_regularized__safe_typewell` with strict-CV RMSE 13.4745, MAE 9.5547, and analog-distance stress RMSE 13.8284.

Best same-notebook safe/typewell baseline: `xgb_square_regularized__safe_typewell` with RMSE 13.4745.

Best pseudo-top interaction candidate: `xgb_pseudohuber_reference__pseudo_interactions` with RMSE 13.4815.

## Interaction Design

Interactions include pseudo-top margin by progress/distance, entropy by analog distance/progress, nearest-top offset by distance, confidence by typewell error, pseudo-top spread by stress, and simple early/late/boundary regime flags.

## Leakage Notes

Direct formation columns are labels only for fold-safe pseudo-top proxy training. Interaction features are derived only from held-out pseudo-top predictions and hidden-test-safe row/typewell/state columns.

## Decision

Recommendation: `iterate_before_package`.

## Next Move

If interactions do not improve strict CV, the next route should be Bayesian/mixture priors over stratigraphic regimes rather than more flat feature engineering.
