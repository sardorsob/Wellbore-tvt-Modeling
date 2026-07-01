# Bayesian Regime Partial-Pooling Model

## Result

Best overall candidate: `xgb_square_global__reference_regime_ridge_alpha80__reference_confidence_s100` with strict-CV RMSE 13.0953, MAE 9.3669, and analog-distance stress RMSE 13.4038.

Best residual partial-pooling candidate: `xgb_square_global__pp_resid_ridge_alpha20__resid_s025` with RMSE 13.4272.

Best direct partial-pooling candidate: `xgb_square_global__pp_direct_ridge_alpha220__direct_s025` with RMSE 13.2049.

## Model Structure

Pseudo-top probabilities define soft formation regimes. Shared Ridge residual and direct mean models use probability-by-feature interactions, with the Ridge penalty acting like a Gaussian shrinkage prior toward the global XGBoost mean. Confidence and margin schedules control how strongly the shared regime layer can move each row.

## Leakage Notes

Direct formation columns are used only as fold-local labels for pseudo-top proxy training. The partial-pooling predictors use held-out pseudo-top probabilities and hidden-test-safe row/typewell/state features.

## Decision

Recommendation: `diagnostic_iterate_before_package`.

## Next Move

If partial pooling does not reduce both strict and stress RMSE, move toward data-state revisions rather than adding more shrinkage: pseudo-top labels or analog grouping likely need more signal before Bayesian hierarchy helps.
