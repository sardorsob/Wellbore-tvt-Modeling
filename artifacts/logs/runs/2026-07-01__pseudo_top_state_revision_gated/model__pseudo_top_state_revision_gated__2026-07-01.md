# Gated Pseudo-Top State Revision

## Result

Best overall candidate: `base__xgb_square_global__regime_ridge_alpha80__confidence_s100` with strict-CV RMSE 13.0928, MAE 9.3596, and analog-distance stress RMSE 13.4006.

Best confidence-gated regime mean candidate: `base__xgb_square_global__regime_ridge_alpha80__confidence_s100` with RMSE 13.0928.

## Model Structure

Pseudo-top predictions are corrected only through light, stress-gated, uncertainty-gated, or hybrid fold-safe residual shifts before forming variant-specific regime probabilities.

## Leakage Notes

Direct formation columns are used only as fold-local labels for pseudo-top proxy training. Regime probabilities and regime means use held-out pseudo-top predictions and hidden-test-safe features.

## Decision

Recommendation: `diagnostic_only`.

## Next Move

Best stress variant `hybrid_s75__xgb_square_global__regime_ridge_alpha220__s050` reached stress RMSE 13.3710 but strict RMSE 13.1309. Since the best strict candidate is the unchanged base regime mean, stop state-correction schedules and switch to a different state model or public-transfer calibration.
