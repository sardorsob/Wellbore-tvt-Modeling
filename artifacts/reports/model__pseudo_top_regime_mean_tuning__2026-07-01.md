# Pseudo-Top Regime Mean Tuning

## Result

Best overall candidate: `xgb_square_global__regime_ridge_alpha40__confidence_s100` with strict-CV RMSE 13.0910, MAE 9.3597, and analog-distance stress RMSE 13.4065.

Best confidence-gated regime mean candidate: `xgb_square_global__regime_ridge_alpha40__confidence_s100` with RMSE 13.0910.

## Model Structure

Pseudo-top probabilities define soft formation regimes. Each regime gets a weighted Ridge MAP mean over legal row/typewell/state features. Predictions blend the regime-specific mean with a global safe/typewell mean, using either fixed blend weights or mixture-confidence weights.

## Leakage Notes

Direct formation columns are used only as fold-local labels for pseudo-top proxy training. Regime probabilities and regime means use held-out pseudo-top predictions and hidden-test-safe features.

## Decision

Recommendation: `iterate_before_package`.

## Next Move

If this does not beat the first regime-mean baseline by a useful margin, move to boosted stacking or pseudo-top inference tuning rather than widening the same grid again.
