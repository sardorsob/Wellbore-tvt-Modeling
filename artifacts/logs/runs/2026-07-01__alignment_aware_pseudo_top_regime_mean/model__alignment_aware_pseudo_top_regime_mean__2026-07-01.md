# Alignment-Aware Pseudo-Top Regime Mean

## Result

Best overall candidate: `xgb_square_global__regime_ridge_alpha80__confidence_s100` with strict-CV RMSE 13.0838, MAE 9.3398, and analog-distance stress RMSE 13.4303.

Best confidence-gated regime mean candidate: `xgb_square_global__regime_ridge_alpha80__confidence_s100` with RMSE 13.0838.

## Model Structure

Pseudo-top probabilities define soft formation regimes. Local GR/typewell alignment features are included in the pseudo-top proxy and regime mean. Each regime gets a weighted Ridge MAP mean over legal row/typewell/state/alignment features. Predictions blend the regime-specific mean with a global safe/typewell mean, using either fixed blend weights or mixture-confidence weights.

## Leakage Notes

Direct formation columns are used only as fold-local labels for pseudo-top proxy training. Regime probabilities and regime means use held-out pseudo-top predictions and hidden-test-safe features.

## Decision

Recommendation: `iterate_before_package`.

## Next Move

If alignment-aware pseudo-top features do not improve both strict and stress metrics, move to a different pseudo-top model class or Bayesian partial pooling rather than more alignment descriptors.
