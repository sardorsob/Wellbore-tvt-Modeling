# Pseudo-Top Transition-Distance Model

## Result

Best overall transition-distance candidate: `reference__xgb_square_global__regime_ridge_alpha80__confidence_s100` with strict-CV RMSE 13.0928, MAE 9.3596, and analog-distance stress RMSE 13.4006.

Best confidence-gated regime mean candidate: `reference__xgb_square_global__regime_ridge_alpha80__confidence_s100` with RMSE 13.0928.

Best non-reference transition-distance variant: `transition_regime__xgb_square_global__regime_ridge_alpha80__confidence_s100` with RMSE 13.0950.

## Model Structure

Pseudo-top probabilities define soft formation regimes. Each regime gets a weighted Ridge MAP mean over legal row/typewell/state features. Predictions blend the regime-specific mean with a global safe/typewell mean, using either fixed blend weights or mixture-confidence weights.

## Leakage Notes

Direct formation columns are used only as fold-local labels for pseudo-top proxy training. Regime probabilities and regime means use held-out pseudo-top predictions and hidden-test-safe features.

## Decision

Recommendation: `diagnostic_only`.

## Next Move

Because transition-distance features do not beat the direct regime-mean anchor, stop this representation as a tabular add-on and move to a different target or sequence architecture.
