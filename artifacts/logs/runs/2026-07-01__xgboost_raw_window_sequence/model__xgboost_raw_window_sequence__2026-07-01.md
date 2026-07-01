# XGBoost Raw Window Sequence

## Result

Best overall candidate: `xgb_square_global__raw_window_regime_ridge_alpha80__confidence_s100` with strict-CV RMSE 13.1664, MAE 9.4362, and analog-distance stress RMSE 13.4758.

Best confidence-gated regime mean candidate: `xgb_square_global__raw_window_regime_ridge_alpha80__confidence_s100` with RMSE 13.1664.

## Model Structure

Pseudo-top probabilities define soft formation regimes. Raw local GR/typewell window bins are included in the pseudo-top proxy and regime mean. Each regime gets a weighted Ridge MAP mean over legal row/typewell/state/alignment/raw-window features. Predictions blend the regime-specific mean with a global safe/typewell mean, using either fixed blend weights or mixture-confidence weights.

## Leakage Notes

Direct formation columns are used only as fold-local labels for pseudo-top proxy training. Regime probabilities and regime means use held-out pseudo-top predictions and hidden-test-safe features.

## Decision

Recommendation: `diagnostic_only`.

## Next Move

Raw local motif bins worsened both strict and stress metrics. Do not continue flat raw-window tabular features; move to a different architecture or public-transfer calibration.
