# Pseudo-Top Probability Scale Tuning

## Result

Best overall candidate: `xgb_square_global__regime_ridge_alpha80__scale50__confidence_s100` with strict-CV RMSE 13.0928, MAE 9.3596, and analog-distance stress RMSE 13.4006.

Original scale-50 reference in this notebook: `xgb_square_global__regime_ridge_alpha80__scale50__confidence_s100` with strict-CV RMSE 13.0928 and stress RMSE 13.4006.

Best stress candidate: `xgb_square_global__regime_ridge_alpha80__scale50__confidence_s100` with stress RMSE 13.4006.

## Model Structure

Pseudo-top distances are converted into formation probabilities with candidate softmax scales. Each scale feeds the same direct regime-mean recipe: global XGBoost fallback, weighted Ridge regime means, and confidence-weighted blending.

## Leakage Notes

Direct formation columns are used only as fold-local labels for pseudo-top proxy training. Probability-scale features are derived from held-out pseudo-top predictions and hidden-test-safe row/typewell/state features.

## Decision

Recommendation: `diagnostic_only`.

## Next Move

If probability-scale tuning does not improve the regime-mean baseline, move to a structurally different pseudo-top inference model or Bayesian partial pooling.
