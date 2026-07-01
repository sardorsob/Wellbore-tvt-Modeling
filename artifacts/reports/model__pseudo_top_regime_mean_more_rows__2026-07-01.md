# Pseudo-Top Regime Mean With More Target Rows

## Result

Best overall candidate: `xgb_pseudohuber_global__regime_ridge_alpha80__confidence_s100` with strict-CV RMSE 13.1185, MAE 9.3707, and analog-distance stress RMSE 13.4745.

Best confidence-gated regime mean candidate: `xgb_pseudohuber_global__regime_ridge_alpha80__confidence_s100` with RMSE 13.1185.

## Model Structure

Pseudo-top probabilities define soft formation regimes. Each regime gets a weighted Ridge MAP mean over legal row/typewell/state features. Predictions blend the regime-specific mean with a global safe/typewell mean, using either fixed blend weights or mixture-confidence weights.

## Leakage Notes

Direct formation columns are used only as fold-local labels for pseudo-top proxy training. Regime probabilities and regime means use held-out pseudo-top predictions and hidden-test-safe features.

## Decision

Recommendation: `iterate_before_package`.

## Next Move

If denser target-row sampling improves both strict and stress metrics, consider rebuilding the Kaggle package with the higher row cap; otherwise keep the lighter original package.
