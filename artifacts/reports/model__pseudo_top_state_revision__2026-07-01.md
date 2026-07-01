# Pseudo-Top State Revision

## Result

Best overall candidate: `xgb_square_global__regime_ridge_alpha80__s050` with strict-CV RMSE 13.1194, MAE 9.3682, and analog-distance stress RMSE 13.3883.

Best confidence-gated regime mean candidate: `xgb_square_global__regime_ridge_alpha80__confidence_s100` with RMSE 13.1280.

## Model Structure

Pseudo-top predictions are first corrected by conservative fold-safe residual models, then converted into soft formation regimes for the direct regime-mean model.

## Leakage Notes

Direct formation columns are used only as fold-local labels for pseudo-top proxy training. Regime probabilities and regime means use held-out pseudo-top predictions and hidden-test-safe features.

## Decision

Recommendation: `stress_diagnostic_iterate`.

## Next Move

Because strict RMSE worsened while stress improved, next test a lighter 0.25 shrink state revision or blend the revised probabilities only in high-stress slices.
