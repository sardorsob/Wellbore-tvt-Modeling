# CNN-Regime Calibrated Gate

## Result

Best overall candidate: `xgb_switch_p100` with strict-CV RMSE 12.6364, MAE 8.9642, and analog-distance stress RMSE 13.0089.

Oracle row-switch upper bound: `oracle_continuous_weight` with RMSE 10.8172. This is leakage-only and is not deployable, but it shows the CNN and regime anchor have useful complementary errors.

## Model Structure

The audit recomputes the direct pseudo-top regime-mean anchor and a compact local-window CNN specialist. It then tests uniform CNN residual blends, simple stress/confidence/disagreement gates, fold-heldout calibrated gates, and a leakage-only oracle row switch.

## Leakage Notes

Direct formation columns are used only as fold-local labels for pseudo-top proxy training. Regime probabilities, regime means, CNN windows, disagreement features, and gates use held-out pseudo-top predictions and hidden-test-safe features. The oracle row switch uses validation targets and is diagnostic only.

## Decision

Recommendation: `package_cnn_regime_calibrated_gate`.

## Next Move

If calibrated gates do not beat the uniform blend, return to CNN seed/architecture stability around the bracketed 0.45 weight.
