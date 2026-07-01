# CNN-Regime Dilated Well Bias Tuning

## Result

Best overall candidate: `logit_switch_floor20__well_bias_h015_s30` with strict-CV RMSE 12.4549, MAE 8.8233, and analog-distance stress RMSE 12.7649.

Oracle row-switch upper bound: `oracle_continuous_weight` with RMSE 10.6114. This is leakage-only and is not deployable, but it shows the CNN and regime anchor have useful complementary errors.

## Model Structure

The audit recomputes the direct pseudo-top regime-mean anchor and a compact local-window CNN specialist. It then tests uniform CNN residual blends, simple stress/confidence/disagreement gates, fold-heldout dilated well-bias tuning gates, and a leakage-only oracle row switch.

## Leakage Notes

Direct formation columns are used only as fold-local labels for pseudo-top proxy training. Regime probabilities, regime means, CNN windows, disagreement features, and gates use held-out pseudo-top predictions and hidden-test-safe features. The oracle row switch uses validation targets and is diagnostic only.

## Decision

Recommendation: `package_cnn_regime_dilated_well_bias_tuning`.

## Next Move

If dilated well-bias tuning gates do not beat the uniform blend, return to CNN seed/architecture stability around the bracketed 0.45 weight.
