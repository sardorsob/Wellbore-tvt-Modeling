# CNN-Regime Neural Blend Gate

## Result

Best overall candidate: `logit_switch_floor20__well_bias_h015_s30` with strict-CV RMSE 12.4549, MAE 8.8233, and analog-distance stress RMSE 12.7649.

Oracle row-switch upper bound: `oracle_continuous_weight` with RMSE 10.6114. This is leakage-only and is not deployable, but it shows the CNN and regime anchor have useful complementary errors.

## Model Structure

The audit recomputes the direct pseudo-top regime-mean anchor and a compact local-window CNN specialist. It then tests uniform CNN residual blends, tree/logistic gates, compact neural blend gates trained directly on blended TVT loss, conservative well-bias corrections, and a leakage-only oracle row switch.

## Leakage Notes

Direct formation columns are used only as fold-local labels for pseudo-top proxy training. Regime probabilities, regime means, CNN windows, disagreement features, and gates use held-out pseudo-top predictions and hidden-test-safe features. The oracle row switch uses validation targets and is diagnostic only.

## Decision

Recommendation: `stress_improves_strict_regresses`.

## Next Move

The neural gates did not beat the current tuned logit-switch well-bias candidate on strict RMSE. Move to representation changes rather than narrower gate tuning.
