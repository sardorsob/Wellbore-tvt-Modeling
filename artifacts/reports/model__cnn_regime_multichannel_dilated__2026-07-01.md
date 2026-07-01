# CNN-Regime Multichannel Dilated CNN

## Result

Best overall candidate: `logit_switch_floor20__well_bias_h015_s30` with strict-CV RMSE 12.4984, MAE 8.8420, and analog-distance stress RMSE 12.8224.

Oracle row-switch upper bound: `oracle_continuous_weight` with RMSE 10.7443. This is leakage-only and is not deployable, but it shows the CNN and regime anchor have useful complementary errors.

## Model Structure

The audit recomputes the direct pseudo-top regime-mean anchor and a multichannel dilated local-window CNN specialist. It then tests uniform CNN residual blends, simple stress/confidence/disagreement gates, conservative well-bias corrections, and a leakage-only oracle row switch.

## Leakage Notes

Direct formation columns are used only as fold-local labels for pseudo-top proxy training. Regime probabilities, regime means, CNN windows, disagreement features, and gates use held-out pseudo-top predictions and hidden-test-safe features. The oracle row switch uses validation targets and is diagnostic only.

## Decision

Recommendation: `diagnostic_only_current_best_worse`.

## Next Move

Derivative channels worsened the oracle and deployable candidate, so the next representation work should use a different state or sequence target rather than simple local derivatives.
