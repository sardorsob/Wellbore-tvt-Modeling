# CNN-Regime Dilated Direct

## Result

Best overall candidate: `logit_switch_floor20` with strict-CV RMSE 12.4888, MAE 8.8548, and analog-distance stress RMSE 12.8465.

Oracle row-switch upper bound: `oracle_continuous_weight` with RMSE 10.6114. This is leakage-only and is not deployable, but it shows the CNN and regime anchor have useful complementary errors.

## Model Structure

The audit recomputes the direct pseudo-top regime-mean anchor and a compact local-window CNN specialist. It then tests uniform CNN residual blends, simple stress/confidence/disagreement gates, fold-heldout dilated-direct gates, and a leakage-only oracle row switch.

## Leakage Notes

Direct formation columns are used only as fold-local labels for pseudo-top proxy training. Regime probabilities, regime means, CNN windows, disagreement features, and gates use held-out pseudo-top predictions and hidden-test-safe features. The oracle row switch uses validation targets and is diagnostic only.

## Decision

Recommendation: `package_cnn_regime_dilated_direct`.

## Next Move

The dilated direct-delta CNN improves both strict and analog-distance stress RMSE. Next, package `logit_switch_floor20` for Kaggle or run a focused stress-aware tuning pass around the dilated candidates.
