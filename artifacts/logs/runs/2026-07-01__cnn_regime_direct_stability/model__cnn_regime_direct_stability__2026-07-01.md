# CNN-Regime Direct Stability

## Result

Best overall candidate: `logit_switch_floor20` with strict-CV RMSE 12.5179, MAE 8.8453, and analog-distance stress RMSE 12.9574.

Oracle row-switch upper bound: `oracle_continuous_weight` with RMSE 10.9067. This is leakage-only and is not deployable, but it shows the CNN and regime anchor have useful complementary errors.

## Model Structure

The audit recomputes the direct pseudo-top regime-mean anchor and a compact local-window CNN specialist. It then tests uniform CNN residual blends, simple stress/confidence/disagreement gates, fold-heldout direct-stability gates, and a leakage-only oracle row switch.

## Leakage Notes

Direct formation columns are used only as fold-local labels for pseudo-top proxy training. Regime probabilities, regime means, CNN windows, disagreement features, and gates use held-out pseudo-top predictions and hidden-test-safe features. The oracle row switch uses validation targets and is diagnostic only.

## Decision

Recommendation: `strict_improves_stress_regresses`.

## Next Move

The stronger three-seed direct-delta CNN is a real strict-CV improvement. Next, tune the stress-aware selection around `logit_switch_floor20`, `xgb_weight_soft_s075`, and `xgb_switch_p100`, or package the strongest candidate for a Kaggle public-transfer probe.
