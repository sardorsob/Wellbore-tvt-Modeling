# CNN-Regime Dilated Attention

## Result

Best overall candidate: `logit_switch_floor20` with strict-CV RMSE 12.5366, MAE 8.8463, and analog-distance stress RMSE 12.9839.

Oracle row-switch upper bound: `oracle_continuous_weight` with RMSE 10.7206. This is leakage-only and is not deployable, but it shows the CNN and regime anchor have useful complementary errors.

## Model Structure

The audit recomputes the direct pseudo-top regime-mean anchor and a compact local-window CNN specialist. It then tests uniform CNN residual blends, simple stress/confidence/disagreement gates, fold-heldout dilated-attention gates, and a leakage-only oracle row switch.

## Leakage Notes

Direct formation columns are used only as fold-local labels for pseudo-top proxy training. Regime probabilities, regime means, CNN windows, disagreement features, and gates use held-out pseudo-top predictions and hidden-test-safe features. The oracle row switch uses validation targets and is diagnostic only.

## Decision

Recommendation: `diagnostic_only_current_best_worse`.

## Next Move

Learned attention pooling did not beat the active average-pooled dilated candidate. Keep the average-pooled dilated branch as the active candidate, and treat this notebook as diagnostic evidence against adding attention pooling in this form.
