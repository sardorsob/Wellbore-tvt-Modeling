# CNN-Regime Residual Specialist

## Result

Best overall candidate: `residual_cnn_s085` with strict-CV RMSE 12.7353, MAE 9.0406, and analog-distance stress RMSE 13.1028.

Oracle row-switch upper bound: `oracle_continuous_shrink` with RMSE 11.8132. This is leakage-only and is not deployable, but it shows the CNN and regime anchor have useful complementary errors.

## Model Structure

The audit recomputes the direct pseudo-top regime-mean anchor and a compact local-window CNN specialist. It then tests uniform CNN residual blends, simple stress/confidence/disagreement gates, fold-heldout residual specialists, and a leakage-only oracle row switch.

## Leakage Notes

Direct formation columns are used only as fold-local labels for pseudo-top proxy training. Regime probabilities, regime means, CNN windows, disagreement features, and gates use held-out pseudo-top predictions and hidden-test-safe features. The oracle row switch uses validation targets and is diagnostic only.

## Decision

Recommendation: `diagnostic_only_current_best_worse`.

## Next Move

Residual-targeted CNN corrections did not beat the current CNN/regime gate. Treat this as a diagnostic: the residual oracle is lower than the deployable residual rows, but also worse than the prior direct-CNN oracle, so the next step should improve the original direct-delta specialist or add stress-aware seed stability rather than package this branch.
