# CNN-Regime Blend Wide Grid

## Result

Best overall candidate: `blend_cnn_45` with strict-CV RMSE 12.6497, MAE 8.9938, and analog-distance stress RMSE 12.9956.

Oracle row-switch upper bound: `oracle_row_switch` with RMSE 10.9049. This is leakage-only and is not deployable, but it shows the CNN and regime anchor have useful complementary errors.

## Model Structure

The audit recomputes the direct pseudo-top regime-mean anchor and a compact local-window CNN specialist. It then tests uniform CNN residual blends, simple stress/confidence/disagreement gates, fold-heldout learned gates, and a leakage-only oracle row switch.

## Leakage Notes

Direct formation columns are used only as fold-local labels for pseudo-top proxy training. Regime probabilities, regime means, CNN windows, disagreement features, and gates use held-out pseudo-top predictions and hidden-test-safe features. The oracle row switch uses validation targets and is diagnostic only.

## Decision

Recommendation: `package_cnn_regime_blend_wide_grid`.

## Next Move

Use the strict/stress tradeoff to select the package candidate; if strict keeps improving at 1.00, revisit CNN architecture rather than a simple residual blend.
