# XGBoost Local Sequence Stack

## Result

Best overall candidate: `alignment_regime_blend_reference` with strict-CV RMSE 13.1613, MAE 9.4366, and analog-distance stress RMSE 13.4561.

Alignment-aware regime-mean reference in this notebook: `alignment_regime_blend_reference` with strict-CV RMSE 13.1613.

Best residual-stack candidate: `blend_plus_sequence_ridge_residual_s08` with strict-CV RMSE 13.1626.

## Model Structure

Pseudo-top probabilities define a fold-safe alignment-aware regime-mean baseline. XGBoost and Ridge residual heads then test whether local GR/typewell sequence-window meta-features can improve the regime mean without losing stress-fold behavior.

## Leakage Notes

Direct formation columns are used only as fold-local labels for pseudo-top proxy training. Regime probabilities, regime means, and meta predictions use held-out pseudo-top predictions and hidden-test-safe features.

## Decision

Recommendation: `diagnostic_only`.

## Next Move

If this does not beat the first regime-mean baseline, move to state-revision/top-error correction or a true local-log sequence model.
