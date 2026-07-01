# OOF Candidate Blend Audit

## Result

Best overall candidate: `blend_plus_ridge_residual_s15` with strict-CV RMSE 13.0769, MAE 9.3593, and analog-distance stress RMSE 13.4468.

Regime-mean reference in this notebook: `regime_blend_reference` with strict-CV RMSE 13.0928.

Best residual-stack candidate: `blend_plus_ridge_residual_s15` with strict-CV RMSE 13.0769.

## Model Structure

Pseudo-top probabilities define a fold-safe regime-mean baseline. XGBoost and Ridge residual heads produce OOF candidates, then fixed averages and leave-one-fold-out positive linear meta-blends test whether candidate diversity can improve prediction without fold leakage.

## Leakage Notes

Direct formation columns are used only as fold-local labels for pseudo-top proxy training. Regime probabilities, regime means, and meta predictions use held-out pseudo-top predictions and hidden-test-safe features.

## Decision

Recommendation: `iterate_before_package`.

## Next Move

If meta-blending does not beat the first regime-mean baseline, candidate diversity is likely too weak; move to structurally new features or formation-boundary modeling rather than further local stacking.
