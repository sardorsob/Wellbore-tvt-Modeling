# XGBoost Regime Stress-Gated Stack

## Result

Best overall candidate: `blend_plus_ridge_residual_s15_uniform` with strict-CV RMSE 13.0769, MAE 9.3593, and analog-distance stress RMSE 13.4468.

Regime-mean reference in this notebook: `regime_blend_reference` with strict-CV RMSE 13.0928.

Best residual-stack candidate: `blend_plus_ridge_residual_s15_uniform` with strict-CV RMSE 13.0769.

## Model Structure

Pseudo-top probabilities define a fold-safe regime-mean baseline. XGBoost and Ridge residual heads are gated by pseudo-top confidence, margin, analog stress, and hybrid transfer-risk schedules before being added to the regime mean.

## Leakage Notes

Direct formation columns are used only as fold-local labels for pseudo-top proxy training. Regime probabilities, regime means, and meta predictions use held-out pseudo-top predictions and hidden-test-safe features.

## Decision

Recommendation: `iterate_before_package`.

## Next Move

If stress gating does not beat the first regime-mean baseline, move to pseudo-top inference tuning or a structurally different Bayesian partial-pooling model.
