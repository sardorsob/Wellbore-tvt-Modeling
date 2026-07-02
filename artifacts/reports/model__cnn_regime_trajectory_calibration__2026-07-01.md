# CNN-Regime Trajectory Calibration

## Result

Best overall candidate: `logit_switch_floor25__well_bias_h005_s35` with strict-CV RMSE 11.6724, MAE 8.2637, and analog-distance stress RMSE 11.9379.

Oracle row-switch upper bound: `oracle_continuous_weight` with RMSE 9.8460. This is leakage-only and is not deployable, but it shows the CNN and regime anchor have useful complementary errors.

## Model Structure

The audit recomputes the direct pseudo-top regime-mean anchor and compact local-window CNN specialist with hidden-test-safe well-trajectory context features. It then tests uniform CNN blends, simple stress/confidence/disagreement gates, conservative well-bias corrections, and a leakage-only oracle row switch.

## Leakage Notes

Direct formation columns are used only as fold-local labels for pseudo-top proxy training. Regime probabilities, regime means, CNN windows, disagreement features, and gates use held-out pseudo-top predictions and hidden-test-safe features. The oracle row switch uses validation targets and is diagnostic only.

## Decision

Recommendation: `package_cnn_regime_trajectory_calibration`.

## Next Move

Trajectory calibration tests whether wider logit floors, switch powers, and conservative well-bias shrinkage can recover more of the new oracle gap.
