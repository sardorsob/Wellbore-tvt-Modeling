# CNN-Regime Trajectory Expansion

## Result

Best overall candidate: `xgb_switch_p100__well_bias_h005_s25` with strict-CV RMSE 11.7138, MAE 8.2026, and analog-distance stress RMSE 11.8366.

Oracle row-switch upper bound: `oracle_continuous_weight` with RMSE 10.0207. This is leakage-only and is not deployable, but it shows the CNN and regime anchor have useful complementary errors.

## Model Structure

The audit recomputes the direct pseudo-top regime-mean anchor and compact local-window CNN specialist with hidden-test-safe well-trajectory context features. It then tests uniform CNN blends, simple stress/confidence/disagreement gates, conservative well-bias corrections, and a leakage-only oracle row switch.

## Leakage Notes

Direct formation columns are used only as fold-local labels for pseudo-top proxy training. Regime probabilities, regime means, CNN windows, disagreement features, and gates use held-out pseudo-top predictions and hidden-test-safe features. The oracle row switch uses validation targets and is diagnostic only.

## Decision

Recommendation: `stress_improves_strict_regresses`.

## Next Move

Trajectory expansion tests whether richer hidden-test-safe within-well context features can improve the calibrated trajectory baseline.
