# Pseudo-Top Gated Regime Model

## Result

Best overall candidate: `xgb_pseudohuber_mean__ridge_pseudo_state_residual__uncertainty_gate_conservative` with strict-CV RMSE 13.4694, MAE 9.5158, and analog-distance stress RMSE 13.7665.

Best mean-only candidate: `xgb_square_mean__ridge_pseudo_state_residual__mean_only` with RMSE 13.4745.

Best residual-corrected candidate: `xgb_pseudohuber_mean__ridge_pseudo_state_residual__uncertainty_gate_conservative` with RMSE 13.4694.

## Gate Behavior

The gated candidates use pseudo-top margin, confidence, entropy, ensemble spread, and state/typewell descriptors to shrink residual corrections. Conservative gates clip weights at 0.35; open gates clip at 0.50.

## Leakage Notes

Direct formation columns are used only as fold-local labels for pseudo-top proxy training. Every pseudo-top feature is predicted for held-out rows by models trained on other folds.

## Decision

Recommendation: `iterate_before_package`.

## Next Move

If residual gating improves over mean-only, package only after checking public-transfer risk; otherwise move the pseudo-top uncertainty into a stronger mean model or a Bayesian prior rather than another post-hoc residual head.
