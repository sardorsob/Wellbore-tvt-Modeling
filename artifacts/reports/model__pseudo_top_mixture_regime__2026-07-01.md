# Pseudo-Top Bayesian Mixture Regime

## Result

Best overall candidate: `xgb_pseudohuber_mean__ridge_alpha650__mixture_conf_s050` with strict-CV RMSE 13.4437, MAE 9.5108, and analog-distance stress RMSE 13.7593.

Best mean-only candidate: `xgb_square_mean__ridge_alpha650__mean_only` with RMSE 13.4745.

Best MAP mixture-prior candidate: `xgb_pseudohuber_mean__ridge_alpha650__mixture_conf_s050` with RMSE 13.4437.

## Model Structure

Pseudo-top distances are converted into soft formation probabilities. The residual correction uses Ridge MAP estimates over formation probability, progress, distance, and near-top basis terms, with conservative shrinkage schedules driven by mixture confidence and pseudo-top spread.

## Leakage Notes

Direct formation columns are used only as fold-local labels for pseudo-top proxy training. Mixture probabilities and corrections are generated from held-out pseudo-top predictions and hidden-test-safe row/typewell/state columns.

## Decision

Recommendation: `iterate_before_package`.

## Next Move

If this MAP mixture does not move strict CV, the next Bayesian route should model the TVT mean directly by stratigraphic regime instead of fitting a residual correction after a strong tree mean.
