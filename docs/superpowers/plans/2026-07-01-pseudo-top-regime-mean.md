# Pseudo-Top Regime Mean Plan

## Goal

Test whether fold-safe pseudo-top probabilities are stronger as direct stratigraphic-regime mean weights than as residual priors or flat tree features.

## Steps

- [x] Reuse the fold-safe pseudo-top inference from the previous Bayesian mixture screen.
- [x] Fit formation/regime-specific weighted Ridge MAP means on strict training folds.
- [x] Blend regime means with a global XGBoost mean using pseudo-top confidence and conservative fallback.
- [x] Export strict-CV, stress-CV, zone, analog-bin, well, proxy-quality, and feature-preview artifacts.
- [x] Record assumptions, decisions, experiment results, and feature log entries.

## Result

Best candidate: `xgb_square_global__regime_ridge_alpha80__confidence_s100`

- Strict RMSE: **13.0928**
- Analog-distance stress RMSE: **13.4006**
- Improvement over pseudo-top Bayesian mixture: **0.3510** strict RMSE
- Improvement over safe/typewell reference: **0.3579** strict RMSE

## Decision

Package this candidate next as a Kaggle-safe notebook. The direct regime-mean formulation is the first pseudo-top path with enough local margin to justify a public-score probe.
