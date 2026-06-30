# Stratigraphic State Feature Screen

Run ID: `2026-06-30__stratigraphic_state_feature_screen`

## Best Non-Oracle Candidate

- Model: `huber_delta`
- Feature set: `strat_known_state`
- strict_hybrid_fold RMSE: **14.9922**
- Delta vs same-model safe base: **-0.0391**
- Delta vs current local safe-XGBoost reference 13.4178: **1.5744**

## Important Same-Model Signals

- Huber `strat_known_state` improved over Huber `safe_base` by **-0.0391 RMSE** (15.0313 -> 14.9922).
- XGBoost squared-error `strat_known_state` improved over XGBoost squared-error `safe_base` by **-0.7738 RMSE** (22.9672 -> 22.1933), but this local consumer is much worse than the current packaged XGBoost reference and is not a submission candidate.

## Oracle Diagnostic

- Best oracle row: `hist_gradient` / `oracle_direct_formations` with RMSE **19.9792**.
- The oracle diagnostic is not useful in this notebook because the consumer/training setup differs from the earlier strict-CV XGBoost oracle run where direct formation features helped strongly.

## Decision

Do not package this screen for Kaggle. Keep the known-zone stratigraphic state features as directional evidence, but revise the next screen to plug them into the exact current public-best XGBoost package/training recipe before judging leaderboard potential.