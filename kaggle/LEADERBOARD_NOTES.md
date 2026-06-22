# Leaderboard Notes

## Public Leaderboard

| Date | Ref | Notebook | Local CV RMSE | Public LB | Notes |
|---|---|---|---:|---:|---|
| 2026-06-22 | `53960276` | `sardorbsobirov/wellbore-huber-hsgp-residual` v1 | 14.3586 | 14.514 | Huber signed-delta mean plus additive Hilbert residual over boundary/progress/slope-instability coordinates. New best public LB, but still far from the top public range. |
| 2026-06-22 | `53932883` | `sardorbsobirov/wellbore-fine-tuned-gam-spline` v1 | 14.8687 | 15.055 | Fine-tuned GAM spline submission using slope-core boundary/trajectory/recent-slope features. New best public LB; use as the current external baseline. |
| 2026-06-22 | `53958582` | `sardorbsobirov/wellbore-additive-hsgp-student-t` v1 | 14.8235 | 15.255 | Kaggle-safe additive HSGP Student-t MAP approximation. Beats Bayesian linear and Huber externally, but still trails the fine-tuned GAM public LB. |
| 2026-06-22 | `53952950` | `sardorbsobirov/wellbore-huber-glm-slope-core` v1 | 14.7928 | 15.464 | Huber signed-delta GLM had the best local diagnostic so far but did not beat the fine-tuned GAM public LB; useful validation-mismatch signal. |
| 2026-06-22 | `53957370` | `sardorbsobirov/wellbore-bayesian-linear-student-t` v1 | 14.8449 | 15.512 | First Bayesian-linear public baseline using a Kaggle-safe Student-t MAP implementation. Worse than Huber and GAM externally, so use as likelihood evidence before HSGP rather than as a promotion candidate. |
| 2026-06-21 | `53924934` | `sardorbsobirov/wellbore-elasticnet-baseline` v1 | 15.7907 | 16.091 | ElasticNet was close to Ridge locally but worse externally; do not promote. |
| 2026-06-21 | `53924036` | `sardorbsobirov/wellbore-random-forest-baseline` v1 | 16.0655 | 15.873 | First Random Forest submission. Public LB beats Ridge even though local CV is worse, so treat the improvement as tentative until validation is refined. |
| 2026-06-21 | `53923555` | `sardorbsobirov/wellbore-linear-ridge-baseline` v2 | 15.7660 | 15.905 | First Ridge baseline. Public score is close to local grouped CV, but slightly worse. |

## Risks

- Public leaderboard may not represent private leaderboard distribution.
- Only 3 test wells are present locally, so public/private split behavior may be high variance.
- Over-tuning to leaderboard submissions can degrade final private score.
- Phase 4 validation EDA found public/local ordering disagreement: Random Forest beats Ridge publicly by **0.032** RMSE, while local CV ranks Random Forest worse by **0.300** RMSE.

## Interpretation Rules

- Prefer models that improve local grouped CV and have plausible error analysis.
- When local CV and public LB disagree, inspect split design and feature behavior before trusting either ranking.
- Treat single small leaderboard improvements as tentative until repeated by local validation.
- Log rejected submissions as well as selected submissions.
- Before promoting a model, check endpoint-hold baseline, fold-level errors, and fold descriptor balance from the active CV protocol.
