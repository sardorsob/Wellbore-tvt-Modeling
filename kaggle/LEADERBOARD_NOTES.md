# Leaderboard Notes

## Public Leaderboard

| Date | Ref | Notebook | Local CV RMSE | Public LB | Notes |
|---|---|---|---:|---:|---|
| 2026-06-21 | `53923555` | `sardorbsobirov/wellbore-linear-ridge-baseline` v2 | 15.7660 | 15.905 | First Ridge baseline. Public score is close to local grouped CV, but slightly worse. |

## Risks

- Public leaderboard may not represent private leaderboard distribution.
- Only 3 test wells are present locally, so public/private split behavior may be high variance.
- Over-tuning to leaderboard submissions can degrade final private score.

## Interpretation Rules

- Prefer models that improve local grouped CV and have plausible error analysis.
- Treat single small leaderboard improvements as tentative until repeated by local validation.
- Log rejected submissions as well as selected submissions.
