# CV Strategy

## Current Status

Status: draft, not yet frozen.

## Current Plan

Use local validation that simulates the test setting by masking `TVT_input` in the evaluation zone of training wells, then measuring RMSE against the true `TVT`.

This follows decision `D003` in `context/DECISIONS.md`.

## Planned Refinement

- Start with deterministic masked-zone validation for the interpolation baseline.
- Add leave-one-well-out or grouped well-level CV before repeated leaderboard submissions.
- Keep preprocessing fit inside the training fold only.
- Track validation RMSE variability across wells, not only global RMSE.

## Leakage Guardrails

- Exclude train-only formation columns from test-safe feature sets unless a model explicitly handles train-only analysis and drops them at inference.
- Do not use test labels or leaderboard feedback to choose features directly.
- Preserve row index mapping for submission ids.
- Treat public leaderboard as a noisy external check, not the validation set.

