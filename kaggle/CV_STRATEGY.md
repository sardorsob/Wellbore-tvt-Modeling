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
- Add Phase 4 fold-similarity diagnostics: compare held-out folds on eval-zone fraction, GR missingness, endpoint TVT, recent slopes, trajectory length, and typewell alignment.
- Investigate why Ridge beat Random Forest locally while Random Forest beat Ridge on public LB before making many boosted-tree leaderboard submissions.
- Add an endpoint-hold diagnostic baseline to future validation reports. Phase 3 sampled target-zone RMSE was **15.5815**, so richer models should demonstrate improvement over this simple boundary baseline.
- Track direct slope extrapolation separately from slope-as-feature. Phase 3 last-50 slope projection was unstable at **104.5322** sampled RMSE.

## Leakage Guardrails

- Exclude train-only formation columns from test-safe feature sets unless a model explicitly handles train-only analysis and drops them at inference.
- Do not use test labels or leaderboard feedback to choose features directly.
- Preserve row index mapping for submission ids.
- Treat public leaderboard as a noisy external check, not the validation set.
- Treat the visible local test wells as submission-shape examples only. Phase 2 found that `000d7d20`, `00bbac68`, and `00e12e8b` also exist in train, so their exact summary descriptors should not be used as hidden-test distribution evidence.
