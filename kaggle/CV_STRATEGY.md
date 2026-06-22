# CV Strategy

## Current Status

Status: Phase 4 EDA complete; protocol is active for the next modeling loop, but still subject to refinement after richer-feature results.

## Current Plan

Use local validation that simulates the test setting by masking `TVT_input` in the evaluation zone of training wells, then measuring RMSE against the true `TVT`.

This follows decision `D003` in `context/DECISIONS.md`.

## Active Reporting Requirements

- Report endpoint-hold RMSE alongside every reportable model. Phase 4 full-row endpoint-hold RMSE is **15.9099**.
- Report grouped fold-level RMSE, MAE, and R2, not only mean RMSE.
- Report descriptor balance by fold: eval rows/fraction, GR missingness, recent slope, endpoint TVT/delta, trajectory length, and typewell alignment.
- Interpret public LB as an external check. Phase 4 found Random Forest beats Ridge publicly by **0.032** RMSE while local CV ranks Random Forest worse by **0.300** RMSE.
- Keep direct slope projection separate from slope-as-feature. Phase 4 full-row last-50 slope projection RMSE is **112.4817**.
- Label target-zone-trained diagnostic runs separately from known-zone-trained baselines. The first GAM-style run uses sampled target-zone grouped CV and should not be treated as directly identical to earlier Ridge/Random Forest CV.
- Treat tuned GAM results as target-zone diagnostic evidence. The best tuned GAM improved the first GAM by only **0.0032** RMSE, so use Kaggle submission sparingly and avoid leaderboard-tuning tiny spline changes.
- Treat robust signed-delta GLM results as target-zone diagnostic evidence until submitted. The Huber GLM scored **14.7928** locally, better than tuned GAM, but still needs a self-contained Kaggle read before external promotion.

## Planned Refinement

- Start with deterministic masked-zone validation for the interpolation baseline.
- Add leave-one-well-out or grouped well-level CV before repeated leaderboard submissions.
- Keep preprocessing fit inside the training fold only.
- Track validation RMSE variability across wells, not only global RMSE.
- Use descriptor-balanced grouped folds as a diagnostic comparison to the current sorted grouped folds.
- Investigate why Ridge beat Random Forest locally while Random Forest beat Ridge on public LB before making many boosted-tree leaderboard submissions.
- Add an endpoint-hold diagnostic baseline to future validation reports. Phase 3 sampled target-zone RMSE was **15.5815**, so richer models should demonstrate improvement over this simple boundary baseline.
- Track direct slope extrapolation separately from slope-as-feature. Phase 3 last-50 slope projection was unstable at **104.5322** sampled RMSE.

## Leakage Guardrails

- Exclude train-only formation columns from test-safe feature sets unless a model explicitly handles train-only analysis and drops them at inference.
- Do not use test labels or leaderboard feedback to choose features directly.
- Preserve row index mapping for submission ids.
- Treat public leaderboard as a noisy external check, not the validation set.
- Treat the visible local test wells as submission-shape examples only. Phase 2 found that `000d7d20`, `00bbac68`, and `00e12e8b` also exist in train, so their exact summary descriptors should not be used as hidden-test distribution evidence.
