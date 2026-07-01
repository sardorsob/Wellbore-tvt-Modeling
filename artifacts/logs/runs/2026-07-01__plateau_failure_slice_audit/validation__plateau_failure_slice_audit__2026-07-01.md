# Plateau Failure Slice Audit

## Score Pattern

| run_key | strict_rmse | stress_rmse | strict_delta_vs_regime_mean | stress_delta_vs_regime_mean |
| --- | --- | --- | --- | --- |
| stress_gated_stack | 13.0769 | 13.4468 | -0.0159 | 0.0462 |
| regime_stack | 13.0769 | 13.4468 | -0.0159 | 0.0462 |
| alignment_aware | 13.0838 | 13.4303 | -0.0089 | 0.0297 |
| regime_tuned | 13.0910 | 13.4065 | -0.0018 | 0.0059 |
| regime_mean | 13.0928 | 13.4006 | 0.0000 | 0.0000 |
| probability_scale | 13.0928 | 13.4006 | 0.0000 | 0.0000 |
| partial_pooling | 13.0953 | 13.4038 | 0.0025 | 0.0032 |
| more_rows | 13.1185 | 13.4745 | 0.0258 | 0.0739 |

## Hard-Well Consensus

| wellname | n_top15_runs | mean_top_rmse | max_top_rmse | runs |
| --- | --- | --- | --- | --- |
| 86454a6f | 8 | 59.8934 | 63.2358 | alignment_aware, more_rows, partial_pooling, probability_scale, regime_mean, regime_stack, regime_tuned, stress_gated_stack |
| 1b1eba53 | 8 | 58.7354 | 60.9001 | alignment_aware, more_rows, partial_pooling, probability_scale, regime_mean, regime_stack, regime_tuned, stress_gated_stack |
| a959858c | 8 | 48.0554 | 48.4765 | alignment_aware, more_rows, partial_pooling, probability_scale, regime_mean, regime_stack, regime_tuned, stress_gated_stack |
| 5f4d2a52 | 8 | 46.7919 | 48.4461 | alignment_aware, more_rows, partial_pooling, probability_scale, regime_mean, regime_stack, regime_tuned, stress_gated_stack |
| ba48188d | 8 | 44.7499 | 49.0458 | alignment_aware, more_rows, partial_pooling, probability_scale, regime_mean, regime_stack, regime_tuned, stress_gated_stack |
| 2fd68f7b | 8 | 44.4659 | 45.1887 | alignment_aware, more_rows, partial_pooling, probability_scale, regime_mean, regime_stack, regime_tuned, stress_gated_stack |
| f88ddb26 | 8 | 44.3256 | 46.8413 | alignment_aware, more_rows, partial_pooling, probability_scale, regime_mean, regime_stack, regime_tuned, stress_gated_stack |
| fef8af96 | 8 | 44.0002 | 44.3781 | alignment_aware, more_rows, partial_pooling, probability_scale, regime_mean, regime_stack, regime_tuned, stress_gated_stack |
| 389ae58f | 8 | 40.4528 | 41.0655 | alignment_aware, more_rows, partial_pooling, probability_scale, regime_mean, regime_stack, regime_tuned, stress_gated_stack |
| f6d009f4 | 8 | 37.2293 | 37.8075 | alignment_aware, more_rows, partial_pooling, probability_scale, regime_mean, regime_stack, regime_tuned, stress_gated_stack |
| 8382f4b6 | 8 | 36.2703 | 36.6413 | alignment_aware, more_rows, partial_pooling, probability_scale, regime_mean, regime_stack, regime_tuned, stress_gated_stack |
| 94d813a4 | 8 | 36.0853 | 37.3556 | alignment_aware, more_rows, partial_pooling, probability_scale, regime_mean, regime_stack, regime_tuned, stress_gated_stack |

## Interpretation

Recent changes are not failing randomly. The direct pseudo-top regime mean remains the best joint strict/stress anchor, while strict-only residual stacks and alignment additions move error into stress-sensitive slices. The partial-pooling experiment is especially useful because it shows that sharing the current pseudo-top regime signal through one high-dimensional Ridge layer weakens the useful formation-specific mean.

## Next Plan

1. Stop simple shrinkage, row-density, and confidence-schedule sweeps.
2. Build a state-revision notebook that audits whether pseudo-top errors are systematic by well, progress zone, analog distance, GR motif, or typewell lag.
3. If systematic state errors exist, train a second-stage state corrector before TVT modeling.
4. If state errors are not systematic enough, try a local-log sequence model that predicts target deltas from known-zone and target-row windows directly.
