# Hierarchical Empirical Bayes Offsets

Date: 2026-06-23

## Purpose

Test conservative contextual residual offsets on top of the current Huber + additive Hilbert residual model.

## Priors And Assumptions

- Corrections are residual offsets centered on zero and clipped to +/- 12.0 TVT units before shrinkage.
- Candidate shrinkage factors are fixed before scoring and evaluated out of fold.
- Well-level effects are predicted from test-safe descriptors, not raw well IDs.
- This is empirical Bayes / partial-pooling screening, not a full posterior model.

## Results

| candidate | kind | shrink | alpha | prior | rmse_mean | rmse_std | mae_mean | mean_abs_fold_bias | mean_offset_abs | offset_std_mean | rmse_delta_vs_sample_baseline | rmse_delta_vs_full_reference |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| global_bias_shrink_050 | global | 0.5000 | 0.0000 | global residual mean with 50 percent shrinkage to zero | 14.7378 | 0.4843 | 10.4181 | 0.3118 | 0.1344 | 0.0000 | -0.0008 | 0.3792 |
| global_bias_shrink_025 | global | 0.2500 | 0.0000 | global residual mean with 75 percent shrinkage to zero | 14.7380 | 0.4859 | 10.4154 | 0.3075 | 0.0672 | 0.0000 | -0.0006 | 0.3794 |
| well_intercept_progress_slope_a100_s025 | intercept_progress | 0.2500 | 100.0000 | descriptor intercept plus progress slope, strongly shrunk | 14.7380 | 0.4808 | 10.4203 | 0.3065 | 0.0684 | 0.4241 | -0.0005 | 0.3794 |
| baseline_no_offset | baseline | 0.0000 | 0.0000 | zero correction | 14.7385 | 0.4871 | 10.4131 | 0.3032 | 0.0000 | 0.0000 | 0.0000 | 0.3799 |
| well_intercept_ridge_a100_s025 | well_intercept | 0.2500 | 100.0000 | ridge descriptor intercept with stronger coefficient shrinkage | 14.7454 | 0.4879 | 10.4268 | 0.3070 | 0.0691 | 0.3323 | 0.0069 | 0.3868 |
| row_context_ridge_a100_s025 | row_context | 0.2500 | 100.0000 | row-level contextual residual with strong zero shrinkage | 14.7458 | 0.4852 | 10.4282 | 0.2929 | 0.0832 | 0.3782 | 0.0073 | 0.3872 |
| well_intercept_ridge_a10_s025 | well_intercept | 0.2500 | 10.0000 | ridge descriptor intercept with strong zero shrinkage | 14.7463 | 0.4843 | 10.4261 | 0.2842 | 0.0973 | 0.3726 | 0.0077 | 0.3877 |
| well_intercept_huber_s025 | well_intercept_huber | 0.2500 | 0.0100 | robust descriptor intercept with strong zero shrinkage | 14.7554 | 0.4880 | 10.4291 | 0.2821 | 0.1150 | 0.4094 | 0.0168 | 0.3968 |
| well_intercept_ridge_a100_s050 | well_intercept | 0.5000 | 100.0000 | ridge descriptor intercept with moderate zero shrinkage | 14.7605 | 0.4855 | 10.4493 | 0.3108 | 0.1382 | 0.6645 | 0.0219 | 0.4019 |

## Findings

| finding | value | modeling_implication |
| --- | --- | --- |
| best_candidate | global_bias_shrink_050 RMSE 14.7378 vs baseline sample RMSE 14.7385 | Promote only if improvement is meaningful and fold-stable. |
| sample_baseline_delta | best candidate delta vs sample no-offset baseline -0.0008 | Negative means the offset improves the same sampled residual frame. |
| best_bias_change | median absolute well-bias delta -0.0891 | Offset should reduce well bias without worsening row-level RMSE. |

## Decision

Best candidate `global_bias_shrink_050` scored **14.7378** RMSE, delta **-0.0008** vs the sample no-offset baseline and **0.3792** vs the stored full Huber + Hilbert reference. Use the promotion gate in `context/STATUS.md` before moving to contextual Huber hierarchy.

## Artifacts

- Fold metrics: `C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\model__hierarchical_empirical_bayes_offsets_fold_metrics__2026-06-23.csv`
- Summary: `C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\model__hierarchical_empirical_bayes_offsets_summary__2026-06-23.csv`
- Residual sample: `C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\model__hierarchical_empirical_bayes_offsets_residual_sample__2026-06-23.csv`
- Well diagnostics: `C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\model__hierarchical_empirical_bayes_offsets_well_diagnostics__2026-06-23.csv`
- Findings: `C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\model__hierarchical_empirical_bayes_offsets_findings__2026-06-23.csv`
- Comparison row: `C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\model__hierarchical_empirical_bayes_offsets_comparison__2026-06-23.csv`
