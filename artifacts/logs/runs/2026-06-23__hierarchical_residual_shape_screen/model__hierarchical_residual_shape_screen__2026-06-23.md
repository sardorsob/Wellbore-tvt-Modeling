# Hierarchical Residual Shape Screen

Date: 2026-06-23

## Purpose

Test shrinkage-controlled residual-shape corrections over evaluation progress, boundary distance, and slope-instability coordinates.

## Priors And Assumptions

- Corrections are residual-shape offsets centered on zero and clipped to +/- 12.0 TVT units before shrinkage.
- Binned arms use empirical-Bayes count shrinkage toward zero or global residual shape.
- Regularized arms use Ridge/Huber penalties and fixed shrinkage before scoring.
- Every arm trains on other folds only and scores the held-out fold.

## Results

| candidate | kind | shrink | prior_strength | alpha | prior | rmse_mean | rmse_std | mae_mean | mean_abs_fold_bias | mean_shape_correction_abs | shape_correction_std_mean | rmse_delta_vs_sample_baseline | rmse_delta_vs_full_reference |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| row_linear_ridge_a100_s025 | ridge | 0.2500 | 0.0000 | 100.0000 | regularized linear row-shape residual with 75 percent zero shrinkage | 14.7382 | 0.4912 | 10.4137 | 0.3062 | 0.0677 | 0.1507 | -0.0003 | 0.3796 |
| baseline_no_shape | baseline | 0.0000 | 0.0000 | 0.0000 | zero residual-shape correction | 14.7385 | 0.4871 | 10.4131 | 0.3032 | 0.0000 | 0.0000 | 0.0000 | 0.3799 |
| distance_bins_p100_s025 | single_bin | 0.2500 | 100.0000 | 0.0000 | boundary-distance-bin means shrunk toward zero | 14.7386 | 0.4862 | 10.4153 | 0.3064 | 0.0570 | 0.0428 | 0.0000 | 0.3800 |
| progress_spline_ridge_a300_s025 | progress_spline | 0.2500 | 0.0000 | 300.0000 | smooth progress residual shape with strong coefficient shrinkage | 14.7386 | 0.4860 | 10.4159 | 0.3066 | 0.0686 | 0.0270 | 0.0000 | 0.3800 |
| row_huber_s025 | huber | 0.2500 | 0.0000 | 0.0500 | robust linear row-shape residual with 75 percent zero shrinkage | 14.7391 | 0.4877 | 10.4194 | 0.3666 | 0.0825 | 0.0969 | 0.0005 | 0.3805 |
| progress_bins_p200_s025 | single_bin | 0.2500 | 200.0000 | 0.0000 | progress-bin means shrunk by 200 pseudo-rows and 75 percent to zero | 14.7398 | 0.4857 | 10.4165 | 0.3063 | 0.0457 | 0.0655 | 0.0013 | 0.3812 |
| progress_bins_p50_s025 | single_bin | 0.2500 | 50.0000 | 0.0000 | progress-bin means shrunk by 50 pseudo-rows and 75 percent to zero | 14.7403 | 0.4853 | 10.4177 | 0.3084 | 0.0588 | 0.0851 | 0.0018 | 0.3817 |
| progress_distance_bins_p100_s025 | two_dim_bin | 0.2500 | 100.0000 | 0.0000 | 2D progress-distance cell means with strong shrinkage | 14.7413 | 0.4853 | 10.4176 | 0.3019 | 0.0529 | 0.0871 | 0.0028 | 0.3827 |
| row_poly2_ridge_a300_s025 | poly2_ridge | 0.2500 | 0.0000 | 300.0000 | quadratic row-shape residual with strong coefficient shrinkage | 14.7443 | 0.4934 | 10.4200 | 0.3064 | 0.0682 | 0.1737 | 0.0057 | 0.3857 |
| slope_instability_bins_p100_s025 | single_bin | 0.2500 | 100.0000 | 0.0000 | slope-instability-bin means shrunk toward zero | 14.7490 | 0.4944 | 10.4211 | 0.2951 | 0.0633 | 0.1435 | 0.0105 | 0.3904 |
| additive_bins_p100_s025 | additive_bins | 0.2500 | 100.0000 | 0.0000 | additive progress/distance/slope bin deviations with strong shrinkage | 14.7516 | 0.4937 | 10.4241 | 0.2968 | 0.0714 | 0.1766 | 0.0130 | 0.3930 |

## Findings

| finding | value | modeling_implication |
| --- | --- | --- |
| best_candidate | row_linear_ridge_a100_s025 RMSE 14.7382 vs sample baseline 14.7385 | Promote only if the same-sample gain clears the hierarchy gate. |
| sample_baseline_delta | best candidate delta -0.0003 | Negative means row-shape correction improved the same residual frame. |
| well_bias_change | median absolute well-bias delta -0.0065 | Residual-shape correction should not reduce RMSE by creating new well-level bias. |

## Decision

Best candidate `row_linear_ridge_a100_s025` scored **14.7382** RMSE, delta **-0.0003** vs the sample no-shape baseline and **0.3796** vs the stored full Huber + Hilbert reference.

## Artifacts

- Fold metrics: `C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\model__hierarchical_residual_shape_fold_metrics__2026-06-23.csv`
- Summary: `C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\model__hierarchical_residual_shape_summary__2026-06-23.csv`
- Residual sample: `C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\model__hierarchical_residual_shape_residual_sample__2026-06-23.csv`
- Well diagnostics: `C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\model__hierarchical_residual_shape_well_diagnostics__2026-06-23.csv`
- Findings: `C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\model__hierarchical_residual_shape_findings__2026-06-23.csv`
- Comparison row: `C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\model__hierarchical_residual_shape_comparison__2026-06-23.csv`
