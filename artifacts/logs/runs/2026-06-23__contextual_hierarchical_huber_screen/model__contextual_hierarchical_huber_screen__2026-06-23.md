# Contextual Hierarchical Huber Screen

Date: 2026-06-23

## Purpose

Test whether test-safe well descriptors can modulate row-level residual shape through regularized contextual interactions.

## Priors And Assumptions

- Corrections are contextual residual offsets centered on zero and clipped to +/- 12.0 TVT units before shrinkage.
- Descriptor-by-row interactions approximate contextual hierarchy without raw well-ID effects.
- Ridge/Huber penalties and fixed shrinkage are the empirical-Bayes prior toward zero correction.
- Every arm trains on other folds only and scores the held-out fold.

## Results

| candidate | kind | alpha | shrink | prior | rmse_mean | rmse_std | mae_mean | mean_abs_fold_bias | mean_context_correction_abs | context_correction_std_mean | rmse_delta_vs_sample_baseline | rmse_delta_vs_full_reference |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| context_poly2_ridge_a2000_s025 | context_poly2 | 2000.0000 | 0.2500 | quadratic contextual surface with very strong ridge | 14.7322 | 0.4757 | 10.4222 | 0.2832 | 0.0498 | 0.3670 | -0.0064 | 0.3736 |
| baseline_no_context | baseline | 0.0000 | 0.0000 | zero contextual correction | 14.7385 | 0.4871 | 10.4131 | 0.3032 | 0.0000 | 0.0000 | 0.0000 | 0.3799 |
| context_row_interactions_ridge_a1500_s010 | context_row_interactions | 1500.0000 | 0.1000 | descriptor-by-row interactions with very strong ridge and 90 percent shrinkage | 14.7411 | 0.4903 | 10.4160 | 0.2993 | 0.0249 | 0.0822 | 0.0026 | 0.3825 |
| context_row_interactions_ridge_a500_s025 | context_row_interactions | 500.0000 | 0.2500 | descriptor-by-row interactions with strong ridge and 75 percent shrinkage | 14.7447 | 0.4951 | 10.4207 | 0.2865 | 0.0611 | 0.2312 | 0.0061 | 0.3861 |
| context_row_interactions_ridge_a1500_s025 | context_row_interactions | 1500.0000 | 0.2500 | descriptor-by-row interactions with very strong ridge and 75 percent shrinkage | 14.7459 | 0.4953 | 10.4208 | 0.2935 | 0.0622 | 0.2055 | 0.0073 | 0.3873 |
| context_main_ridge_a300_s025 | context_main | 300.0000 | 0.2500 | descriptor main effects with strong ridge and 75 percent shrinkage | 14.7481 | 0.4943 | 10.4214 | 0.2947 | 0.0552 | 0.2219 | 0.0095 | 0.3895 |
| context_row_huber_a005_s025 | context_row_huber | 0.0500 | 0.2500 | robust contextual row interactions with Huber loss and zero shrinkage | 14.7521 | 0.4980 | 10.4303 | 0.3441 | 0.0494 | 0.4404 | 0.0135 | 0.3935 |

## Findings

| finding | value | modeling_implication |
| --- | --- | --- |
| best_candidate | context_poly2_ridge_a2000_s025 RMSE 14.7322 vs sample baseline 14.7385 | Promote only if contextual interactions clear the hierarchy gate. |
| sample_baseline_delta | best candidate delta -0.0064 | Negative means descriptor-by-row context improved the same residual frame. |
| well_bias_change | median absolute well-bias delta 0.0143 | Context should not reduce row RMSE by moving bias between wells. |

## Decision

Best candidate `context_poly2_ridge_a2000_s025` scored **14.7322** RMSE, delta **-0.0064** vs the sample no-context baseline and **0.3736** vs the stored full Huber + Hilbert reference.

## Artifacts

- Fold metrics: `C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\model__contextual_hierarchical_huber_fold_metrics__2026-06-23.csv`
- Summary: `C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\model__contextual_hierarchical_huber_summary__2026-06-23.csv`
- Residual sample: `C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\model__contextual_hierarchical_huber_residual_sample__2026-06-23.csv`
- Well diagnostics: `C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\model__contextual_hierarchical_huber_well_diagnostics__2026-06-23.csv`
- Findings: `C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\model__contextual_hierarchical_huber_findings__2026-06-23.csv`
- Comparison row: `C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\model__contextual_hierarchical_huber_comparison__2026-06-23.csv`
