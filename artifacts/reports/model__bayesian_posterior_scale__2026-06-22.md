# Bayesian Posterior Diagnostics And Targeted Scale

Date: 2026-06-22

## Key Findings
- **best_scale_candidate**: Best targeted scale candidate is hetero_boundary_progress_autodelta with sampled grouped RMSE 14.8626 and 80% coverage 0.780.
  - Modeling implication: Use this result to decide whether heteroskedastic scale structure improves the Bayesian linear baseline enough to keep tuning.
- **fixed_reference_comparison**: The fixed Student-t reference scored RMSE 14.9129; best targeted scale delta is -0.0503.
  - Modeling implication: If the delta is small or positive, move scale complexity into HSGP/hierarchical models rather than standalone Bayesian linear notebooks.
- **uncertainty_error_alignment**: Best candidate mean Spearman correlation between posterior SD and absolute error is 0.315.
  - Modeling implication: Low correlation means intervals are calibrated globally but weak at ranking difficult rows.
- **well_bias_signal**: The 90th percentile absolute well-level mean residual for the best candidate is 18.60 TVT units.
  - Modeling implication: Persistent well-level signed bias supports hierarchical well/intercept structure after the HSGP baseline.
- **hsgp_readiness**: Best targeted scale RMSE vs Huber GLM local reference is 0.0697, with coverage gap from nominal 80% of 0.020.
  - Modeling implication: The next Bayesian notebook should add smooth residual functions rather than only changing linear likelihood scale.

## Best Candidate
- Candidate: `hetero_boundary_progress_autodelta`
- Guide: `autodelta`
- Scale family: `boundary_progress`
- RMSE mean/std: 14.8626 / 0.5019
- 80% coverage/width: 0.780 / 31.95
- Posterior SD vs absolute error Spearman: 0.315

## Artifacts
- Fold metrics: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\model__bayesian_posterior_scale_fold_metrics__2026-06-22.csv
- Summary: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\model__bayesian_posterior_scale_summary__2026-06-22.csv
- Posterior diagnostics: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\model__bayesian_posterior_scale_posterior_diagnostics__2026-06-22.csv
- Residual sample: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\model__bayesian_posterior_scale_residual_sample__2026-06-22.csv
- Bin diagnostics: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\model__bayesian_posterior_scale_bin_diagnostics__2026-06-22.csv
- Well diagnostics: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\model__bayesian_posterior_scale_well_diagnostics__2026-06-22.csv
- Findings: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\model__bayesian_posterior_scale_findings__2026-06-22.csv
- rmse_comparison: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\figures\model__bayesian_posterior_scale_rmse_comparison__2026-06-22.png
- coverage_rmse_by_boundary: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\figures\model__bayesian_posterior_scale_coverage_rmse_by_boundary__2026-06-22.png
- residual_bias_by_progress: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\figures\model__bayesian_posterior_scale_residual_bias_by_progress__2026-06-22.png
- posterior_sd_vs_abs_error: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\figures\model__bayesian_posterior_scale_sd_vs_abs_error__2026-06-22.png
- well_bias_distribution: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\figures\model__bayesian_posterior_scale_well_bias_distribution__2026-06-22.png
