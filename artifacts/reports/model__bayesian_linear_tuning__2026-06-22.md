# Bayesian Linear Tuning

Date: 2026-06-22

## Key Findings
- **best_tuned_bayesian_candidate**: Best tuned Bayesian candidate is student_t_fixed3_autodelta with sampled grouped RMSE 14.8449.
  - Modeling implication: Use this to decide whether Bayesian linear tuning meaningfully improved the first Bayesian baseline.
- **baseline_comparison**: Best tuned Bayesian RMSE 14.8449 vs previous Bayesian best 14.9752 and Huber GLM 14.7928.
  - Modeling implication: If Bayesian linear still trails Huber, move to HSGP residual structure rather than more linear tuning.
- **coverage_check**: Best tuned Bayesian 80% coverage is 0.769 with mean width 31.12 TVT units.
  - Modeling implication: Coverage remains the main Bayesian value even if posterior mean RMSE is not leading.

## Best Candidate
- Candidate: `student_t_fixed3_autodelta`
- Guide: `autodelta`
- Likelihood: `student_t_fixed3`
- Heteroskedastic: `False`
- RMSE mean/std: 14.8449 / 0.5205
- 80% coverage/width: 0.769 / 31.12

## Artifacts
- Fold metrics: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\model__bayesian_linear_tuning_fold_metrics__2026-06-22.csv
- Summary: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\model__bayesian_linear_tuning_summary__2026-06-22.csv
- Posterior diagnostics: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\model__bayesian_linear_tuning_posterior_diagnostics__2026-06-22.csv
- Residual sample: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\model__bayesian_linear_tuning_residual_sample__2026-06-22.csv
- Findings: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\model__bayesian_linear_tuning_findings__2026-06-22.csv
- rmse_comparison: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\figures\model__bayesian_linear_tuning_rmse_comparison__2026-06-22.png
- fold_rmse: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\figures\model__bayesian_linear_tuning_fold_rmse__2026-06-22.png
- coverage_width: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\figures\model__bayesian_linear_tuning_coverage_width__2026-06-22.png
- uncertainty_by_boundary: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\figures\model__bayesian_linear_tuning_uncertainty_by_boundary__2026-06-22.png
