# Bayesian Linear Robust Baseline

Date: 2026-06-22

## Key Findings
- **best_bayesian_candidate**: Best Bayesian linear candidate is bayes_student_t_heteroskedastic with sampled grouped RMSE 14.9752.
  - Modeling implication: Compare the posterior mean against Huber GLM and GAM before considering Kaggle packaging.
- **uncertainty_calibration**: Best candidate 80% coverage is 0.783 with mean 80% interval width 32.72 TVT units.
  - Modeling implication: Use coverage and width to decide whether Bayesian uncertainty is calibrated enough for HSGP priors.
- **robust_likelihood_context**: Huber GLM local RMSE reference is 14.7928; fine-tuned GAM public LB is 15.055.
  - Modeling implication: A Bayesian model needs either better RMSE or better calibrated uncertainty to justify added complexity.

## Best Candidate
- Candidate: `bayes_student_t_heteroskedastic`
- Likelihood: `student_t`
- Heteroskedastic: `True`
- RMSE mean/std: 14.9752 / 0.5352
- 80% coverage/width: 0.783 / 32.72
- 95% coverage/width: 0.933 / 57.79

## Artifacts
- Fold metrics: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\model__bayesian_linear_robust_fold_metrics__2026-06-22.csv
- Summary: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\model__bayesian_linear_robust_summary__2026-06-22.csv
- Posterior diagnostics: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\model__bayesian_linear_robust_posterior_diagnostics__2026-06-22.csv
- Residual sample: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\model__bayesian_linear_robust_residual_sample__2026-06-22.csv
- Findings: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\model__bayesian_linear_robust_findings__2026-06-22.csv
- rmse_comparison: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\figures\model__bayesian_linear_robust_rmse_comparison__2026-06-22.png
- fold_rmse: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\figures\model__bayesian_linear_robust_fold_rmse__2026-06-22.png
- coverage_width: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\figures\model__bayesian_linear_robust_coverage_width__2026-06-22.png
- uncertainty_by_boundary: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\figures\model__bayesian_linear_robust_uncertainty_by_boundary__2026-06-22.png
