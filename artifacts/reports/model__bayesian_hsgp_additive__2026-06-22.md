# Additive HSGP Residual Tuning

Date: 2026-06-22

## Key Findings
- **best_additive_hsgp_candidate**: Best additive HSGP candidate is additive_dps_m8_se_autodelta with sampled grouped RMSE 14.8235.
  - Modeling implication: Use this as the first additive-HSGP reference before trying Huber/GAM residual means.
- **linear_reference_delta**: Best additive HSGP RMSE delta versus same-notebook linear reference is -0.0698.
  - Modeling implication: A negative delta means additive smooth terms add value beyond the linear Student-t mean.
- **joint_hsgp_delta**: Best additive HSGP RMSE delta versus first joint HSGP is -0.0762.
  - Modeling implication: This compares additive one-dimensional terms against the previous joint boundary/progress surface.
- **bayesian_best_delta**: Best additive HSGP RMSE delta versus prior tuned Bayesian best is -0.0214.
  - Modeling implication: If still positive, the next HSGP should change the mean model rather than only residual basis geometry.
- **huber_reference_delta**: Best additive HSGP RMSE delta versus Huber GLM local reference is 0.0307.
  - Modeling implication: This decides whether to proceed to Huber/GAM residual HSGP before Kaggle packaging.

## Best Candidate
- Candidate: `additive_dps_m8_se_autodelta`
- Guide: `autodelta`
- Terms: `distance_log1p+eval_progress+abs_projected_delta_10`
- Basis total: 24
- RMSE mean/std: 14.8235 / 0.5186
- 80% coverage/width: 0.765 / 30.97

## Artifacts
- Fold metrics: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\model__bayesian_hsgp_additive_fold_metrics__2026-06-22.csv
- Summary: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\model__bayesian_hsgp_additive_summary__2026-06-22.csv
- Posterior diagnostics: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\model__bayesian_hsgp_additive_posterior_diagnostics__2026-06-22.csv
- Residual sample: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\model__bayesian_hsgp_additive_residual_sample__2026-06-22.csv
- Findings: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\model__bayesian_hsgp_additive_findings__2026-06-22.csv
- rmse_comparison: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\figures\model__bayesian_hsgp_additive_rmse_comparison__2026-06-22.png
- fold_rmse: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\figures\model__bayesian_hsgp_additive_fold_rmse__2026-06-22.png
- hsgp_residual_by_boundary: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\figures\model__bayesian_hsgp_additive_residual_by_boundary__2026-06-22.png
- residual_by_progress: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\figures\model__bayesian_hsgp_additive_residual_by_progress__2026-06-22.png
- coverage_width: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\figures\model__bayesian_hsgp_additive_coverage_width__2026-06-22.png
