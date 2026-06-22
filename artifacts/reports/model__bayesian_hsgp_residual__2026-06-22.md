# Bayesian HSGP Residual Baseline

Date: 2026-06-22

## Key Findings
- **best_hsgp_candidate**: Best HSGP candidate is hsgp_bp_m5_autodelta with sampled grouped RMSE 14.8997.
  - Modeling implication: Use this as the first check of whether smooth GP residual structure improves Bayesian point predictions.
- **linear_reference_delta**: Best HSGP RMSE delta versus same-notebook linear reference is -0.0310.
  - Modeling implication: A negative delta supports expanding HSGP; a small or positive delta means revisit coordinates, priors, or validation before making it more complex.
- **bayesian_baseline_delta**: Best HSGP RMSE delta versus prior tuned Bayesian best is 0.0547.
  - Modeling implication: This controls for notebook-to-notebook stochastic and implementation differences.
- **huber_reference_delta**: Best HSGP RMSE delta versus Huber GLM local reference is 0.1069.
  - Modeling implication: If HSGP still trails Huber, use residual diagnostics to decide between richer HSGP and boosted trees.
- **posterior_calibration**: Best HSGP 80% coverage is 0.767 with mean width 31.10.
  - Modeling implication: Coverage and width indicate whether HSGP uncertainty remains usable for later hierarchical extensions.

## Best Candidate
- Candidate: `hsgp_bp_m5_autodelta`
- Guide: `autodelta`
- Coordinate set: `boundary_progress`
- Basis count: 25
- RMSE mean/std: 14.8997 / 0.5109
- 80% coverage/width: 0.767 / 31.10
- HSGP alpha/length: 0.770 / 0.802
- Current best public LB remains GAM: 15.055

## Artifacts
- Fold metrics: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\model__bayesian_hsgp_residual_fold_metrics__2026-06-22.csv
- Summary: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\model__bayesian_hsgp_residual_summary__2026-06-22.csv
- Posterior diagnostics: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\model__bayesian_hsgp_residual_posterior_diagnostics__2026-06-22.csv
- Residual sample: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\model__bayesian_hsgp_residual_sample__2026-06-22.csv
- Findings: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\model__bayesian_hsgp_residual_findings__2026-06-22.csv
- rmse_comparison: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\figures\model__bayesian_hsgp_residual_rmse_comparison__2026-06-22.png
- fold_rmse: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\figures\model__bayesian_hsgp_residual_fold_rmse__2026-06-22.png
- hsgp_residual_by_boundary: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\figures\model__bayesian_hsgp_residual_by_boundary__2026-06-22.png
- residual_by_progress: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\figures\model__bayesian_hsgp_residual_by_progress__2026-06-22.png
- coverage_width: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\figures\model__bayesian_hsgp_residual_coverage_width__2026-06-22.png
