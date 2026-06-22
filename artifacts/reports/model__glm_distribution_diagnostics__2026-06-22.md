# GLM Distribution Diagnostics

Date: 2026-06-22

## Key Findings
- **best_glm_candidate**: Best GLM/robust candidate is robust_delta_huber with sampled grouped RMSE 14.7928.
  - Modeling implication: Use this result to decide whether GLM likelihood changes beat the current GAM spline baseline.
- **gaussian_delta_reference**: Gaussian Ridge on slope-core delta scored RMSE 14.8398, compared with endpoint hold 15.5782.
  - Modeling implication: A simple continuous signed-delta model remains a strong reference for later Bayesian likelihood choices.
- **classic_glm_validity**: Beta and Dirichlet families are not fit because TVT is a scalar depth response, not a bounded probability or composition vector.
  - Modeling implication: Prefer Gaussian/Student-t/robust or positive continuous families over count/proportion families for Bayesian modeling.
- **gam_reference**: Current submitted GAM spline has local diagnostic RMSE 14.8687 and public LB 15.055.
  - Modeling implication: A GLM candidate should clear this local hurdle before becoming a Kaggle submission candidate.

## Best Candidate
- Candidate: `robust_delta_huber`
- Family: `robust_gaussian`
- Target mode: `delta_boundary`
- RMSE mean/std: 14.7928 / 0.4716

## Artifacts
- Fold metrics: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\model__glm_distribution_fold_metrics__2026-06-22.csv
- Summary: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\model__glm_distribution_summary__2026-06-22.csv
- Candidate table: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\model__glm_distribution_candidates__2026-06-22.csv
- Family validity: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\model__glm_distribution_family_validity__2026-06-22.csv
- Target diagnostics: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\model__glm_distribution_target_diagnostics__2026-06-22.csv
- Residual sample: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\model__glm_distribution_residual_sample__2026-06-22.csv
- Findings: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\model__glm_distribution_findings__2026-06-22.csv
- rmse_comparison: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\figures\model__glm_distribution_rmse_comparison__2026-06-22.png
- fold_rmse: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\figures\model__glm_distribution_fold_rmse__2026-06-22.png
- residuals: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\figures\model__glm_distribution_residuals__2026-06-22.png
- target_distributions: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\figures\model__glm_distribution_target_histograms__2026-06-22.png
