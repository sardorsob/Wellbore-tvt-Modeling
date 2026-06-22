# GAM-Style Spline Tuning

Date: 2026-06-21

## Key Findings
- **best_tuned_gam**: Best tuned candidate is core_fine__delta_boundary__k7__d3__a60 with sampled grouped RMSE 14.8687.
  - Modeling implication: Use this candidate as the current GAM submission candidate if fold behavior is stable.
- **phase1_comparison**: Phase 2 best RMSE 14.8687 vs Phase 1 best boundary_trajectory_slopes RMSE 14.8719.
  - Modeling implication: Submit only if the tuned model improves materially and does not concentrate gains in one fold.
- **endpoint_comparison**: Best tuned GAM RMSE 14.8687 vs endpoint-hold RMSE 15.5782.
  - Modeling implication: Endpoint hold remains the minimum useful local hurdle for this diagnostic protocol.
- **target_mode_signal**: Best target mode by RMSE is delta_boundary; feature family leader is slopes_core.
  - Modeling implication: Carry these choices into a self-contained Kaggle notebook before trying more complex GAM libraries.

## Best Candidate
- Candidate: `core_fine__delta_boundary__k7__d3__a60`
- Feature family: `slopes_core`
- Target mode: `delta_boundary`
- Spline: knots=7, degree=3, alpha=60
- RMSE mean/std: 14.8687 / 0.5012

## Artifacts
- Fold metrics: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\model__gam_spline_tuning_fold_metrics__2026-06-21.csv
- Summary: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\model__gam_spline_tuning_summary__2026-06-21.csv
- Candidate table: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\model__gam_spline_tuning_candidates__2026-06-21.csv
- Residual sample: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\model__gam_spline_tuning_residual_sample__2026-06-21.csv
- Findings: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\model__gam_spline_tuning_findings__2026-06-21.csv
- Feature-family summary: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\model__gam_spline_tuning_family_summary__2026-06-21.csv
- Target summary: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\model__gam_spline_tuning_target_summary__2026-06-21.csv
- rmse_comparison: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\figures\model__gam_spline_tuning_rmse_comparison__2026-06-21.png
- fold_rmse: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\figures\model__gam_spline_tuning_fold_rmse__2026-06-21.png
- best_residuals: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\figures\model__gam_spline_tuning_best_residuals__2026-06-21.png
- residual_by_boundary: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\figures\model__gam_spline_tuning_residual_by_boundary__2026-06-21.png
