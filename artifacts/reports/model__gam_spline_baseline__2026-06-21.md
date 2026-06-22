# GAM-Style Additive Spline Baseline

Date: 2026-06-21

## Key Findings
- **best_gam_variant**: Best GAM-style variant is boundary_trajectory_slopes with sampled grouped RMSE 14.8719.
  - Modeling implication: Use this as the first additive-smooth baseline before boosted-tree tuning.
- **endpoint_comparison**: Best GAM-style RMSE 14.8719 vs sampled endpoint-hold RMSE 15.5782.
  - Modeling implication: A useful model must clear endpoint hold under the active validation protocol.
- **slope_projection_check**: Direct last-50 slope projection remains unstable with sampled RMSE 104.6157.
  - Modeling implication: Slope should remain a feature, not a standalone extrapolator.
- **historical_reference**: Reference local CV: Ridge 15.7660, Random Forest 16.0655; this GAM run uses sampled target-zone CV, so compare directionally, not as a final leaderboard proxy.
  - Modeling implication: If GAM is competitive, create a self-contained Kaggle notebook and rerun against the hidden-test workflow.

## Artifacts
- Fold metrics: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\model__gam_spline_fold_metrics__2026-06-21.csv
- Summary: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\model__gam_spline_summary__2026-06-21.csv
- Variant table: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\model__gam_spline_variants__2026-06-21.csv
- Residual sample: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\model__gam_spline_residual_sample__2026-06-21.csv
- Findings: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\model__gam_spline_findings__2026-06-21.csv
- rmse_comparison: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\figures\model__gam_spline_rmse_comparison__2026-06-21.png
- fold_rmse: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\figures\model__gam_spline_fold_rmse__2026-06-21.png
- best_residuals: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\figures\model__gam_spline_best_residuals__2026-06-21.png
- residual_by_boundary: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\figures\model__gam_spline_residual_by_boundary__2026-06-21.png
