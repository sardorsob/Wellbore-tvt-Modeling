# HSGP Residuals on GAM and Huber Means

Date: 2026-06-22

## Key Findings
- **best_candidate**: Best residual-HSGP sweep RMSE 14.3586.
  - Modeling implication: Use this to decide whether to package a Kaggle notebook.
- **gam_hsgp_delta**: Best GAM+HSGP 46.0657 vs GAM mean-only 46.0910.
  - Modeling implication: Negative means residual HSGP improved the public-leading GAM direction locally.
- **huber_hsgp_delta**: Best Huber+HSGP 14.3586 vs Huber mean-only 14.4172.
  - Modeling implication: Negative means residual HSGP improved the locally strong Huber direction.
- **public_baseline_context**: Current best public LB is fine-tuned GAM 15.055; additive HSGP public LB is 15.255.
  - Modeling implication: Only package if local diagnostics show a plausible gain over GAM mean-only.

## Best Candidate
- Candidate: `huber_plus_hsgp_dps_m8`
- Mean model: `huber`
- Coordinates: `distance_log1p+eval_progress+abs_projected_delta_10`
- RMSE mean/std: 14.3586 / 0.5713
- GAM residual-HSGP delta vs GAM mean-only: -0.0253
- Huber residual-HSGP delta vs Huber mean-only: -0.0585

## Artifacts
- Fold metrics: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\model__hsgp_residual_on_gam_huber_fold_metrics__2026-06-22.csv
- Summary: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\model__hsgp_residual_on_gam_huber_summary__2026-06-22.csv
- Diagnostics: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\model__hsgp_residual_on_gam_huber_diagnostics__2026-06-22.csv
- Residual sample: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\model__hsgp_residual_on_gam_huber_residual_sample__2026-06-22.csv
- Findings: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\model__hsgp_residual_on_gam_huber_findings__2026-06-22.csv
- rmse_comparison: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\figures\model__hsgp_residual_on_gam_huber_rmse_comparison__2026-06-22.png
- fold_rmse: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\figures\model__hsgp_residual_on_gam_huber_fold_rmse__2026-06-22.png
- correction_by_boundary: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\figures\model__hsgp_residual_on_gam_huber_correction_boundary__2026-06-22.png
- residual_by_slope_instability: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\figures\model__hsgp_residual_on_gam_huber_residual_slope_instability__2026-06-22.png
