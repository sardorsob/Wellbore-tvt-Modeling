# Phase 3 Feature Signal EDA

Date: 2026-06-21

## Key Findings
- **boundary_endpoint_signal**: Endpoint hold diagnostic RMSE is 15.5815; the best feature-family probe is all_candidate with RMSE 14.6236.
  - Modeling implication: Boundary and endpoint features should be first-class inputs for boosted trees and Bayesian generative models.
- **recent_slope_projection**: Last-50 slope projection diagnostic RMSE is 104.5322.
  - Modeling implication: Recent slopes are useful context, but should be combined with endpoint, position, and other well-relative features rather than used as a standalone extrapolator.
- **feature_screening**: Top absolute Spearman feature for target-zone delta is tvt_slope_last_10 (0.199); top mutual-information feature is z_slope_last_100 (0.514).
  - Modeling implication: Use the ranked feature table to choose candidate groups for the richer feature builder instead of guessing from raw columns.
- **gr_availability_signal**: Median target-zone delta gap between GR-missing and GR-present sampled rows is 0.3100.
  - Modeling implication: GR missingness indicators and rolling availability summaries should remain in tree/boosted feature candidates.
- **typewell_signal**: Typewell residual feature gr_minus_tw_proxy has Spearman delta correlation 0.019.
  - Modeling implication: Typewell residuals should be tested as robust, smoothed features rather than raw matching signals.
- **visible_test_drift_caveat**: Largest visible train/test descriptor KS statistic is 0.351 for typewell_gr_corr, but the visible test wells overlap train wells.
  - Modeling implication: Use visible-test drift diagnostics only for notebook robustness and submission-shape checks; Phase 4 should evaluate held-out fold similarity instead.

## Artifacts
- Candidate feature sample: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\eda__phase3_candidate_feature_sample_preview__2026-06-21.csv
- Candidate feature signal: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\eda__phase3_candidate_feature_signal__2026-06-21.csv
- Feature group probe metrics: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\eda__phase3_feature_group_probe_metrics__2026-06-21.csv
- Feature group probe summary: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\eda__phase3_feature_group_probe_summary__2026-06-21.csv
- Visible test drift: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\eda__phase3_visible_test_drift__2026-06-21.csv
- GR missingness signal: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\eda__phase3_gr_missing_signal__2026-06-21.csv
- Findings table: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\eda__phase3_findings__2026-06-21.csv
- feature_signal_rank: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\figures\eda__phase3_feature_signal_rank__2026-06-21.png
- group_probe_rmse: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\figures\eda__phase3_group_probe_rmse__2026-06-21.png
- boundary_delta_curve: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\figures\eda__phase3_boundary_delta_curve__2026-06-21.png
- gr_missing_signal: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\figures\eda__phase3_gr_missing_signal__2026-06-21.png
- typewell_residual_signal: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\figures\eda__phase3_typewell_residual_signal__2026-06-21.png
