# Phase 4 Validation EDA

Date: 2026-06-21

## Key Findings
- **endpoint_baseline**: Full-row endpoint-hold RMSE is 15.9099; direct last-50 slope projection RMSE is 112.4817.
  - Modeling implication: Endpoint hold should be included as a validation baseline. Direct recent-slope extrapolation should not be used as a standalone model.
- **fold_balance**: Current sorted folds have eval-row CV 0.003 and endpoint-RMSE range 1.466; balanced descriptor folds have eval-row CV 0.001 and endpoint-RMSE range 2.428.
  - Modeling implication: Future CV should use descriptor-balanced grouped folds or at least report descriptor balance; sorted grouped folds are acceptable only as a baseline diagnostic.
- **local_vs_public_mismatch**: Random Forest public LB beats Ridge by 0.032, while local CV ranks Random Forest worse than Ridge by 0.300.
  - Modeling implication: Treat the Random Forest public gain as tentative until the validation split is refined and repeated on richer features.
- **leaderboard_gap**: Largest absolute public-minus-CV gap is 0.300 for elasticnet_alpha_0p001_l1_0p2.
  - Modeling implication: Avoid leaderboard-only tuning; use public LB as a coarse external check after local diagnostics agree.
- **recommended_protocol**: Use grouped well-level folds, descriptor balance reporting, endpoint-hold baseline, and model fold-level error tables for each reportable modeling notebook.
  - Modeling implication: The next LightGBM/XGBoost pass should compare against Ridge, Random Forest, endpoint hold, and balanced-fold diagnostics before Kaggle submission.

## Artifacts
- Endpoint baselines by well: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\eda__phase4_endpoint_baselines_by_well__2026-06-21.csv
- Fold assignments: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\eda__phase4_fold_assignments__2026-06-21.csv
- Fold descriptor summary: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\eda__phase4_fold_descriptor_summary__2026-06-21.csv
- Fold variability: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\eda__phase4_fold_variability__2026-06-21.csv
- Model fold metrics: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\eda__phase4_model_fold_metrics__2026-06-21.csv
- CV / leaderboard gap: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\eda__phase4_cv_leaderboard_gap__2026-06-21.csv
- Descriptor PCA table: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\eda__phase4_well_descriptor_pca__2026-06-21.csv
- Findings table: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\eda__phase4_findings__2026-06-21.csv
- model_fold_rmse: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\figures\eda__phase4_model_fold_rmse__2026-06-21.png
- fold_balance: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\figures\eda__phase4_fold_balance__2026-06-21.png
- cv_vs_leaderboard: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\figures\eda__phase4_cv_vs_leaderboard__2026-06-21.png
- endpoint_vs_slope: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\figures\eda__phase4_endpoint_vs_slope__2026-06-21.png
- descriptor_pca: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\figures\eda__phase4_descriptor_pca__2026-06-21.png
