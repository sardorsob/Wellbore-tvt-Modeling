# Phase 2 Geological Domain EDA

Date: 2026-06-21

## Key Findings
- **evaluation_zone_geometry**: Train eval_frac median 0.740; test eval_frac range 0.674-0.796.
  - Modeling implication: Models must extrapolate through a long trailing block, so endpoint and within-well-position features are likely important.
- **tvt_smoothness**: Median known-zone dTVT/dMD is 0.3200; median last-50 known slope is 0.0000.
  - Modeling implication: Smooth extrapolation assumptions are plausible, but fold-level residuals should check slope changes near the boundary.
- **gr_missingness**: Mean GR missing fraction is 0.294 in train and 0.215 in test.
  - Modeling implication: GR imputation and missingness indicators should be explicit in tree/boosted features.
- **typewell_alignment**: Median horizontal/typewell GR correlation at known TVT positions is 0.812.
  - Modeling implication: Typewell alignment may be useful, but likely noisy enough to require robust features rather than raw direct matching.
- **train_only_formations**: Formation median order monotone for 0.000 of train wells with finite medians.
  - Modeling implication: Formation tops are useful for generative understanding but remain excluded from test-safe feature sets.
- **local_test_overlap**: Local test wells overlapping train wells: ['000d7d20', '00bbac68', '00e12e8b'].
  - Modeling implication: Local test wells appear to be example/diagnostic wells, so use them for submission-shape checks but not as evidence of hidden-test distribution.
- **test_similarity**: Nearest train summary distances for test wells: {'000d7d20': 0.0, '00bbac68': 0.0, '00e12e8b': 0.0}.
  - Modeling implication: Validation should inspect whether held-out folds resemble the three public test wells on summary descriptors.

## Artifacts
- Train summary: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\eda__phase2_train_well_summary__2026-06-21.csv
- Test summary: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\eda__phase2_test_well_summary__2026-06-21.csv
- Typewell summary: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\eda__phase2_typewell_summary__2026-06-21.csv
- Formation summary: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\eda__phase2_formation_summary__2026-06-21.csv
- Typewell alignment: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\eda__phase2_typewell_alignment__2026-06-21.csv
- Test similarity: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\eda__phase2_test_train_similarity__2026-06-21.csv
- Findings table: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\eda__phase2_findings__2026-06-21.csv
- eval_fraction: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\figures\eda__phase2_eval_fraction__2026-06-21.png
- gr_distribution: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\figures\eda__phase2_gr_distribution__2026-06-21.png
- tvt_md_samples: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\figures\eda__phase2_tvt_md_samples__2026-06-21.png
- boundary_extrapolation: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\figures\eda__phase2_boundary_extrapolation__2026-06-21.png
- typewell_alignment: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\figures\eda__phase2_typewell_alignment__2026-06-21.png
- xy_footprint: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\figures\eda__phase2_xy_footprint__2026-06-21.png
- test_similarity: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\figures\eda__phase2_test_similarity__2026-06-21.png
