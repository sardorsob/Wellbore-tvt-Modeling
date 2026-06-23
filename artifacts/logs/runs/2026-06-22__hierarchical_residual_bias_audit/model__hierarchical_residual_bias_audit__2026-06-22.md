# Hierarchical Residual Bias Audit

Date: 2026-06-22

## Purpose

Audit whether the current Huber + additive Hilbert residual model leaves well-level residual structure that could justify contextual partial pooling.

## Input

- Residual sample: `C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\model__hsgp_residual_on_gam_huber_residual_sample__2026-06-22.csv`
- Candidate audited: `huber_plus_hsgp_dps_m8`
- Sampled residual rows: 5,400
- Wells: 773

## Key Metrics

- Current local diagnostic reference: **14.3586** RMSE
- Current public LB reference: **14.514** RMSE
- Median absolute well mean residual: **6.890** TVT units
- 90th percentile absolute well mean residual: **18.929** TVT units
- Max absolute well mean residual: **56.454** TVT units

## Strongest Descriptor Signals

| descriptor | target | n_wells | spearman |
| --- | --- | --- | --- |
| max_distance_from_boundary_md | mae | 773 | 0.1745 |
| max_distance_log1p | mae | 773 | 0.1745 |
| max_distance_from_boundary_md | rmse | 773 | 0.1704 |
| max_distance_log1p | rmse | 773 | 0.1704 |
| xy_span | mae | 773 | 0.1661 |

## Fold Summary

| fold | n_rows | n_wells | mean_residual | median_residual | residual_std | mae | p90_abs_residual | rmse | mean_residual_correction | residual_correction_std | abs_mean_residual |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1800 | 258 | -0.0516 | -0.8610 | 14.1800 | 9.9784 | 22.3290 | 14.1762 | 0.1543 | 1.3947 | 0.0516 |
| 2 | 1800 | 258 | 0.4079 | 0.0203 | 15.0150 | 10.3618 | 23.3110 | 15.0164 | 0.0425 | 1.3616 | 0.4079 |
| 3 | 1800 | 257 | 0.4500 | 0.6077 | 15.0205 | 10.8990 | 23.3852 | 15.0231 | -0.0331 | 1.4928 | 0.4500 |

## Decision

Continue to `01_empirical_bayes_offsets.ipynb`, but keep the first correction small and shrinkage-controlled. The audit shows enough well-level signed residual variation to test contextual offsets, while the descriptor correlations should guide which known-zone and geometry summaries enter first.

## Artifacts

- Well summary: `C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\model__hierarchical_residual_bias_well_summary__2026-06-22.csv`
- Descriptor correlations: `C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\model__hierarchical_residual_bias_descriptor_correlations__2026-06-22.csv`
- Fold summary: `C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\model__hierarchical_residual_bias_fold_summary__2026-06-22.csv`
- Findings: `C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\model__hierarchical_residual_bias_findings__2026-06-22.csv`
- Comparison row: `C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\model__hierarchical_residual_bias_comparison__2026-06-22.csv`
