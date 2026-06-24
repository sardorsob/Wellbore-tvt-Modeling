# Typewell Alignment Feature Screen

Run ID: `2026-06-23__typewell_alignment_feature_screen`

## Result

Best candidate: `analog_direct_s010`

- Same-sample baseline RMSE: **14.7385**
- Best RMSE: **14.6966**
- Delta vs same-sample baseline: **-0.0419**
- Current full local reference: **14.3586**
- Current public reference: **14.514**

## Interpretation

This screen tests research-guided alignment features as residual corrections on the bounded Huber + additive Hilbert residual sample. The result should be interpreted against the same-sample no-alignment baseline, not directly against the full local diagnostic reference.

## Decision

Promote only if the best alignment arm clears the **0.05 RMSE** screen gate and does not worsen fold-level diagnostics. If it fails, use the feature correlations and analog diagnostics to redesign alignment rather than stacking more residual models.
