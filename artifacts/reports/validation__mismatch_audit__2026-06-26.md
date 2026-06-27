# Validation Mismatch Audit - 2026-06-26

## Summary

The Huber + additive Hilbert residual submission remains the public baseline at **14.514**. The direct analog-gated extension scored **14.603** on public LB, worsening the external score by **0.089** even though the bounded residual sample showed a **0.1003 RMSE** local improvement.

## Decision

Pause deeper hierarchy and particle/state-space modeling. The current bottleneck is validation transfer, not model expressiveness. The next notebook should design stricter folds that stress far-analog, clustered/geologic, and high-extrapolation cases before boosted-tree tuning.

## Evidence

- Analog gate submission ref: `53999839`; public LB **14.603**.
- Current public baseline ref: `53960276`; public LB **14.514**.
- Local analog diagnostic was on a bounded residual sample, not the same full local CV used for the original Huber + Hilbert result.
- Shallow hierarchy screens failed the promotion gate with gains below **0.05 RMSE**.

## Next Steps

1. Build `01_stricter_cv_fold_design.ipynb` with cluster, far-analog, and high-extrapolation folds.
2. Re-score Huber + Hilbert and analog-gated variants under those folds.
3. Start LightGBM/XGBoost only after strict folds identify feature families that transfer.
4. Revisit hierarchy or particle models only if strict-fold residuals show coherent well-level or sequential latent structure.
