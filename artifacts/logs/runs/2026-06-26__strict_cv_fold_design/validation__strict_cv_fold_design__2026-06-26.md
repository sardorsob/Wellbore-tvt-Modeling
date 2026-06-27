# Strict CV Fold Design - 2026-06-26

## Summary

This notebook creates reusable well-level fold assignments that deliberately stress descriptor clusters, far analogs, and high-extrapolation wells. The recommended default for the next modeling notebooks is `strict_hybrid_fold` because it balances row count while distributing high-stress wells across folds.

## Fold Designs

- `descriptor_cluster_fold`: holds out descriptor clusters built from known-zone geometry, GR summaries, trajectory span, and target-zone shape descriptors.
- `analog_distance_fold`: buckets wells by selected analog distance to stress weak analog transfer.
- `extrapolation_fold`: buckets wells by target-zone length/share/span/gap stress.
- `strict_hybrid_fold`: greedily balances rows while spreading combined analog and extrapolation stress.

## Recommendation

Use `strict_hybrid_fold` as the first strict-CV column for LightGBM/XGBoost. Treat `descriptor_cluster_fold`, `analog_distance_fold`, and `extrapolation_fold` as follow-up diagnostic stress tests for any candidate that beats the current Huber + Hilbert baseline locally.

## Key Diagnostics

- Hybrid row imbalance: **1.000**.
- Hybrid well imbalance: **1.024**.
- Hybrid mean nearest-training descriptor distance: **2.436**.
- Strictest diagnostic fold design by score: `analog_distance_fold`.

## Next Steps

1. Re-score Huber + Hilbert and analog-gated variants with `strict_hybrid_fold` if feasible.
2. Start `02_trees/06_lightgbm_strict_cv.ipynb` using this fold assignment table.
3. Promote a boosted-tree notebook only if it improves strict CV by at least **0.10 RMSE** or explains the analog public/local mismatch.
