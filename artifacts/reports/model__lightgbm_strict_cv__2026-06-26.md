# LightGBM Strict-CV Screen - 2026-06-26

## Summary

Best default strict-CV candidate: `lgbm_medium_l2` with `strict_hybrid_fold` RMSE **13.2363** (std **0.8224**). The sampled last-known hold baseline is **15.5734**, so the LightGBM delta vs hold is **-2.3371**.

## Analog-Distance Stress Test

The same `lgbm_medium_l2` configuration scored **13.4453** RMSE on `analog_distance_fold`, with last-known hold RMSE **15.5833** and delta vs hold **-2.1380**.

## Decision

Decision: **prepare Kaggle-safe LightGBM package after one reproducibility/tuning pass**. The result is strong enough to keep pursuing boosted trees, but Kaggle packaging should wait until one reproducibility/tuning pass confirms this was not created by a sampled-row artifact.

## Notes

- Target is signed TVT delta from the last known TVT, transformed back to TVT for RMSE.
- Rows are deterministically sampled at up to **160** target rows per well.
- The strongest features include extrapolation stress, last-known TVT, formation distance, GR summaries, typewell errors, and selected analog distance.
