# XGBoost Strict-CV Screen - 2026-06-27

## Summary

Best default strict-CV candidate: `xgb_pseudohuber_shallow_robust` with `strict_hybrid_fold` RMSE **12.7504** (std **0.7300**). The sampled last-known hold baseline is **15.5734**, so the XGBoost delta vs hold is **-2.8229**.

## Comparison To LightGBM

The best XGBoost candidate is **-0.4859** RMSE versus LightGBM strict-CV reference **13.2363**. The analog-distance stress score is **12.9116**, which is **-0.5337** versus LightGBM stress reference **13.4453**.

## Decision

Decision: **promote XGBoost for Kaggle-safe packaging or inference-safe tuning**.

## Notes

- Target is signed TVT delta from the last known TVT, transformed back to TVT for RMSE.
- Rows are deterministically sampled at up to **160** target rows per well.
- Local formation-distance features are included for strict-CV screening; Kaggle packages must remove them or replace them with hidden-test-safe analogs.
