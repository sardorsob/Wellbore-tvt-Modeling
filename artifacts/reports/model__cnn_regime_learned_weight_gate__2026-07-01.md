# CNN-Regime Learned Weight Gate

## Result

Best deployable candidate: `xgb_weight_soft_s100` with strict-CV RMSE 12.6471, MAE 8.9554, and analog-distance stress RMSE 12.9475.

Leakage-only upper bound: `oracle_continuous_weight` with strict-CV RMSE 10.8284. This uses validation targets to define row-wise optimal weights and is not packageable.

## Model Structure

The notebook recomputes the direct pseudo-top regime-mean anchor and two-seed local-window CNN specialist. Static blends are compared with fold-heldout XGBoost models that predict a clipped row-wise CNN blend weight from hidden-test-safe state, confidence, disagreement, and raw-window diagnostics.

## Leakage Notes

Direct formation columns are used only as fold-local labels for pseudo-top proxy training. Learned weight models train on other folds only. `oracle_continuous_weight` and `oracle_row_switch` use validation targets and are diagnostic upper bounds only.

## Decision

Recommendation: `package_cnn_regime_learned_weight_gate`.

## Next Move

Use `xgb_weight_soft_s100` as the current local package candidate or run a package-focused stability check before Kaggle submission.
