# XGBoost Inference-Safe Tuning Report

Run id: `2026-06-28__xgboost_inference_safe_tuning`

## Result

- Best legal strict-CV model: `pseudo_shallow_more_trees__safe_typewell`
- Best legal strict-CV RMSE: `13.4178`
- Best legal analog-distance stress RMSE: `13.7533`
- Best legal feature set: `safe_typewell`
- Safe-base RMSE: `13.4399`
- Best proxy-formation RMSE: `13.5503`
- Proxy minus base RMSE: `0.1103`
- Oracle direct-formation diagnostic RMSE: `12.9597`
- Legal minus oracle RMSE gap: `0.4581`

## Decision

Recommendation: `do_not_package_yet`.

Best legal tuning improved local strict CV only marginally, while proxy-formation features underperformed safe base.

## Interpretation

The direct formation feature set remains a diagnostic ceiling only. It confirms formation distance is valuable, but the current fold-safe proxy approach does not recover that signal and should not be packaged as-is.

## Next Move

Move to residual stacking or a stronger hidden-test-safe structural proxy instead of submitting this proxy-formation recipe.
