# XGBoost Known-State Feature Injection Report

Run id: `2026-06-30__xgboost_known_state_features`

## Result

- Best legal strict-CV model: `pseudo_shallow_more_trees__safe_typewell`
- Best legal strict-CV RMSE: `13.4334`
- Best legal analog-distance stress RMSE: `13.8243`
- Best legal feature set: `safe_typewell`
- Safe-base RMSE: `13.4784`
- Best proxy-formation RMSE: `13.5518`
- Proxy minus base RMSE: `0.0734`
- Oracle direct-formation diagnostic RMSE: `12.9353`
- Legal minus oracle RMSE gap: `0.4981`

## Decision

Recommendation: `do_not_package_yet`.

Best legal tuning did not clear the promotion gate; known-state features require exact-package or Bayesian follow-up only if diagnostics justify it.

## Interpretation

The direct formation feature set remains a diagnostic ceiling only. It confirms formation distance is valuable, but the current fold-safe proxy approach does not recover that signal and should not be packaged as-is.

## Next Move

Move to residual stacking or a stronger hidden-test-safe structural proxy instead of submitting this proxy-formation recipe.
