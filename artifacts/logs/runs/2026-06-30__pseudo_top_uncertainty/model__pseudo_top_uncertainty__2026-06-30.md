# Pseudo-Top Uncertainty Feature Screen

## Result

Best legal strict-CV candidate: `xgb_square_regularized__safe_typewell` with RMSE 13.4506, MAE 9.5273, and analog-distance stress RMSE 13.8870.

Best pseudo-top feature candidate: `xgb_pseudohuber_known_state_recipe__safe_pseudo_top_uncertainty` with strict-CV RMSE 13.5891.

## Pseudo-Top Quality

Nearest pseudo-top agreement with direct nearest formation labels was 0.5804 under `strict_hybrid_fold` and 0.5823 under `analog_distance_fold`.

Mean top-elevation RMSE across formation proxies was 35.0660.

## Downstream RMSE

Pseudo-top features did not use direct formation columns. Oracle diagnostics remain separate and should only be read as a recoverable-signal ceiling.

## Leakage Notes

Formation top columns were used only as fold-local labels and oracle diagnostics. For each validation fold, pseudo-top models were trained on the complementary wells and predicted held-out rows. The exported legal pseudo features are therefore hidden-test-compatible if the same fitting recipe is rerun inside a Kaggle notebook.

## Decision

Recommendation: `iterate_before_package`.

## Next Move

Use this screen to decide whether to integrate pseudo-top uncertainty into the gated Bayesian regime, because the gate can use margin and uncertainty as shrinkage controls rather than raw predictors.
