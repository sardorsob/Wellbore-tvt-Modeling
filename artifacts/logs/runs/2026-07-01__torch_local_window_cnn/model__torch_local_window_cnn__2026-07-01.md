# Torch Local Window CNN

## Result

Best overall candidate: `torch_cnn_direct_delta` with strict-CV RMSE 13.6857, MAE 9.8216, and analog-distance stress RMSE 14.5202.

Best confidence-gated regime mean candidate: `torch_cnn_direct_delta` with RMSE 13.6857.

## Model Structure

Pseudo-top probabilities define soft formation regimes. Raw local GR/typewell window bins are included in the pseudo-top proxy and regime mean. Each regime gets a weighted Ridge MAP mean over legal row/typewell/state/alignment/local-window features. Predictions blend the regime-specific mean with a global safe/typewell mean, using either fixed blend weights or mixture-confidence weights.

## Leakage Notes

Direct formation columns are used only as fold-local labels for pseudo-top proxy training. Regime probabilities and regime means use held-out pseudo-top predictions and hidden-test-safe features.

## Decision

Recommendation: `diagnostic_only`.

## Next Move

If alignment-aware pseudo-top features do not improve both strict and stress metrics, move to a different pseudo-top model class or Bayesian partial pooling rather than more alignment descriptors.
