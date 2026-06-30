# Stratigraphic State Feature Screen

Run ID: `2026-06-30__stratigraphic_state_feature_screen`
XGBoost available: `True`

## Best Non-Oracle Candidate

- Model: `huber_delta`
- Feature set: `strat_known_state`
- strict_hybrid_fold RMSE: **14.9922**
- Delta vs local safe XGBoost reference 13.4178: **1.5744**

## Oracle Diagnostic

- Best oracle model: `hist_gradient` / `oracle_direct_formations`
- Oracle strict_hybrid_fold RMSE: **19.9792**

## Interpretation

This screen tests whether known-zone stratigraphic descriptors recover formation-state signal without using hidden-test-unsafe formation columns. Promote only if non-oracle gains beat the existing safe XGBoost reference and improve stress slices.