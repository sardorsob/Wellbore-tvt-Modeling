# Gated Bayesian Stratigraphic-Regime Model

Run id: `2026-06-30__gated_bayesian_regime`

## Result

- Best candidate: `huber_plus_gate_conservative`
- Best strict-hybrid RMSE: **14.1072**
- Best gate candidate: `huber_plus_gate_conservative` at **14.1072**
- Huber mean-only RMSE: **14.5053**
- GAM mean-only RMSE: **25.8177**
- Fixed HSGP shrinkage RMSE: **14.4838**
- Best minus Huber mean-only: **-0.3981**
- Recommendation: `curiosity_package`

## Interpretation

The model treats known-zone state as a gate over residual experts rather than as raw predictive columns. The priors deliberately shrink the gate toward low correction magnitude, so positive movement here is stronger evidence than an unconstrained residual learner.

## Artifacts
- summary: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\model__gated_bayesian_regime_summary__2026-06-30.csv
- fold_metrics: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\model__gated_bayesian_regime_fold_metrics__2026-06-30.csv
- gate_diagnostics: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\model__gated_bayesian_regime_gate_diagnostics__2026-06-30.csv
- gate_weight_sample: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\model__gated_bayesian_regime_gate_weight_sample__2026-06-30.csv
- zone_summary: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\model__gated_bayesian_regime_zone_summary__2026-06-30.csv
- analog_bin_summary: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\model__gated_bayesian_regime_analog_bin_summary__2026-06-30.csv
- well_diagnostics: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\model__gated_bayesian_regime_well_diagnostics__2026-06-30.csv
- residual_sample: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\model__gated_bayesian_regime_residual_sample__2026-06-30.csv
- findings: C:\Users\sardo\OneDrive\Desktop\Classes\Projects\Wellbore-tvt-Modeling\artifacts\tables\model__gated_bayesian_regime_findings__2026-06-30.csv
