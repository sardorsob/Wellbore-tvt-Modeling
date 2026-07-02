# Changelog

## 2026-06-22

- Added and ran `notebooks/modeling/07_hierarchical/00_residual_bias_audit.ipynb`; exported well summary, descriptor correlations, fold summary, findings, comparison row, report, and run bundle.
- Recorded A29: residual bias exists after Huber + additive Hilbert residuals, but descriptor signal is modest, so the next empirical Bayes offsets should be conservative.
- Distributed the hierarchical modeling plan across existing context files instead of keeping a standalone roadmap document.
- Added D014 for the staged contextual hierarchy decision and R8/R9 for hierarchy-specific risks.
- Updated `context/STATUS.md` with the hierarchical task ladder, promotion gates, and next modeling steps.
- Added assumption A28 and a planned `07_hierarchical` experiments entry.

## 2026-06-23

- Added self-contained `kaggle/huber_hilbert_analog_submission/` notebook and locally validated a 14,151-row submission CSV.
- Added and ran `notebooks/modeling/08_alignment/06_alignment_hsgp_residual.ipynb`.
- Recorded A43: direct analog transfer carries the alignment signal; learned residual heads do not improve on it.
- Added and ran `notebooks/modeling/08_alignment/05_analog_selection_refinement.ipynb`.
- Recorded A42: log-plus-trajectory nearest-analog selection clears the alignment promotion gate with **0.1003** same-sample RMSE gain.
- Added and ran `notebooks/modeling/08_alignment/04_spatial_residual_correlation.ipynb`.
- Recorded A41: spatial residual correlation is weak under current descriptors, so kriging/spatial GP is deferred.
- Added and ran `notebooks/modeling/08_alignment/03_progress_gated_analog.ipynb`.
- Recorded A40: progress-gated analog correction confirms the late-zone pattern but does not unlock a larger gain.
- Added and ran `notebooks/modeling/08_alignment/02_pseudo_zone_huber.ipynb`.
- Recorded A39: learned pseudo-zone analog heads help late target-zone rows but do not beat fixed nearest-analog shrink.
- Added and ran `notebooks/modeling/08_alignment/01_multi_analog_zone_blend.ipynb`.
- Recorded A38: distance-weighted multi-analog transfer is useful but does not beat the simpler nearest-analog direct-transfer screen.
- Added and ran `notebooks/modeling/08_alignment/00_typewell_alignment_feature_screen.ipynb`.
- Recorded A37: nearest-analog target-zone transfer improved the same residual sample by **0.0419** RMSE, just below the feature-screen promotion gate.
- Updated status and experiments to continue alignment modeling with multi-analog or pseudo-zone analog features instead of submitting the first direct-transfer screen.
- Added and ran `notebooks/modeling/07_hierarchical/01_empirical_bayes_offsets.ipynb`.
- Recorded A30: simple contextual well offsets do not improve enough; best same-sample gain was only **0.0008** RMSE and descriptor-based offsets worsened RMSE.
- Updated status and experiments to mark empirical Bayes offsets as a negative screen, not a promotion candidate.
- Added and ran `notebooks/modeling/07_hierarchical/03_hierarchical_residual_slope.ipynb`.
- Recorded A31: shallow residual-shape corrections do not improve enough; best same-sample gain was only **0.0003** RMSE.
- Updated status and experiments to mark residual-shape hierarchy as a negative screen, not a promotion candidate.
- Added and ran `notebooks/modeling/07_hierarchical/02_contextual_hierarchical_huber.ipynb`.
- Recorded A32: contextual descriptor interactions show weak positive signal (**0.0064** same-sample RMSE gain), but still fail the hierarchy promotion gate.

## [Unreleased]

- Added and ran `notebooks/modeling/06_bayesian/12_bayesian_regime_partial_pooling.ipynb`; exported summary, fold metrics, regime diagnostics, zone/analog-bin summaries, well diagnostics, findings, report, model-comparison row, and run bundle.
- Recorded A70 and D039: shared partial pooling over current pseudo-top probabilities does not improve the direct regime mean, so the next move should revise state inference or model architecture rather than sweep more shrinkage.
- Added and ran `notebooks/modeling/09_validation/02_plateau_failure_slice_audit.ipynb`; exported score summary, hard-well consensus, zone and analog comparisons, findings, report, and run bundle.
- Recorded A71 and D040: plateau errors concentrate in persistent hard wells, so the next notebook should target state-error modeling rather than schedule tuning.
- Added and ran `notebooks/modeling/02_trees/13_xgboost_local_sequence_stack.ipynb`; exported summary, fold metrics, diagnostics, findings, report, model-comparison row, and run bundle.
- Recorded A72 and D041: scalar local-sequence stack features worsened the regime mean, so the next modeling branch should correct pseudo-top/state inference directly.
- Added and ran `notebooks/modeling/08_alignment/13_pseudo_top_state_revision.ipynb`; exported revision-quality diagnostics, model summaries, findings, report, model-comparison row, and run bundle.
- Recorded A73 and D042: top-state correction improves analog stress but hurts strict RMSE, so continue only with lighter or gated state revision.
- Added and ran `notebooks/modeling/08_alignment/14_pseudo_top_state_revision_gated.ipynb`; exported variant-specific revision-quality diagnostics, summaries, findings, report, model-comparison row, and run bundle.
- Recorded A74 and D043: gated top residual correction only trades strict RMSE for stress improvement, so stop correction-schedule tuning.
- Added and ran `notebooks/modeling/02_trees/14_xgboost_raw_window_sequence.ipynb`; exported raw-window feature preview, summaries, findings, report, model-comparison row, and run bundle.
- Recorded A75 and D044: raw local motif bins worsen the current tabular regime-mean recipe, so stop flat local-window additions.
- Added and ran `notebooks/modeling/09_validation/03_oof_candidate_blend_audit.ipynb`; exported OOF blend summaries, fold metrics, diagnostics, findings, report, model-comparison row, and run bundle.
- Recorded A76 and D045: same-family candidate averaging/meta-blending does not beat the residual-stack plateau, so the next branch needs a new state target or representation.
- Added and ran `notebooks/modeling/08_alignment/15_pseudo_top_transition_distance.ipynb`; exported transition-distance summaries, diagnostics, findings, report, model-comparison row, and run bundle.
- Recorded A77 and D046: transition-distance tabular features do not beat the direct regime-mean reference, so stop pseudo-top scalar column engineering.
- Added and ran `notebooks/modeling/05_neural_network/05_torch_local_window_cnn.ipynb`; exported CNN summaries, diagnostics, findings, report, model-comparison row, and run bundle.
- Recorded A78 and D047: direct local-window CNN is worse overall but has different hard-well errors, so future neural use should be gated/residualized only.
- Added and ran `notebooks/modeling/09_validation/04_cnn_regime_gated_specialist.ipynb`; exported CNN/regime blend and gate summaries, diagnostics, findings, report, model-comparison row, and run bundle.
- Recorded A79 and D048: `blend_cnn_20` improves strict RMSE to **12.7816** and stress to **13.1223**, while oracle switching at **10.7716** exposes a new calibration/gating path.
- Added and ran `notebooks/modeling/09_validation/05_cnn_regime_blend_tuning.ipynb`; exported seed-averaged CNN blend-grid summaries, diagnostics, findings, report, model-comparison row, and run bundle.
- Recorded A80 and D049: `blend_cnn_50` improves strict RMSE to **12.6514**, but the optimum is not bracketed because the best strict score sits at the upper grid edge.
- Added and ran `notebooks/modeling/09_validation/06_cnn_regime_blend_wide_grid.ipynb`; exported wide CNN blend-grid summaries, diagnostics, findings, report, model-comparison row, and run bundle.
- Recorded A81 and D050: `blend_cnn_45` brackets the current deployable optimum with strict RMSE **12.6497** and stress RMSE **12.9956**.
- Added and ran `notebooks/modeling/09_validation/07_cnn_regime_learned_weight_gate.ipynb`; exported learned CNN/regime weight summaries, diagnostics, findings, report, model-comparison row, and run bundle.
- Recorded A82 and D051: `xgb_weight_soft_s100` slightly improves the current deployable candidate to **12.6471** strict RMSE and **12.9475** stress RMSE.

### Changed
- Added and ran `notebooks/modeling/06_bayesian/07_gated_bayesian_regime.ipynb`; exported gated Bayesian regime metrics, gate diagnostics, report, model comparison row, and run bundle.
- Added self-contained Kaggle package `kaggle/gated_bayesian_regime_submission/`, locally validated a **14,151** row submission, pushed version 1, and submitted it as `54211837` with score pending.
- Recorded A58 and D027: known-zone state is useful as a Bayesian residual-regime gate even though it failed as raw XGBoost columns.
- Added and ran `notebooks/modeling/02_trees/10_xgboost_known_state_features.ipynb`; exported exact-recipe XGBoost state-feature diagnostics, report, model comparison row, and run bundle.
- Recorded A57 and D026: raw known-zone state features do not improve the current inference-safe XGBoost recipe, so they should move into gated/Bayesian structure rather than a flat tree feature set.
- Added and ran `notebooks/modeling/08_alignment/07_stratigraphic_state_feature_screen.ipynb`; exported summary, fold metrics, stress-slice diagnostics, feature importance, prediction sample, report, and run bundle.
- Added tested reusable stratigraphic preprocessing helpers for known-zone state summaries, sequence resampling, DTW distance, lagged correlation, and alignment uncertainty.
- Recorded A56 and D025: first stratigraphic-state screen found directional known-zone state signal, but did not clear Kaggle packaging gates.
- Recorded a 2026-06-30 domain research reset focused on TVT as a stratigraphic-state prediction problem, including formation-top uncertainty, log/DTW alignment, geosteering-style continuity, and validation-regime mismatch.
- Added decision `D024`, assumptions `A51`-`A55`, risk `R11`, stratigraphic-transfer CV requirements, glossary terms, and a feature backlog for the next alignment/validation notebook.
- Recorded the inference-safe XGBoost Kaggle result (`ref=54129475`, public LB **14.472**) as the new best public submission.
- Recorded the inference-safe LightGBM Kaggle result (`ref=54097453`, public LB **14.687**) as a transfer miss versus Huber + Hilbert **14.514**.
- Recorded the 2026-06-23 research pass across geologic alignment, well-top recommender systems, DTW/log alignment, HSGP/GP kernels, multitask GP, Student-t GP likelihoods, and geostatistical residual modeling; updated assumptions, risks, decisions, and task ladders to pivot from shallow residual hierarchy to fold-safe typewell/analog feature screening.
- Pushed and submitted the self-contained Huber + additive Hilbert residual notebook; public LB scored **14.514**, the current best external result.
- Ran HSGP residual diagnostics on GAM/Huber means and recorded that Huber + additive Hilbert residuals reach **14.3586** local RMSE, while the reconstructed GAM residual path was unstable.
- Pushed and submitted the self-contained additive HSGP Student-t MAP notebook; public LB scored **15.255**, better than Bayesian linear and Huber but worse than fine-tuned GAM.
- Ran additive NumPyro/JAX HSGP residual tuning and recorded that one-dimensional boundary/progress/slope-instability smooth terms beat the first joint HSGP but still trail the Huber GLM local diagnostic.
- Ran the first NumPyro/JAX HSGP residual baseline and recorded that a small boundary/progress HSGP improves its same-notebook linear reference but not the prior tuned Bayesian best.
- Pushed and submitted the self-contained Bayesian linear Student-t MAP notebook; public LB scored **15.512**, worse than Huber GLM and fine-tuned GAM.
- Ran Bayesian posterior predictive diagnostics with targeted heteroskedastic scale models and recorded that boundary/progress scale helps calibration but does not beat the tuned fixed Student-t Bayesian baseline.
- Ran Bayesian linear tuning and recorded that fixed-tail Student-t AutoDelta improves the first Bayesian baseline but still trails the Huber GLM local diagnostic.
- Ran the first NumPyro/JAX Bayesian linear robust baseline and recorded that heteroskedastic Student-t improves uncertainty calibration but not point-RMSE versus Huber/GAM.
- Pushed and submitted the self-contained Huber GLM slope-core notebook; public LB scored **15.464**, worse than the fine-tuned GAM despite better local diagnostic RMSE.
- Ran GLM distribution diagnostics and recorded that robust signed-delta Huber regression beats the tuned GAM local diagnostic, while Beta/Dirichlet are invalid target geometries.
- Pushed and submitted the self-contained fine-tuned GAM spline Kaggle notebook; public LB improved to **15.055**.
- Ran a tuned GAM-style spline sweep and recorded that slope-core splines plateau around **14.8687** sampled target-zone CV RMSE.
- Ran the first GAM-style additive spline baseline and recorded that the best boundary/trajectory/slope additive model beats sampled endpoint hold under target-zone grouped CV.
- Completed Phase 4 validation EDA and recorded the validation protocol for endpoint-hold baselines, fold descriptor balance, model fold metrics, and public leaderboard gap interpretation.
- Completed Phase 3 feature-signal EDA and recorded the resulting assumptions for endpoint hold, unstable direct slope extrapolation, additive feature-family signal, GR availability, and cautious typewell residual use.
- Completed Phase 2 geological/domain EDA and recorded the resulting assumptions for long trailing-zone extrapolation, GR missingness, typewell alignment, train-only formation caveats, and local-test overlap.
- Recorded the ElasticNet Kaggle submission and leaderboard result (`ref=53924934`, public LB `16.091`), including the decision not to promote it.
- Ran the ElasticNet / Huber / median Quantile linear-family sweep and recorded that the best ElasticNet candidate did not beat Ridge locally.
- Ran the first KNN neighbor baseline and recorded the decision to reject vanilla global KNN on the current simple feature space.
- Recorded the Random Forest Kaggle submission and leaderboard result (`ref=53924036`, public LB `15.873`), including the validation mismatch against Ridge.
- Added first tree-baseline experiment results for Random Forest and Extra Trees, including exported metrics, plots, run bundles, and comparison-table rows.
- Logged Phase 0-1 structural EDA results from `notebooks/eda/01_phase0_phase1_data_and_schema.ipynb` into `structure.md`, `ASSUMPTIONS.md`, `DECISIONS.md`, `DATASETS.md`, and `STATUS.md`.
- Generated and ran first baseline modeling notebooks; updated structure, status, experiments, and feature log with run IDs and RMSEs.

### Added
- Alignment-aware pseudo-top regime-mean notebook and artifacts. Local GR/typewell DTW and correlation features improved strict RMSE slightly to **13.0838** but worsened stress to **13.4303**, so the original regime-mean package remains preferred.
- Pseudo-top regime-mean more-rows notebook and artifacts. Increasing the target-row cap to **220** worsened strict and stress RMSE versus the original **120**-row regime mean.
- Pseudo-top probability-scale tuning notebook and artifacts. Original scale **50** remained best on strict and stress RMSE, closing the simple probability-calibration path.
- Stress-gated XGBoost regime-mean stack notebook and artifacts. Gates reduced some stress damage but did not beat the original regime mean on both strict and stress metrics.
- XGBoost regime-mean stack notebook and artifacts. Best candidate `blend_plus_ridge_residual_s15` scored **13.0769** strict RMSE but worsened analog stress to **13.4468**, so it is not package-ready.
- Pseudo-top regime-mean tuning notebook and artifacts. Best candidate `xgb_square_global__regime_ridge_alpha40__confidence_s100` scored **13.0910** strict RMSE and **13.4065** analog stress RMSE, a negligible strict gain with slightly worse stress versus the packaged regime-mean baseline.
- Self-contained pseudo-top regime-mean Kaggle package under `kaggle/pseudo_top_regime_mean_submission/`; local execution completed and passed submission row-order, non-null, and finite-prediction assertions.
- Pseudo-top regime-mean notebook and artifacts. Best candidate `xgb_square_global__regime_ridge_alpha80__confidence_s100` scored **13.0928** strict RMSE and **13.4006** analog stress RMSE, clearing the local packaging gate for the next Kaggle-safe probe.
- Pseudo-top Bayesian mixture-regime notebook and artifacts. Best candidate `xgb_pseudohuber_mean__ridge_alpha650__mixture_conf_s050` scored **13.4437** strict RMSE and **13.7593** analog stress RMSE, the first pseudo-top-family result to edge past the strict safe/typewell reference.
- Pseudo-top interaction XGBoost notebook and artifacts. Best interaction candidate scored **13.4815** strict RMSE and **13.8681** analog stress RMSE, trailing the same-notebook safe/typewell baseline **13.4745**, so this closes the first flat pseudo-top interaction path.
- Pseudo-top gated residual-regime notebook and artifacts. Best candidate `xgb_pseudohuber_mean__ridge_pseudo_state_residual__uncertainty_gate_conservative` scored **13.4694** strict RMSE and **13.7665** analog stress RMSE; it improves same-notebook mean-only slightly but is not package-ready.
- Fold-safe pseudo-top uncertainty screen, artifacts, report, feature log, and model-comparison row. Mean proxy top RMSE was **35.0660** and nearest-top agreement was **0.5804**, but best pseudo-top downstream RMSE **13.5891** trailed the safe/typewell XGBoost result **13.4506**, so no Kaggle package was promoted.
- Self-contained XGBoost Ridge residual-stack Kaggle package under `kaggle/xgboost_residual_stack_submission/`; Kaggle version 2 completed with 14,151 prediction rows and submission `54157692` scored public LB **14.583**, worse than current XGBoost strict inference-safe **14.472**.
- Nested XGBoost residual-stacking notebook, strict/stress nested OOF diagnostics, reports, model-comparison row, and run bundle. Best `ridge_residual_s025` scored **13.2548** strict-hybrid RMSE and **13.5632** analog-distance stress RMSE, clearing both promotion gates for the next Kaggle package.
- Inference-safe XGBoost tuning notebook, proxy-formation diagnostics, legal/oracle feature-set comparison, stress-fold metrics, report, model-comparison row, and run bundle. Best legal `safe_typewell` candidate scored **13.4178** strict-hybrid RMSE and **13.7533** analog-distance stress RMSE; proxy-formation features underperformed safe base, so no Kaggle package was promoted.
- Self-contained inference-safe XGBoost strict-CV Kaggle package under `kaggle/xgboost_strict_submission/`; Kaggle version 1 completed with 14,151 prediction rows and submission `54129475` scored public LB **14.472**.
- XGBoost strict-CV notebook, stress-fold diagnostics, feature importance, residual sample, reports, model-comparison row, and run bundle. Best `xgb_pseudohuber_shallow_robust` scored **12.7504** on `strict_hybrid_fold` and **12.9116** on `analog_distance_fold`.
- Self-contained inference-safe LightGBM strict-CV Kaggle package under `kaggle/lightgbm_strict_submission/`; Kaggle version 1 completed with 14,151 prediction rows and submission `54097453` scored public LB **14.687**.
- LightGBM strict-CV notebook, stress-fold diagnostics, feature importance, reports, model-comparison row, and run bundle. Best `lgbm_medium_l2` scored **13.2363** on `strict_hybrid_fold` and **13.4453** on `analog_distance_fold`.
- Strict CV fold-design notebook, fold-assignment tables, descriptor/stress diagnostics, report, and run bundle. Recommended `strict_hybrid_fold` for first boosted-tree strict CV and `analog_distance_fold` as the strictest diagnostic stress test.
- Validation mismatch audit notebook, exported transfer/analog/decision tables, run bundle, and report after analog-gate public LB **14.603** failed to beat Huber + Hilbert **14.514**.
- Kaggle notebook version 1 submission for Huber + additive Hilbert plus direct analog gate (`ref=53999839`); Kaggle run completed with 14,151 prediction rows and public LB **14.603**.
- Self-contained Huber + additive Hilbert residual Kaggle submission notebook, kernel metadata, and local validation CSV.
- HSGP residual-on-GAM/Huber notebook, residual diagnostics, figures, report, model-comparison row, and run bundle.
- Self-contained additive HSGP Student-t MAP Kaggle submission notebook, kernel metadata, and local validation CSV.
- Additive HSGP residual tuning notebook, posterior diagnostics, residual figures, report, model-comparison row, and run bundle.
- HSGP residual baseline notebook, posterior diagnostics, residual figures, report, model-comparison row, and run bundle.
- Self-contained Bayesian linear Student-t MAP Kaggle submission notebook, kernel metadata, and local validation CSV.
- Bayesian posterior diagnostics and targeted-scale notebook, bin/well diagnostics, calibration figures, report, model-comparison row, and run bundle.
- Bayesian linear tuning notebook, posterior diagnostics, residual/coverage figures, report, model-comparison row, and run bundle.
- Bayesian linear robust notebook, posterior predictive diagnostics, calibration figures, report, model-comparison row, and run bundle.
- Self-contained Huber GLM Kaggle submission notebook, kernel metadata, and local validation CSV.
- GLM distribution diagnostic notebook, family-validity table, target diagnostics, residual/RMSE figures, report, model-comparison row, and run bundle.
- Self-contained fine-tuned GAM spline Kaggle submission notebook, kernel metadata, and local validation CSV.
- Tuned GAM-style spline notebook, candidate/fold/family/target summaries, residual diagnostics, report, model-comparison row, and run bundle.
- GAM-style modeling folder, executable additive spline baseline notebook, fold metrics, summary tables, residual diagnostics, report, model-comparison row, and run bundle.
- Phase 4 EDA notebook execution artifacts: endpoint baseline table, fold assignment and descriptor-balance tables, model fold metrics, CV/leaderboard gap table, descriptor PCA table, diagnostic figures, findings report, and run bundle.
- Phase 3 EDA notebook execution artifacts: bounded candidate feature sample preview, feature signal table, diagnostic feature-family probe tables, visible-test drift table, GR missingness table, diagnostic figures, findings report, and run bundle.
- Phase 2 EDA notebook execution artifacts: train/test/typewell/formation summary tables, diagnostic figures, findings table, markdown report, and run bundle.
- Self-contained Kaggle ElasticNet submission notebook and locally validated submission CSV.
- Scikit-learn linear model factory, focused tests, executable ElasticNet/Huber/Quantile notebook, and reportable run artifacts.
- KNN neighbor model factory, focused tests, executable KNN notebook, and reportable KNN run artifacts.
- Self-contained Kaggle Random Forest submission notebook and kernel metadata under `kaggle/random_forest_submission/`.
- Locally validated Random Forest submission CSV under `artifacts/reports/`.
- Reusable simple tabular feature builder and deterministic row sampler for train/test-safe model baselines.
- Random Forest and Extra Trees model factory utilities plus focused tests.
- Self-contained Kaggle Ridge submission notebook and kernel metadata under `kaggle/linear_ridge_submission/`.
- Locally validated Ridge submission CSV under `artifacts/reports/`.
- First Kaggle code-competition submission log entry for notebook version 2 (`ref=53923555`, public LB `15.905`).
- Agent/project rules for deleting one-off notebook generators and logging Bayesian generative assumptions.
- NumPy OLS/Ridge/Lasso baseline model utilities and regression metric/validation helpers.
- Tests for metrics, validation masks/folds, linear models, and run-bundle writing.
- Run-bundle fallback config serialization when PyYAML is unavailable.
- Local agent rules in `.agents/AGENTS.md`, including the rule to never add AI co-author trailers to commits.
- Ignored local workflow reference path `context/_external/`.
- Kaggle operations docs under `kaggle/`: competition summary, rules summary, CV strategy, feature log, submission log, and leaderboard notes.
- `context/EXPERIMENTS.md` as the reportable run registry.
- Full repository scaffold per `rules_templet.md`
- `requirements.txt` with JAX, NumpPyro, Flax, XGBoost, LightGBM, PyMC, ArviZ
- `src/` package: io, preprocessing, modeling, evaluation, viz, utils
- All config files (baseline, xgboost, lgbm, bayesian_gp, bnn, ensemble)
- CLI scripts: download_data, train, predict, eval
- Context docs: PROJECT_BRIEF, GLOSSARY, DATASETS, INTERFACES, ASSUMPTIONS, DECISIONS, RISKS, STATUS
- Notebook stubs for EDA, modeling (6 approaches), and reporting
- Unit test stubs (loaders, features, metrics, smoke)

## [2026-07-01] - CNN-Regime Calibrated Gate
### Added / Changed / Fixed / Removed
- Added calibrated CNN/regime gate notebook with per-well prediction-context features and fold-heldout switch/weight gates.
- Added calibrated-gate artifacts, report, run bundle, and context notes.

### Results
- Local CV RMSE: **12.6364** (`xgb_switch_p100`)
- Analog-distance stress RMSE: **13.0089**
- Run ID: `2026-07-01__cnn_regime_calibrated_gate`

## [2026-07-01] - CNN-Regime Residual Specialist
### Added / Changed / Fixed / Removed
- Added residual-targeted CNN specialist notebook and run artifacts.
- Corrected promotion logic to exclude leakage-only oracle rows and mark the run diagnostic because it trails current best CNN/regime gates.

### Results
- Local CV RMSE: **12.7353** (`residual_cnn_s085`)
- Analog-distance stress RMSE: **13.1028**
- Run ID: `2026-07-01__cnn_regime_residual_specialist`

## [2026-07-01] - CNN-Regime Direct Stability
### Added / Changed / Fixed / Removed
- Added stronger direct-delta CNN stability notebook with three seed offsets and direct-stability gates.
- Added run artifacts, report, comparison row, and context notes for the new strict-best branch.

### Results
- Local CV RMSE: **12.5179** (`logit_switch_floor20`)
- Analog-distance stress RMSE: **12.9574**
- Stress-balanced candidate: **12.5788** strict / **12.9485** stress (`xgb_weight_soft_s075`)
- Run ID: `2026-07-01__cnn_regime_direct_stability`

## [2026-07-01] - CNN-Regime Dilated Direct
### Added / Changed / Fixed / Removed
- Added dilated direct-delta CNN notebook with three seed offsets and comparable CNN/regime gates.
- Added run artifacts, report, comparison row, and context notes for the new best local branch.

### Results
- Local CV RMSE: **12.4888** (`logit_switch_floor20`)
- Analog-distance stress RMSE: **12.8465**
- Oracle diagnostic RMSE: **10.6114**
- Run ID: `2026-07-01__cnn_regime_dilated_direct`

## [2026-07-01] - CNN-Regime Dilated Pooling
### Added / Changed / Fixed / Removed
- Added dilated CNN avg+max pooling diagnostic notebook and run artifacts.
- Marked the run diagnostic because it trails the active average-pooled dilated branch.

### Results
- Local CV RMSE: **12.5051** (`xgb_weight_medium_s100`)
- Analog-distance stress RMSE: **12.9361**
- Oracle diagnostic RMSE: **10.6897**
- Run ID: `2026-07-01__cnn_regime_dilated_pooling`

## [2026-07-01] - CNN-Regime Dilated Attention
### Added / Changed / Fixed / Removed
- Added dilated CNN average-plus-attention pooling diagnostic notebook and run artifacts.
- Marked the run diagnostic because it trails the active average-pooled dilated branch.

### Results
- Local CV RMSE: **12.5366** (`logit_switch_floor20`)
- Analog-distance stress RMSE: **12.9839**
- Oracle diagnostic RMSE: **10.7206**
- Run ID: `2026-07-01__cnn_regime_dilated_attention`

## [2026-07-01] - CNN-Regime Dilated Well Bias
### Added / Changed / Fixed / Removed
- Added nested well-level residual bias calibration on top of the active dilated direct-CNN branch.
- Added run artifacts, report, comparison row, and context notes for the new best local branch.

### Results
- Local CV RMSE: **12.4568** (`logit_switch_floor20__well_bias_s25`)
- Analog-distance stress RMSE: **12.7684**
- Run ID: `2026-07-01__cnn_regime_dilated_well_bias`

## [2026-07-01] - CNN-Regime Dilated Well-Bias Tuning
### Added / Changed / Fixed / Removed
- Added conservative well-bias shrinkage tuning around the active dilated direct-CNN branch.
- Restricted the tuning candidate grid after the first failed aggregation attempt to keep notebook execution bounded.
- Added run artifacts, report, comparison row, and context notes for the new best local branch.

### Results
- Local CV RMSE: **12.4549** (`logit_switch_floor20__well_bias_h015_s30`)
- Analog-distance stress RMSE: **12.7649**
- Run ID: `2026-07-01__cnn_regime_dilated_well_bias_tuning`

## [2026-07-01] - CNN-Regime Neural Gate
### Added / Changed / Fixed / Removed
- Added compact neural blend-gate audit on top of the active dilated direct-CNN/regime branch.
- Added conservative well-bias corrections for selected neural/tree gate candidates.
- Recorded the run as diagnostic because the neural gates did not beat the active tuned logit-switch well-bias candidate.

### Results
- Local CV RMSE: **12.4549** (`logit_switch_floor20__well_bias_h015_s30`, unchanged active best)
- Best neural-derived corrected RMSE: **12.4886** (`neural_gate_p003_floor25__well_bias_h015_s30`)
- Analog-distance stress RMSE: **12.7649** for active best; **12.7519** stress for the best stress-friendlier tree-gated correction with worse strict RMSE
- Run ID: `2026-07-01__cnn_regime_neural_gate`

## [2026-07-01] - CNN-Regime Multichannel Dilated CNN
### Added / Changed / Fixed / Removed
- Added multichannel dilated CNN audit with first- and second-derivative local-window channels.
- Kept the same fold-safe regime anchor, candidate gates, and conservative well-bias correction for comparison.
- Recorded the run as diagnostic because derivative channels worsened strict, stress, and oracle RMSE.

### Results
- Local CV RMSE: **12.4984** (`logit_switch_floor20__well_bias_h015_s30`)
- Analog-distance stress RMSE: **12.8224**
- Oracle diagnostic RMSE: **10.7443**
- Run ID: `2026-07-01__cnn_regime_multichannel_dilated`

## [2026-07-01] - CNN-Regime Well Trajectory Context
### Added / Changed / Fixed / Removed
- Added hidden-test-safe within-well trajectory context features to the active pseudo-top/regime/CNN stack.
- Abandoned the uncommitted GRU sequence attempt after two timed-out runs and removed its partial artifacts.
- Added run artifacts, report, comparison row, and context notes for the new active local best.

### Results
- Local CV RMSE: **11.6868** (`logit_switch_floor20__well_bias_h015_s30`)
- Analog-distance stress RMSE: **11.9598**
- Oracle diagnostic RMSE: **9.8460**
- Run ID: `2026-07-01__cnn_regime_well_trajectory_context`

## [2026-07-01] - CNN-Regime Trajectory Calibration
### Added / Changed / Fixed / Removed
- Added focused calibration notebook around the successful trajectory-context representation.
- Expanded logit-switch floors, XGBoost switch powers, and conservative well-bias shrinkage/regularization.
- Added run artifacts, report, comparison row, and context notes for the new active local best.

### Results
- Local CV RMSE: **11.6724** (`logit_switch_floor25__well_bias_h005_s35`)
- Analog-distance stress RMSE: **11.9379**
- Stress-best deployable row: **11.7019** strict / **11.9158** stress (`xgb_weight_soft_s100__well_bias_h005_s35`)
- Run ID: `2026-07-01__cnn_regime_trajectory_calibration`

---

## Milestone format

```text
## [YYYY-MM-DD] - <Short milestone name>
### Added / Changed / Fixed / Removed
- ...

### Results
- Local CV RMSE: ...
- Kaggle LB RMSE: ...
- Run ID: ...
```
