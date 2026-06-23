# Hierarchical Modeling Roadmap

**Date:** 2026-06-22
**Status:** Proposed next modeling campaign
**Current external baseline:** Huber + additive Hilbert residual, public LB **14.514**

---

## Goal

Build a staged hierarchical modeling track that can improve the current Huber + additive Hilbert residual model without jumping directly into a hard-to-debug full hierarchical HSGP.

The immediate objective is not to create the most complex model first. The objective is to learn whether persistent well-level residual structure is predictable from test-safe context, then add hierarchy only where it transfers under grouped validation.

## Current Evidence

- The best public model is `wellbore-huber-hsgp-residual`, public LB **14.514**.
- The best local diagnostic model is `huber_plus_hsgp_dps_m8`, sampled grouped RMSE **14.3586**.
- Additive HSGP terms help more than simply increasing joint HSGP basis size.
- The strongest residual geometry so far uses `distance_log1p`, `eval_progress`, and `abs_projected_delta_10`.
- Huber signed-delta mean structure transferred better than Bayesian-linear mean structure.
- Pure local micro-tuning is producing small gains; the next useful gain likely needs better well-level structure, better validation, or a new feature view.

## Research Notes

- Riutort-Mayol et al. show HSGP is a reduced-rank GP approximation and emphasize selecting enough basis functions and boundary factor, then diagnosing whether the approximation covers the target domain well.
- PyMC's HSGP implementation frames HSGP as a latent GP component that can be combined with non-Gaussian likelihoods, which supports our Student-t / robust residual framing.
- Stan's regression guide reinforces robust likelihoods, hierarchical regression, and posterior predictive checks as first-class workflow pieces rather than afterthoughts.
- GPyTorch and GPflow document multitask/coregionalized GP patterns. These are appealing for well-level structure, but direct well-ID task effects are risky because hidden test wells may be unseen.
- Geostatistical modeling practice treats spatial/reservoir prediction as uncertainty-bearing and often uses blind-well style validation. Our grouped-well CV is the closest local analogue and should remain the main guardrail.

## Modeling Principle

Only use hierarchy that can generalize to hidden wells.

Direct random effects keyed by train well IDs can improve local fits but are not Kaggle-safe unless the hidden test wells are represented by observable descriptors. The hierarchy should therefore be contextual:

- infer partial-pooling structure from train wells,
- predict held-out or hidden well effects from known-zone summaries and trajectory descriptors,
- shrink uncertain well effects toward the global mean,
- keep row-level HSGP residual terms as additive corrections, not as a replacement for the robust mean.

## Approach Options

### Recommended: Contextual Hierarchy Ladder

Start with residual bias audits, then build empirical Bayes and Bayesian partial-pooling models where well-level intercepts/slopes are predicted from test-safe well descriptors.

**Pros:** Most aligned with the current evidence; lower implementation risk; directly tests whether hierarchy transfers to unseen wells.

**Cons:** Requires careful descriptor engineering and validation before the model becomes more expressive.

### Alternative: Full Hierarchical HSGP Immediately

Create a large model with well-level random effects, additive HSGP terms, and hierarchical priors over length scales/amplitudes.

**Pros:** Matches the long-term Bayesian idea and may capture richer structure.

**Cons:** Easy to overfit, slow, and difficult to diagnose. It can also create a false local gain if well-ID effects do not transfer to hidden wells.

### Alternative: Boosted Tree Detour

Try LightGBM/XGBoost on the current slope-core and residual features before hierarchy.

**Pros:** Fast, strong tabular baseline, useful feature-importance audit.

**Cons:** Does not directly answer the current Bayesian/hierarchy question, and previous tree baselines were weak without richer local features.

## Folder Plan

Create a separate modeling folder:

`notebooks/modeling/07_hierarchical/`

This keeps the hierarchical track distinct from `06_bayesian/` while still allowing later notebooks to reuse Huber, Student-t, and HSGP lessons.

## Notebook Roadmap

### 00 Residual Bias Audit

**Notebook:** `notebooks/modeling/07_hierarchical/00_residual_bias_audit.ipynb`

**Purpose:** Diagnose whether the current best model has systematic residual structure by well, fold, boundary distance, progress, slope instability, GR availability, and trajectory descriptors.

**Artifacts:**

- `artifacts/tables/model__hierarchical_residual_bias_well_summary__YYYY-MM-DD.csv`
- `artifacts/tables/model__hierarchical_residual_bias_descriptor_correlations__YYYY-MM-DD.csv`
- `artifacts/tables/model__hierarchical_residual_bias_fold_summary__YYYY-MM-DD.csv`
- `artifacts/reports/model__hierarchical_residual_bias_audit__YYYY-MM-DD.md`

**Decision gate:** Continue to hierarchy only if well-level residual bias or slope bias is material and partly explainable by test-safe descriptors.

### 01 Empirical Bayes Offsets

**Notebook:** `notebooks/modeling/07_hierarchical/01_empirical_bayes_offsets.ipynb`

**Purpose:** Add a small partial-pooling correction on top of the current Huber + Hilbert residual prediction.

**Candidate corrections:**

- global shrinkage intercept,
- well-level intercept predicted from known-zone descriptors,
- well-level intercept plus `eval_progress` slope,
- ridge/Huber empirical Bayes correction using out-of-fold residual targets.

**Decision gate:** Promote only if grouped CV improves by at least **0.05 RMSE** and no fold gets materially worse.

### 02 Contextual Hierarchical Huber

**Notebook:** `notebooks/modeling/07_hierarchical/02_contextual_hierarchical_huber.ipynb`

**Purpose:** Fold contextual well effects into the robust signed-delta Huber mean before residual HSGP.

**Candidate hierarchy:**

- global Huber slope-core mean,
- descriptor-predicted well intercept,
- descriptor-predicted well progress slope,
- shrinkage controlled by well-level residual variance.

**Decision gate:** Continue only if the hierarchy improves Huber mean local RMSE and does not erase the residual-HSGP gain.

### 03 Hierarchical Residual Slope

**Notebook:** `notebooks/modeling/07_hierarchical/03_hierarchical_residual_slope.ipynb`

**Purpose:** Model the residual trend over `eval_progress` and `distance_log1p` as a partially pooled well-level correction.

**Candidate hierarchy:**

- global residual trend,
- contextual well residual slope,
- Student-t residual likelihood,
- conservative shrinkage toward zero correction.

**Decision gate:** Keep only if residual slope hierarchy beats the simpler empirical Bayes offset model.

### 04 Contextual Hierarchy Plus Hilbert Residual

**Notebook:** `notebooks/modeling/07_hierarchical/04_contextual_hierarchy_hilbert_residual.ipynb`

**Purpose:** Combine the best contextual hierarchy with the current additive Hilbert residual geometry.

**Candidate structure:**

- robust Huber signed-delta mean,
- contextual well intercept/slope correction,
- additive HSGP residual terms over `distance_log1p`, `eval_progress`, and `abs_projected_delta_10`,
- Student-t or Huber-style residual objective.

**Decision gate:** Package for Kaggle only if local CV improves by at least **0.10 RMSE** or the diagnostic shows a clear public-generalization reason.

### 05 Multitask GP Screen

**Notebook:** `notebooks/modeling/07_hierarchical/05_multitask_gp_screen.ipynb`

**Purpose:** Test whether a low-rank task/coregionalization view is worth a full implementation.

**Candidate structure:**

- tasks represented by well descriptors, not raw well IDs,
- low-rank latent well embedding,
- additive row-level GP/HSGP residual,
- grouped-well validation with no hidden ID leakage.

**Decision gate:** Continue only if the screen beats contextual hierarchy or reveals a reusable descriptor embedding.

### 06 Full Hierarchical HSGP

**Notebook:** `notebooks/modeling/07_hierarchical/06_hierarchical_hsgp_full.ipynb`

**Purpose:** Build the full hierarchical Hilbert-space GP only after the earlier notebooks identify transferable well-level structure.

**Candidate structure:**

- contextual partial pooling over well intercepts/slopes,
- hierarchical priors over HSGP amplitudes and length scales,
- additive HSGP basis terms,
- Student-t likelihood,
- posterior predictive diagnostics and RMSE scoring.

**Decision gate:** Kaggle submit only if it beats the current public baseline locally and passes residual diagnostics.

## Task Checklist

- [ ] Create `notebooks/modeling/07_hierarchical/`.
- [ ] Run `00_residual_bias_audit.ipynb` and export the four required artifacts.
- [ ] Update `context/ASSUMPTIONS.md` with residual-bias findings.
- [ ] Update `context/EXPERIMENTS.md` with the residual audit entry.
- [ ] Commit the audit notebook and artifacts.
- [ ] Run `01_empirical_bayes_offsets.ipynb`.
- [ ] Compare empirical Bayes offsets against the current Huber + Hilbert residual baseline.
- [ ] Commit the first successful or instructive hierarchy result.
- [ ] Run `02_contextual_hierarchical_huber.ipynb` only after the empirical Bayes audit identifies useful descriptors.
- [ ] Run `03_hierarchical_residual_slope.ipynb` if residual slope bias remains after offset correction.
- [ ] Run `04_contextual_hierarchy_hilbert_residual.ipynb` after the mean hierarchy is stable.
- [ ] Package and submit a Kaggle notebook only after local diagnostics justify it.
- [ ] Use `05_multitask_gp_screen.ipynb` and `06_hierarchical_hsgp_full.ipynb` as later steps, not first steps.

## Artifact Standards

Every hierarchy notebook should export:

- fold metrics,
- candidate summary,
- residual sample,
- well-level diagnostics,
- concise markdown report,
- model comparison row,
- run bundle under `artifacts/logs/runs/`.

Every report should state:

- local RMSE mean and standard deviation,
- endpoint-hold comparison,
- comparison against public baseline **14.514**,
- whether the model is Kaggle-submission worthy,
- which assumptions were supported or rejected.

## Kaggle Submission Gate

Do not submit every local improvement. Submit when one of these is true:

- local RMSE improves by at least **0.10** over the current best local diagnostic,
- a smaller local gain fixes a known public-generalization problem,
- the notebook represents a materially new model family and we need one public read.

When submitted, update:

- `kaggle/SUBMISSIONS.md`,
- `kaggle/LEADERBOARD_NOTES.md`,
- `context/STATUS.md`,
- `context/EXPERIMENTS.md`,
- `context/CHANGELOG.md`.

## Risks

- Well-ID random effects can leak local validation structure and fail on hidden wells.
- Public leaderboard variance can reward leaderboard-specific tuning.
- Full HSGP hierarchy can become slow enough that iteration quality drops.
- Residual corrections can overfit one fold if they are not shrinkage-controlled.
- Reconstructing prior model features incorrectly can produce misleading comparisons.

## Sources

- Riutort-Mayol, Buerkner, Andersen, Solin, and Vehtari, "Practical Hilbert space approximate Bayesian Gaussian processes for probabilistic programming": https://arxiv.org/abs/2004.11408
- PyMC HSGP documentation: https://www.pymc.io/projects/docs/en/stable/api/gp/generated/pymc.gp.HSGP.html
- Stan User's Guide, regression and hierarchical modeling sections: https://mc-stan.org/docs/stan-users-guide/regression.html
- GPyTorch multitask GP regression tutorial: https://docs.gpytorch.ai/en/stable/examples/03_Multitask_Exact_GPs/Multitask_GP_Regression.html
- GPflow coregionalization tutorial: https://gpflow.github.io/GPflow/develop/notebooks/advanced/coregionalisation.html
- NumPyro Gaussian process example: https://num.pyro.ai/en/stable/examples/gp.html
- Chaki et al., well tops guided reservoir property prediction: https://arxiv.org/abs/1509.07079
