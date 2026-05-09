# Project Brief

## Competition

**Name:** ROGII Wellbore Geology Prediction
**Platform:** Kaggle
**Slug:** `rogii-wellbore-geology-prediction`
**Timeline:** May 5 – August 5, 2026

## Problem

Horizontal drilling requires knowing the geological position (TVT) at each foot of the wellbore.
The challenge: TVT is only partially observed (known up to the "kick-off" point, then missing in the lateral/evaluation zone).
The model must predict missing TVT values using trajectory data, gamma ray logs, and vertical type-well references.

## Target

`TVT` — True Vertical Thickness in feet. One value per row of the horizontal well lateral.

## Metric

**RMSE** (Root Mean Squared Error). Lower is better. Public leaderboard uses a hidden test set.

## Success criteria

| Milestone | Target |
|---|---|
| Valid baseline submission | Any finite RMSE |
| Beat naive interpolation | RMSE < interpolation baseline |
| Top-50% leaderboard | Competitive RMSE |
| Top-25% leaderboard | Stretch goal |

## Constraints

- Kaggle Notebook runtime ≤ 9 hours (CPU or GPU)
- No internet access in Kaggle Notebook
- External data allowed if publicly available
- Final submission must be produced by a committed Kaggle Notebook

## Approach philosophy

Progress through a structured ladder of approaches:
1. **Baseline** — interpolation / forward-fill on TVT_input
2. **Classical ML** — XGBoost, LightGBM with hand-crafted features
3. **Bayesian GP** — NumpPyro/JAX Gaussian Process for uncertainty-aware TVT estimation
4. **BNN** — Bayesian Neural Network (Flax + NumpPyro) for non-linear structure
5. **Standard NN** — deterministic Flax network, faster iteration
6. **Ensemble** — weighted combination of best approaches

Each approach is evaluated with the same validation protocol (masked evaluation zone).
