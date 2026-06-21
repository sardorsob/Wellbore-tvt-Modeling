"""Regression metrics used by notebooks and experiment summaries."""

from __future__ import annotations

import numpy as np


def _as_arrays(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    """Convert inputs to aligned float arrays."""
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if true.shape != pred.shape:
        raise ValueError(f"Shape mismatch: {true.shape} != {pred.shape}")
    return true, pred


def rmse(y_true, y_pred) -> float:
    """Return root mean squared error."""
    true, pred = _as_arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((true - pred) ** 2)))


def mae(y_true, y_pred) -> float:
    """Return mean absolute error."""
    true, pred = _as_arrays(y_true, y_pred)
    return float(np.mean(np.abs(true - pred)))


def r2(y_true, y_pred) -> float:
    """Return coefficient of determination."""
    true, pred = _as_arrays(y_true, y_pred)
    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    if ss_tot == 0.0:
        return 0.0 if ss_res > 0.0 else 1.0
    return 1.0 - ss_res / ss_tot


def metrics_summary(y_true, y_pred) -> dict[str, float]:
    """Return the core regression metric bundle."""
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "r2": r2(y_true, y_pred),
    }

