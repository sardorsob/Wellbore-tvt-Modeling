"""Small NumPy linear regression models for first-pass baselines."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


class Standardizer:
    """Train-only mean/std scaler with safe handling for constant columns."""

    def __init__(self):
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "Standardizer":
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        scale = X.std(axis=0)
        self.scale_ = np.where(scale == 0.0, 1.0, scale)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Standardizer is not fitted.")
        return (np.asarray(X, dtype=float) - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class LinearRegressor:
    """OLS, Ridge, or Lasso regressor with a shared fit/predict interface."""

    def __init__(
        self,
        kind: str = "ols",
        alpha: float = 1.0,
        max_iter: int = 2000,
        tol: float = 1e-6,
    ):
        if kind not in {"ols", "ridge", "lasso"}:
            raise ValueError("kind must be one of: ols, ridge, lasso")
        self.kind = kind
        self.alpha = float(alpha)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.intercept_: float = 0.0
        self.coef_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressor":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if self.kind == "lasso":
            self._fit_lasso(X, y)
        else:
            self._fit_closed_form(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Model is not fitted.")
        return self.intercept_ + np.asarray(X, dtype=float) @ self.coef_

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as handle:
            pickle.dump(self, handle)

    @classmethod
    def load(cls, path: str | Path) -> "LinearRegressor":
        with open(path, "rb") as handle:
            return pickle.load(handle)

    def _fit_closed_form(self, X: np.ndarray, y: np.ndarray) -> None:
        ones = np.ones((X.shape[0], 1))
        X_design = np.hstack([ones, X])
        penalty = np.eye(X_design.shape[1]) * (self.alpha if self.kind == "ridge" else 0.0)
        penalty[0, 0] = 0.0
        coef = np.linalg.pinv(X_design.T @ X_design + penalty) @ X_design.T @ y
        self.intercept_ = float(coef[0])
        self.coef_ = coef[1:]

    def _fit_lasso(self, X: np.ndarray, y: np.ndarray) -> None:
        x_mean = X.mean(axis=0)
        y_mean = float(y.mean())
        X_centered = X - x_mean
        centered_y = y - y_mean
        coef = np.zeros(X.shape[1], dtype=float)
        squared_norms = np.sum(X_centered * X_centered, axis=0)

        for _ in range(self.max_iter):
            previous = coef.copy()
            for j in range(X.shape[1]):
                if squared_norms[j] == 0.0:
                    coef[j] = 0.0
                    continue
                residual = centered_y - X_centered @ coef + X_centered[:, j] * coef[j]
                rho = float(X_centered[:, j] @ residual)
                coef[j] = _soft_threshold(rho, self.alpha * X.shape[0]) / squared_norms[j]
            if np.max(np.abs(coef - previous)) < self.tol:
                break

        self.intercept_ = y_mean - float(x_mean @ coef)
        self.coef_ = coef


def _soft_threshold(value: float, penalty: float) -> float:
    if value > penalty:
        return value - penalty
    if value < -penalty:
        return value + penalty
    return 0.0
