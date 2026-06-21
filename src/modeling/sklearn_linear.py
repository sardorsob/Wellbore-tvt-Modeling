"""Scikit-learn linear model factories for robust regression baselines."""

from __future__ import annotations


def make_sklearn_linear_model(model_name: str, params: dict | None = None):
    """Create a scikit-learn linear or robust regressor."""
    from sklearn.linear_model import ElasticNet, HuberRegressor, QuantileRegressor

    params = dict(params or {})

    if model_name == "elasticnet":
        params.setdefault("alpha", 0.01)
        params.setdefault("l1_ratio", 0.5)
        params.setdefault("max_iter", 2000)
        params.setdefault("random_state", 42)
        return ElasticNet(**params)
    if model_name == "huber":
        params.setdefault("epsilon", 1.35)
        params.setdefault("alpha", 0.0001)
        params.setdefault("max_iter", 200)
        return HuberRegressor(**params)
    if model_name == "quantile":
        params.setdefault("quantile", 0.5)
        params.setdefault("alpha", 0.01)
        params.setdefault("solver", "highs")
        return QuantileRegressor(**params)
    raise ValueError("model_name must be one of: elasticnet, huber, quantile")
