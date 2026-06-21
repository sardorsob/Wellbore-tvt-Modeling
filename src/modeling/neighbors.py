"""Neighbor-based regression factories for local TVT baselines."""

from __future__ import annotations


def make_neighbor_model(model_name: str, params: dict | None = None):
    """Create a scikit-learn neighbor regressor."""
    from sklearn.neighbors import KNeighborsRegressor

    params = dict(params or {})
    params.setdefault("n_neighbors", 15)
    params.setdefault("weights", "distance")
    params.setdefault("algorithm", "auto")
    params.setdefault("n_jobs", -1)

    if model_name == "knn":
        return KNeighborsRegressor(**params)
    raise ValueError("model_name must be one of: knn")
