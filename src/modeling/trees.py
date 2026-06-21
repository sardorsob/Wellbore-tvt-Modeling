"""Tree ensemble factories for tabular TVT baselines."""

from __future__ import annotations


def make_tree_model(model_name: str, params: dict | None = None):
    """Create a scikit-learn tree ensemble regressor."""
    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

    params = dict(params or {})
    params.setdefault("n_estimators", 80)
    params.setdefault("max_depth", 18)
    params.setdefault("min_samples_leaf", 5)
    params.setdefault("n_jobs", -1)
    params.setdefault("random_state", 42)

    if model_name == "random_forest":
        return RandomForestRegressor(**params)
    if model_name == "extra_trees":
        return ExtraTreesRegressor(**params)
    raise ValueError("model_name must be one of: random_forest, extra_trees")

