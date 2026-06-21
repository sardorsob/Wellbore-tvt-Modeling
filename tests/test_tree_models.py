import numpy as np

from src.modeling.trees import make_tree_model


def test_make_tree_model_builds_random_forest_regressor():
    model = make_tree_model("random_forest", {"n_estimators": 3, "random_state": 7})

    assert model.__class__.__name__ == "RandomForestRegressor"
    assert model.n_estimators == 3
    assert model.random_state == 7


def test_make_tree_model_builds_extra_trees_regressor():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 1.0, 2.0, 3.0])
    model = make_tree_model("extra_trees", {"n_estimators": 5, "random_state": 7})

    model.fit(X, y)
    pred = model.predict(X)

    assert pred.shape == y.shape
