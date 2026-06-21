import numpy as np

from src.modeling.sklearn_linear import make_sklearn_linear_model


def test_make_sklearn_linear_model_builds_elasticnet():
    model = make_sklearn_linear_model("elasticnet", {"alpha": 0.1, "l1_ratio": 0.25})

    assert model.__class__.__name__ == "ElasticNet"
    assert model.alpha == 0.1
    assert model.l1_ratio == 0.25


def test_make_sklearn_linear_model_builds_huber_regressor():
    model = make_sklearn_linear_model("huber", {"epsilon": 1.5, "alpha": 0.01})

    assert model.__class__.__name__ == "HuberRegressor"
    assert model.epsilon == 1.5
    assert model.alpha == 0.01


def test_make_sklearn_linear_model_builds_quantile_regressor():
    model = make_sklearn_linear_model("quantile", {"quantile": 0.5, "alpha": 0.01})

    assert model.__class__.__name__ == "QuantileRegressor"
    assert model.quantile == 0.5
    assert model.alpha == 0.01


def test_sklearn_linear_model_predicts_training_shape():
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    model = make_sklearn_linear_model("elasticnet", {"alpha": 0.001, "l1_ratio": 0.5})

    model.fit(X, y)
    pred = model.predict(X)

    assert pred.shape == y.shape
