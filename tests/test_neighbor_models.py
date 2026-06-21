import numpy as np

from src.modeling.neighbors import make_neighbor_model


def test_make_neighbor_model_builds_knn_regressor():
    model = make_neighbor_model("knn", {"n_neighbors": 3, "weights": "distance"})

    assert model.__class__.__name__ == "KNeighborsRegressor"
    assert model.n_neighbors == 3
    assert model.weights == "distance"


def test_make_neighbor_model_predicts_training_shape():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 1.0, 2.0, 3.0])
    model = make_neighbor_model("knn", {"n_neighbors": 2, "weights": "uniform"})

    model.fit(X, y)
    pred = model.predict(X)

    assert pred.shape == y.shape
