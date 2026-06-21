import numpy as np

from src.modeling.linear import LinearRegressor, Standardizer


def test_ols_recovers_simple_linear_relationship():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = 2.0 + 3.0 * X[:, 0]

    model = LinearRegressor(kind="ols").fit(X, y)

    np.testing.assert_allclose(model.predict(X), y, atol=1e-8)


def test_ridge_keeps_intercept_unpenalized():
    X = np.array([[0.0], [0.0], [0.0]])
    y = np.array([5.0, 5.0, 5.0])

    model = LinearRegressor(kind="ridge", alpha=10.0).fit(X, y)

    np.testing.assert_allclose(model.predict(X), y, atol=1e-8)


def test_lasso_shrinks_irrelevant_feature_to_zero():
    X = np.array(
        [
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
            [3.0, 1.0],
            [4.0, 1.0],
        ]
    )
    y = 1.0 + 2.0 * X[:, 0]

    model = LinearRegressor(kind="lasso", alpha=0.05, max_iter=5000, tol=1e-10).fit(X, y)

    assert abs(model.coef_[1]) < 1e-8
    np.testing.assert_allclose(model.predict(X), y, atol=0.25)


def test_standardizer_uses_training_statistics_for_transform():
    train = np.array([[1.0, 10.0], [3.0, 10.0], [5.0, 10.0]])
    valid = np.array([[7.0, 10.0]])

    scaler = Standardizer().fit(train)
    train_scaled = scaler.transform(train)
    valid_scaled = scaler.transform(valid)

    np.testing.assert_allclose(train_scaled.mean(axis=0), np.array([0.0, 0.0]), atol=1e-12)
    np.testing.assert_allclose(valid_scaled, np.array([[2.44948974, 0.0]]), atol=1e-8)

