import numpy as np

from src.evaluation.metrics import mae, metrics_summary, r2, rmse


def test_rmse_computes_root_mean_squared_error():
    y_true = np.array([1.0, 2.0, 4.0])
    y_pred = np.array([1.0, 4.0, 1.0])

    assert rmse(y_true, y_pred) == np.sqrt(13.0 / 3.0)


def test_metrics_summary_returns_core_regression_metrics():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.5, 2.5])

    summary = metrics_summary(y_true, y_pred)

    assert summary["rmse"] == rmse(y_true, y_pred)
    assert summary["mae"] == mae(y_true, y_pred)
    assert summary["r2"] == r2(y_true, y_pred)

