import numpy as np
import pandas as pd

from src.preprocessing.tabular import make_simple_feature_frame, sample_training_rows


def test_make_simple_feature_frame_uses_test_safe_columns():
    df = pd.DataFrame(
        {
            "row_idx": [0, 1, 2],
            "MD": [100.0, 101.0, 102.0],
            "X": [0.0, 3.0, 6.0],
            "Y": [0.0, 4.0, 8.0],
            "Z": [-10.0, -10.5, -11.0],
            "GR": [10.0, None, 500.0],
            "TVT_input": [1.0, 2.0, None],
            "TVT": [1.0, 2.0, 3.0],
            "ANCC": [99.0, 99.0, 99.0],
        }
    )

    features = make_simple_feature_frame(df)

    assert "TVT" not in features.columns
    assert "ANCC" not in features.columns
    assert features["GR"].max() <= 300.0
    assert features["horizontal_dist"].iloc[-1] == 10.0
    assert features["known_zone"].tolist() == [1.0, 1.0, 0.0]


def test_sample_training_rows_is_deterministic_and_ordered():
    X = np.arange(20.0).reshape(10, 2)
    y = np.arange(10.0)

    X_sample, y_sample = sample_training_rows(X, y, max_rows=4)

    np.testing.assert_array_equal(y_sample, np.array([0.0, 3.0, 6.0, 9.0]))
    np.testing.assert_array_equal(X_sample[:, 0], np.array([0.0, 6.0, 12.0, 18.0]))

