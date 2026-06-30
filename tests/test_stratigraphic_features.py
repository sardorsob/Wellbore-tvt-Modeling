import numpy as np
import pandas as pd

from src.preprocessing.stratigraphic import (
    alignment_uncertainty,
    best_lag_correlation,
    dtw_distance,
    resample_sequence,
    summarize_known_zone_state,
)


def test_resample_sequence_interpolates_to_fixed_length():
    values = np.array([0.0, 10.0, 20.0])
    result = resample_sequence(values, length=5)
    assert result.shape == (5,)
    assert np.allclose(result[[0, -1]], [0.0, 20.0])
    assert np.isfinite(result).all()


def test_resample_sequence_handles_empty_or_nan_input():
    assert np.allclose(resample_sequence([], length=4), np.zeros(4))
    assert np.allclose(resample_sequence([np.nan, np.nan], length=3), np.zeros(3))


def test_dtw_distance_is_zero_for_identical_sequences():
    seq = np.array([1.0, 2.0, 3.0, 2.0])
    assert dtw_distance(seq, seq) == 0.0


def test_best_lag_correlation_finds_shifted_match():
    target = np.array([0.0, 1.0, 4.0, 9.0, 0.0])
    template = np.array([1.0, 4.0, 9.0])
    result = best_lag_correlation(target, template, max_lag=2)
    assert result["best_lag"] == 1
    assert result["best_corr"] > 0.99


def test_alignment_uncertainty_reports_margin_and_entropy():
    scores = np.array([0.9, 0.7, 0.1])
    result = alignment_uncertainty(scores)
    assert result["alignment_margin"] > 0
    assert result["alignment_entropy"] > 0
    assert result["alignment_score_best"] == 0.9


def test_summarize_known_zone_state_uses_only_known_rows():
    frame = pd.DataFrame(
        {
            "MD": [100.0, 110.0, 120.0, 130.0],
            "X": [0.0, 1.0, 2.0, 3.0],
            "Y": [0.0, 0.0, 0.0, 0.0],
            "Z": [-10.0, -11.0, -12.0, -13.0],
            "GR": [50.0, 55.0, np.nan, 80.0],
            "TVT_input": [5.0, 6.0, np.nan, np.nan],
        }
    )
    result = summarize_known_zone_state(frame)
    assert result["known_row_count"] == 2
    assert result["known_tvt_last"] == 6.0
    assert result["known_gr_coverage"] == 1.0
