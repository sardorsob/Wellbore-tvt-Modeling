"""Interpolation baseline model.

Predicts TVT in the evaluation zone by:
1. Forward-filling the last known TVT_input value.
2. Linear interpolation between known TVT_input endpoints.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class InterpolationBaseline:
    """Simple interpolation baseline — no ML, no training.

    Strategy
    --------
    For each row in the evaluation zone (TVT_input is NaN):
    - Use linear interpolation between the last known TVT_input before
      the gap and the first known TVT_input after the gap.
    - If no value exists after, fall back to forward fill.
    - If no value exists before, fall back to backward fill.
    """

    def fit(self, *args, **kwargs) -> "InterpolationBaseline":
        return self  # stateless

    def predict_well(self, df: pd.DataFrame) -> np.ndarray:
        """Predict TVT for all rows of a single well.

        Parameters
        ----------
        df : horizontal well DataFrame with TVT_input column

        Returns
        -------
        np.ndarray of predicted TVT aligned to df index
        """
        tvt = df["TVT_input"].copy()
        # Interpolate NaN spans linearly, then fill edges
        tvt_filled = (
            tvt.interpolate(method="linear", limit_direction="both")
              .ffill()
              .bfill()
        )
        return tvt_filled.values

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Compatibility stub — baseline works per-well, not on feature matrix."""
        raise NotImplementedError("Use predict_well() for the interpolation baseline.")

    def save(self, path: str) -> None:
        pass  # stateless

    def load(self, path: str) -> None:
        pass  # stateless


def predict_all_wells_baseline(
    wells: dict[str, dict[str, pd.DataFrame]],
    sample_submission: pd.DataFrame,
) -> pd.DataFrame:
    """Run baseline interpolation across all test wells.

    Parameters
    ----------
    wells : output of loaders.load_all_wells (test split)
    sample_submission : DataFrame with columns [id, tvt, wellname, row_idx]

    Returns
    -------
    pd.DataFrame with columns [id, tvt] matching sample_submission row order.
    """
    model = InterpolationBaseline()
    predictions = {}

    for wellname, data in wells.items():
        df = data["horizontal"]
        preds = model.predict_well(df)
        for row_idx, tvt_pred in zip(df.index, preds):
            id_str = f"{wellname}_{row_idx}"
            predictions[id_str] = tvt_pred

    submission = sample_submission[["id"]].copy()
    submission["tvt"] = submission["id"].map(predictions)

    n_missing = submission["tvt"].isna().sum()
    if n_missing > 0:
        logger.warning("%d submission rows have no prediction — filling with 0.0", n_missing)
        submission["tvt"] = submission["tvt"].fillna(0.0)

    return submission
