"""Scaling and imputation utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def impute_gr(df: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """Fill GR NaN with centered rolling median.

    Parameters
    ----------
    df : DataFrame with a GR column
    window : rolling window size (rows)
    """
    df = df.copy()
    df["GR"] = df["GR"].fillna(
        df["GR"].rolling(window, center=True, min_periods=1).median()
    )
    return df


def clip_gr(df: pd.DataFrame, lo: float = 0.0, hi: float = 300.0) -> pd.DataFrame:
    """Clip GR to physical range [lo, hi] API units."""
    df = df.copy()
    df["GR"] = df["GR"].clip(lower=lo, upper=hi)
    return df


def scale_features(
    X_train: np.ndarray,
    X_val: np.ndarray | None = None,
    X_test: np.ndarray | None = None,
) -> tuple:
    """Fit StandardScaler on X_train and transform all splits.

    Returns
    -------
    (scaler, X_train_scaled, X_val_scaled | None, X_test_scaled | None)
    """
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val) if X_val is not None else None
    X_test_s = scaler.transform(X_test) if X_test is not None else None
    return scaler, X_train_s, X_val_s, X_test_s


def fill_eval_zone_features(df: pd.DataFrame) -> pd.DataFrame:
    """Forward/backward fill feature NaN that arise in the evaluation zone.

    After building features, some lag/rolling features will be NaN at the edges
    of the evaluation zone.  This is safe because we never include the true TVT
    target in those features.
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].ffill().bfill()
    return df
