"""Simple test-safe tabular features for first-pass models."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_simple_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Build a compact numeric feature matrix from train/test-safe columns."""
    out = pd.DataFrame(index=df.index)
    out["row_idx"] = df["row_idx"].astype(float)
    out["MD"] = df["MD"].astype(float)
    out["X"] = df["X"].astype(float)
    out["Y"] = df["Y"].astype(float)
    out["Z"] = df["Z"].astype(float)
    out["GR"] = _safe_gr(df["GR"])
    out["TVT_input_ffill"] = df["TVT_input"].ffill()
    out["known_zone"] = df["TVT_input"].notna().astype(float)
    out["md_from_start"] = out["MD"] - out["MD"].min()
    out["md_norm"] = out["md_from_start"] / (out["md_from_start"].max() + 1e-9)
    out["dX"] = out["X"].diff().fillna(0.0)
    out["dY"] = out["Y"].diff().fillna(0.0)
    out["dZ"] = out["Z"].diff().fillna(0.0)
    out["horizontal_step"] = np.sqrt(out["dX"] ** 2 + out["dY"] ** 2)
    out["horizontal_dist"] = out["horizontal_step"].cumsum()
    return out


def sample_training_rows(
    X: np.ndarray,
    y: np.ndarray,
    max_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a deterministic row-spaced sample when arrays exceed max_rows."""
    if len(y) <= max_rows:
        return X, y
    idx = np.linspace(0, len(y) - 1, max_rows).astype(int)
    return X[idx], y[idx]


def _safe_gr(series: pd.Series) -> pd.Series:
    gr = series.astype(float).clip(lower=0, upper=300)
    return gr.fillna(gr.rolling(21, center=True, min_periods=1).median()).ffill().bfill()

