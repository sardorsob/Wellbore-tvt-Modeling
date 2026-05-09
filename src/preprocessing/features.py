"""Feature engineering pipeline for horizontal well + typewell data."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Features safe to use in both train and test
_BASE_FEATURES = ["MD", "X", "Y", "Z", "GR", "TVT_input", "row_idx"]

# Lag windows for TVT_input (measured in rows = 1 ft increments)
_LAG_WINDOWS = [1, 2, 5, 10, 20, 50]
_ROLLING_WINDOWS = [5, 10, 20, 50]


def build_features(
    df_horiz: pd.DataFrame,
    df_typewell: pd.DataFrame,
    lag_windows: list[int] = _LAG_WINDOWS,
    rolling_windows: list[int] = _ROLLING_WINDOWS,
) -> pd.DataFrame:
    """Build feature matrix for one well.

    Parameters
    ----------
    df_horiz : horizontal well DataFrame (from loaders.load_horizontal_well)
    df_typewell : typewell DataFrame (from loaders.load_typewell)
    lag_windows : row-lag values for TVT_input
    rolling_windows : rolling-window sizes for GR and TVT_input

    Returns
    -------
    pd.DataFrame aligned to df_horiz index; no TVT column included.
    """
    df = df_horiz.copy()

    # ── GR cleaning ──────────────────────────────────────────────────────────
    df["GR"] = df["GR"].clip(lower=0, upper=300)
    df["GR"] = df["GR"].fillna(df["GR"].rolling(21, center=True, min_periods=1).median())

    # ── TVT_input lags ───────────────────────────────────────────────────────
    for lag in lag_windows:
        df[f"tvt_input_lag_{lag}"] = df["TVT_input"].shift(lag)
        df[f"tvt_input_lag_neg{lag}"] = df["TVT_input"].shift(-lag)

    # ── TVT_input rolling stats ───────────────────────────────────────────────
    for w in rolling_windows:
        df[f"tvt_input_roll_mean_{w}"] = (
            df["TVT_input"].rolling(w, center=True, min_periods=1).mean()
        )
        df[f"tvt_input_roll_std_{w}"] = (
            df["TVT_input"].rolling(w, center=True, min_periods=1).std()
        )

    # ── GR rolling stats ──────────────────────────────────────────────────────
    for w in rolling_windows:
        df[f"gr_roll_mean_{w}"] = df["GR"].rolling(w, center=True, min_periods=1).mean()
        df[f"gr_roll_std_{w}"] = df["GR"].rolling(w, center=True, min_periods=1).std()

    # ── MD-derived features ───────────────────────────────────────────────────
    df["md_diff"] = df["MD"].diff().fillna(0)
    df["dz_dmd"] = df["Z"].diff().fillna(0)  # inclination proxy
    df["dx_dmd"] = df["X"].diff().fillna(0)
    df["dy_dmd"] = df["Y"].diff().fillna(0)

    # Cumulative horizontal displacement
    df["horiz_dist"] = np.sqrt(df["dx_dmd"] ** 2 + df["dy_dmd"] ** 2).cumsum()

    # ── Typewell join features ────────────────────────────────────────────────
    if not df_typewell.empty and "GR" in df_typewell.columns and "TVT" in df_typewell.columns:
        tw_features = _typewell_features(df, df_typewell)
        df = df.join(tw_features)

    # ── TVT_input forward/backward fill (as feature, not target) ─────────────
    df["tvt_input_ffill"] = df["TVT_input"].ffill()
    df["tvt_input_bfill"] = df["TVT_input"].bfill()

    # ── Known-zone flag ───────────────────────────────────────────────────────
    df["is_known_zone"] = (~df["TVT_input"].isna()).astype(int)

    # ── Position within well ──────────────────────────────────────────────────
    df["md_pct"] = (df["MD"] - df["MD"].min()) / (df["MD"].max() - df["MD"].min() + 1e-9)

    return df


def _typewell_features(df_horiz: pd.DataFrame, df_typewell: pd.DataFrame) -> pd.DataFrame:
    """Interpolate typewell GR at the TVT_input positions of horizontal rows.

    For unknown-zone rows, uses ffill of TVT_input as a proxy TVT position.
    """
    tw = df_typewell.sort_values("TVT").dropna(subset=["TVT", "GR"])
    tw_interp = interp1d(
        tw["TVT"].values,
        tw["GR"].values,
        bounds_error=False,
        fill_value=(tw["GR"].iloc[0], tw["GR"].iloc[-1]),
    )

    tvt_proxy = df_horiz["TVT_input"].ffill().bfill()
    tw_gr = tw_interp(tvt_proxy.values)

    result = pd.DataFrame(
        {"tw_gr_at_tvt": tw_gr, "gr_minus_tw_gr": df_horiz["GR"].values - tw_gr},
        index=df_horiz.index,
    )

    # Encode dominant formation at each depth
    if "Geology" in df_typewell.columns:
        formations = df_typewell.sort_values("TVT")[["TVT", "Geology"]].dropna()
        # Nearest-neighbor formation assignment
        idxs = np.searchsorted(formations["TVT"].values, tvt_proxy.values, side="right") - 1
        idxs = np.clip(idxs, 0, len(formations) - 1)
        result["tw_formation"] = formations["Geology"].iloc[idxs].values

    return result


def build_feature_matrix(
    wells: dict[str, dict[str, pd.DataFrame]],
    include_target: bool = True,
) -> pd.DataFrame:
    """Build a concatenated feature matrix across all wells.

    Parameters
    ----------
    wells : output of loaders.load_all_wells
    include_target : if True, add TVT column (for training data)

    Returns
    -------
    pd.DataFrame with ``wellname`` column and all features.
    """
    frames = []
    for wellname, data in wells.items():
        df_hw = data["horizontal"]
        df_tw = data.get("typewell", pd.DataFrame())
        feat = build_features(df_hw, df_tw)
        feat["wellname"] = wellname
        if include_target and "TVT" in df_hw.columns:
            feat["TVT"] = df_hw["TVT"].values
        frames.append(feat)

    combined = pd.concat(frames, ignore_index=False)
    logger.info(
        "Feature matrix: %d rows, %d cols across %d wells",
        len(combined), combined.shape[1], len(wells),
    )
    return combined


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return feature columns, excluding target, id, and metadata columns."""
    exclude = {"TVT", "WELLNAME", "wellname", "row_idx", "Geology", "tw_formation"}
    exclude.update({"ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"})
    return [c for c in df.columns if c not in exclude and df[c].dtype != object]
