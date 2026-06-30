"""Fold-safe stratigraphic-state feature helpers."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def _finite_array(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    return arr[np.isfinite(arr)]


def resample_sequence(values: Iterable[float], length: int = 32) -> np.ndarray:
    """Resample a finite sequence to a fixed length with linear interpolation."""
    arr = _finite_array(values)
    if length <= 0:
        raise ValueError("length must be positive")
    if arr.size == 0:
        return np.zeros(length, dtype=float)
    if arr.size == 1:
        return np.full(length, float(arr[0]), dtype=float)
    x_old = np.linspace(0.0, 1.0, arr.size)
    x_new = np.linspace(0.0, 1.0, length)
    return np.interp(x_new, x_old, arr)


def robust_standardize(values: Iterable[float]) -> np.ndarray:
    """Robustly center and scale a numeric sequence."""
    arr = np.asarray(list(values), dtype=float)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr, dtype=float)
    med = float(np.nanmedian(arr[finite]))
    mad = float(np.nanmedian(np.abs(arr[finite] - med)))
    scale = 1.4826 * mad if mad > 1e-9 else float(np.nanstd(arr[finite]))
    if not np.isfinite(scale) or scale <= 1e-9:
        scale = 1.0
    out = (np.where(finite, arr, med) - med) / scale
    return np.clip(out, -8.0, 8.0)


def dtw_distance(a: Iterable[float], b: Iterable[float]) -> float:
    """Compute a compact dynamic-time-warping distance between two sequences."""
    x = robust_standardize(a)
    y = robust_standardize(b)
    if x.size == 0 or y.size == 0:
        return 0.0
    cost = np.full((x.size + 1, y.size + 1), np.inf)
    cost[0, 0] = 0.0
    for i in range(1, x.size + 1):
        for j in range(1, y.size + 1):
            step = abs(x[i - 1] - y[j - 1])
            cost[i, j] = step + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    return float(cost[x.size, y.size] / max(x.size, y.size))


def best_lag_correlation(
    target: Iterable[float],
    template: Iterable[float],
    max_lag: int = 8,
) -> dict[str, float]:
    """Find the best short lag correlation between target and template sequences."""
    x = robust_standardize(target)
    y = robust_standardize(template)
    if x.size == 0 or y.size == 0:
        return {"best_lag": 0.0, "best_corr": 0.0, "second_corr": 0.0}
    rows: list[tuple[int, float]] = []
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            x_part = x[lag : lag + y.size]
            y_part = y[: x_part.size]
        else:
            y_part = y[-lag : -lag + x.size]
            x_part = x[: y_part.size]
        if x_part.size < 3 or y_part.size < 3:
            corr = 0.0
        else:
            corr = float(np.corrcoef(x_part, y_part)[0, 1])
            if not np.isfinite(corr):
                corr = 0.0
        rows.append((lag, corr))
    rows.sort(key=lambda item: item[1], reverse=True)
    second = rows[1][1] if len(rows) > 1 else rows[0][1]
    return {"best_lag": float(rows[0][0]), "best_corr": float(rows[0][1]), "second_corr": float(second)}


def alignment_uncertainty(scores: Iterable[float]) -> dict[str, float]:
    """Summarize ranked alignment scores with margin and entropy."""
    arr = _finite_array(scores)
    if arr.size == 0:
        return {
            "alignment_score_best": 0.0,
            "alignment_score_second": 0.0,
            "alignment_margin": 0.0,
            "alignment_entropy": 0.0,
        }
    ordered = np.sort(arr)[::-1]
    best = float(ordered[0])
    second = float(ordered[1]) if ordered.size > 1 else 0.0
    shifted = ordered - np.max(ordered)
    probs = np.exp(shifted)
    probs = probs / probs.sum()
    entropy = float(-(probs * np.log(probs + 1e-12)).sum())
    return {
        "alignment_score_best": best,
        "alignment_score_second": second,
        "alignment_margin": best - second,
        "alignment_entropy": entropy,
    }


def summarize_known_zone_state(frame: pd.DataFrame) -> dict[str, float]:
    """Summarize known-zone geometry and log coverage without target-zone labels."""
    known = frame[frame["TVT_input"].notna()].copy()
    if known.empty:
        return {
            "known_row_count": 0.0,
            "known_tvt_last": 0.0,
            "known_tvt_slope": 0.0,
            "known_gr_coverage": 0.0,
            "known_md_span": 0.0,
            "known_xy_span": 0.0,
            "known_z_drop": 0.0,
        }
    md_span = float(known["MD"].iloc[-1] - known["MD"].iloc[0]) if "MD" in known else 0.0
    tvt_delta = float(known["TVT_input"].iloc[-1] - known["TVT_input"].iloc[0])
    xy_span = float(np.hypot(known["X"].iloc[-1] - known["X"].iloc[0], known["Y"].iloc[-1] - known["Y"].iloc[0]))
    z_drop = float(known["Z"].iloc[-1] - known["Z"].iloc[0])
    return {
        "known_row_count": float(len(known)),
        "known_tvt_last": float(known["TVT_input"].iloc[-1]),
        "known_tvt_slope": tvt_delta / max(abs(md_span), 1.0),
        "known_gr_coverage": float(known["GR"].notna().mean()) if "GR" in known else 0.0,
        "known_md_span": md_span,
        "known_xy_span": xy_span,
        "known_z_drop": z_drop,
    }
