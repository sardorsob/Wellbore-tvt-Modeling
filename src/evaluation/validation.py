"""Validation helpers for well-level TVT prediction experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd


def target_zone_mask(df: pd.DataFrame) -> pd.Series:
    """Rows to score: `TVT_input` missing but true `TVT` available."""
    return df["TVT_input"].isna() & df["TVT"].notna()


def known_zone_mask(df: pd.DataFrame) -> pd.Series:
    """Rows available for supervised fitting within a training well."""
    return df["TVT_input"].notna() & df["TVT"].notna()


def group_well_folds(wells: list[str], n_splits: int = 5) -> list[tuple[list[str], list[str]]]:
    """Return deterministic well-level folds with no train/validation overlap."""
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    wells = sorted(wells)
    if not wells:
        raise ValueError("At least one well is required")

    fold_count = min(n_splits, len(wells))
    valid_groups = [list(group) for group in np.array_split(wells, fold_count)]
    folds = []
    for valid in valid_groups:
        valid_set = set(valid)
        train = [well for well in wells if well not in valid_set]
        folds.append((train, valid))
    return folds


def leave_one_well_out_cv(wells: list[str]) -> list[tuple[list[str], list[str]]]:
    """Return one fold per held-out well."""
    return group_well_folds(wells, n_splits=len(wells))


def masked_eval_zone_cv(wells: list[str], n_splits: int = 5) -> list[tuple[list[str], list[str]]]:
    """Alias for grouped masked-zone validation used by legacy notebooks."""
    return group_well_folds(wells, n_splits=n_splits)

