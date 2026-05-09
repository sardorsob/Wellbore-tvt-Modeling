"""Well data loaders — discover and load horizontal well + typewell CSV files."""

from pathlib import Path

import pandas as pd

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Columns expected only in training files — excluded at inference time
_TRAIN_ONLY_COLS = {"ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA", "TVT"}


def discover_wells(directory: str | Path) -> list[str]:
    """Return sorted list of well names found in directory.

    Discovers wells by globbing ``*__horizontal_well.csv``.
    """
    directory = Path(directory)
    files = sorted(directory.glob("*__horizontal_well.csv"))
    wells = [f.name.replace("__horizontal_well.csv", "") for f in files]
    logger.info("Discovered %d wells in %s", len(wells), directory)
    return wells


def load_horizontal_well(path: str | Path) -> pd.DataFrame:
    """Load a horizontal well CSV, preserving original row index.

    Returns
    -------
    pd.DataFrame
        Index is the original CSV row index (0-based).
        Column ``row_idx`` is a copy of the index for safe joins.
    """
    df = pd.read_csv(path)
    df.index.name = "row_idx"
    df["row_idx"] = df.index
    return df


def load_typewell(path: str | Path) -> pd.DataFrame:
    """Load a typewell CSV.

    Returns
    -------
    pd.DataFrame
        Columns: TVT (depth index), GR, Geology.
    """
    df = pd.read_csv(path)
    return df


def load_all_wells(
    directory: str | Path,
    wells: list[str] | None = None,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Load all wells in directory.

    Parameters
    ----------
    directory : path to train/ or test/
    wells : optional list of well names; if None, discovers all

    Returns
    -------
    dict mapping well name → {"horizontal": df, "typewell": df}
    """
    directory = Path(directory)
    if wells is None:
        wells = discover_wells(directory)

    result = {}
    for well in wells:
        hw_path = directory / f"{well}__horizontal_well.csv"
        tw_path = directory / f"{well}__typewell.csv"

        if not hw_path.exists():
            logger.warning("Missing horizontal well file for %s", well)
            continue

        result[well] = {
            "horizontal": load_horizontal_well(hw_path),
            "typewell": load_typewell(tw_path) if tw_path.exists() else pd.DataFrame(),
        }

    logger.info("Loaded %d wells from %s", len(result), directory)
    return result


def load_sample_submission(path: str | Path) -> pd.DataFrame:
    """Load sample_submission.csv.

    Returns
    -------
    pd.DataFrame with columns [id, tvt].
    Adds helper columns ``wellname`` and ``row_idx`` parsed from id.
    """
    df = pd.read_csv(path)
    # Parse id = {WELLNAME}_{row_index}
    split = df["id"].str.rsplit("_", n=1, expand=True)
    df["wellname"] = split[0]
    df["row_idx"] = split[1].astype(int)
    return df


def get_test_column_names(directory: str | Path) -> set[str]:
    """Return the union of column names across all test horizontal well files.

    Use this to verify which features are actually available at inference time.
    """
    directory = Path(directory)
    wells = discover_wells(directory)
    if not wells:
        return set()
    cols: set[str] = set()
    for well in wells:
        path = directory / f"{well}__horizontal_well.csv"
        header = pd.read_csv(path, nrows=0)
        cols.update(header.columns.tolist())
    return cols
