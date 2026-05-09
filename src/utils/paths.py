"""Path resolution — auto-detects Kaggle vs. local environment."""

from pathlib import Path

_KAGGLE_INPUT = Path("/kaggle/input/rogii-wellbore-geology-prediction")
_KAGGLE_OUTPUT = Path("/kaggle/working")
_REPO_ROOT = Path(__file__).resolve().parents[2]


def is_kaggle() -> bool:
    return _KAGGLE_INPUT.exists()


def get_base_dir() -> Path:
    """Root of competition data (train/, test/, sample_submission.csv)."""
    if is_kaggle():
        return _KAGGLE_INPUT
    return _REPO_ROOT / "data" / "raw"


def get_data_dir(split: str = "train") -> Path:
    """Return path to train/ or test/ directory."""
    assert split in ("train", "test"), f"Unknown split: {split}"
    return get_base_dir() / split


def get_output_dir() -> Path:
    """Directory for writing submission.csv."""
    if is_kaggle():
        return _KAGGLE_OUTPUT
    return _REPO_ROOT / "artifacts" / "reports"


def get_artifacts_dir(subdir: str = "") -> Path:
    """Return path inside artifacts/."""
    base = _REPO_ROOT / "artifacts"
    return base / subdir if subdir else base


def get_repo_root() -> Path:
    return _REPO_ROOT


def get_sample_submission_path() -> Path:
    return get_base_dir() / "sample_submission.csv"
