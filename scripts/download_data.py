"""Download ROGII Wellbore Geology Prediction competition files via Kaggle API."""

from __future__ import annotations

import argparse
import tarfile
import zipfile
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi


COMPETITION = "rogii-wellbore-geology-prediction"


def _repo_root() -> Path:
    """Directory containing ``src/`` (parent of ``scripts/``)."""
    return Path(__file__).resolve().parents[1]


def _extract_archives(dest: Path) -> None:
    """KaggleApi no longer accepts ``unzip=``; expand bundles locally."""

    zip_path = dest / f"{COMPETITION}.zip"
    zips = [zip_path] if zip_path.is_file() else sorted(dest.glob("*.zip"))
    if not zips:
        raise FileNotFoundError(
            f"No .zip bundle found under {dest} after download "
            f"(expected {zip_path.name})."
        )

    for zpath in zips:
        with zipfile.ZipFile(zpath, "r") as zf:
            zf.extractall(dest)
        zpath.unlink()

    for tpath in sorted(dest.glob("*.tar.gz")) + sorted(dest.glob("*.tgz")):
        with tarfile.open(tpath, "r:gz") as tf:
            tf.extractall(dest)
        tpath.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Download {COMPETITION} into data/raw.")
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="Destination directory (default: <repo>/data/raw)",
    )
    args = parser.parse_args()

    # Default dest does not import ``src``: ``python scripts/download_data.py`` puts ``scripts/`` on sys.path,
    # not the repo root, so ``import src`` would fail without extra sys.path hacks.
    dest = args.dest if args.dest is not None else _repo_root() / "data" / "raw"
    dest.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()
    # ``competition_download_files`` signature (kaggle>=1.6): competition, path, force, quiet — no ``unzip``.
    api.competition_download_files(COMPETITION, path=str(dest), force=False, quiet=False)
    _extract_archives(dest)
    print(f"Download complete → {dest.resolve()}")


if __name__ == "__main__":
    main()
