import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from src.utils.run_bundle import RunBundle


def test_run_bundle_writes_required_files():
    tmp_path = Path(tempfile.mkdtemp())

    with patch("src.utils.run_bundle.get_artifacts_dir", lambda subdir="": tmp_path / subdir):
        with RunBundle("unit", {"model": {"name": "baseline"}}, seed=7) as bundle:
            bundle.add_input("data/raw/train")
            bundle.add_output("artifacts/tables/example.csv")
            bundle.add_metric("rmse", 1.25)
            bundle.set_notes("unit-test bundle")

    required = {
        "meta.json",
        "config.yaml",
        "seeds.json",
        "outputs.json",
        "metrics.json",
        "notes.md",
    }
    assert required.issubset({path.name for path in bundle.bundle_dir.iterdir()})
    assert json.loads((bundle.bundle_dir / "metrics.json").read_text())["rmse"] == 1.25
    assert "baseline" in (bundle.bundle_dir / "config.yaml").read_text()
