"""Run bundle writer — records per-experiment metadata to artifacts/logs/runs/."""

import json
import platform
import subprocess
from datetime import datetime
from pathlib import Path

import yaml

from .paths import get_artifacts_dir
from .seeds import make_seed_dict


def _git_short_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _make_run_id(tag: str) -> str:
    ts = datetime.now().strftime("%Y-%m-%d__%H%M")
    sha = _git_short_sha()
    return f"{ts}__{tag}__{sha}"


class RunBundle:
    """Context manager that writes a run bundle on exit.

    Usage::

        with RunBundle("xgboost_baseline", config) as rb:
            model.fit(X_train, y_train)
            rb.add_metric("rmse_cv", 2.34)
            rb.add_output("artifacts/models/xgb.pkl")
    """

    def __init__(self, tag: str, config: dict, seed: int = 42):
        self.run_id = _make_run_id(tag)
        self.config = config
        self.seed = seed
        self.bundle_dir: Path = get_artifacts_dir(f"logs/runs/{self.run_id}")
        self._metrics: dict = {}
        self._outputs: list = []
        self._inputs: list = []
        self._notes: str = ""

    def __enter__(self):
        self.bundle_dir.mkdir(parents=True, exist_ok=True)
        self._write_meta()
        self._write_config()
        self._write_seeds()
        return self

    def __exit__(self, *_):
        self._write_metrics()
        self._write_outputs()
        self._write_notes()

    # ── public helpers ────────────────────────────────────────────────────────

    def add_metric(self, name: str, value: float) -> None:
        self._metrics[name] = value

    def add_output(self, path: str) -> None:
        self._outputs.append(str(path))

    def add_input(self, path: str) -> None:
        self._inputs.append(str(path))

    def set_notes(self, notes: str) -> None:
        self._notes = notes

    # ── internal writers ──────────────────────────────────────────────────────

    def _write_meta(self) -> None:
        meta = {
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            "user": platform.node(),
            "python": platform.python_version(),
            "os": platform.system(),
            "git_sha": _git_short_sha(),
        }
        (self.bundle_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    def _write_config(self) -> None:
        (self.bundle_dir / "config.yaml").write_text(
            yaml.dump(self.config, default_flow_style=False)
        )

    def _write_seeds(self) -> None:
        (self.bundle_dir / "seeds.json").write_text(
            json.dumps(make_seed_dict(self.seed), indent=2)
        )

    def _write_metrics(self) -> None:
        (self.bundle_dir / "metrics.json").write_text(
            json.dumps(self._metrics, indent=2)
        )

    def _write_outputs(self) -> None:
        payload = {
            "inputs": self._inputs,
            "outputs": self._outputs,
        }
        (self.bundle_dir / "outputs.json").write_text(json.dumps(payload, indent=2))

    def _write_notes(self) -> None:
        (self.bundle_dir / "notes.md").write_text(self._notes)
