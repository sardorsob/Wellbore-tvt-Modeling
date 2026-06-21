"""Generate a self-contained Kaggle linear Ridge submission notebook."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KERNEL_DIR = ROOT / "kaggle" / "linear_ridge_submission"
NOTEBOOK_PATH = KERNEL_DIR / "linear_ridge_submission.ipynb"
METADATA_PATH = KERNEL_DIR / "kernel-metadata.json"


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(True),
    }


SETUP = r'''
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

KAGGLE_ROOT = Path("/kaggle")
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_OUTPUT = Path("/kaggle/working")

def resolve_base_dir():
    known = Path("/kaggle/input/rogii-wellbore-geology-prediction")
    local = Path.cwd() / "data" / "raw"
    for candidate in [known, local]:
        if (candidate / "sample_submission.csv").exists() and (candidate / "train").exists() and (candidate / "test").exists():
            return candidate
    if KAGGLE_INPUT_ROOT.exists():
        for sample_path in KAGGLE_INPUT_ROOT.rglob("sample_submission.csv"):
            candidate = sample_path.parent
            if (candidate / "train").exists() and (candidate / "test").exists():
                return candidate
        children = sorted(str(path) for path in KAGGLE_INPUT_ROOT.glob("*"))
        raise FileNotFoundError(f"Could not find competition data under /kaggle/input. Children: {children}")
    raise FileNotFoundError("Could not find Kaggle or local competition data.")

BASE_DIR = resolve_base_dir()

if KAGGLE_ROOT.exists():
    OUTPUT_PATH = KAGGLE_OUTPUT / "submission.csv"
else:
    OUTPUT_PATH = Path.cwd() / "artifacts" / "reports" / f"submission__linear_ridge_alpha_10__{date.today().isoformat()}.csv"

TRAIN_DIR = BASE_DIR / "train"
TEST_DIR = BASE_DIR / "test"
SAMPLE_PATH = BASE_DIR / "sample_submission.csv"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

RIDGE_ALPHA = 10.0
MAX_TRAIN_ROWS = 25_000

print("BASE_DIR:", BASE_DIR)
print("OUTPUT_PATH:", OUTPUT_PATH)
'''


HELPERS = r'''
def discover_wells(directory):
    return sorted(path.name.replace("__horizontal_well.csv", "") for path in Path(directory).glob("*__horizontal_well.csv"))


def load_horizontal_well(path):
    df = pd.read_csv(path)
    df.index.name = "row_idx"
    df["row_idx"] = df.index
    return df


def load_sample_submission(path):
    sample = pd.read_csv(path)
    split = sample["id"].str.rsplit("_", n=1, expand=True)
    sample["wellname"] = split[0]
    sample["row_idx"] = split[1].astype(int)
    return sample


def safe_gr(series):
    gr = series.astype(float).clip(lower=0, upper=300)
    return gr.fillna(gr.rolling(21, center=True, min_periods=1).median()).ffill().bfill()


def make_feature_frame(df):
    out = pd.DataFrame(index=df.index)
    out["row_idx"] = df["row_idx"].astype(float)
    out["MD"] = df["MD"].astype(float)
    out["X"] = df["X"].astype(float)
    out["Y"] = df["Y"].astype(float)
    out["Z"] = df["Z"].astype(float)
    out["GR"] = safe_gr(df["GR"])
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


class Standardizer:
    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        scale = X.std(axis=0)
        self.scale_ = np.where(scale == 0.0, 1.0, scale)
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=float) - self.mean_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class RidgeRegressor:
    def __init__(self, alpha=10.0):
        self.alpha = float(alpha)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        design = np.hstack([np.ones((X.shape[0], 1)), X])
        penalty = np.eye(design.shape[1]) * self.alpha
        penalty[0, 0] = 0.0
        coef = np.linalg.pinv(design.T @ design + penalty) @ design.T @ y
        self.intercept_ = float(coef[0])
        self.coef_ = coef[1:]
        return self

    def predict(self, X):
        return self.intercept_ + np.asarray(X, dtype=float) @ self.coef_


def sample_training_rows(X, y, max_rows):
    if len(y) <= max_rows:
        return X, y
    idx = np.linspace(0, len(y) - 1, max_rows).astype(int)
    return X[idx], y[idx]
'''


TRAIN = r'''
train_frames = []
train_wells = discover_wells(TRAIN_DIR)

for well in train_wells:
    df = load_horizontal_well(TRAIN_DIR / f"{well}__horizontal_well.csv")
    mask = df["TVT_input"].notna() & df["TVT"].notna()
    features = make_feature_frame(df).loc[mask]
    features["target"] = df.loc[mask, "TVT"].to_numpy(dtype=float)
    train_frames.append(features)

train_df = pd.concat(train_frames, axis=0)
y_train = train_df.pop("target").to_numpy(dtype=float)
feature_columns = list(train_df.columns)
feature_medians = train_df.median(numeric_only=True)
X_train = train_df.fillna(feature_medians).to_numpy(dtype=float)
X_train, y_train = sample_training_rows(X_train, y_train, MAX_TRAIN_ROWS)

scaler = Standardizer()
X_train_scaled = scaler.fit_transform(X_train)
model = RidgeRegressor(alpha=RIDGE_ALPHA).fit(X_train_scaled, y_train)

print("train_wells:", len(train_wells))
print("train_rows_used:", len(y_train))
print("n_features:", len(feature_columns))
'''


PREDICT = r'''
sample = load_sample_submission(SAMPLE_PATH)
test_wells = discover_wells(TEST_DIR)
predictions = {}

for well in test_wells:
    df = load_horizontal_well(TEST_DIR / f"{well}__horizontal_well.csv")
    features = make_feature_frame(df).reindex(columns=feature_columns)
    X_test = features.fillna(feature_medians).to_numpy(dtype=float)
    pred = model.predict(scaler.transform(X_test))
    for row_idx, value in zip(df.index, pred):
        predictions[f"{well}_{row_idx}"] = float(value)

submission = sample[["id"]].copy()
submission["tvt"] = submission["id"].map(predictions)

assert list(submission.columns) == ["id", "tvt"]
assert len(submission) == len(sample)
assert submission["id"].equals(sample["id"])
assert submission["tvt"].notna().all()
assert np.isfinite(submission["tvt"].to_numpy()).all()

submission.to_csv(OUTPUT_PATH, index=False)

print("test_wells:", len(test_wells))
print("submission_rows:", len(submission))
print("submission_path:", OUTPUT_PATH)
submission.head()
'''


def write_notebook() -> None:
    cells = [
        markdown(
            "# Linear Ridge Submission\n\n"
            "Self-contained Kaggle notebook for the first Ridge baseline. "
            "It trains on known-zone training rows and writes `/kaggle/working/submission.csv`."
        ),
        markdown("## 1. Setup\n\nResolve Kaggle/local paths and set the first Ridge baseline constants. (Paths and constants.)"),
        code(SETUP),
        markdown("## 2. Helpers\n\nSmall loaders, feature builder, scaler, and closed-form Ridge regression. (Self-contained model code.)"),
        code(HELPERS),
        markdown("## 3. Train Ridge Baseline\n\nFit `ridge_alpha_10` on known-zone rows from the training wells. (Training.)"),
        code(TRAIN),
        markdown("## 4. Write Submission\n\nPredict all sample-submission ids, validate shape/order/finite values, and save `submission.csv`. (Competition output.)"),
        code(PREDICT),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    KERNEL_DIR.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2), encoding="utf-8")


def write_metadata() -> None:
    metadata = {
        "id": "sardorbsobirov/wellbore-linear-ridge-baseline",
        "title": "Wellbore Linear Ridge Baseline",
        "code_file": NOTEBOOK_PATH.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": False,
        "competition_sources": ["rogii-wellbore-geology-prediction"],
        "dataset_sources": [],
        "kernel_sources": [],
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    write_notebook()
    write_metadata()


if __name__ == "__main__":
    main()
