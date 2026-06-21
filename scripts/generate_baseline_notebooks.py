"""Generate executable baseline modeling notebooks."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


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


def write_notebook(path: Path, cells: list[dict]) -> None:
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")


COMMON_SETUP = r'''
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path.cwd().resolve()
if not (ROOT / "src").is_dir():
    for parent in ROOT.parents:
        if (parent / "src").is_dir():
            ROOT = parent
            break
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from src.evaluation.metrics import metrics_summary, rmse
from src.evaluation.validation import group_well_folds, known_zone_mask, target_zone_mask
from src.io.loaders import discover_wells, load_all_wells
from src.modeling.baseline import InterpolationBaseline
from src.modeling.linear import LinearRegressor, Standardizer
from src.utils.paths import get_artifacts_dir, get_data_dir
from src.utils.run_bundle import RunBundle
from src.utils.seeds import set_global_seeds

set_global_seeds(42)
RUN_DATE = date.today().isoformat()
N_SPLITS = 3
MAX_TRAIN_ROWS_PER_FOLD = 25_000

train_dir = get_data_dir("train")
figures_dir = get_artifacts_dir("figures")
tables_dir = get_artifacts_dir("tables")
models_dir = get_artifacts_dir("models")
figures_dir.mkdir(parents=True, exist_ok=True)
tables_dir.mkdir(parents=True, exist_ok=True)
models_dir.mkdir(parents=True, exist_ok=True)

wells = discover_wells(train_dir)
well_bundle = load_all_wells(train_dir, wells=wells)
folds = group_well_folds(wells, n_splits=N_SPLITS)

print(f"train wells: {len(wells)}")
print(f"folds: {len(folds)}")
'''


HELPERS = r'''
def _safe_gr(series):
    gr = series.clip(lower=0, upper=300)
    return gr.fillna(gr.rolling(21, center=True, min_periods=1).median()).ffill().bfill()


def make_feature_frame(df):
    out = pd.DataFrame(index=df.index)
    out["row_idx"] = df["row_idx"].astype(float)
    out["MD"] = df["MD"].astype(float)
    out["X"] = df["X"].astype(float)
    out["Y"] = df["Y"].astype(float)
    out["Z"] = df["Z"].astype(float)
    out["GR"] = _safe_gr(df["GR"].astype(float))
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


def train_valid_arrays(well_bundle, train_wells, valid_wells):
    train_frames = []
    valid_frames = []
    valid_targets = []
    valid_meta = []

    for well in train_wells:
        df = well_bundle[well]["horizontal"]
        mask = known_zone_mask(df)
        feat = make_feature_frame(df).loc[mask]
        feat["target"] = df.loc[mask, "TVT"].to_numpy()
        train_frames.append(feat)

    for well in valid_wells:
        df = well_bundle[well]["horizontal"]
        mask = target_zone_mask(df)
        feat = make_feature_frame(df).loc[mask]
        valid_frames.append(feat)
        valid_targets.append(df.loc[mask, "TVT"].to_numpy())
        valid_meta.extend([well] * int(mask.sum()))

    train_df = pd.concat(train_frames, axis=0)
    valid_df = pd.concat(valid_frames, axis=0)
    y_train = train_df.pop("target").to_numpy(dtype=float)
    y_valid = np.concatenate(valid_targets).astype(float)

    medians = train_df.median(numeric_only=True)
    X_train = train_df.fillna(medians).to_numpy(dtype=float)
    X_valid = valid_df.fillna(medians).to_numpy(dtype=float)
    return X_train, y_train, X_valid, y_valid, valid_meta, list(train_df.columns)


def sample_training_rows(X, y, max_rows):
    if len(y) <= max_rows:
        return X, y
    idx = np.linspace(0, len(y) - 1, max_rows).astype(int)
    return X[idx], y[idx]


def append_model_comparison(row):
    path = tables_dir / "model_comparison.csv"
    if path.exists():
        table = pd.read_csv(path)
        table = pd.concat([table, pd.DataFrame([row])], ignore_index=True)
    else:
        table = pd.DataFrame([row])
    table = table.drop_duplicates(["model_family"], keep="last")
    table.to_csv(path, index=False)
    return path


def save_metric_bars(metrics_df, path, title):
    width, height = 920, 360
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((24, 18), title, fill="black")
    values = metrics_df["rmse"].to_numpy(dtype=float)
    labels = metrics_df["model"].astype(str).tolist()
    max_value = max(float(values.max()), 1.0)
    left, top, bar_h = 190, 70, 34
    for i, (label, value) in enumerate(zip(labels, values)):
        y = top + i * 52
        draw.text((24, y + 8), label[:22], fill="black")
        bar_w = int((width - left - 80) * value / max_value)
        draw.rectangle((left, y, left + bar_w, y + bar_h), fill="#4c78a8")
        draw.text((left + bar_w + 8, y + 8), f"{value:.4f}", fill="black")
    image.save(path)
    return path


def save_residual_strip(y_true, y_pred, path, title):
    width, height = 920, 360
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((24, 18), title, fill="black")
    residual = np.asarray(y_pred) - np.asarray(y_true)
    if residual.size > 1400:
        idx = np.linspace(0, residual.size - 1, 1400).astype(int)
        residual = residual[idx]
    lo, hi = np.percentile(residual, [1, 99])
    bound = max(abs(float(lo)), abs(float(hi)), 1.0)
    mid = height // 2
    draw.line((40, mid, width - 30, mid), fill="#444444")
    for i, value in enumerate(residual):
        x = 40 + int(i * (width - 80) / max(len(residual) - 1, 1))
        y = mid - int((float(value) / bound) * 120)
        color = "#d95f02" if value > 0 else "#1b9e77"
        draw.line((x, mid, x, y), fill=color)
    draw.text((24, height - 34), f"sampled residuals; clipped at +/-{bound:.2f}", fill="black")
    image.save(path)
    return path
'''


BASELINE_EVAL = r'''
start = time.time()
model = InterpolationBaseline()
fold_rows = []
all_true = []
all_pred = []

for fold_idx, (_, valid_wells) in enumerate(folds, start=1):
    fold_true = []
    fold_pred = []
    for well in valid_wells:
        df = well_bundle[well]["horizontal"]
        mask = target_zone_mask(df)
        pred = model.predict_well(df)[mask.to_numpy()]
        true = df.loc[mask, "TVT"].to_numpy(dtype=float)
        fold_true.append(true)
        fold_pred.append(pred)
    y_true = np.concatenate(fold_true)
    y_pred = np.concatenate(fold_pred)
    summary = metrics_summary(y_true, y_pred)
    summary.update({"fold": fold_idx, "n_wells": len(valid_wells), "n_rows": len(y_true)})
    fold_rows.append(summary)
    all_true.append(y_true)
    all_pred.append(y_pred)

metrics_df = pd.DataFrame(fold_rows)
overall_true = np.concatenate(all_true)
overall_pred = np.concatenate(all_pred)
overall = metrics_summary(overall_true, overall_pred)
train_time_s = time.time() - start

metrics_df
'''


BASELINE_ARTIFACTS = r'''
config = {
    "model": "InterpolationBaseline",
    "validation": "3-fold grouped by well; score TVT_input-missing rows",
    "n_splits": N_SPLITS,
}
with RunBundle("interpolation_baseline", config) as run:
    table_path = tables_dir / f"model__interpolation_baseline_metrics__{run.run_id}.csv"
    figure_path = figures_dir / f"model__interpolation_baseline_residuals__{run.run_id}.png"
    metrics_df.to_csv(table_path, index=False)
    save_residual_strip(overall_true, overall_pred, figure_path, "Interpolation baseline residuals")
    run.add_input(str(train_dir))
    run.add_output(str(table_path))
    run.add_output(str(figure_path))
    run.add_metric("cv_rmse_mean", float(metrics_df["rmse"].mean()))
    run.add_metric("cv_rmse_std", float(metrics_df["rmse"].std(ddof=0)))
    run.add_metric("overall_rmse", float(overall["rmse"]))
    run.set_notes(
        "Naive interpolation baseline. Predicts evaluation-zone TVT by linear interpolation/forward fill of TVT_input."
    )

comparison_path = append_model_comparison(
    {
        "run_id": run.run_id,
        "model_family": "baseline",
        "model_name": "interpolation",
        "cv_rmse_mean": float(metrics_df["rmse"].mean()),
        "cv_rmse_std": float(metrics_df["rmse"].std(ddof=0)),
        "train_time_s": float(train_time_s),
        "n_features": 1,
        "notes": "Stateless interpolation/ffill baseline",
    }
)

print("run_id:", run.run_id)
print("overall_rmse:", overall["rmse"])
print("metrics:", table_path)
print("figure:", figure_path)
print("comparison:", comparison_path)
'''


LINEAR_EVAL = r'''
start = time.time()
candidate_specs = [
    ("ols", "ols", 0.0),
    ("ridge_alpha_0.1", "ridge", 0.1),
    ("ridge_alpha_1", "ridge", 1.0),
    ("ridge_alpha_10", "ridge", 10.0),
    ("ridge_alpha_100", "ridge", 100.0),
    ("lasso_alpha_0.001", "lasso", 0.001),
    ("lasso_alpha_0.01", "lasso", 0.01),
    ("lasso_alpha_0.1", "lasso", 0.1),
]
fold_metric_rows = []
prediction_rows = []
feature_columns = None

for fold_idx, (train_wells, valid_wells) in enumerate(folds, start=1):
    X_train, y_train, X_valid, y_valid, valid_meta, feature_columns = train_valid_arrays(
        well_bundle, train_wells, valid_wells
    )
    X_train, y_train = sample_training_rows(X_train, y_train, MAX_TRAIN_ROWS_PER_FOLD)
    scaler = Standardizer()
    X_train_s = scaler.fit_transform(X_train)
    X_valid_s = scaler.transform(X_valid)

    for model_name, kind, alpha in candidate_specs:
        print(f"fold {fold_idx}: fitting {model_name} on {len(y_train)} rows")
        model = LinearRegressor(kind=kind, alpha=alpha, max_iter=100, tol=1e-4).fit(X_train_s, y_train)
        pred = model.predict(X_valid_s)
        summary = metrics_summary(y_valid, pred)
        summary.update(
            {
                "fold": fold_idx,
                "model": model_name,
                "kind": kind,
                "alpha": alpha,
                "n_train_rows": len(y_train),
                "n_valid_rows": len(y_valid),
                "n_valid_wells": len(valid_wells),
            }
        )
        fold_metric_rows.append(summary)
        prediction_rows.append(
            pd.DataFrame(
                {
                    "fold": fold_idx,
                    "model": model_name,
                    "wellname": valid_meta,
                    "y_true": y_valid,
                    "y_pred": pred,
                    "residual": pred - y_valid,
                }
            )
        )

fold_metrics = pd.DataFrame(fold_metric_rows)
model_summary = (
    fold_metrics.groupby(["model", "kind", "alpha"], as_index=False)
    .agg(
        rmse=("rmse", "mean"),
        rmse_std=("rmse", lambda x: x.std(ddof=0)),
        mae=("mae", "mean"),
        r2=("r2", "mean"),
        n_valid_rows=("n_valid_rows", "sum"),
    )
    .sort_values("rmse", ascending=True)
)
best_row = model_summary.iloc[0].to_dict()
predictions = pd.concat(prediction_rows, ignore_index=True)
best_predictions = predictions[predictions["model"] == best_row["model"]]
train_time_s = time.time() - start

model_summary
'''


LINEAR_FINAL = r'''
all_train_wells = wells
X_all, y_all, _, _, _, feature_columns = train_valid_arrays(well_bundle, all_train_wells[:-1], all_train_wells[-1:])
X_all, y_all = sample_training_rows(X_all, y_all, MAX_TRAIN_ROWS_PER_FOLD)
final_scaler = Standardizer()
X_all_s = final_scaler.fit_transform(X_all)
final_model = LinearRegressor(
    kind=str(best_row["kind"]),
    alpha=float(best_row["alpha"]),
    max_iter=150,
    tol=1e-4,
).fit(X_all_s, y_all)

model_dir = models_dir / "linear"
model_dir.mkdir(parents=True, exist_ok=True)

config = {
    "model_family": "linear",
    "candidates": [name for name, _, _ in candidate_specs],
    "validation": "3-fold grouped by well; train known-zone rows; score target-zone rows",
    "max_train_rows_per_fold": MAX_TRAIN_ROWS_PER_FOLD,
}
with RunBundle("linear_ols_ridge_lasso", config) as run:
    model_path = model_dir / f"model__linear_{best_row['model']}__{run.run_id}.pkl"
    metrics_path = tables_dir / f"model__linear_ols_ridge_lasso_metrics__{run.run_id}.csv"
    fold_path = tables_dir / f"model__linear_ols_ridge_lasso_fold_metrics__{run.run_id}.csv"
    bar_path = figures_dir / f"model__linear_rmse_comparison__{run.run_id}.png"
    residual_path = figures_dir / f"model__linear_best_residuals__{run.run_id}.png"
    final_model.save(model_path)
    model_summary.to_csv(metrics_path, index=False)
    fold_metrics.to_csv(fold_path, index=False)
    save_metric_bars(model_summary, bar_path, "Linear baseline RMSE by candidate")
    save_residual_strip(
        best_predictions["y_true"].to_numpy(),
        best_predictions["y_pred"].to_numpy(),
        residual_path,
        f"Best linear residuals: {best_row['model']}",
    )
    run.add_input(str(train_dir))
    run.add_output(str(metrics_path))
    run.add_output(str(fold_path))
    run.add_output(str(bar_path))
    run.add_output(str(residual_path))
    run.add_output(str(model_path))
    run.add_metric("best_cv_rmse_mean", float(best_row["rmse"]))
    run.add_metric("best_cv_rmse_std", float(best_row["rmse_std"]))
    run.add_metric("best_cv_mae_mean", float(best_row["mae"]))
    run.add_metric("best_cv_r2_mean", float(best_row["r2"]))
    run.set_notes(
        f"First linear baseline sweep. Best candidate: {best_row['model']} with grouped CV RMSE {best_row['rmse']:.6f}."
    )

comparison_path = append_model_comparison(
    {
        "run_id": run.run_id,
        "model_family": "linear",
        "model_name": str(best_row["model"]),
        "cv_rmse_mean": float(best_row["rmse"]),
        "cv_rmse_std": float(best_row["rmse_std"]),
        "train_time_s": float(train_time_s),
        "n_features": len(feature_columns),
        "notes": "OLS/Ridge/Lasso NumPy sweep; grouped well CV",
    }
)

print("run_id:", run.run_id)
print("best_model:", best_row["model"])
print("best_rmse:", best_row["rmse"])
print("metrics:", metrics_path)
print("figures:", bar_path, residual_path)
print("model:", model_path)
print("comparison:", comparison_path)
'''


def main() -> None:
    write_notebook(
        ROOT / "notebooks/modeling/00_baseline/00_interpolation_baseline.ipynb",
        [
            markdown("# Modeling - Interpolation Baseline\n\nA deliberately naive anchor model. It predicts target-zone TVT by linearly interpolating or forward-filling the known `TVT_input` series within each well."),
            markdown("## 1. Setup\n\nLoad project paths, data, metric helpers, and the train wells. (Imports and constants.)"),
            code(COMMON_SETUP),
            markdown("## 2. Shared Helpers\n\nSmall notebook helpers for artifact writing and simple residual visualizations. (Artifact utilities.)"),
            code(HELPERS),
            markdown("## 3. Grouped Masked-Zone Evaluation\n\nScore only rows where `TVT_input` is missing and true `TVT` exists. (Competition-like validation.)"),
            code(BASELINE_EVAL),
            markdown("## 4. Artifacts And Run Bundle\n\nExport fold metrics, a residual plot, the run bundle, and the master model-comparison row. (Reportable outputs.)"),
            code(BASELINE_ARTIFACTS),
        ],
    )

    write_notebook(
        ROOT / "notebooks/modeling/01_linear/01_ols_ridge_lasso.ipynb",
        [
            markdown("# Modeling - OLS, Ridge, And Lasso\n\nFirst regression baseline sweep using compact NumPy implementations. OLS is the plain linear anchor; Ridge adds L2 shrinkage; Lasso adds L1 shrinkage for sparse coefficients."),
            markdown("## 1. Setup\n\nLoad project paths, data, metric helpers, and the train wells. (Imports and constants.)"),
            code(COMMON_SETUP),
            markdown("## 2. Feature Helpers\n\nBuild a simple test-safe numeric feature matrix from each well. (Trajectory, GR, and known-TVt context features.)"),
            code(HELPERS),
            markdown("## 3. Linear Regression Sweep\n\nTrain OLS, Ridge, and Lasso candidates on known-zone rows and validate on held-out wells' target-zone rows. (Grouped CV.)"),
            code(LINEAR_EVAL),
            markdown("## 4. Final Fit, Artifacts, And Run Bundle\n\nFit the selected candidate on a capped known-zone training sample, export metrics and figures, and record the run. (Reportable outputs.)"),
            code(LINEAR_FINAL),
        ],
    )


if __name__ == "__main__":
    main()
