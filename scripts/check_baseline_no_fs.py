#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


DATASETS = [
    "mogcn_brca",
    "cancersd_stad",
    "mogonet_brca_optional",
]

MODELS = [
    "logistic_regression",
    "linear_svm",
    "random_forest",
    "gaussian_nb",
]

METRIC_COLS = [
    "accuracy",
    "balanced_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_f1",
]


def find_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def read_json_required(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def assert_metric_range(df: pd.DataFrame, name: str, metric_cols: List[str]) -> None:
    for col in metric_cols:
        if col not in df.columns:
            raise ValueError(f"{name}: missing metric column {col}")
        if df[col].isna().any():
            raise ValueError(f"{name}: metric column {col} contains NaN")
        bad = df[(df[col] < 0) | (df[col] > 1)]
        if len(bad) > 0:
            raise ValueError(f"{name}: metric column {col} contains values outside [0, 1]")


def check_confusion_matrix(path: Path, expected_n_test: int) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing confusion matrix: {path}")

    cm = pd.read_csv(path, index_col=0)

    if cm.shape[0] != cm.shape[1]:
        raise ValueError(f"{path}: confusion matrix is not square: {cm.shape}")

    total = int(cm.to_numpy().sum())
    if total != int(expected_n_test):
        raise ValueError(
            f"{path}: confusion matrix total={total}, expected test samples={expected_n_test}"
        )


def main() -> None:
    repo_root = find_repo_root()
    out_dir = repo_root / "results" / "experiments" / "baseline_no_fs"

    print(f"Checking baseline results in: {out_dir}")

    metadata = read_json_required(out_dir / "run_metadata.json")
    cv_results = read_csv_required(out_dir / "cv_results.csv")
    cv_summary = read_csv_required(out_dir / "cv_summary.csv")
    test_results = read_csv_required(out_dir / "test_results.csv")
    best_models = read_csv_required(out_dir / "best_models_by_cv.csv")
    label_mappings = read_csv_required(out_dir / "label_mappings.csv")

    datasets = metadata["datasets"]
    models = metadata["models"]

    expected_cv_rows = len(datasets) * len(models) * 5
    expected_test_rows = len(datasets) * len(models)

    if len(cv_results) != expected_cv_rows:
        raise ValueError(
            f"cv_results has {len(cv_results)} rows, expected {expected_cv_rows}"
        )

    if len(test_results) != expected_test_rows:
        raise ValueError(
            f"test_results has {len(test_results)} rows, expected {expected_test_rows}"
        )

    if len(best_models) != len(datasets):
        raise ValueError(
            f"best_models_by_cv has {len(best_models)} rows, expected {len(datasets)}"
        )

    assert_metric_range(cv_results, "cv_results", METRIC_COLS)
    assert_metric_range(test_results, "test_results", METRIC_COLS)

    cv_summary_metric_cols = [
        col
        for col in cv_summary.columns
        if col.startswith("cv_") and (col.endswith("_mean") or col.endswith("_std"))
    ]
    for col in cv_summary_metric_cols:
        if cv_summary[col].isna().any():
            raise ValueError(f"cv_summary: column {col} contains NaN")

    seen_pairs = set(zip(test_results["dataset"], test_results["model"]))
    expected_pairs = set((d, m) for d in datasets for m in models)

    if seen_pairs != expected_pairs:
        missing = expected_pairs - seen_pairs
        extra = seen_pairs - expected_pairs
        raise ValueError(f"test result pairs mismatch. missing={missing}, extra={extra}")

    for dataset in datasets:
        split_test = read_csv_required(repo_root / "data" / "splits" / dataset / "test.csv")
        expected_n_test = len(split_test)

        for model in models:
            cm_path = out_dir / "confusion_matrices" / f"{dataset}__{model}.csv"
            check_confusion_matrix(cm_path, expected_n_test)

    if label_mappings[["dataset", "label_id", "label"]].duplicated().any():
        raise ValueError("label_mappings contains duplicated dataset/label_id/label rows")

    print("PASS: baseline result files are valid.")
    print("")

    print("Best baseline model per dataset according to CV macro F1:")
    cols = [
        "dataset",
        "model",
        "cv_macro_f1_mean",
        "cv_macro_f1_std",
        "cv_accuracy_mean",
        "cv_accuracy_std",
    ]
    print(best_models[cols].to_string(index=False))
    print("")

    print("Final test results:")
    test_cols = [
        "dataset",
        "model",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "cv_macro_f1_mean",
        "cv_macro_f1_std",
    ]
    print(test_results[test_cols].sort_values(["dataset", "macro_f1"], ascending=[True, False]).to_string(index=False))


if __name__ == "__main__":
    main()
