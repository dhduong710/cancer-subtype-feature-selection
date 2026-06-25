#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pandas as pd


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
            raise ValueError(f"{name}: metric column {col} outside [0, 1]")


def check_confusion_matrix(path: Path, expected_n_test: int) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing confusion matrix: {path}")

    cm = pd.read_csv(path, index_col=0)

    if cm.shape[0] != cm.shape[1]:
        raise ValueError(f"{path}: confusion matrix is not square")

    total = int(cm.to_numpy().sum())
    if total != int(expected_n_test):
        raise ValueError(
            f"{path}: confusion matrix total={total}, expected={expected_n_test}"
        )


def main() -> None:
    repo_root = find_repo_root()
    out_dir = repo_root / "results" / "experiments" / "embedded_feature_selection"

    print(f"Checking embedded feature selection results in: {out_dir}")

    metadata = read_json_required(out_dir / "run_metadata.json")
    cv_results = read_csv_required(out_dir / "cv_results.csv")
    cv_summary = read_csv_required(out_dir / "cv_summary.csv")
    test_results = read_csv_required(out_dir / "test_results.csv")
    best_by_dataset = read_csv_required(out_dir / "best_configs_by_dataset.csv")
    best_by_dataset_selector = read_csv_required(out_dir / "best_configs_by_dataset_selector.csv")
    label_mappings = read_csv_required(out_dir / "label_mappings.csv")

    datasets = metadata["datasets"]
    selectors = metadata["selectors"]
    k_values = metadata["k_values"]
    models = metadata["models"]

    expected_cv_rows = len(datasets) * len(selectors) * len(k_values) * len(models) * 5
    expected_test_rows = len(datasets) * len(selectors) * len(k_values) * len(models)
    expected_best_dataset_rows = len(datasets)
    expected_best_dataset_selector_rows = len(datasets) * len(selectors)

    if len(cv_results) != expected_cv_rows:
        raise ValueError(
            f"cv_results has {len(cv_results)} rows, expected {expected_cv_rows}"
        )

    if len(test_results) != expected_test_rows:
        raise ValueError(
            f"test_results has {len(test_results)} rows, expected {expected_test_rows}"
        )

    if len(best_by_dataset) != expected_best_dataset_rows:
        raise ValueError(
            f"best_configs_by_dataset has {len(best_by_dataset)} rows, "
            f"expected {expected_best_dataset_rows}"
        )

    if len(best_by_dataset_selector) != expected_best_dataset_selector_rows:
        raise ValueError(
            f"best_configs_by_dataset_selector has {len(best_by_dataset_selector)} rows, "
            f"expected {expected_best_dataset_selector_rows}"
        )

    assert_metric_range(cv_results, "cv_results", METRIC_COLS)
    assert_metric_range(test_results, "test_results", METRIC_COLS)

    for df_name, df in [
        ("cv_results", cv_results),
        ("test_results", test_results),
    ]:
        if (df["n_selected_features"] <= 0).any():
            raise ValueError(f"{df_name}: n_selected_features must be positive")
        if (df["n_selected_features"] > df["n_original_features"]).any():
            raise ValueError(f"{df_name}: selected features exceed original features")

    expected_pairs = set(
        (d, s, int(k), m)
        for d in datasets
        for s in selectors
        for k in k_values
        for m in models
    )
    seen_test_pairs = set(
        (row.dataset, row.selector, int(row.k), row.model)
        for row in test_results.itertuples(index=False)
    )

    if seen_test_pairs != expected_pairs:
        missing = expected_pairs - seen_test_pairs
        extra = seen_test_pairs - expected_pairs
        raise ValueError(f"test result config mismatch. missing={missing}, extra={extra}")

    for dataset in datasets:
        split_test = read_csv_required(repo_root / "data" / "splits" / dataset / "test.csv")
        expected_n_test = len(split_test)

        for selector in selectors:
            for k in k_values:
                selected_path = (
                    out_dir
                    / "selected_features"
                    / f"{dataset}__{selector}__k{k}.csv"
                )
                selected = read_csv_required(selected_path)

                if len(selected) != int(k):
                    raise ValueError(
                        f"{selected_path}: has {len(selected)} selected features, expected {k}"
                    )

                if selected["feature_name"].duplicated().any():
                    raise ValueError(f"{selected_path}: duplicated feature_name")

                if selected["rank"].min() != 1 or selected["rank"].max() != int(k):
                    raise ValueError(f"{selected_path}: invalid rank range")

                if selected["score"].isna().any():
                    raise ValueError(f"{selected_path}: score contains NaN")

                for model in models:
                    cm_path = (
                        out_dir
                        / "confusion_matrices"
                        / f"{dataset}__{selector}__k{k}__{model}.csv"
                    )
                    check_confusion_matrix(cm_path, expected_n_test)

    if label_mappings[["dataset", "label_id", "label"]].duplicated().any():
        raise ValueError("label_mappings contains duplicated rows")

    print("PASS: embedded feature selection result files are valid.")
    print("")

    print("Best embedded configuration per dataset according to CV Macro F1:")
    cols = [
        "dataset",
        "selector",
        "k",
        "model",
        "cv_macro_f1_mean",
        "cv_macro_f1_std",
        "cv_accuracy_mean",
        "cv_accuracy_std",
    ]
    print(best_by_dataset[cols].to_string(index=False))
    print("")

    print("Best embedded configuration per dataset and selector:")
    cols2 = [
        "dataset",
        "selector",
        "k",
        "model",
        "cv_macro_f1_mean",
        "cv_macro_f1_std",
    ]
    print(best_by_dataset_selector[cols2].to_string(index=False))


if __name__ == "__main__":
    main()
