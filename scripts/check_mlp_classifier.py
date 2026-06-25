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
    out_dir = repo_root / "results" / "experiments" / "mlp_classifier"

    print(f"Checking MLP classifier results in: {out_dir}")

    metadata = read_json_required(out_dir / "run_metadata.json")
    config_plan = read_csv_required(out_dir / "config_plan.csv")
    cv_results = read_csv_required(out_dir / "cv_results.csv")
    cv_summary = read_csv_required(out_dir / "cv_summary.csv")
    test_results = read_csv_required(out_dir / "test_results.csv")
    best_mlp = read_csv_required(out_dir / "best_mlp_by_dataset.csv")
    label_mappings = read_csv_required(out_dir / "label_mappings.csv")

    expected_cv_rows = len(config_plan) * 5
    expected_test_rows = len(config_plan)
    expected_best_rows = len(metadata["datasets"])

    if len(cv_results) != expected_cv_rows:
        raise ValueError(
            f"cv_results has {len(cv_results)} rows, expected {expected_cv_rows}"
        )

    if len(test_results) != expected_test_rows:
        raise ValueError(
            f"test_results has {len(test_results)} rows, expected {expected_test_rows}"
        )

    if len(best_mlp) != expected_best_rows:
        raise ValueError(
            f"best_mlp_by_dataset has {len(best_mlp)} rows, expected {expected_best_rows}"
        )

    assert_metric_range(cv_results, "cv_results", METRIC_COLS)
    assert_metric_range(test_results, "test_results", METRIC_COLS)

    for df_name, df in [("cv_results", cv_results), ("test_results", test_results)]:
        if (df["n_selected_features"] <= 0).any():
            raise ValueError(f"{df_name}: n_selected_features must be positive")
        if (df["n_selected_features"] > df["n_original_features"]).any():
            raise ValueError(f"{df_name}: selected features exceed original features")
        if df["config_id"].isna().any():
            raise ValueError(f"{df_name}: config_id contains NaN")

    expected_config_ids = set(config_plan["config_id"])
    cv_config_ids = set(cv_results["config_id"])
    test_config_ids = set(test_results["config_id"])

    if cv_config_ids != expected_config_ids:
        missing = expected_config_ids - cv_config_ids
        extra = cv_config_ids - expected_config_ids
        raise ValueError(f"CV config mismatch. missing={missing}, extra={extra}")

    if test_config_ids != expected_config_ids:
        missing = expected_config_ids - test_config_ids
        extra = test_config_ids - expected_config_ids
        raise ValueError(f"Test config mismatch. missing={missing}, extra={extra}")

    for _, row in config_plan.iterrows():
        dataset = row["dataset"]
        config_id = row["config_id"]

        split_test = read_csv_required(repo_root / "data" / "splits" / dataset / "test.csv")
        expected_n_test = len(split_test)

        cm_path = out_dir / "confusion_matrices" / f"{config_id}__mlp.csv"
        check_confusion_matrix(cm_path, expected_n_test)

        if row["method_family"] != "Baseline":
            selected_path = out_dir / "selected_features" / f"{config_id}.csv"
            selected = read_csv_required(selected_path)
            expected_k = int(row["k"])

            if len(selected) != expected_k:
                raise ValueError(
                    f"{selected_path}: has {len(selected)} features, expected {expected_k}"
                )

            if selected["feature_name"].duplicated().any():
                raise ValueError(f"{selected_path}: duplicated feature_name")

            if selected["rank"].min() != 1 or selected["rank"].max() != expected_k:
                raise ValueError(f"{selected_path}: invalid rank range")

    if label_mappings[["dataset", "label_id", "label"]].duplicated().any():
        raise ValueError("label_mappings contains duplicated rows")

    print("PASS: MLP classifier result files are valid.")
    print("")

    print("Best MLP configuration per dataset according to CV Macro F1:")
    cols = [
        "dataset",
        "method_family",
        "selector",
        "k",
        "model",
        "n_selected_features",
        "cv_macro_f1_mean",
        "cv_macro_f1_std",
        "cv_accuracy_mean",
        "cv_accuracy_std",
    ]
    print(best_mlp[cols].to_string(index=False))
    print("")

    print("MLP final test results:")
    test_cols = [
        "dataset",
        "method_family",
        "selector",
        "k",
        "n_selected_features",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "cv_macro_f1_mean",
        "cv_macro_f1_std",
    ]
    print(
        test_results[test_cols]
        .sort_values(["dataset", "macro_f1"], ascending=[True, False])
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
