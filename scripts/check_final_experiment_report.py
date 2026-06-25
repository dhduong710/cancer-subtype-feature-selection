#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_FILES = [
    "final_experiment_report.md",
    "final_comparison_table.csv",
    "best_overall_by_cv.csv",
    "best_descriptive_test.csv",
    "mlp_vs_best_classical.csv",
    "figures/cv_macro_f1_comparison.png",
    "figures/test_macro_f1_comparison.png",
    "figures/selected_features_comparison.png",
]


METRIC_COLS = [
    "cv_accuracy_mean",
    "cv_macro_f1_mean",
    "test_accuracy",
    "test_macro_f1",
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> None:
    root = repo_root()
    report_dir = root / "reports" / "final_experiments"

    print(f"Checking final experiment report in: {report_dir}")

    for rel in REQUIRED_FILES:
        path = report_dir / rel
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")
        if path.stat().st_size == 0:
            raise ValueError(f"File is empty: {path}")

    final = pd.read_csv(report_dir / "final_comparison_table.csv")
    best_cv = pd.read_csv(report_dir / "best_overall_by_cv.csv")
    best_test = pd.read_csv(report_dir / "best_descriptive_test.csv")
    mlp_compare = pd.read_csv(report_dir / "mlp_vs_best_classical.csv")

    expected_datasets = {
        "mogcn_brca",
        "cancersd_stad",
        "mogonet_brca_optional",
    }

    expected_groups = {
        "Classical baseline",
        "Classical Filter",
        "Classical Wrapper",
        "Classical Embedded",
        "MLP best",
    }

    if set(final["dataset"]) != expected_datasets:
        raise ValueError("final_comparison_table.csv does not contain all expected datasets")

    for dataset in expected_datasets:
        groups = set(final[final["dataset"] == dataset]["experiment_group"])
        if groups != expected_groups:
            raise ValueError(f"{dataset}: groups mismatch: {groups}")

    if len(final) != len(expected_datasets) * len(expected_groups):
        raise ValueError(f"Expected 15 rows in final table, got {len(final)}")

    if len(best_cv) != len(expected_datasets):
        raise ValueError(f"Expected 3 rows in best_overall_by_cv.csv, got {len(best_cv)}")

    if len(best_test) != len(expected_datasets):
        raise ValueError(f"Expected 3 rows in best_descriptive_test.csv, got {len(best_test)}")

    if len(mlp_compare) != len(expected_datasets):
        raise ValueError(f"Expected 3 rows in mlp_vs_best_classical.csv, got {len(mlp_compare)}")

    for col in METRIC_COLS:
        if final[col].isna().any():
            raise ValueError(f"{col} contains NaN")
        bad = final[(final[col] < 0) | (final[col] > 1)]
        if len(bad) > 0:
            raise ValueError(f"{col} contains values outside [0, 1]")

    if (final["n_selected_features"] <= 0).any():
        raise ValueError("n_selected_features must be positive")

    if (final["n_selected_features"] > final["n_original_features"]).any():
        raise ValueError("selected features exceed original features")

    print("PASS: final experiment report files are valid.")
    print("")
    print("Best overall by CV Macro F1:")
    cols = [
        "dataset",
        "experiment_group",
        "method_family",
        "selector",
        "k",
        "model",
        "n_selected_features",
        "cv_macro_f1_mean",
        "cv_macro_f1_std",
        "test_macro_f1",
    ]
    print(best_cv[cols].to_string(index=False))
    print("")
    print("MLP vs best classical:")
    print(mlp_compare.to_string(index=False))


if __name__ == "__main__":
    main()
