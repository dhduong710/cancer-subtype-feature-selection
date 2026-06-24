#!/usr/bin/env python3
"""Generate preprocessing report for MOGONET BRCA optional dataset."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate MOGONET BRCA preprocessing report")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed/mogonet_brca_optional"),
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("results/data_summaries/mogonet_brca_summary.json"),
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("reports/data_preprocessing"),
    )
    parser.add_argument(
        "--deep-check",
        action="store_true",
        help="Load full X.csv to verify missing values, finite values, and constant features",
    )
    return parser.parse_args()


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def load_summary(path: Path) -> Dict[str, Any]:
    require_file(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def markdown_table(rows: List[Dict[str, Any]], columns: List[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = [
        "| " + " | ".join(str(row.get(col, "")) for col in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, sep] + body)


def build_label_table(y: pd.DataFrame) -> pd.DataFrame:
    table = (
        y.groupby(["label_id", "label"])
        .size()
        .reset_index(name="sample_count")
        .sort_values("label_id")
    )
    total = int(table["sample_count"].sum())
    table["percentage"] = (table["sample_count"] / total * 100).round(2)
    return table


def build_split_label_table(y: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    merged = y.merge(metadata[["sample_id", "source_split"]], on="sample_id", how="left")
    table = (
        merged.groupby(["label_id", "label", "source_split"])
        .size()
        .reset_index(name="sample_count")
        .sort_values(["label_id", "source_split"])
    )
    return table


def make_label_distribution_figure(label_table: pd.DataFrame, figure_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(label_table["label"], label_table["sample_count"])
    ax.set_title("MOGONET BRCA subtype distribution")
    ax.set_xlabel("Subtype")
    ax.set_ylabel("Number of samples")
    ax.tick_params(axis="x", rotation=25)

    for idx, value in enumerate(label_table["sample_count"]):
        ax.text(idx, value, str(value), ha="center", va="bottom")

    fig.tight_layout()
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)


def lightweight_validate(
    x_path: Path,
    y: pd.DataFrame,
    metadata: pd.DataFrame,
    features: pd.DataFrame,
    x_train_path: Path,
    x_test_path: Path,
    y_train_path: Path,
    y_test_path: Path,
) -> Dict[str, Any]:
    x_header = pd.read_csv(x_path, nrows=0).columns.tolist()
    x_id_col = x_header[0]
    x_feature_count = len(x_header) - 1

    x_ids = pd.read_csv(x_path, usecols=[x_id_col])[x_id_col].astype(str).tolist()
    y_ids = y["sample_id"].astype(str).tolist()
    metadata_ids = metadata["sample_id"].astype(str).tolist()

    x_train_ids = pd.read_csv(x_train_path, usecols=[x_id_col])[x_id_col].astype(str).tolist()
    x_test_ids = pd.read_csv(x_test_path, usecols=[x_id_col])[x_id_col].astype(str).tolist()
    y_train = pd.read_csv(y_train_path)
    y_test = pd.read_csv(y_test_path)

    train_ids_from_metadata = (
        metadata.loc[metadata["source_split"] == "train", "sample_id"].astype(str).tolist()
    )
    test_ids_from_metadata = (
        metadata.loc[metadata["source_split"] == "test", "sample_id"].astype(str).tolist()
    )

    problems: List[str] = []

    if len(x_ids) != len(y_ids):
        problems.append(f"X sample count ({len(x_ids)}) != y sample count ({len(y_ids)})")
    if x_ids != y_ids:
        problems.append("X sample order does not match y.sample_id order")
    if y_ids != metadata_ids:
        problems.append("y.sample_id order does not match metadata.sample_id order")
    if y["sample_id"].duplicated().any():
        problems.append("Duplicate sample IDs in y.csv")
    if metadata["sample_id"].duplicated().any():
        problems.append("Duplicate sample IDs in metadata.csv")
    if x_feature_count != len(features):
        problems.append(f"X feature count ({x_feature_count}) != feature_names rows ({len(features)})")
    if features["feature_name"].duplicated().any():
        problems.append("Duplicate feature names in feature_names.csv")

    if x_train_ids != train_ids_from_metadata:
        problems.append("X_train sample order does not match metadata train split")
    if x_test_ids != test_ids_from_metadata:
        problems.append("X_test sample order does not match metadata test split")
    if y_train["sample_id"].astype(str).tolist() != train_ids_from_metadata:
        problems.append("y_train sample order does not match metadata train split")
    if y_test["sample_id"].astype(str).tolist() != test_ids_from_metadata:
        problems.append("y_test sample order does not match metadata test split")
    if set(train_ids_from_metadata).intersection(set(test_ids_from_metadata)):
        problems.append("Train/test sample IDs overlap")

    return {
        "status": "PASS" if not problems else "FAIL",
        "problems": problems,
        "x_sample_count": len(x_ids),
        "x_feature_count": x_feature_count,
    }


def deep_validate(x_path: Path) -> Dict[str, Any]:
    X = pd.read_csv(x_path, index_col=0)
    missing_count = int(X.isna().sum().sum())
    finite_values = bool(np.isfinite(X.to_numpy(dtype=float)).all())
    constant_feature_count = int((X.nunique(dropna=False) <= 1).sum())

    problems: List[str] = []
    if missing_count != 0:
        problems.append(f"X.csv contains {missing_count} missing values")
    if not finite_values:
        problems.append("X.csv contains inf or -inf values")
    if constant_feature_count != 0:
        problems.append(f"X.csv still contains {constant_feature_count} constant features")

    return {
        "status": "PASS" if not problems else "FAIL",
        "problems": problems,
        "missing_value_count": missing_count,
        "finite_values": finite_values,
        "constant_feature_count": constant_feature_count,
    }


def write_report(
    report_path: Path,
    summary: Dict[str, Any],
    label_table: pd.DataFrame,
    split_table: pd.DataFrame,
    file_table: pd.DataFrame,
    light_check: Dict[str, Any],
    deep_check: Dict[str, Any] | None,
    figure_rel_path: str,
) -> None:
    label_rows = [
        {
            "Class ID": int(row["label_id"]),
            "Subtype": row["label"],
            "Samples": int(row["sample_count"]),
            "Percentage (%)": row["percentage"],
        }
        for _, row in label_table.iterrows()
    ]

    split_rows = [
        {
            "Class ID": int(row["label_id"]),
            "Subtype": row["label"],
            "Split": row["source_split"],
            "Samples": int(row["sample_count"]),
        }
        for _, row in split_table.iterrows()
    ]

    file_rows = [
        {
            "File": row["file"],
            "Description": row["description"],
            "Size (MB)": row["size_mb"],
        }
        for _, row in file_table.iterrows()
    ]

    validation_rows = [
        {
            "Check": "Lightweight structural validation",
            "Status": light_check["status"],
            "Details": "; ".join(light_check["problems"]) if light_check["problems"] else "No problems found",
        }
    ]

    if deep_check is not None:
        validation_rows.append(
            {
                "Check": "Deep validation",
                "Status": deep_check["status"],
                "Details": "; ".join(deep_check["problems"]) if deep_check["problems"] else "No problems found",
            }
        )

    deep_check_text = ""
    if deep_check is not None:
        deep_check_text = f"""
### Deep validation details

| Metric | Value |
|---|---:|
| Missing values in `X.csv` | {deep_check["missing_value_count"]} |
| All values finite | {deep_check["finite_values"]} |
| Constant features after processing | {deep_check["constant_feature_count"]} |
"""

    label_mapping_rows = "\n".join(
        [f"| {class_id} | {label} |" for class_id, label in summary["label_mapping"].items()]
    )

    report = f"""# MOGONET BRCA Data Preprocessing Report

Generated at: `{datetime.now().isoformat(timespec="seconds")}`

## 1. Dataset objective

This report documents the preprocessing of the MOGONET BRCA omics-1 dataset for the project:

**A comparative study of feature selection method for cancer classification using gene expression data**

This dataset is used as an **optional benchmark dataset**. Unlike MoGCN BRCA and CancerSD STAD, the MOGONET BRCA omics-1 data is already preprocessed and feature-reduced by the original MOGONET dataset. Therefore, it should not be treated as raw gene expression data.

## 2. Raw input data

The raw files used in this step are:

- `1_tr.csv`: training omics-1 feature matrix.
- `1_te.csv`: testing omics-1 feature matrix.
- `1_featname.csv`: feature names.
- `labels_tr.csv`: training labels.
- `labels_te.csv`: testing labels.

| Item | Value |
|---|---:|
| Raw train expression shape | {summary["raw_train_expression_shape_rows_cols"]} |
| Raw test expression shape | {summary["raw_test_expression_shape_rows_cols"]} |
| Raw feature count | {summary["raw_feature_count"]} |
| Train sample count | {summary["train_sample_count"]} |
| Test sample count | {summary["test_sample_count"]} |
| Final sample count | {summary["final_sample_count"]} |

## 3. Preprocessing procedure

The following preprocessing steps were applied:

1. Loaded `1_tr.csv` and `1_te.csv` as the omics-1 feature matrices.
2. Loaded `1_featname.csv` and assigned feature names to the matrices.
3. Loaded `labels_tr.csv` and `labels_te.csv`.
4. Mapped numeric label IDs to BRCA subtype names.
5. Generated synthetic sample IDs because original sample IDs are not provided in the CSV files.
6. Preserved the original MOGONET train/test split.
7. Concatenated train and test data into a unified `X.csv` and `y.csv`.
8. Checked missing values.
9. Removed all-missing or constant features if any.
10. Exported both combined files and split-specific files.

Important note: no additional global scaling or feature selection was fitted in this preprocessing step.

## 4. Dataset caveat

{summary["data_source_note"]}

{summary["sample_id_note"]}

This means MOGONET BRCA can be used for comparison, but the interpretation should be separated from the two more raw gene expression datasets.

## 5. Processed data summary

| Item | Value |
|---|---:|
| Final sample count | {summary["final_sample_count"]} |
| Final feature count | {summary["final_feature_count"]} |
| Train sample count | {summary["train_sample_count"]} |
| Test sample count | {summary["test_sample_count"]} |
| Missing values before imputation | {summary["missing_value_count_before_imputation"]} |
| Removed all-missing features | {summary["removed_all_missing_feature_count"]} |
| Removed constant features | {summary["removed_constant_feature_count"]} |

## 6. Label mapping

| Class ID | Label |
|---|---|
{label_mapping_rows}

## 7. Class distribution

{markdown_table(label_rows, ["Class ID", "Subtype", "Samples", "Percentage (%)"])}

![MOGONET BRCA subtype distribution]({figure_rel_path})

## 8. Train/test class distribution

{markdown_table(split_rows, ["Class ID", "Subtype", "Split", "Samples"])}

## 9. Output files

{markdown_table(file_rows, ["File", "Description", "Size (MB)"])}

The processed files follow this format:

- `X.csv`: combined samples × features matrix.
- `y.csv`: combined labels.
- `metadata.csv`: sample-level metadata including source split.
- `feature_names.csv`: retained feature names.
- `X_train.csv`, `y_train.csv`: preserved original training split.
- `X_test.csv`, `y_test.csv`: preserved original testing split.

## 10. Validation

{markdown_table(validation_rows, ["Check", "Status", "Details"])}

{deep_check_text}

## 11. Conclusion

The MOGONET BRCA omics-1 dataset was successfully preprocessed. The final dataset contains **{summary["final_sample_count"]} samples** and **{summary["final_feature_count"]} preprocessed features** across **{len(label_table)} BRCA molecular subtypes**. Since this dataset is already feature-reduced, it should be used as an optional benchmark rather than as the main raw gene expression dataset.

"""

    report_path.write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()

    processed_dir = args.processed_dir
    report_dir = args.report_dir
    asset_dir = report_dir / "assets"

    report_dir.mkdir(parents=True, exist_ok=True)
    asset_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "X": processed_dir / "X.csv",
        "y": processed_dir / "y.csv",
        "metadata": processed_dir / "metadata.csv",
        "features": processed_dir / "feature_names.csv",
        "X_train": processed_dir / "X_train.csv",
        "X_test": processed_dir / "X_test.csv",
        "y_train": processed_dir / "y_train.csv",
        "y_test": processed_dir / "y_test.csv",
    }

    for path in paths.values():
        require_file(path)

    summary = load_summary(args.summary_json)

    y = pd.read_csv(paths["y"])
    metadata = pd.read_csv(paths["metadata"])
    features = pd.read_csv(paths["features"])

    label_table = build_label_table(y)
    split_table = build_split_label_table(y, metadata)

    label_table_path = asset_dir / "mogonet_brca_label_distribution.csv"
    split_table_path = asset_dir / "mogonet_brca_split_label_distribution.csv"
    figure_path = asset_dir / "mogonet_brca_label_distribution.png"

    label_table.to_csv(label_table_path, index=False)
    split_table.to_csv(split_table_path, index=False)
    make_label_distribution_figure(label_table, figure_path)

    file_table = pd.DataFrame(
        [
            {
                "file": str(paths["X"]),
                "description": "Combined preprocessed feature matrix, samples x features",
                "size_mb": round(file_size_mb(paths["X"]), 2),
            },
            {
                "file": str(paths["y"]),
                "description": "Combined subtype labels",
                "size_mb": round(file_size_mb(paths["y"]), 4),
            },
            {
                "file": str(paths["metadata"]),
                "description": "Sample metadata with source split",
                "size_mb": round(file_size_mb(paths["metadata"]), 4),
            },
            {
                "file": str(paths["features"]),
                "description": "Retained feature names",
                "size_mb": round(file_size_mb(paths["features"]), 4),
            },
            {
                "file": str(paths["X_train"]),
                "description": "Original MOGONET training feature matrix",
                "size_mb": round(file_size_mb(paths["X_train"]), 2),
            },
            {
                "file": str(paths["X_test"]),
                "description": "Original MOGONET testing feature matrix",
                "size_mb": round(file_size_mb(paths["X_test"]), 2),
            },
            {
                "file": str(paths["y_train"]),
                "description": "Original MOGONET training labels",
                "size_mb": round(file_size_mb(paths["y_train"]), 4),
            },
            {
                "file": str(paths["y_test"]),
                "description": "Original MOGONET testing labels",
                "size_mb": round(file_size_mb(paths["y_test"]), 4),
            },
            {
                "file": str(args.summary_json),
                "description": "Preprocessing summary",
                "size_mb": round(file_size_mb(args.summary_json), 4),
            },
        ]
    )

    light_check = lightweight_validate(
        x_path=paths["X"],
        y=y,
        metadata=metadata,
        features=features,
        x_train_path=paths["X_train"],
        x_test_path=paths["X_test"],
        y_train_path=paths["y_train"],
        y_test_path=paths["y_test"],
    )
    deep_check = deep_validate(paths["X"]) if args.deep_check else None

    report_path = report_dir / "mogonet_brca_preprocessing_report.md"
    figure_rel_path = "assets/mogonet_brca_label_distribution.png"

    write_report(
        report_path=report_path,
        summary=summary,
        label_table=label_table,
        split_table=split_table,
        file_table=file_table,
        light_check=light_check,
        deep_check=deep_check,
        figure_rel_path=figure_rel_path,
    )

    print("Report generated successfully.")
    print(f"Markdown report: {report_path}")
    print(f"Label table: {label_table_path}")
    print(f"Split label table: {split_table_path}")
    print(f"Figure: {figure_path}")

    final_status = light_check["status"]
    if deep_check is not None and deep_check["status"] == "FAIL":
        final_status = "FAIL"

    print(f"Validation status: {final_status}")

    if final_status == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()