#!/usr/bin/env python3
"""Generate preprocessing report for CancerSD STAD dataset."""

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
    parser = argparse.ArgumentParser(description="Generate CancerSD STAD preprocessing report")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed/cancersd_stad"),
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("results/data_summaries/cancersd_stad_summary.json"),
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("reports/data_preprocessing"),
    )
    parser.add_argument(
        "--deep-check",
        action="store_true",
        help="Load full X.csv to verify missing values, finite values, and constant genes",
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


def make_label_distribution_figure(label_table: pd.DataFrame, figure_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(label_table["label"], label_table["sample_count"])
    ax.set_title("CancerSD STAD subtype distribution")
    ax.set_xlabel("Subtype")
    ax.set_ylabel("Number of samples")

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
) -> Dict[str, Any]:
    x_header = pd.read_csv(x_path, nrows=0).columns.tolist()
    x_id_col = x_header[0]
    x_feature_count = len(x_header) - 1

    x_ids = pd.read_csv(x_path, usecols=[x_id_col])[x_id_col].astype(str).tolist()
    y_ids = y["sample_id"].astype(str).tolist()
    metadata_ids = metadata["sample_id"].astype(str).tolist()

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
    if features["gene_name"].duplicated().any():
        problems.append("Duplicate gene names in feature_names.csv")

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
    constant_gene_count = int((X.nunique(dropna=False) <= 1).sum())

    problems: List[str] = []
    if missing_count != 0:
        problems.append(f"X.csv contains {missing_count} missing values")
    if not finite_values:
        problems.append("X.csv contains inf or -inf values")
    if constant_gene_count != 0:
        problems.append(f"X.csv still contains {constant_gene_count} constant genes")

    return {
        "status": "PASS" if not problems else "FAIL",
        "problems": problems,
        "missing_value_count": missing_count,
        "finite_values": finite_values,
        "constant_gene_count": constant_gene_count,
    }


def write_report(
    report_path: Path,
    summary: Dict[str, Any],
    label_table: pd.DataFrame,
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
| Constant genes after processing | {deep_check["constant_gene_count"]} |
"""

    report = f"""# CancerSD STAD Data Preprocessing Report

Generated at: `{datetime.now().isoformat(timespec="seconds")}`

## 1. Dataset objective

This report documents the preprocessing of the CancerSD STAD mRNA dataset for the project:

**A comparative study of feature selection method for cancer classification using gene expression data**

The goal of this preprocessing step is to convert the raw STAD mRNA expression data into a clean machine learning format suitable for feature selection and cancer subtype classification experiments.

## 2. Raw input data

The raw CancerSD STAD files used in this step are:

- `mRNA.zip`: mRNA expression data.
- `patient_diagnose.csv`: sample-level diagnosis and subtype information.

The original mRNA matrix is organized as **genes × samples**, so it was transposed into **samples × genes**.

| Item | Value |
|---|---:|
| Raw mRNA shape after transpose | {summary["raw_expression_shape_samples_genes_after_transpose"]} |
| Raw diagnosis shape after transpose | {summary["raw_label_shape_samples_metadata_after_transpose"]} |
| Raw mRNA sample count | {summary["expression_sample_count"]} |
| Diagnosis sample count | {summary["label_sample_count"]} |
| Raw gene feature count | {summary["raw_gene_count"]} |

## 3. Preprocessing procedure

The following preprocessing steps were applied:

1. Loaded `mRNA.zip` and extracted the mRNA expression table.
2. Transposed mRNA expression from genes × samples to samples × genes.
3. Loaded `patient_diagnose.csv`.
4. Transposed diagnosis information from aspects × samples to samples × metadata.
5. Used the `subtype` row as the classification label.
6. Standardized sample identifiers by converting them to strings and trimming spaces.
7. Converted all gene expression values to numeric values.
8. Checked missing values in the expression matrix.
9. Removed genes with no useful variation:
   - all-missing genes: {summary["removed_all_missing_gene_count"]}
   - constant genes: {summary["removed_constant_gene_count"]}
10. Matched expression samples with diagnosis labels.
11. Exported the cleaned dataset into a unified format.

Important note: no global standardization, train-test split, or feature selection was fitted in this preprocessing step. These operations must be fitted only on the training data inside the later machine learning pipeline to avoid data leakage.

## 4. Sample matching result

| Item | Value |
|---|---:|
| Expression samples | {summary["expression_sample_count"]} |
| Diagnosis samples | {summary["label_sample_count"]} |
| Matched samples | {summary["matched_sample_count"]} |
| Expression-only samples removed | {summary["expression_only_sample_count"]} |
| Label-only samples removed | {summary["label_only_sample_count"]} |

The diagnosis file contains {summary["label_sample_count"]} samples, but only {summary["matched_sample_count"]} samples have matching mRNA expression profiles. Therefore, {summary["label_only_sample_count"]} label-only samples were excluded from the final gene expression classification dataset.

## 5. Processed data summary

| Item | Value |
|---|---:|
| Final sample count | {summary["final_sample_count"]} |
| Final gene count | {summary["final_gene_count"]} |
| Missing values before imputation | {summary["missing_value_count_before_imputation"]} |
| Removed all-missing genes | {summary["removed_all_missing_gene_count"]} |
| Removed constant genes | {summary["removed_constant_gene_count"]} |

## 6. Label mapping

| Label | Class ID |
|---|---:|
{chr(10).join([f"| {label} | {class_id} |" for label, class_id in summary["label_mapping"].items()])}

## 7. Class distribution

{markdown_table(label_rows, ["Class ID", "Subtype", "Samples", "Percentage (%)"])}

![CancerSD STAD subtype distribution]({figure_rel_path})

## 8. Output files

{markdown_table(file_rows, ["File", "Description", "Size (MB)"])}

The processed files follow this format:

- `X.csv`: rows are samples, columns are mRNA gene expression features.
- `y.csv`: sample identifier, subtype label, and numeric label ID.
- `metadata.csv`: sample-level metadata including subtype, stage, race, and class ID.
- `feature_names.csv`: final list of retained gene names.

## 9. Validation

{markdown_table(validation_rows, ["Check", "Status", "Details"])}

{deep_check_text}

## 10. Conclusion

The CancerSD STAD dataset was successfully preprocessed. The final dataset contains **{summary["final_sample_count"]} samples** and **{summary["final_gene_count"]} mRNA gene expression features** across **{len(label_table)} STAD molecular subtypes**. The processed dataset is ready for downstream feature selection and cancer subtype classification experiments.

"""

    report_path.write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()

    processed_dir = args.processed_dir
    report_dir = args.report_dir
    asset_dir = report_dir / "assets"

    report_dir.mkdir(parents=True, exist_ok=True)
    asset_dir.mkdir(parents=True, exist_ok=True)

    x_path = processed_dir / "X.csv"
    y_path = processed_dir / "y.csv"
    metadata_path = processed_dir / "metadata.csv"
    feature_path = processed_dir / "feature_names.csv"

    for path in [x_path, y_path, metadata_path, feature_path]:
        require_file(path)

    summary = load_summary(args.summary_json)

    y = pd.read_csv(y_path)
    metadata = pd.read_csv(metadata_path)
    features = pd.read_csv(feature_path)

    label_table = build_label_table(y)

    label_table_path = asset_dir / "cancersd_stad_label_distribution.csv"
    figure_path = asset_dir / "cancersd_stad_label_distribution.png"

    label_table.to_csv(label_table_path, index=False)
    make_label_distribution_figure(label_table, figure_path)

    file_table = pd.DataFrame(
        [
            {
                "file": str(x_path),
                "description": "Processed mRNA expression matrix, samples x genes",
                "size_mb": round(file_size_mb(x_path), 2),
            },
            {
                "file": str(y_path),
                "description": "Subtype labels",
                "size_mb": round(file_size_mb(y_path), 4),
            },
            {
                "file": str(metadata_path),
                "description": "Sample metadata",
                "size_mb": round(file_size_mb(metadata_path), 4),
            },
            {
                "file": str(feature_path),
                "description": "Retained gene names",
                "size_mb": round(file_size_mb(feature_path), 4),
            },
            {
                "file": str(args.summary_json),
                "description": "Preprocessing summary",
                "size_mb": round(file_size_mb(args.summary_json), 4),
            },
        ]
    )

    light_check = lightweight_validate(x_path, y, metadata, features)
    deep_check = deep_validate(x_path) if args.deep_check else None

    report_path = report_dir / "cancersd_stad_preprocessing_report.md"
    figure_rel_path = "assets/cancersd_stad_label_distribution.png"

    write_report(
        report_path=report_path,
        summary=summary,
        label_table=label_table,
        file_table=file_table,
        light_check=light_check,
        deep_check=deep_check,
        figure_rel_path=figure_rel_path,
    )

    print("Report generated successfully.")
    print(f"Markdown report: {report_path}")
    print(f"Label table: {label_table_path}")
    print(f"Figure: {figure_path}")

    final_status = light_check["status"]
    if deep_check is not None and deep_check["status"] == "FAIL":
        final_status = "FAIL"

    print(f"Validation status: {final_status}")

    if final_status == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()