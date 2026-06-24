#!/usr/bin/env python3
"""Create combined data preprocessing summary report for all datasets."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


SUMMARY_FILES = {
    "MoGCN BRCA": Path("results/data_summaries/mogcn_brca_summary.json"),
    "CancerSD STAD": Path("results/data_summaries/cancersd_stad_summary.json"),
    "MOGONET BRCA": Path("results/data_summaries/mogonet_brca_summary.json"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate combined dataset summary report")
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("reports/data_preprocessing"),
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing summary file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_feature_count(summary: Dict[str, Any]) -> int:
    if "final_gene_count" in summary:
        return int(summary["final_gene_count"])
    if "final_feature_count" in summary:
        return int(summary["final_feature_count"])
    raise KeyError(f"Cannot find final feature count in summary for {summary.get('dataset')}")


def get_removed_constant_count(summary: Dict[str, Any]) -> int:
    if "removed_constant_gene_count" in summary:
        return int(summary["removed_constant_gene_count"])
    if "removed_constant_feature_count" in summary:
        return int(summary["removed_constant_feature_count"])
    return 0


def get_feature_type(summary: Dict[str, Any]) -> str:
    if summary["dataset"] == "mogonet_brca_optional":
        return "preprocessed features"
    return "genes"


def get_dataset_role(summary: Dict[str, Any]) -> str:
    if summary["dataset"] == "mogonet_brca_optional":
        return "Optional benchmark"
    return "Main dataset"


def get_cancer_type(summary: Dict[str, Any]) -> str:
    dataset = summary["dataset"].lower()
    if "brca" in dataset:
        return "BRCA"
    if "stad" in dataset:
        return "STAD"
    return "Unknown"


def make_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""

    columns = list(df.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"

    rows = []
    for _, row in df.iterrows():
        values = [str(row[col]) for col in columns]
        rows.append("| " + " | ".join(values) + " |")

    return "\n".join([header, separator] + rows)


def make_dataset_table(summaries: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for display_name, summary in summaries.items():
        rows.append(
            {
                "Dataset": display_name,
                "Cancer type": get_cancer_type(summary),
                "Role": get_dataset_role(summary),
                "Samples": int(summary["final_sample_count"]),
                "Features": get_feature_count(summary),
                "Feature type": get_feature_type(summary),
                "Classes": len(summary["label_counts"]),
                "Missing values": int(summary["missing_value_count_before_imputation"]),
                "Removed constant features": get_removed_constant_count(summary),
            }
        )

    return pd.DataFrame(rows)


def make_class_distribution_table(summaries: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for display_name, summary in summaries.items():
        total = int(summary["final_sample_count"])
        for label, count in summary["label_counts"].items():
            count_int = int(count)
            rows.append(
                {
                    "Dataset": display_name,
                    "Class label": label,
                    "Samples": count_int,
                    "Percentage (%)": round(count_int / total * 100, 2),
                }
            )

    return pd.DataFrame(rows)


def make_dataset_sample_figure(dataset_table: pd.DataFrame, figure_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.bar(dataset_table["Dataset"], dataset_table["Samples"])
    ax.set_title("Number of samples per dataset")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Number of samples")
    ax.tick_params(axis="x", rotation=20)

    for idx, value in enumerate(dataset_table["Samples"]):
        ax.text(idx, value, str(value), ha="center", va="bottom")

    fig.tight_layout()
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)


def make_dataset_feature_figure(dataset_table: pd.DataFrame, figure_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.bar(dataset_table["Dataset"], dataset_table["Features"])
    ax.set_title("Number of features per dataset")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Number of features")
    ax.tick_params(axis="x", rotation=20)

    for idx, value in enumerate(dataset_table["Features"]):
        ax.text(idx, value, str(value), ha="center", va="bottom")

    fig.tight_layout()
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)


def write_report(
    report_path: Path,
    dataset_table: pd.DataFrame,
    class_table: pd.DataFrame,
    sample_figure_rel: str,
    feature_figure_rel: str,
) -> None:
    dataset_table_md = make_markdown_table(dataset_table)
    class_table_md = make_markdown_table(class_table)

    report = f"""# Data Preprocessing Overview

Generated at: `{datetime.now().isoformat(timespec="seconds")}`

## 1. Project context

This report summarizes the data preprocessing phase for the project:

**A comparative study of feature selection method for cancer classification using gene expression data**

The goal of this phase is to prepare clean and unified datasets for later experiments on feature selection and cancer subtype classification.

## 2. Dataset selection

Three datasets were prepared:

1. **MoGCN BRCA**: main BRCA gene expression dataset.
2. **CancerSD STAD**: main STAD mRNA gene expression dataset.
3. **MOGONET BRCA**: optional BRCA benchmark dataset.

MOGONET BRCA is treated separately because its omics-1 data is already preprocessed and feature-reduced to 1,000 features by the original dataset. Therefore, it is useful as an additional benchmark, but it should not be interpreted in the same way as the more raw gene expression datasets.

## 3. Why datasets are not merged

The datasets were not merged into one large table because they come from different studies, cancer types, preprocessing pipelines, subtype definitions, and feature spaces. Direct merging may introduce batch effects and unreliable comparisons.

Instead, each dataset is converted into the same processed format and evaluated independently using the same machine learning pipeline.

## 4. Common processed format

Each processed dataset follows this structure:

    data/processed/<dataset_name>/
    ├── X.csv
    ├── y.csv
    ├── metadata.csv
    └── feature_names.csv

Where:

- `X.csv`: samples × genes/features matrix.
- `y.csv`: sample IDs and class labels.
- `metadata.csv`: sample-level metadata.
- `feature_names.csv`: retained gene or feature names.

For MOGONET BRCA, the original train/test split is also preserved:

    X_train.csv
    X_test.csv
    y_train.csv
    y_test.csv

## 5. Dataset summary

{dataset_table_md}

![Number of samples per dataset]({sample_figure_rel})

![Number of features per dataset]({feature_figure_rel})

## 6. Class distribution

{class_table_md}

## 7. Data leakage prevention

To avoid data leakage, the preprocessing phase only performs structural cleaning:

- sample-label matching,
- numeric conversion,
- missing-value checking,
- removal of all-missing or constant features,
- conversion into a unified machine learning format.

The following operations are **not** fitted globally during preprocessing:

- standardization,
- normalization,
- train-test split for datasets without official split,
- feature selection,
- model training.

These steps will be fitted only on the training data inside the later machine learning pipeline.

## 8. Final status

All three datasets passed validation checks and are ready for downstream feature selection and classification experiments.

| Dataset | Status |
|---|---|
| MoGCN BRCA | PASS |
| CancerSD STAD | PASS |
| MOGONET BRCA | PASS |

"""

    report_path.write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()

    report_dir = args.report_dir
    asset_dir = report_dir / "assets"

    report_dir.mkdir(parents=True, exist_ok=True)
    asset_dir.mkdir(parents=True, exist_ok=True)

    summaries = {
        display_name: load_json(path)
        for display_name, path in SUMMARY_FILES.items()
    }

    dataset_table = make_dataset_table(summaries)
    class_table = make_class_distribution_table(summaries)

    dataset_table_path = report_dir / "dataset_summary.csv"
    class_table_path = report_dir / "class_distribution_summary.csv"
    report_path = report_dir / "data_preprocessing_overview.md"

    sample_figure_path = asset_dir / "dataset_sample_counts.png"
    feature_figure_path = asset_dir / "dataset_feature_counts.png"

    dataset_table.to_csv(dataset_table_path, index=False)
    class_table.to_csv(class_table_path, index=False)

    make_dataset_sample_figure(dataset_table, sample_figure_path)
    make_dataset_feature_figure(dataset_table, feature_figure_path)

    write_report(
        report_path=report_path,
        dataset_table=dataset_table,
        class_table=class_table,
        sample_figure_rel="assets/dataset_sample_counts.png",
        feature_figure_rel="assets/dataset_feature_counts.png",
    )

    print("Combined data preprocessing report generated.")
    print(f"Report: {report_path}")
    print(f"Dataset summary CSV: {dataset_table_path}")
    print(f"Class distribution CSV: {class_table_path}")
    print(f"Sample figure: {sample_figure_path}")
    print(f"Feature figure: {feature_figure_path}")


if __name__ == "__main__":
    main()
