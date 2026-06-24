#!/usr/bin/env python3
"""Preprocess CancerSD STAD mRNA dataset.

Input expected:
  data/raw/cancersd_stad/mRNA.zip
  data/raw/cancersd_stad/patient_diagnose.csv

Output:
  data/processed/cancersd_stad/X.csv
  data/processed/cancersd_stad/y.csv
  data/processed/cancersd_stad/metadata.csv
  data/processed/cancersd_stad/feature_names.csv
  results/data_summaries/cancersd_stad_summary.json
  results/data_summaries/cancersd_stad_summary.csv
"""

from __future__ import annotations

import argparse
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess CancerSD STAD mRNA dataset")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/cancersd_stad"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed/cancersd_stad"))
    parser.add_argument("--summary-dir", type=Path, default=Path("results/data_summaries"))
    parser.add_argument("--expression-file", default="mRNA.zip")
    parser.add_argument("--label-file", default="patient_diagnose.csv")
    parser.add_argument("--gene-col", default="mRNA")
    parser.add_argument("--label-aspect", default="subtype")
    return parser.parse_args()


def normalize_sample_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


def read_mrna_expression(path: Path, gene_col: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing expression file: {path}")

    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path, "r") as z:
            csv_files = [name for name in z.namelist() if name.lower().endswith(".csv")]
            if not csv_files:
                raise ValueError(f"No CSV file found inside {path}")
            if len(csv_files) > 1:
                print(f"Warning: multiple CSV files found in {path}. Using {csv_files[0]}")
            with z.open(csv_files[0]) as f:
                df = pd.read_csv(f)
    else:
        df = pd.read_csv(path)

    df.columns = df.columns.astype(str).str.strip()

    if gene_col not in df.columns:
        raise ValueError(f"Expression file is missing gene column: {gene_col}")

    df[gene_col] = df[gene_col].astype(str).str.strip()

    if df[gene_col].eq("").any():
        raise ValueError("Expression file contains blank gene names")
    if df[gene_col].duplicated().any():
        dupes = df.loc[df[gene_col].duplicated(), gene_col].tolist()
        raise ValueError(f"Duplicate gene names in expression file, examples: {dupes[:10]}")

    sample_cols = [c for c in df.columns if c != gene_col]
    if len(sample_cols) == 0:
        raise ValueError("Expression file contains no sample columns")

    # Raw CancerSD mRNA format: rows are genes, columns are samples.
    # Convert to standard ML format: rows are samples, columns are genes.
    X = df.set_index(gene_col)[sample_cols].T
    X.index = normalize_sample_id(pd.Series(X.index, index=X.index)).values
    X.index.name = "sample_id"
    X.columns = X.columns.astype(str).str.strip()

    X = X.apply(pd.to_numeric, errors="coerce")

    return X


def read_and_transform_labels(path: Path, label_aspect: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing label file: {path}")

    labels_raw = pd.read_csv(path)
    labels_raw.columns = labels_raw.columns.astype(str).str.strip()

    if "aspect" not in labels_raw.columns:
        raise ValueError("patient_diagnose.csv is missing required column: aspect")

    labels_raw["aspect"] = labels_raw["aspect"].astype(str).str.strip()

    if label_aspect not in set(labels_raw["aspect"]):
        raise ValueError(f"Cannot find label aspect '{label_aspect}' in patient_diagnose.csv")

    # Raw CancerSD label format: rows are aspects, columns are samples.
    # Convert to sample-level metadata.
    labels = labels_raw.set_index("aspect").T.reset_index().rename(columns={"index": "sample_id"})
    labels["sample_id"] = normalize_sample_id(labels["sample_id"])

    if labels["sample_id"].eq("").any():
        raise ValueError("Label file contains blank sample IDs")
    if labels["sample_id"].duplicated().any():
        dupes = labels.loc[labels["sample_id"].duplicated(), "sample_id"].tolist()
        raise ValueError(f"Duplicate sample IDs in label file, examples: {dupes[:10]}")

    labels[label_aspect] = labels[label_aspect].astype(str).str.strip()

    if labels[label_aspect].eq("").any():
        raise ValueError("Label file contains blank subtype labels")

    return labels


def clean_expression(X: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    raw_gene_count = int(X.shape[1])
    n_missing_before = int(X.isna().sum().sum())

    all_missing_cols = X.columns[X.isna().all()].tolist()
    if all_missing_cols:
        X = X.drop(columns=all_missing_cols)

    if X.isna().any().any():
        medians = X.median(axis=0, skipna=True)
        X = X.fillna(medians)
        X = X.fillna(0.0)

    nunique = X.nunique(dropna=False)
    constant_cols = nunique[nunique <= 1].index.tolist()
    if constant_cols:
        X = X.drop(columns=constant_cols)

    stats = {
        "raw_gene_count": raw_gene_count,
        "removed_all_missing_gene_count": len(all_missing_cols),
        "removed_constant_gene_count": len(constant_cols),
        "missing_value_count_before_imputation": n_missing_before,
        "final_gene_count_before_alignment": int(X.shape[1]),
    }
    return X, stats


def align_expression_and_labels(
    X: pd.DataFrame,
    labels: pd.DataFrame,
    label_aspect: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    expression_ids = set(X.index.astype(str))
    label_ids = set(labels["sample_id"].astype(str))

    common_ids = sorted(expression_ids.intersection(label_ids))
    if not common_ids:
        raise ValueError("No overlapping sample IDs between expression and label files")

    expression_only = sorted(expression_ids - label_ids)
    label_only = sorted(label_ids - expression_ids)

    common_set = set(common_ids)
    ordered_common_ids = [sid for sid in X.index.astype(str).tolist() if sid in common_set]

    X_aligned = X.loc[ordered_common_ids].copy()
    metadata = labels.set_index("sample_id").loc[ordered_common_ids].reset_index().copy()

    y = metadata[["sample_id", label_aspect]].rename(columns={label_aspect: "label"}).copy()

    # Deterministic label mapping.
    label_names = sorted(y["label"].unique().tolist())
    label_to_id = {label: idx for idx, label in enumerate(label_names)}
    y["label_id"] = y["label"].map(label_to_id).astype(int)

    metadata.insert(1, "dataset", "cancersd_stad")
    metadata.insert(2, "cancer_type", "STAD")
    metadata = metadata.rename(columns={label_aspect: "label"})
    metadata["label_id"] = y["label_id"].values

    stats = {
        "expression_sample_count": int(len(expression_ids)),
        "label_sample_count": int(len(label_ids)),
        "matched_sample_count": int(len(common_ids)),
        "expression_only_sample_count": int(len(expression_only)),
        "label_only_sample_count": int(len(label_only)),
    }

    return X_aligned, y, metadata, stats


def write_outputs(
    X: pd.DataFrame,
    y: pd.DataFrame,
    metadata: pd.DataFrame,
    args: argparse.Namespace,
    summary: Dict,
) -> None:
    args.processed_dir.mkdir(parents=True, exist_ok=True)
    args.summary_dir.mkdir(parents=True, exist_ok=True)

    X.to_csv(args.processed_dir / "X.csv", index_label="sample_id")
    y.to_csv(args.processed_dir / "y.csv", index=False)
    metadata.to_csv(args.processed_dir / "metadata.csv", index=False)

    feature_names = pd.DataFrame({"gene_name": X.columns.tolist()})
    feature_names.to_csv(args.processed_dir / "feature_names.csv", index=False)

    with open(args.summary_dir / "cancersd_stad_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    flat_rows = []
    for key, value in summary.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                flat_rows.append({"metric": f"{key}.{subkey}", "value": subvalue})
        else:
            flat_rows.append({"metric": key, "value": value})

    pd.DataFrame(flat_rows).to_csv(args.summary_dir / "cancersd_stad_summary.csv", index=False)


def main() -> None:
    args = parse_args()

    expression_path = args.raw_dir / args.expression_file
    label_path = args.raw_dir / args.label_file

    X_raw = read_mrna_expression(expression_path, args.gene_col)
    labels_raw = read_and_transform_labels(label_path, args.label_aspect)

    X, expr_stats = clean_expression(X_raw)
    X, y, metadata, align_stats = align_expression_and_labels(X, labels_raw, args.label_aspect)

    if X.isna().any().any():
        raise ValueError("Processed X still contains missing values")
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Processed X contains inf or -inf values")
    if X.index.duplicated().any():
        raise ValueError("Processed X contains duplicate sample IDs")
    if y["sample_id"].duplicated().any():
        raise ValueError("Processed y contains duplicate sample IDs")
    if X.index.tolist() != y["sample_id"].tolist():
        raise ValueError("X and y sample orders are not aligned")

    label_counts = y["label"].value_counts().sort_index().to_dict()
    class_counts = y["label_id"].value_counts().sort_index().to_dict()

    summary = {
        "dataset": "cancersd_stad",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_expression_shape_samples_genes_after_transpose": [int(X_raw.shape[0]), int(X_raw.shape[1])],
        "raw_label_shape_samples_metadata_after_transpose": [int(labels_raw.shape[0]), int(labels_raw.shape[1])],
        **expr_stats,
        **align_stats,
        "final_sample_count": int(X.shape[0]),
        "final_gene_count": int(X.shape[1]),
        "label_mapping": {
            str(label): int(label_id)
            for label, label_id in sorted(
                zip(y["label"], y["label_id"]),
                key=lambda pair: pair[1],
            )
        },
        "label_counts": {str(k): int(v) for k, v in label_counts.items()},
        "class_id_counts": {str(k): int(v) for k, v in class_counts.items()},
        "metadata_columns": metadata.columns.tolist(),
        "preprocessing_notes": (
            "CancerSD STAD mRNA data was transposed from genes x samples to samples x genes. "
            "patient_diagnose.csv was transposed from aspects x samples to samples x metadata. "
            "Only structural cleaning was applied: sample-label alignment, numeric conversion, "
            "missing-value safety check/imputation if needed, and constant gene removal. "
            "No global scaling, normalization, train-test split, or feature selection was fitted here."
        ),
    }

    # Deduplicate label_mapping in case dict construction above receives repeated labels.
    summary["label_mapping"] = {
        label: int(y.loc[y["label"] == label, "label_id"].iloc[0])
        for label in sorted(y["label"].unique().tolist())
    }

    write_outputs(X, y, metadata, args, summary)

    print("CancerSD STAD preprocessing completed.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()