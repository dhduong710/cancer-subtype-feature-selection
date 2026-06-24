#!/usr/bin/env python3
"""Preprocess MoGCN BRCA gene expression dataset.

Input expected:
  data/raw/mogcn_brca/fpkm_data.csv
  data/raw/mogcn_brca/sample_classes.csv

Output:
  data/processed/mogcn_brca/X.csv
  data/processed/mogcn_brca/y.csv
  data/processed/mogcn_brca/metadata.csv
  data/processed/mogcn_brca/feature_names.csv
  results/data_summaries/mogcn_brca_summary.json
  results/data_summaries/mogcn_brca_summary.csv
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess MoGCN BRCA dataset")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/mogcn_brca"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed/mogcn_brca"))
    parser.add_argument("--summary-dir", type=Path, default=Path("results/data_summaries"))
    parser.add_argument("--expression-file", default="fpkm_data.csv")
    parser.add_argument("--label-file", default="sample_classes.csv")
    parser.add_argument("--sample-id-col", default="Sample_ID")
    parser.add_argument("--label-col", default="PAM50Call_RNAseq")
    parser.add_argument("--class-col", default="class")
    return parser.parse_args()


def normalize_sample_id(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()


def read_inputs(args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame]:
    expr_path = args.raw_dir / args.expression_file
    label_path = args.raw_dir / args.label_file

    if not expr_path.exists():
        raise FileNotFoundError(f"Missing expression file: {expr_path}")
    if not label_path.exists():
        raise FileNotFoundError(f"Missing label file: {label_path}")

    expression = pd.read_csv(expr_path)
    labels = pd.read_csv(label_path)
    return expression, labels


def clean_labels(labels: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    labels = labels.copy()
    labels.columns = labels.columns.astype(str).str.strip()

    required = {args.sample_id_col, args.label_col, args.class_col}
    missing = required.difference(labels.columns)
    if missing:
        raise ValueError(f"Label file is missing required columns: {sorted(missing)}")

    labels[args.sample_id_col] = normalize_sample_id(labels[args.sample_id_col])
    labels[args.label_col] = labels[args.label_col].astype(str).str.strip()

    if labels[args.sample_id_col].eq("").any():
        raise ValueError("Label file contains blank sample IDs")
    if labels[args.sample_id_col].duplicated().any():
        dupes = labels.loc[labels[args.sample_id_col].duplicated(), args.sample_id_col].tolist()
        raise ValueError(f"Duplicate sample IDs in label file, examples: {dupes[:10]}")

    labels = labels[[args.sample_id_col, args.label_col, args.class_col]].rename(
        columns={
            args.sample_id_col: "sample_id",
            args.label_col: "label",
            args.class_col: "label_id",
        }
    )
    labels["label_id"] = pd.to_numeric(labels["label_id"], errors="raise").astype(int)
    return labels


def clean_expression(expression: pd.DataFrame, args: argparse.Namespace) -> Tuple[pd.DataFrame, Dict[str, int]]:
    expression = expression.copy()
    expression.columns = expression.columns.astype(str).str.strip()

    if args.sample_id_col not in expression.columns:
        raise ValueError(f"Expression file is missing sample ID column: {args.sample_id_col}")

    expression[args.sample_id_col] = normalize_sample_id(expression[args.sample_id_col])
    if expression[args.sample_id_col].eq("").any():
        raise ValueError("Expression file contains blank sample IDs")
    if expression[args.sample_id_col].duplicated().any():
        dupes = expression.loc[expression[args.sample_id_col].duplicated(), args.sample_id_col].tolist()
        raise ValueError(f"Duplicate sample IDs in expression file, examples: {dupes[:10]}")

    gene_cols = [c for c in expression.columns if c != args.sample_id_col]
    if len(gene_cols) == 0:
        raise ValueError("Expression file contains no gene columns")
    if pd.Index(gene_cols).duplicated().any():
        dupes = pd.Index(gene_cols)[pd.Index(gene_cols).duplicated()].tolist()
        raise ValueError(f"Duplicate gene columns, examples: {dupes[:10]}")

    X = expression.set_index(args.sample_id_col)[gene_cols]
    X = X.apply(pd.to_numeric, errors="coerce")

    n_missing_before = int(X.isna().sum().sum())

    all_missing_cols = X.columns[X.isna().all()].tolist()
    if all_missing_cols:
        X = X.drop(columns=all_missing_cols)

    # For MoGCN BRCA this should not change anything because there are no missing values.
    # It is included as a safety step for reproducibility.
    if X.isna().any().any():
        medians = X.median(axis=0, skipna=True)
        X = X.fillna(medians)
        X = X.fillna(0.0)

    # Remove genes that carry no information.
    nunique = X.nunique(dropna=False)
    constant_cols = nunique[nunique <= 1].index.tolist()
    if constant_cols:
        X = X.drop(columns=constant_cols)

    stats = {
        "raw_gene_count": len(gene_cols),
        "removed_all_missing_gene_count": len(all_missing_cols),
        "removed_constant_gene_count": len(constant_cols),
        "missing_value_count_before_imputation": n_missing_before,
        "final_gene_count_before_alignment": int(X.shape[1]),
    }
    return X, stats


def align_expression_and_labels(X: pd.DataFrame, y: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    expression_ids = set(X.index)
    label_ids = set(y["sample_id"])
    common_ids = sorted(expression_ids.intersection(label_ids))

    if not common_ids:
        raise ValueError("No overlapping sample IDs between expression and label files")

    expression_only = sorted(expression_ids - label_ids)
    label_only = sorted(label_ids - expression_ids)

    # Keep expression file order.
    ordered_common_ids = [sid for sid in X.index.tolist() if sid in set(common_ids)]
    X_aligned = X.loc[ordered_common_ids].copy()
    y_aligned = y.set_index("sample_id").loc[ordered_common_ids].reset_index().copy()

    stats = {
        "expression_sample_count": int(len(expression_ids)),
        "label_sample_count": int(len(label_ids)),
        "matched_sample_count": int(len(common_ids)),
        "expression_only_sample_count": int(len(expression_only)),
        "label_only_sample_count": int(len(label_only)),
    }
    return X_aligned, y_aligned, stats


def write_outputs(
    X: pd.DataFrame,
    y: pd.DataFrame,
    args: argparse.Namespace,
    summary: Dict,
) -> None:
    args.processed_dir.mkdir(parents=True, exist_ok=True)
    args.summary_dir.mkdir(parents=True, exist_ok=True)

    X.to_csv(args.processed_dir / "X.csv", index_label="sample_id")
    y.to_csv(args.processed_dir / "y.csv", index=False)

    metadata = y[["sample_id", "label", "label_id"]].copy()
    metadata.insert(1, "dataset", "mogcn_brca")
    metadata.insert(2, "cancer_type", "BRCA")
    metadata.to_csv(args.processed_dir / "metadata.csv", index=False)

    feature_names = pd.DataFrame({"gene_name": X.columns.tolist()})
    feature_names.to_csv(args.processed_dir / "feature_names.csv", index=False)

    with open(args.summary_dir / "mogcn_brca_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    flat_rows = []
    for key, value in summary.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                flat_rows.append({"metric": f"{key}.{subkey}", "value": subvalue})
        else:
            flat_rows.append({"metric": key, "value": value})
    pd.DataFrame(flat_rows).to_csv(args.summary_dir / "mogcn_brca_summary.csv", index=False)


def main() -> None:
    args = parse_args()

    expression_raw, labels_raw = read_inputs(args)
    labels = clean_labels(labels_raw, args)
    X, expr_stats = clean_expression(expression_raw, args)
    X, y, align_stats = align_expression_and_labels(X, labels)

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
        "dataset": "mogcn_brca",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_expression_shape_rows_cols": list(expression_raw.shape),
        "raw_label_shape_rows_cols": list(labels_raw.shape),
        **expr_stats,
        **align_stats,
        "final_sample_count": int(X.shape[0]),
        "final_gene_count": int(X.shape[1]),
        "label_counts": {str(k): int(v) for k, v in label_counts.items()},
        "class_id_counts": {str(k): int(v) for k, v in class_counts.items()},
        "preprocessing_notes": (
            "Only structural cleaning was applied: sample-label alignment, numeric conversion, "
            "missing-value safety check/imputation if needed, and constant/all-zero gene removal. "
            "No global scaling, normalization, train-test split, or feature selection was fitted here."
        ),
    }

    write_outputs(X, y, args, summary)

    print("MoGCN BRCA preprocessing completed.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()