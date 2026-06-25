#!/usr/bin/env python3
"""Preprocess MOGONET BRCA omics-1 dataset.

Input expected:
  data/raw/mogonet_brca_optional/1_tr.csv
  data/raw/mogonet_brca_optional/1_te.csv
  data/raw/mogonet_brca_optional/1_featname.csv
  data/raw/mogonet_brca_optional/labels_tr.csv
  data/raw/mogonet_brca_optional/labels_te.csv

Output:
  data/processed/mogonet_brca_optional/X.csv
  data/processed/mogonet_brca_optional/y.csv
  data/processed/mogonet_brca_optional/metadata.csv
  data/processed/mogonet_brca_optional/feature_names.csv
  data/processed/mogonet_brca_optional/X_train.csv
  data/processed/mogonet_brca_optional/X_test.csv
  data/processed/mogonet_brca_optional/y_train.csv
  data/processed/mogonet_brca_optional/y_test.csv
  results/data_summaries/mogonet_brca_summary.json
  results/data_summaries/mogonet_brca_summary.csv
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


LABEL_MAPPING = {
    0: "Normal-like",
    1: "Basal-like",
    2: "HER2-enriched",
    3: "Luminal A",
    4: "Luminal B",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess MOGONET BRCA omics-1 dataset")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/mogonet_brca_optional"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed/mogonet_brca_optional"))
    parser.add_argument("--summary-dir", type=Path, default=Path("results/data_summaries"))
    parser.add_argument("--train-expression-file", default="1_tr.csv")
    parser.add_argument("--test-expression-file", default="1_te.csv")
    parser.add_argument("--feature-file", default="1_featname.csv")
    parser.add_argument("--train-label-file", default="labels_tr.csv")
    parser.add_argument("--test-label-file", default="labels_te.csv")
    return parser.parse_args()


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def read_no_header_csv(path: Path) -> pd.DataFrame:
    require_file(path)
    return pd.read_csv(path, header=None)


def make_sample_ids(prefix: str, n: int) -> list[str]:
    return [f"{prefix}_{i:04d}" for i in range(1, n + 1)]


def read_feature_names(path: Path, expected_count: int) -> pd.DataFrame:
    feat = read_no_header_csv(path)

    if feat.shape[1] != 1:
        raise ValueError(f"Feature file should have exactly 1 column, got shape {feat.shape}")

    feat_names = feat.iloc[:, 0].astype(str).str.strip()

    if feat_names.eq("").any():
        raise ValueError("Feature file contains blank feature names")
    if feat_names.duplicated().any():
        dupes = feat_names[feat_names.duplicated()].tolist()
        raise ValueError(f"Duplicate feature names, examples: {dupes[:10]}")
    if len(feat_names) != expected_count:
        raise ValueError(
            f"Feature count mismatch: feature file has {len(feat_names)}, "
            f"but expression has {expected_count} columns"
        )

    feature_df = pd.DataFrame(
        {
            "feature_index": list(range(expected_count)),
            "feature_name": feat_names.tolist(),
        }
    )

    # Many MOGONET BRCA omics-1 features have format SYMBOL|ENTREZ_ID.
    parsed = feature_df["feature_name"].str.split("|", n=1, expand=True)
    if parsed.shape[1] == 2:
        feature_df["gene_symbol"] = parsed[0].replace({"?": np.nan})
        feature_df["entrez_id"] = parsed[1]
    else:
        feature_df["gene_symbol"] = feature_df["feature_name"]
        feature_df["entrez_id"] = np.nan

    return feature_df


def read_labels(path: Path, expected_count: int, split_name: str) -> pd.DataFrame:
    labels_raw = read_no_header_csv(path)

    if labels_raw.shape[1] != 1:
        raise ValueError(f"{path} should have exactly 1 label column, got shape {labels_raw.shape}")
    if labels_raw.shape[0] != expected_count:
        raise ValueError(
            f"Label count mismatch for {split_name}: labels={labels_raw.shape[0]}, "
            f"samples={expected_count}"
        )

    label_id = pd.to_numeric(labels_raw.iloc[:, 0], errors="raise").astype(int)

    unknown = sorted(set(label_id.tolist()) - set(LABEL_MAPPING.keys()))
    if unknown:
        raise ValueError(f"Unknown label IDs in {split_name}: {unknown}")

    labels = pd.DataFrame(
        {
            "label_id": label_id,
            "label": label_id.map(LABEL_MAPPING),
        }
    )
    return labels


def build_split_dataset(
    expression: pd.DataFrame,
    labels: pd.DataFrame,
    feature_names: list[str],
    split_name: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X = expression.copy()
    X = X.apply(pd.to_numeric, errors="coerce")
    X.columns = feature_names

    prefix = "MOGONET_BRCA_TR" if split_name == "train" else "MOGONET_BRCA_TE"
    sample_ids = make_sample_ids(prefix, X.shape[0])
    X.index = sample_ids
    X.index.name = "sample_id"

    y = labels.copy()
    y.insert(0, "sample_id", sample_ids)

    metadata = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "dataset": "mogonet_brca_optional",
            "cancer_type": "BRCA",
            "source_split": split_name,
            "label": y["label"].values,
            "label_id": y["label_id"].values,
        }
    )

    return X, y, metadata


def clean_combined_expression(X: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    raw_feature_count = int(X.shape[1])
    missing_before = int(X.isna().sum().sum())

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
        "raw_feature_count": raw_feature_count,
        "removed_all_missing_feature_count": len(all_missing_cols),
        "removed_constant_feature_count": len(constant_cols),
        "missing_value_count_before_imputation": missing_before,
        "final_feature_count": int(X.shape[1]),
    }
    return X, stats


def write_outputs(
    X: pd.DataFrame,
    y: pd.DataFrame,
    metadata: pd.DataFrame,
    feature_df: pd.DataFrame,
    summary: Dict,
    args: argparse.Namespace,
) -> None:
    args.processed_dir.mkdir(parents=True, exist_ok=True)
    args.summary_dir.mkdir(parents=True, exist_ok=True)

    X.to_csv(args.processed_dir / "X.csv", index_label="sample_id")
    y.to_csv(args.processed_dir / "y.csv", index=False)
    metadata.to_csv(args.processed_dir / "metadata.csv", index=False)

    kept_features = feature_df[feature_df["feature_name"].isin(X.columns)].copy()
    kept_features.to_csv(args.processed_dir / "feature_names.csv", index=False)

    train_ids = metadata.loc[metadata["source_split"] == "train", "sample_id"].tolist()
    test_ids = metadata.loc[metadata["source_split"] == "test", "sample_id"].tolist()

    X.loc[train_ids].to_csv(args.processed_dir / "X_train.csv", index_label="sample_id")
    X.loc[test_ids].to_csv(args.processed_dir / "X_test.csv", index_label="sample_id")
    y[y["sample_id"].isin(train_ids)].to_csv(args.processed_dir / "y_train.csv", index=False)
    y[y["sample_id"].isin(test_ids)].to_csv(args.processed_dir / "y_test.csv", index=False)

    with open(args.summary_dir / "mogonet_brca_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    flat_rows = []
    for key, value in summary.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                flat_rows.append({"metric": f"{key}.{subkey}", "value": subvalue})
        else:
            flat_rows.append({"metric": key, "value": value})

    pd.DataFrame(flat_rows).to_csv(args.summary_dir / "mogonet_brca_summary.csv", index=False)


def main() -> None:
    args = parse_args()

    train_expr = read_no_header_csv(args.raw_dir / args.train_expression_file)
    test_expr = read_no_header_csv(args.raw_dir / args.test_expression_file)

    if train_expr.shape[1] != test_expr.shape[1]:
        raise ValueError(
            f"Train/test feature mismatch: train={train_expr.shape[1]}, test={test_expr.shape[1]}"
        )

    feature_df = read_feature_names(args.raw_dir / args.feature_file, expected_count=train_expr.shape[1])
    feature_names = feature_df["feature_name"].tolist()

    train_labels = read_labels(args.raw_dir / args.train_label_file, train_expr.shape[0], "train")
    test_labels = read_labels(args.raw_dir / args.test_label_file, test_expr.shape[0], "test")

    X_train, y_train, meta_train = build_split_dataset(
        train_expr,
        train_labels,
        feature_names,
        split_name="train",
    )
    X_test, y_test, meta_test = build_split_dataset(
        test_expr,
        test_labels,
        feature_names,
        split_name="test",
    )

    X = pd.concat([X_train, X_test], axis=0)
    y = pd.concat([y_train, y_test], axis=0, ignore_index=True)
    metadata = pd.concat([meta_train, meta_test], axis=0, ignore_index=True)

    X, expr_stats = clean_combined_expression(X)

    # Keep y and metadata aligned with X after possible feature filtering.
    if X.isna().any().any():
        raise ValueError("Processed X still contains missing values")
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Processed X contains inf or -inf values")
    if X.index.duplicated().any():
        raise ValueError("Processed X contains duplicate synthetic sample IDs")
    if y["sample_id"].duplicated().any():
        raise ValueError("Processed y contains duplicate sample IDs")
    if X.index.tolist() != y["sample_id"].tolist():
        raise ValueError("X and y sample orders are not aligned")
    if metadata["sample_id"].tolist() != y["sample_id"].tolist():
        raise ValueError("metadata and y sample orders are not aligned")

    label_counts = y["label"].value_counts().sort_index().to_dict()
    class_counts = y["label_id"].value_counts().sort_index().to_dict()
    train_label_counts = y_train["label"].value_counts().sort_index().to_dict()
    test_label_counts = y_test["label"].value_counts().sort_index().to_dict()

    summary = {
        "dataset": "mogonet_brca_optional",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_source_note": (
            "MOGONET BRCA omics-1 is already preprocessed and feature-reduced by the original "
            "MOGONET dataset. It is used as an optional benchmark dataset, not as raw gene expression."
        ),
        "raw_train_expression_shape_rows_cols": [int(train_expr.shape[0]), int(train_expr.shape[1])],
        "raw_test_expression_shape_rows_cols": [int(test_expr.shape[0]), int(test_expr.shape[1])],
        "raw_train_label_shape_rows_cols": [int(train_labels.shape[0]), int(train_labels.shape[1])],
        "raw_test_label_shape_rows_cols": [int(test_labels.shape[0]), int(test_labels.shape[1])],
        **expr_stats,
        "final_sample_count": int(X.shape[0]),
        "final_feature_count": int(X.shape[1]),
        "train_sample_count": int(len(y_train)),
        "test_sample_count": int(len(y_test)),
        "label_mapping": {str(k): v for k, v in LABEL_MAPPING.items()},
        "label_counts": {str(k): int(v) for k, v in label_counts.items()},
        "class_id_counts": {str(k): int(v) for k, v in class_counts.items()},
        "train_label_counts": {str(k): int(v) for k, v in train_label_counts.items()},
        "test_label_counts": {str(k): int(v) for k, v in test_label_counts.items()},
        "sample_id_note": (
            "Original sample IDs are not included in the MOGONET CSV files. "
            "Synthetic sample IDs were generated while preserving the original train/test split."
        ),
        "preprocessing_notes": (
            "Loaded omics-1 train/test matrices, assigned feature names from 1_featname.csv, "
            "mapped numeric labels to BRCA PAM50 subtype names, generated synthetic sample IDs, "
            "preserved the original train/test split, checked missing values, and removed constant "
            "or all-missing features if any. No additional global scaling or feature selection was fitted here."
        ),
    }

    write_outputs(X, y, metadata, feature_df, summary, args)

    print("MOGONET BRCA preprocessing completed.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()