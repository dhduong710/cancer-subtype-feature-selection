#!/usr/bin/env python3
"""Validate processed CancerSD STAD outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check processed CancerSD STAD dataset")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed/cancersd_stad"))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    x_path = args.processed_dir / "X.csv"
    y_path = args.processed_dir / "y.csv"
    meta_path = args.processed_dir / "metadata.csv"
    feature_path = args.processed_dir / "feature_names.csv"

    for path in [x_path, y_path, meta_path, feature_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing processed file: {path}")

    X = pd.read_csv(x_path, index_col=0)
    y = pd.read_csv(y_path)
    metadata = pd.read_csv(meta_path)
    features = pd.read_csv(feature_path)

    problems = []

    if X.shape[0] != y.shape[0]:
        problems.append(f"X rows ({X.shape[0]}) != y rows ({y.shape[0]})")
    if X.index.astype(str).tolist() != y["sample_id"].astype(str).tolist():
        problems.append("X index order does not match y.sample_id order")
    if y["sample_id"].duplicated().any():
        problems.append("Duplicate sample IDs in y")
    if X.index.duplicated().any():
        problems.append("Duplicate sample IDs in X")
    if X.columns.duplicated().any():
        problems.append("Duplicate gene columns in X")
    if features["gene_name"].duplicated().any():
        problems.append("Duplicate gene names in feature_names.csv")
    if X.shape[1] != features.shape[0]:
        problems.append(f"X columns ({X.shape[1]}) != feature count ({features.shape[0]})")
    if X.isna().any().any():
        problems.append("X contains missing values")
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        problems.append("X contains inf or -inf")

    constant_gene_count = int((X.nunique(dropna=False) <= 1).sum())

    try:
        train_ids, test_ids = train_test_split(
            y["sample_id"],
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=y["label"],
        )
        y_train = y[y["sample_id"].isin(train_ids)]
        y_test = y[y["sample_id"].isin(test_ids)]

        split_report = {
            "test_size": args.test_size,
            "random_state": args.random_state,
            "train_sample_count": int(len(train_ids)),
            "test_sample_count": int(len(test_ids)),
            "train_label_counts": {
                str(k): int(v)
                for k, v in y_train["label"].value_counts().sort_index().to_dict().items()
            },
            "test_label_counts": {
                str(k): int(v)
                for k, v in y_test["label"].value_counts().sort_index().to_dict().items()
            },
        }
    except ValueError as exc:
        problems.append(f"Stratified split failed: {exc}")
        split_report = None

    report = {
        "status": "PASS" if not problems else "FAIL",
        "problems": problems,
        "X_shape_samples_genes": [int(X.shape[0]), int(X.shape[1])],
        "y_shape_rows_cols": [int(y.shape[0]), int(y.shape[1])],
        "metadata_shape_rows_cols": [int(metadata.shape[0]), int(metadata.shape[1])],
        "feature_count": int(features.shape[0]),
        "missing_value_count": int(X.isna().sum().sum()),
        "constant_gene_count_after_processing": constant_gene_count,
        "label_counts": {
            str(k): int(v)
            for k, v in y["label"].value_counts().sort_index().to_dict().items()
        },
        "class_id_counts": {
            str(k): int(v)
            for k, v in y["label_id"].value_counts().sort_index().to_dict().items()
        },
        "example_stratified_split": split_report,
    }

    print(json.dumps(report, indent=2, ensure_ascii=False))

    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()