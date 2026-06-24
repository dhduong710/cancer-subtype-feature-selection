#!/usr/bin/env python3
"""Validate processed MOGONET BRCA outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check processed MOGONET BRCA dataset")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed/mogonet_brca_optional"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    paths = {
        "X": args.processed_dir / "X.csv",
        "y": args.processed_dir / "y.csv",
        "metadata": args.processed_dir / "metadata.csv",
        "features": args.processed_dir / "feature_names.csv",
        "X_train": args.processed_dir / "X_train.csv",
        "X_test": args.processed_dir / "X_test.csv",
        "y_train": args.processed_dir / "y_train.csv",
        "y_test": args.processed_dir / "y_test.csv",
    }

    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing processed file {name}: {path}")

    X = pd.read_csv(paths["X"], index_col=0)
    y = pd.read_csv(paths["y"])
    metadata = pd.read_csv(paths["metadata"])
    features = pd.read_csv(paths["features"])

    X_train = pd.read_csv(paths["X_train"], index_col=0)
    X_test = pd.read_csv(paths["X_test"], index_col=0)
    y_train = pd.read_csv(paths["y_train"])
    y_test = pd.read_csv(paths["y_test"])

    problems = []

    if X.shape[0] != y.shape[0]:
        problems.append(f"X rows ({X.shape[0]}) != y rows ({y.shape[0]})")
    if X.index.astype(str).tolist() != y["sample_id"].astype(str).tolist():
        problems.append("X index order does not match y.sample_id order")
    if metadata["sample_id"].astype(str).tolist() != y["sample_id"].astype(str).tolist():
        problems.append("metadata sample order does not match y.sample_id order")
    if y["sample_id"].duplicated().any():
        problems.append("Duplicate sample IDs in y")
    if X.index.duplicated().any():
        problems.append("Duplicate sample IDs in X")
    if X.columns.duplicated().any():
        problems.append("Duplicate feature columns in X")
    if features["feature_name"].duplicated().any():
        problems.append("Duplicate feature names in feature_names.csv")
    if X.shape[1] != features.shape[0]:
        problems.append(f"X columns ({X.shape[1]}) != feature count ({features.shape[0]})")
    if X.isna().any().any():
        problems.append("X contains missing values")
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        problems.append("X contains inf or -inf values")

    constant_feature_count = int((X.nunique(dropna=False) <= 1).sum())

    train_ids_from_meta = metadata.loc[metadata["source_split"] == "train", "sample_id"].astype(str).tolist()
    test_ids_from_meta = metadata.loc[metadata["source_split"] == "test", "sample_id"].astype(str).tolist()

    if X_train.index.astype(str).tolist() != train_ids_from_meta:
        problems.append("X_train sample IDs do not match metadata train split")
    if X_test.index.astype(str).tolist() != test_ids_from_meta:
        problems.append("X_test sample IDs do not match metadata test split")
    if y_train["sample_id"].astype(str).tolist() != train_ids_from_meta:
        problems.append("y_train sample IDs do not match metadata train split")
    if y_test["sample_id"].astype(str).tolist() != test_ids_from_meta:
        problems.append("y_test sample IDs do not match metadata test split")
    if set(train_ids_from_meta).intersection(set(test_ids_from_meta)):
        problems.append("Train/test sample IDs overlap")
    if X_train.shape[1] != X.shape[1] or X_test.shape[1] != X.shape[1]:
        problems.append("Train/test feature count does not match full X feature count")

    report = {
        "status": "PASS" if not problems else "FAIL",
        "problems": problems,
        "X_shape_samples_features": [int(X.shape[0]), int(X.shape[1])],
        "X_train_shape_samples_features": [int(X_train.shape[0]), int(X_train.shape[1])],
        "X_test_shape_samples_features": [int(X_test.shape[0]), int(X_test.shape[1])],
        "y_shape_rows_cols": [int(y.shape[0]), int(y.shape[1])],
        "metadata_shape_rows_cols": [int(metadata.shape[0]), int(metadata.shape[1])],
        "feature_count": int(features.shape[0]),
        "missing_value_count": int(X.isna().sum().sum()),
        "constant_feature_count_after_processing": constant_feature_count,
        "label_counts": {
            str(k): int(v)
            for k, v in y["label"].value_counts().sort_index().to_dict().items()
        },
        "class_id_counts": {
            str(k): int(v)
            for k, v in y["label_id"].value_counts().sort_index().to_dict().items()
        },
        "train_label_counts": {
            str(k): int(v)
            for k, v in y_train["label"].value_counts().sort_index().to_dict().items()
        },
        "test_label_counts": {
            str(k): int(v)
            for k, v in y_test["label"].value_counts().sort_index().to_dict().items()
        },
    }

    print(json.dumps(report, indent=2, ensure_ascii=False))

    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()