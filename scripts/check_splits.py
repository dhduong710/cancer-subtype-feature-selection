#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd


DATASETS = ["mogcn_brca", "cancersd_stad", "mogonet_brca_optional"]


def find_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def read_csv_required(path: Path, required_cols: Set[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")

    df = pd.read_csv(path)
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    df["sample_id"] = df["sample_id"].astype(str)
    return df


def read_processed_ids(processed_dir: Path, dataset_name: str) -> Dict[str, Set[str]]:
    if dataset_name == "mogonet_brca_optional":
        x_train = pd.read_csv(processed_dir / "X_train.csv", usecols=["sample_id"])["sample_id"].astype(str)
        x_test = pd.read_csv(processed_dir / "X_test.csv", usecols=["sample_id"])["sample_id"].astype(str)

        return {
            "all": set(x_train) | set(x_test),
            "official_train": set(x_train),
            "official_test": set(x_test),
        }

    x_all = pd.read_csv(processed_dir / "X.csv", usecols=["sample_id"])["sample_id"].astype(str)
    return {"all": set(x_all)}


def assert_no_duplicates(df: pd.DataFrame, name: str) -> None:
    duplicated = df[df["sample_id"].duplicated()]["sample_id"].tolist()
    if duplicated:
        raise ValueError(f"{name}: duplicated sample_id detected: {duplicated[:10]}")


def class_counts(df: pd.DataFrame) -> Dict[str, int]:
    return df["label"].value_counts().sort_index().astype(int).to_dict()


def validate_dataset(repo_root: Path, dataset_name: str) -> Dict[str, object]:
    processed_dir = repo_root / "data" / "processed" / dataset_name
    split_dir = repo_root / "data" / "splits" / dataset_name

    required_label_cols = {"sample_id", "label", "label_id"}

    train = read_csv_required(split_dir / "train.csv", required_label_cols)
    val = read_csv_required(split_dir / "val.csv", required_label_cols)
    test = read_csv_required(split_dir / "test.csv", required_label_cols)
    train_val = read_csv_required(split_dir / "train_val.csv", required_label_cols)
    all_splits = read_csv_required(split_dir / "all_splits.csv", required_label_cols | {"split"})
    cv_folds = read_csv_required(split_dir / "cv_folds.csv", required_label_cols | {"fold", "cv_role"})

    for name, df in [
        ("train", train),
        ("val", val),
        ("test", test),
        ("train_val", train_val),
    ]:
        assert_no_duplicates(df, f"{dataset_name}/{name}")

    train_ids = set(train["sample_id"])
    val_ids = set(val["sample_id"])
    test_ids = set(test["sample_id"])
    train_val_ids = set(train_val["sample_id"])

    if train_ids & val_ids:
        raise ValueError(f"{dataset_name}: train/val overlap detected")
    if train_ids & test_ids:
        raise ValueError(f"{dataset_name}: train/test overlap detected")
    if val_ids & test_ids:
        raise ValueError(f"{dataset_name}: val/test overlap detected")
    if train_ids | val_ids != train_val_ids:
        raise ValueError(f"{dataset_name}: train_val is not exactly train union val")

    processed_ids = read_processed_ids(processed_dir, dataset_name)

    if train_ids | val_ids | test_ids != processed_ids["all"]:
        missing = sorted(processed_ids["all"] - (train_ids | val_ids | test_ids))[:10]
        extra = sorted((train_ids | val_ids | test_ids) - processed_ids["all"])[:10]
        raise ValueError(
            f"{dataset_name}: split coverage mismatch. missing={missing}, extra={extra}"
        )

    if dataset_name == "mogonet_brca_optional":
        if not train_val_ids <= processed_ids["official_train"]:
            raise ValueError(f"{dataset_name}: train/val contain samples outside official train")
        if test_ids != processed_ids["official_test"]:
            raise ValueError(f"{dataset_name}: test split does not equal official test")

    all_split_ids = set(all_splits["sample_id"])
    if all_split_ids != train_ids | val_ids | test_ids:
        raise ValueError(f"{dataset_name}: all_splits.csv inconsistent with train/val/test files")

    full_labels = set(all_splits["label"])
    for split_name, df in [("train", train), ("val", val), ("test", test)]:
        missing_labels = full_labels - set(df["label"])
        if missing_labels:
            raise ValueError(
                f"{dataset_name}: {split_name} misses labels {sorted(missing_labels)}"
            )

    if set(cv_folds["sample_id"]) != train_val_ids:
        raise ValueError(f"{dataset_name}: cv_folds sample IDs must equal train_val IDs")
    if set(cv_folds["sample_id"]) & test_ids:
        raise ValueError(f"{dataset_name}: cv_folds contain test samples")

    n_folds = int(cv_folds["fold"].nunique())

    for fold in sorted(cv_folds["fold"].unique()):
        fold_df = cv_folds[cv_folds["fold"] == fold]
        fold_train = fold_df[fold_df["cv_role"] == "train"]
        fold_val = fold_df[fold_df["cv_role"] == "val"]

        fold_train_ids = set(fold_train["sample_id"])
        fold_val_ids = set(fold_val["sample_id"])

        if fold_train_ids & fold_val_ids:
            raise ValueError(f"{dataset_name}: CV fold {fold} train/val overlap")
        if fold_train_ids | fold_val_ids != train_val_ids:
            raise ValueError(f"{dataset_name}: CV fold {fold} does not cover train_val")
        if set(fold_val["label"]) != full_labels:
            raise ValueError(f"{dataset_name}: CV fold {fold} validation misses at least one class")

    val_occurrence = (
        cv_folds[cv_folds["cv_role"] == "val"]["sample_id"].value_counts().sort_index()
    )
    if not (val_occurrence == 1).all():
        bad = val_occurrence[val_occurrence != 1].head(10).to_dict()
        raise ValueError(
            f"{dataset_name}: each train_val sample must appear once as CV val. Bad={bad}"
        )

    return {
        "dataset": dataset_name,
        "status": "PASS",
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "n_train_val": len(train_val),
        "n_cv_folds": n_folds,
        "train_counts": class_counts(train),
        "val_counts": class_counts(val),
        "test_counts": class_counts(test),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=DATASETS)
    args = parser.parse_args()

    repo_root = find_repo_root()
    summaries: List[Dict[str, object]] = []

    for dataset_name in args.datasets:
        print(f"Checking splits for {dataset_name}...")
        summary = validate_dataset(repo_root, dataset_name)
        summaries.append(summary)

        print(
            f"  PASS: train={summary['n_train']}, val={summary['n_val']}, "
            f"test={summary['n_test']}, CV={summary['n_cv_folds']} folds"
        )

    out_path = repo_root / "data" / "splits" / "split_validation_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)

    print("\nAll split checks passed.")
    print(f"Validation summary saved to: {out_path}")


if __name__ == "__main__":
    main()