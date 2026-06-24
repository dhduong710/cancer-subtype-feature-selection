#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


DATASETS: Dict[str, Dict[str, object]] = {
    "mogcn_brca": {
        "official_split": False,
        "train_size": 0.70,
        "val_size": 0.15,
        "test_size": 0.15,
        "cv_folds": 5,
    },
    "cancersd_stad": {
        "official_split": False,
        "train_size": 0.70,
        "val_size": 0.15,
        "test_size": 0.15,
        "cv_folds": 5,
    },
    "mogonet_brca_optional": {
        "official_split": True,
        "official_train_val_size": 0.20,
        "cv_folds": 5,
    },
}


def find_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def read_labels(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing label file: {path}")

    y = pd.read_csv(path)
    required = {"sample_id", "label", "label_id"}
    missing = required - set(y.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    y = y[["sample_id", "label", "label_id"]].copy()
    y["sample_id"] = y["sample_id"].astype(str)
    y["label"] = y["label"].astype(str)
    return y


def read_x_sample_ids(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing expression file: {path}")
    return pd.read_csv(path, usecols=["sample_id"])["sample_id"].astype(str).tolist()


def assert_x_y_match(x_path: Path, y_df: pd.DataFrame, dataset_name: str) -> None:
    x_ids = read_x_sample_ids(x_path)
    y_ids = y_df["sample_id"].tolist()

    if len(x_ids) != len(set(x_ids)):
        raise ValueError(f"{dataset_name}: duplicated sample_id in {x_path.name}")
    if len(y_ids) != len(set(y_ids)):
        raise ValueError(f"{dataset_name}: duplicated sample_id in label file")
    if set(x_ids) != set(y_ids):
        missing_in_x = sorted(set(y_ids) - set(x_ids))[:10]
        missing_in_y = sorted(set(x_ids) - set(y_ids))[:10]
        raise ValueError(
            f"{dataset_name}: X/y sample_id mismatch. "
            f"missing_in_x={missing_in_x}, missing_in_y={missing_in_y}"
        )


def stratified_train_val_test(
    y: pd.DataFrame,
    train_size: float,
    val_size: float,
    test_size: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total = train_size + val_size + test_size
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"train/val/test sizes must sum to 1. Current sum={total}")

    train_val_df, test_df = train_test_split(
        y,
        test_size=test_size,
        stratify=y["label_id"],
        random_state=random_state,
        shuffle=True,
    )

    relative_val_size = val_size / (train_size + val_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val_size,
        stratify=train_val_df["label_id"],
        random_state=random_state,
        shuffle=True,
    )

    return (
        train_df.sort_values("sample_id").reset_index(drop=True),
        val_df.sort_values("sample_id").reset_index(drop=True),
        test_df.sort_values("sample_id").reset_index(drop=True),
    )


def stratified_train_val(
    y_train_official: pd.DataFrame,
    val_size: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df, val_df = train_test_split(
        y_train_official,
        test_size=val_size,
        stratify=y_train_official["label_id"],
        random_state=random_state,
        shuffle=True,
    )
    return (
        train_df.sort_values("sample_id").reset_index(drop=True),
        val_df.sort_values("sample_id").reset_index(drop=True),
    )


def add_split_column(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    out = df.copy()
    out.insert(0, "split", split_name)
    return out


def label_count_table(split_frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for split_name, df in split_frames.items():
        counts = (
            df.groupby(["label_id", "label"], dropna=False)
            .size()
            .reset_index(name="n_samples")
            .sort_values(["label_id", "label"])
        )
        counts.insert(0, "split", split_name)
        counts.insert(1, "total_samples", len(df))
        rows.append(counts)
    return pd.concat(rows, ignore_index=True)


def create_cv_folds(train_val_df: pd.DataFrame, n_splits: int, random_state: int) -> pd.DataFrame:
    min_class_count = train_val_df["label_id"].value_counts().min()
    if min_class_count < n_splits:
        raise ValueError(
            f"Cannot create {n_splits}-fold stratified CV: "
            f"smallest class in train_val has only {min_class_count} samples."
        )

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rows = []
    X_dummy = train_val_df["sample_id"].values
    y_labels = train_val_df["label_id"].values

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_dummy, y_labels), start=1):
        fold_train = train_val_df.iloc[train_idx].copy()
        fold_val = train_val_df.iloc[val_idx].copy()

        fold_train.insert(0, "cv_role", "train")
        fold_train.insert(0, "fold", fold_idx)

        fold_val.insert(0, "cv_role", "val")
        fold_val.insert(0, "fold", fold_idx)

        rows.extend([fold_train, fold_val])

    folds = pd.concat(rows, ignore_index=True)
    return folds.sort_values(["fold", "cv_role", "sample_id"]).reset_index(drop=True)


def cv_summary_table(cv_folds: pd.DataFrame) -> pd.DataFrame:
    return (
        cv_folds.groupby(["fold", "cv_role", "label_id", "label"], dropna=False)
        .size()
        .reset_index(name="n_samples")
        .sort_values(["fold", "cv_role", "label_id", "label"])
    )


def save_split_files(
    out_dir: Path,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_val_df: pd.DataFrame,
    cv_folds: pd.DataFrame,
    split_summary: pd.DataFrame,
    cv_summary: pd.DataFrame,
    metadata: Dict[str, object],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)
    train_val_df.to_csv(out_dir / "train_val.csv", index=False)

    all_splits = pd.concat(
        [
            add_split_column(train_df, "train"),
            add_split_column(val_df, "val"),
            add_split_column(test_df, "test"),
        ],
        ignore_index=True,
    )
    all_splits.to_csv(out_dir / "all_splits.csv", index=False)

    cv_folds.to_csv(out_dir / "cv_folds.csv", index=False)
    split_summary.to_csv(out_dir / "split_summary.csv", index=False)
    cv_summary.to_csv(out_dir / "cv_fold_summary.csv", index=False)

    with open(out_dir / "split_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def process_dataset(
    repo_root: Path,
    dataset_name: str,
    cfg: Dict[str, object],
    random_state: int,
) -> Dict[str, object]:
    processed_dir = repo_root / "data" / "processed" / dataset_name
    out_dir = repo_root / "data" / "splits" / dataset_name

    official_split = bool(cfg["official_split"])
    n_splits = int(cfg["cv_folds"])

    if official_split:
        y_train_official = read_labels(processed_dir / "y_train.csv")
        y_test = read_labels(processed_dir / "y_test.csv")

        assert_x_y_match(processed_dir / "X_train.csv", y_train_official, dataset_name)
        assert_x_y_match(processed_dir / "X_test.csv", y_test, dataset_name)

        overlap = set(y_train_official["sample_id"]) & set(y_test["sample_id"])
        if overlap:
            raise ValueError(
                f"{dataset_name}: official train/test overlap detected: {sorted(overlap)[:10]}"
            )

        train_df, val_df = stratified_train_val(
            y_train_official,
            val_size=float(cfg["official_train_val_size"]),
            random_state=random_state,
        )
        test_df = y_test.sort_values("sample_id").reset_index(drop=True)
        train_val_df = y_train_official.sort_values("sample_id").reset_index(drop=True)
        split_strategy = "official_test_preserved_train_val_split_from_official_train"

    else:
        y_all = read_labels(processed_dir / "y.csv")
        assert_x_y_match(processed_dir / "X.csv", y_all, dataset_name)

        train_df, val_df, test_df = stratified_train_val_test(
            y_all,
            train_size=float(cfg["train_size"]),
            val_size=float(cfg["val_size"]),
            test_size=float(cfg["test_size"]),
            random_state=random_state,
        )
        train_val_df = (
            pd.concat([train_df, val_df], ignore_index=True)
            .sort_values("sample_id")
            .reset_index(drop=True)
        )
        split_strategy = "stratified_70_15_15_holdout"

    cv_folds = create_cv_folds(train_val_df, n_splits=n_splits, random_state=random_state)

    split_summary = label_count_table(
        {
            "train": train_df,
            "val": val_df,
            "test": test_df,
            "train_val": train_val_df,
        }
    )
    cv_summary = cv_summary_table(cv_folds)

    metadata = {
        "dataset": dataset_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "random_state": random_state,
        "official_split": official_split,
        "split_strategy": split_strategy,
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "n_train_val": int(len(train_val_df)),
        "cv_folds": n_splits,
        "notes": [
            "Test split must stay untouched until final evaluation.",
            "Feature selection, scaling, and model fitting must be fitted inside each training split only to avoid data leakage.",
            "CV folds are created only on train_val, never on test.",
        ],
    }

    save_split_files(
        out_dir=out_dir,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        train_val_df=train_val_df,
        cv_folds=cv_folds,
        split_summary=split_summary,
        cv_summary=cv_summary,
        metadata=metadata,
    )

    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--datasets", nargs="*", default=list(DATASETS.keys()))
    args = parser.parse_args()

    repo_root = find_repo_root()
    all_metadata = []

    for dataset_name in args.datasets:
        if dataset_name not in DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASETS)}")

        print(f"Creating splits for {dataset_name}...")
        metadata = process_dataset(repo_root, dataset_name, DATASETS[dataset_name], args.random_state)
        all_metadata.append(metadata)

        print(
            f"  done: train={metadata['n_train']}, val={metadata['n_val']}, "
            f"test={metadata['n_test']}, train_val={metadata['n_train_val']}, "
            f"cv_folds={metadata['cv_folds']}"
        )

    out_root = repo_root / "data" / "splits"
    out_root.mkdir(parents=True, exist_ok=True)

    with open(out_root / "split_creation_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, indent=2, ensure_ascii=False)

    print("\nAll splits created successfully.")
    print(f"Output directory: {out_root}")


if __name__ == "__main__":
    main()