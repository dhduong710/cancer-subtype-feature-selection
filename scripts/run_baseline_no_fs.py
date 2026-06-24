#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


DATASETS = [
    "mogcn_brca",
    "cancersd_stad",
    "mogonet_brca_optional",
]

DEFAULT_MODELS = [
    "logistic_regression",
    "linear_svm",
    "random_forest",
    "gaussian_nb",
]


def find_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def load_dataset_matrix(repo_root: Path, dataset: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    processed_dir = repo_root / "data" / "processed" / dataset

    x_path = processed_dir / "X.csv"
    y_path = processed_dir / "y.csv"

    if x_path.exists() and y_path.exists():
        X = pd.read_csv(x_path)
        y = pd.read_csv(y_path)
    else:
        x_train_path = processed_dir / "X_train.csv"
        x_test_path = processed_dir / "X_test.csv"
        y_train_path = processed_dir / "y_train.csv"
        y_test_path = processed_dir / "y_test.csv"

        X = pd.concat(
            [pd.read_csv(x_train_path), pd.read_csv(x_test_path)],
            ignore_index=True,
        )
        y = pd.concat(
            [pd.read_csv(y_train_path), pd.read_csv(y_test_path)],
            ignore_index=True,
        )

    if "sample_id" not in X.columns:
        raise ValueError(f"{dataset}: X file must contain sample_id column")

    required_y_cols = {"sample_id", "label", "label_id"}
    missing = required_y_cols - set(y.columns)
    if missing:
        raise ValueError(f"{dataset}: y file missing columns: {sorted(missing)}")

    X["sample_id"] = X["sample_id"].astype(str)
    y["sample_id"] = y["sample_id"].astype(str)

    if X["sample_id"].duplicated().any():
        raise ValueError(f"{dataset}: duplicated sample_id in X")
    if y["sample_id"].duplicated().any():
        raise ValueError(f"{dataset}: duplicated sample_id in y")

    if set(X["sample_id"]) != set(y["sample_id"]):
        raise ValueError(f"{dataset}: X/y sample_id mismatch")

    y = y[["sample_id", "label", "label_id"]].copy()

    return X, y


def align_by_sample_ids(
    X: pd.DataFrame,
    y: pd.DataFrame,
    sample_ids: List[str],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    sample_ids = [str(s) for s in sample_ids]

    X_indexed = X.set_index("sample_id", drop=True)
    y_indexed = y.set_index("sample_id", drop=True)

    missing_x = sorted(set(sample_ids) - set(X_indexed.index))
    missing_y = sorted(set(sample_ids) - set(y_indexed.index))

    if missing_x or missing_y:
        raise ValueError(
            f"Missing sample IDs. missing_x={missing_x[:5]}, missing_y={missing_y[:5]}"
        )

    X_part = X_indexed.loc[sample_ids]
    y_part = y_indexed.loc[sample_ids]

    feature_names = list(X_part.columns)

    return (
        X_part.to_numpy(dtype=np.float32),
        y_part["label_id"].to_numpy(dtype=int),
        feature_names,
    )


def get_label_mapping(y: pd.DataFrame) -> pd.DataFrame:
    mapping = (
        y[["label_id", "label"]]
        .drop_duplicates()
        .sort_values("label_id")
        .reset_index(drop=True)
    )
    return mapping


def build_model(model_name: str, random_state: int) -> Pipeline:
    if model_name == "logistic_regression":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    OneVsRestClassifier(
                        LogisticRegression(
                            solver="liblinear",
                            class_weight="balanced",
                            max_iter=5000,
                            random_state=random_state,
                        ),
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    if model_name == "linear_svm":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LinearSVC(
                        C=1.0,
                        class_weight="balanced",
                        max_iter=20000,
                        random_state=random_state,
                    ),
                ),
            ]
        )

    if model_name == "random_forest":
        return Pipeline(
            steps=[
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=300,
                        criterion="gini",
                        class_weight="balanced",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    if model_name == "gaussian_nb":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("classifier", GaussianNB()),
            ]
        )

    raise ValueError(f"Unknown model: {model_name}")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
    }


def run_cv_for_dataset_model(
    dataset: str,
    model_name: str,
    X: pd.DataFrame,
    y: pd.DataFrame,
    cv_folds: pd.DataFrame,
    random_state: int,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []

    for fold in sorted(cv_folds["fold"].unique()):
        fold_df = cv_folds[cv_folds["fold"] == fold]
        train_ids = (
            fold_df[fold_df["cv_role"] == "train"]["sample_id"]
            .astype(str)
            .tolist()
        )
        val_ids = (
            fold_df[fold_df["cv_role"] == "val"]["sample_id"]
            .astype(str)
            .tolist()
        )

        X_train, y_train, _ = align_by_sample_ids(X, y, train_ids)
        X_val, y_val, _ = align_by_sample_ids(X, y, val_ids)

        model = build_model(model_name, random_state=random_state)

        start = time.time()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        elapsed = time.time() - start

        metrics = compute_metrics(y_val, y_pred)

        row = {
            "dataset": dataset,
            "model": model_name,
            "fold": int(fold),
            "n_train": int(len(y_train)),
            "n_val": int(len(y_val)),
            "n_features": int(X_train.shape[1]),
            "runtime_seconds": round(float(elapsed), 4),
        }
        row.update(metrics)
        rows.append(row)

    return rows


def summarize_cv(cv_results: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "accuracy",
        "balanced_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
        "runtime_seconds",
    ]

    summary = (
        cv_results.groupby(["dataset", "model"])[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )

    summary.columns = [
        "_".join(col).rstrip("_") if isinstance(col, tuple) else col
        for col in summary.columns
    ]

    summary = summary.rename(
        columns={
            "accuracy_mean": "cv_accuracy_mean",
            "accuracy_std": "cv_accuracy_std",
            "balanced_accuracy_mean": "cv_balanced_accuracy_mean",
            "balanced_accuracy_std": "cv_balanced_accuracy_std",
            "macro_precision_mean": "cv_macro_precision_mean",
            "macro_precision_std": "cv_macro_precision_std",
            "macro_recall_mean": "cv_macro_recall_mean",
            "macro_recall_std": "cv_macro_recall_std",
            "macro_f1_mean": "cv_macro_f1_mean",
            "macro_f1_std": "cv_macro_f1_std",
            "weighted_f1_mean": "cv_weighted_f1_mean",
            "weighted_f1_std": "cv_weighted_f1_std",
            "runtime_seconds_mean": "cv_runtime_seconds_mean",
            "runtime_seconds_std": "cv_runtime_seconds_std",
        }
    )

    return summary.sort_values(
        ["dataset", "cv_macro_f1_mean", "cv_accuracy_mean"],
        ascending=[True, False, False],
    ).reset_index(drop=True)


def run_final_test_for_dataset_model(
    repo_root: Path,
    dataset: str,
    model_name: str,
    X: pd.DataFrame,
    y: pd.DataFrame,
    train_val_ids: List[str],
    test_ids: List[str],
    label_ids_sorted: List[int],
    random_state: int,
    out_dir: Path,
) -> Dict[str, object]:
    X_train_val, y_train_val, _ = align_by_sample_ids(X, y, train_val_ids)
    X_test, y_test, _ = align_by_sample_ids(X, y, test_ids)

    model = build_model(model_name, random_state=random_state)

    start = time.time()
    model.fit(X_train_val, y_train_val)
    y_pred = model.predict(X_test)
    elapsed = time.time() - start

    metrics = compute_metrics(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred, labels=label_ids_sorted)

    cm_dir = out_dir / "confusion_matrices"
    cm_dir.mkdir(parents=True, exist_ok=True)

    label_mapping = get_label_mapping(y)
    id_to_label = {
        int(row["label_id"]): str(row["label"])
        for _, row in label_mapping.iterrows()
    }
    labels_for_cm = [id_to_label[i] for i in label_ids_sorted]

    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{label}" for label in labels_for_cm],
        columns=[f"pred_{label}" for label in labels_for_cm],
    )
    cm_df.to_csv(cm_dir / f"{dataset}__{model_name}.csv")

    row = {
        "dataset": dataset,
        "model": model_name,
        "n_train_val": int(len(y_train_val)),
        "n_test": int(len(y_test)),
        "n_features": int(X_train_val.shape[1]),
        "runtime_seconds": round(float(elapsed), 4),
    }
    row.update(metrics)

    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=DATASETS)
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    repo_root = find_repo_root()
    out_dir = repo_root / "results" / "experiments" / "baseline_no_fs"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_cv_rows: List[Dict[str, object]] = []
    all_test_rows: List[Dict[str, object]] = []
    all_label_mappings: List[pd.DataFrame] = []

    print("Running baseline experiments without feature selection.")
    print(f"Datasets: {args.datasets}")
    print(f"Models: {args.models}")
    print(f"Output directory: {out_dir}")
    print("")

    for dataset in args.datasets:
        print(f"Loading dataset: {dataset}")
        X, y = load_dataset_matrix(repo_root, dataset)
        label_mapping = get_label_mapping(y)
        label_mapping.insert(0, "dataset", dataset)
        all_label_mappings.append(label_mapping)

        split_dir = repo_root / "data" / "splits" / dataset
        train_val = read_csv_required(split_dir / "train_val.csv")
        test = read_csv_required(split_dir / "test.csv")
        cv_folds = read_csv_required(split_dir / "cv_folds.csv")

        train_val["sample_id"] = train_val["sample_id"].astype(str)
        test["sample_id"] = test["sample_id"].astype(str)
        cv_folds["sample_id"] = cv_folds["sample_id"].astype(str)

        train_val_ids = train_val["sample_id"].tolist()
        test_ids = test["sample_id"].tolist()
        label_ids_sorted = sorted(y["label_id"].astype(int).unique().tolist())

        print(
            f"  samples={len(y)}, features={X.shape[1] - 1}, "
            f"train_val={len(train_val_ids)}, test={len(test_ids)}"
        )

        for model_name in args.models:
            print(f"  CV: {dataset} / {model_name}")
            cv_rows = run_cv_for_dataset_model(
                dataset=dataset,
                model_name=model_name,
                X=X,
                y=y,
                cv_folds=cv_folds,
                random_state=args.random_state,
            )
            all_cv_rows.extend(cv_rows)

            print(f"  Final test: {dataset} / {model_name}")
            test_row = run_final_test_for_dataset_model(
                repo_root=repo_root,
                dataset=dataset,
                model_name=model_name,
                X=X,
                y=y,
                train_val_ids=train_val_ids,
                test_ids=test_ids,
                label_ids_sorted=label_ids_sorted,
                random_state=args.random_state,
                out_dir=out_dir,
            )
            all_test_rows.append(test_row)

        print("")

    cv_results = pd.DataFrame(all_cv_rows)
    cv_summary = summarize_cv(cv_results)
    test_results = pd.DataFrame(all_test_rows)

    test_results = test_results.merge(
        cv_summary[
            [
                "dataset",
                "model",
                "cv_accuracy_mean",
                "cv_accuracy_std",
                "cv_macro_f1_mean",
                "cv_macro_f1_std",
                "cv_weighted_f1_mean",
                "cv_weighted_f1_std",
            ]
        ],
        on=["dataset", "model"],
        how="left",
    )

    best_models = (
        cv_summary.sort_values(
            ["dataset", "cv_macro_f1_mean", "cv_accuracy_mean"],
            ascending=[True, False, False],
        )
        .groupby("dataset", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )

    label_mappings = pd.concat(all_label_mappings, ignore_index=True)

    cv_results.to_csv(out_dir / "cv_results.csv", index=False)
    cv_summary.to_csv(out_dir / "cv_summary.csv", index=False)
    test_results.to_csv(out_dir / "test_results.csv", index=False)
    best_models.to_csv(out_dir / "best_models_by_cv.csv", index=False)
    label_mappings.to_csv(out_dir / "label_mappings.csv", index=False)

    metadata = {
        "experiment_name": "baseline_no_feature_selection",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "random_state": args.random_state,
        "datasets": args.datasets,
        "models": args.models,
        "notes": [
            "This experiment uses no feature selection.",
            "Standardization is fitted inside each training fold through sklearn Pipeline.",
            "Cross-validation is performed only on train_val.",
            "Final test evaluation trains on train_val and evaluates once on test.",
            "These results are baseline controls for later feature selection experiments.",
        ],
    }

    with open(out_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("Baseline experiment completed.")
    print(f"Saved results to: {out_dir}")


if __name__ == "__main__":
    main()
