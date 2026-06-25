#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, f_classif
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

DEFAULT_WRAPPER_SELECTORS = [
    "rfe_logistic",
    "rfe_linear_svm",
]

DEFAULT_K_VALUES = [50, 100, 200]

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
        X = pd.concat(
            [
                pd.read_csv(processed_dir / "X_train.csv"),
                pd.read_csv(processed_dir / "X_test.csv"),
            ],
            ignore_index=True,
        )
        y = pd.concat(
            [
                pd.read_csv(processed_dir / "y_train.csv"),
                pd.read_csv(processed_dir / "y_test.csv"),
            ],
            ignore_index=True,
        )

    if "sample_id" not in X.columns:
        raise ValueError(f"{dataset}: X must contain sample_id column")

    required_y_cols = {"sample_id", "label", "label_id"}
    missing = required_y_cols - set(y.columns)
    if missing:
        raise ValueError(f"{dataset}: y missing columns: {sorted(missing)}")

    X["sample_id"] = X["sample_id"].astype(str)
    y["sample_id"] = y["sample_id"].astype(str)

    if X["sample_id"].duplicated().any():
        raise ValueError(f"{dataset}: duplicated sample_id in X")
    if y["sample_id"].duplicated().any():
        raise ValueError(f"{dataset}: duplicated sample_id in y")
    if set(X["sample_id"]) != set(y["sample_id"]):
        raise ValueError(f"{dataset}: X/y sample_id mismatch")

    return X, y[["sample_id", "label", "label_id"]].copy()


def get_feature_names(X: pd.DataFrame) -> List[str]:
    return [col for col in X.columns if col != "sample_id"]


def get_label_mapping(y: pd.DataFrame) -> pd.DataFrame:
    return (
        y[["label_id", "label"]]
        .drop_duplicates()
        .sort_values("label_id")
        .reset_index(drop=True)
    )


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


def sanitize_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    return np.nan_to_num(
        scores,
        nan=-np.inf,
        posinf=np.finfo(np.float64).max,
        neginf=-np.inf,
    )


def top_k_indices_from_scores(scores: np.ndarray, k: int) -> np.ndarray:
    scores = sanitize_scores(scores)
    k = min(int(k), len(scores))
    if k <= 0:
        raise ValueError(f"k must be positive. Got k={k}")
    return np.argsort(scores)[::-1][:k]


def make_rfe_estimator(selector_name: str, random_state: int):
    if selector_name == "rfe_logistic":
        return LogisticRegression(
            solver="lbfgs",
            class_weight="balanced",
            max_iter=3000,
            random_state=random_state,
        )

    if selector_name == "rfe_linear_svm":
        return LinearSVC(
            C=0.5,
            class_weight="balanced",
            max_iter=30000,
            dual="auto",
            random_state=random_state,
        )

    raise ValueError(f"Unknown wrapper selector: {selector_name}")


def coefficient_importance(estimator) -> np.ndarray:
    if not hasattr(estimator, "coef_"):
        raise ValueError("RFE estimator has no coef_ attribute")

    coef = np.asarray(estimator.coef_)

    if coef.ndim == 1:
        return np.abs(coef)

    return np.mean(np.abs(coef), axis=0)


def wrapper_select_features(
    selector_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    k: int,
    prefilter_top_m: int,
    random_state: int,
) -> Tuple[np.ndarray, pd.DataFrame, float]:
    start = time.time()

    n_original_features = X_train.shape[1]
    prefilter_top_m = min(int(prefilter_top_m), n_original_features)
    k = min(int(k), prefilter_top_m)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        anova_scores, _ = f_classif(X_train_scaled, y_train)

    anova_scores = sanitize_scores(anova_scores)
    prefilter_indices = top_k_indices_from_scores(anova_scores, prefilter_top_m)

    X_prefiltered = X_train_scaled[:, prefilter_indices]

    estimator = make_rfe_estimator(selector_name, random_state=random_state)

    rfe = RFE(
        estimator=estimator,
        n_features_to_select=k,
        step=0.20,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rfe.fit(X_prefiltered, y_train)

    selected_positions = np.where(rfe.support_)[0]
    selected_original_indices = prefilter_indices[selected_positions]

    selected_importance = coefficient_importance(rfe.estimator_)

    info = pd.DataFrame(
        {
            "feature_index": selected_original_indices.astype(int),
            "prefilter_anova_score": anova_scores[selected_original_indices],
            "rfe_importance": selected_importance,
        }
    )

    info = info.sort_values(
        ["rfe_importance", "prefilter_anova_score"],
        ascending=[False, False],
    ).reset_index(drop=True)
    info.insert(0, "rank", np.arange(1, len(info) + 1))

    elapsed = time.time() - start

    return selected_original_indices.astype(int), info, float(elapsed)


def build_classifier(model_name: str, random_state: int) -> Pipeline:
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
                        dual="auto",
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


def save_selected_features(
    out_dir: Path,
    dataset: str,
    selector_name: str,
    k: int,
    selected_info: pd.DataFrame,
    feature_names: List[str],
) -> None:
    selected_dir = out_dir / "selected_features"
    selected_dir.mkdir(parents=True, exist_ok=True)

    df = selected_info.copy()
    df.insert(0, "dataset", dataset)
    df.insert(1, "selector", selector_name)
    df.insert(2, "k", int(k))
    df["feature_name"] = df["feature_index"].apply(lambda idx: feature_names[int(idx)])

    df = df[
        [
            "dataset",
            "selector",
            "k",
            "rank",
            "feature_index",
            "feature_name",
            "rfe_importance",
            "prefilter_anova_score",
        ]
    ]

    df.to_csv(
        selected_dir / f"{dataset}__{selector_name}__k{k}.csv",
        index=False,
    )


def summarize_cv(cv_results: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "accuracy",
        "balanced_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
        "feature_selection_seconds",
        "model_runtime_seconds",
        "total_runtime_seconds",
    ]

    summary = (
        cv_results.groupby(["dataset", "selector", "k", "model"])[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )

    summary.columns = [
        "_".join(col).rstrip("_") if isinstance(col, tuple) else col
        for col in summary.columns
    ]

    rename_map = {}
    for metric in metric_cols:
        rename_map[f"{metric}_mean"] = f"cv_{metric}_mean"
        rename_map[f"{metric}_std"] = f"cv_{metric}_std"

    summary = summary.rename(columns=rename_map)

    return summary.sort_values(
        ["dataset", "cv_macro_f1_mean", "cv_accuracy_mean"],
        ascending=[True, False, False],
    ).reset_index(drop=True)


def run_cv_for_dataset_selector_k(
    dataset: str,
    selector_name: str,
    k: int,
    models: List[str],
    X: pd.DataFrame,
    y: pd.DataFrame,
    cv_folds: pd.DataFrame,
    prefilter_top_m: int,
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

        selected_indices, _, fs_elapsed = wrapper_select_features(
            selector_name=selector_name,
            X_train=X_train,
            y_train=y_train,
            k=k,
            prefilter_top_m=prefilter_top_m,
            random_state=random_state,
        )

        X_train_selected = X_train[:, selected_indices]
        X_val_selected = X_val[:, selected_indices]

        for model_name in models:
            classifier = build_classifier(model_name, random_state=random_state)

            fit_start = time.time()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                classifier.fit(X_train_selected, y_train)
            y_pred = classifier.predict(X_val_selected)
            fit_elapsed = time.time() - fit_start

            metrics = compute_metrics(y_val, y_pred)

            row = {
                "dataset": dataset,
                "selector": selector_name,
                "k": int(k),
                "model": model_name,
                "fold": int(fold),
                "n_train": int(len(y_train)),
                "n_val": int(len(y_val)),
                "n_original_features": int(X_train.shape[1]),
                "n_prefilter_features": int(min(prefilter_top_m, X_train.shape[1])),
                "n_selected_features": int(len(selected_indices)),
                "feature_selection_seconds": round(float(fs_elapsed), 4),
                "model_runtime_seconds": round(float(fit_elapsed), 4),
                "total_runtime_seconds": round(float(fs_elapsed + fit_elapsed), 4),
            }
            row.update(metrics)
            rows.append(row)

    return rows


def run_final_test_for_dataset_selector_k(
    dataset: str,
    selector_name: str,
    k: int,
    models: List[str],
    X: pd.DataFrame,
    y: pd.DataFrame,
    train_val_ids: List[str],
    test_ids: List[str],
    label_ids_sorted: List[int],
    feature_names: List[str],
    prefilter_top_m: int,
    random_state: int,
    out_dir: Path,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []

    X_train_val, y_train_val, _ = align_by_sample_ids(X, y, train_val_ids)
    X_test, y_test, _ = align_by_sample_ids(X, y, test_ids)

    selected_indices, selected_info, fs_elapsed = wrapper_select_features(
        selector_name=selector_name,
        X_train=X_train_val,
        y_train=y_train_val,
        k=k,
        prefilter_top_m=prefilter_top_m,
        random_state=random_state,
    )

    save_selected_features(
        out_dir=out_dir,
        dataset=dataset,
        selector_name=selector_name,
        k=k,
        selected_info=selected_info,
        feature_names=feature_names,
    )

    X_train_val_selected = X_train_val[:, selected_indices]
    X_test_selected = X_test[:, selected_indices]

    label_mapping = get_label_mapping(y)
    id_to_label = {
        int(row["label_id"]): str(row["label"])
        for _, row in label_mapping.iterrows()
    }
    labels_for_cm = [id_to_label[i] for i in label_ids_sorted]

    cm_dir = out_dir / "confusion_matrices"
    cm_dir.mkdir(parents=True, exist_ok=True)

    for model_name in models:
        classifier = build_classifier(model_name, random_state=random_state)

        fit_start = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            classifier.fit(X_train_val_selected, y_train_val)
        y_pred = classifier.predict(X_test_selected)
        fit_elapsed = time.time() - fit_start

        metrics = compute_metrics(y_test, y_pred)

        cm = confusion_matrix(y_test, y_pred, labels=label_ids_sorted)
        cm_df = pd.DataFrame(
            cm,
            index=[f"true_{label}" for label in labels_for_cm],
            columns=[f"pred_{label}" for label in labels_for_cm],
        )
        cm_df.to_csv(
            cm_dir / f"{dataset}__{selector_name}__k{k}__{model_name}.csv"
        )

        row = {
            "dataset": dataset,
            "selector": selector_name,
            "k": int(k),
            "model": model_name,
            "n_train_val": int(len(y_train_val)),
            "n_test": int(len(y_test)),
            "n_original_features": int(X_train_val.shape[1]),
            "n_prefilter_features": int(min(prefilter_top_m, X_train_val.shape[1])),
            "n_selected_features": int(len(selected_indices)),
            "feature_selection_seconds": round(float(fs_elapsed), 4),
            "model_runtime_seconds": round(float(fit_elapsed), 4),
            "total_runtime_seconds": round(float(fs_elapsed + fit_elapsed), 4),
        }
        row.update(metrics)
        rows.append(row)

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=DATASETS)
    parser.add_argument("--selectors", nargs="*", default=DEFAULT_WRAPPER_SELECTORS)
    parser.add_argument("--k-values", nargs="*", type=int, default=DEFAULT_K_VALUES)
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--prefilter-top-m", type=int, default=1000)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    repo_root = find_repo_root()
    out_dir = repo_root / "results" / "experiments" / "wrapper_rfe"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_cv_rows: List[Dict[str, object]] = []
    all_test_rows: List[Dict[str, object]] = []
    all_label_mappings: List[pd.DataFrame] = []

    print("Running wrapper RFE feature selection experiments.")
    print(f"Datasets: {args.datasets}")
    print(f"Wrapper selectors: {args.selectors}")
    print(f"k values: {args.k_values}")
    print(f"Downstream models: {args.models}")
    print(f"ANOVA prefilter top m: {args.prefilter_top_m}")
    print(f"Output directory: {out_dir}")
    print("")

    for dataset in args.datasets:
        print(f"Loading dataset: {dataset}")

        X, y = load_dataset_matrix(repo_root, dataset)
        feature_names = get_feature_names(X)

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

        valid_k_values = [
            k for k in args.k_values if k <= min(args.prefilter_top_m, len(feature_names))
        ]

        if len(valid_k_values) != len(args.k_values):
            skipped = sorted(set(args.k_values) - set(valid_k_values))
            print(f"  skipped k values larger than available prefilter features: {skipped}")

        print(
            f"  samples={len(y)}, features={len(feature_names)}, "
            f"train_val={len(train_val_ids)}, test={len(test_ids)}"
        )

        for selector_name in args.selectors:
            for k in valid_k_values:
                print(f"  Selecting with {selector_name}, k={k}")

                cv_rows = run_cv_for_dataset_selector_k(
                    dataset=dataset,
                    selector_name=selector_name,
                    k=k,
                    models=args.models,
                    X=X,
                    y=y,
                    cv_folds=cv_folds,
                    prefilter_top_m=args.prefilter_top_m,
                    random_state=args.random_state,
                )
                all_cv_rows.extend(cv_rows)

                test_rows = run_final_test_for_dataset_selector_k(
                    dataset=dataset,
                    selector_name=selector_name,
                    k=k,
                    models=args.models,
                    X=X,
                    y=y,
                    train_val_ids=train_val_ids,
                    test_ids=test_ids,
                    label_ids_sorted=label_ids_sorted,
                    feature_names=feature_names,
                    prefilter_top_m=args.prefilter_top_m,
                    random_state=args.random_state,
                    out_dir=out_dir,
                )
                all_test_rows.extend(test_rows)

        print("")

    cv_results = pd.DataFrame(all_cv_rows)
    cv_summary = summarize_cv(cv_results)
    test_results = pd.DataFrame(all_test_rows)

    test_results = test_results.merge(
        cv_summary[
            [
                "dataset",
                "selector",
                "k",
                "model",
                "cv_accuracy_mean",
                "cv_accuracy_std",
                "cv_macro_f1_mean",
                "cv_macro_f1_std",
                "cv_weighted_f1_mean",
                "cv_weighted_f1_std",
            ]
        ],
        on=["dataset", "selector", "k", "model"],
        how="left",
    )

    best_configs_by_dataset = (
        cv_summary.sort_values(
            ["dataset", "cv_macro_f1_mean", "cv_accuracy_mean"],
            ascending=[True, False, False],
        )
        .groupby("dataset", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )

    best_configs_by_dataset_selector = (
        cv_summary.sort_values(
            ["dataset", "selector", "cv_macro_f1_mean", "cv_accuracy_mean"],
            ascending=[True, True, False, False],
        )
        .groupby(["dataset", "selector"], as_index=False)
        .head(1)
        .reset_index(drop=True)
    )

    label_mappings = pd.concat(all_label_mappings, ignore_index=True)

    cv_results.to_csv(out_dir / "cv_results.csv", index=False)
    cv_summary.to_csv(out_dir / "cv_summary.csv", index=False)
    test_results.to_csv(out_dir / "test_results.csv", index=False)
    best_configs_by_dataset.to_csv(out_dir / "best_configs_by_dataset.csv", index=False)
    best_configs_by_dataset_selector.to_csv(
        out_dir / "best_configs_by_dataset_selector.csv",
        index=False,
    )
    label_mappings.to_csv(out_dir / "label_mappings.csv", index=False)

    metadata = {
        "experiment_name": "wrapper_rfe",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "random_state": args.random_state,
        "datasets": args.datasets,
        "selectors": args.selectors,
        "k_values": args.k_values,
        "models": args.models,
        "prefilter_top_m": args.prefilter_top_m,
        "notes": [
            "This experiment evaluates wrapper feature selection using RFE.",
            "ANOVA prefiltering is applied before RFE for computational feasibility.",
            "Prefiltering, scaling, and RFE are fitted only on the training portion of each split.",
            "CV feature selection is fitted on CV train and evaluated on CV validation.",
            "Final test feature selection is fitted on train_val and evaluated once on test.",
            "Test labels are not used for selecting features, models, or k values.",
        ],
    }

    with open(out_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("Wrapper RFE experiment completed.")
    print(f"Saved results to: {out_dir}")


if __name__ == "__main__":
    main()
