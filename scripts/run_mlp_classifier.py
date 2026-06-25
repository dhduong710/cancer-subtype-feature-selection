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
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import RFE, f_classif, mutual_info_classif
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
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


DATASETS = [
    "mogcn_brca",
    "cancersd_stad",
    "mogonet_brca_optional",
]

FAMILY_TO_DIR = {
    "Filter": "filter_feature_selection",
    "Wrapper": "wrapper_rfe",
    "Embedded": "embedded_feature_selection",
}

DEFAULT_FAMILIES = [
    "Baseline",
    "Filter",
    "Wrapper",
    "Embedded",
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


def top_ranked_indices(scores: np.ndarray, k: int) -> np.ndarray:
    scores = sanitize_scores(scores)
    k = min(int(k), len(scores))
    if k <= 0:
        raise ValueError(f"k must be positive. Got k={k}")
    return np.argsort(scores)[::-1][:k].astype(int)


def matrix_indices_from_ranked(ranked: np.ndarray) -> np.ndarray:
    return np.sort(ranked).astype(int)


def compute_filter_scores(
    selector: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int,
) -> np.ndarray:
    if selector == "variance_top_k":
        return sanitize_scores(np.var(X_train, axis=0))

    if selector == "anova_f":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores, _ = f_classif(X_train, y_train)
        return sanitize_scores(scores)

    if selector == "mutual_info":
        scores = mutual_info_classif(
            X_train,
            y_train,
            discrete_features=False,
            n_neighbors=3,
            random_state=random_state,
        )
        return sanitize_scores(scores)

    raise ValueError(f"Unknown filter selector: {selector}")


def make_rfe_estimator(selector: str, random_state: int):
    if selector == "rfe_logistic":
        return LogisticRegression(
            solver="lbfgs",
            class_weight="balanced",
            max_iter=3000,
            random_state=random_state,
        )

    if selector == "rfe_linear_svm":
        return LinearSVC(
            C=0.5,
            class_weight="balanced",
            max_iter=30000,
            dual="auto",
            random_state=random_state,
        )

    raise ValueError(f"Unknown wrapper selector: {selector}")


def coefficient_importance(estimator) -> np.ndarray:
    if not hasattr(estimator, "coef_"):
        raise ValueError("RFE estimator has no coef_ attribute")

    coef = np.asarray(estimator.coef_)

    if coef.ndim == 1:
        return np.abs(coef)

    return np.mean(np.abs(coef), axis=0)


def select_wrapper_rfe(
    selector: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    k: int,
    random_state: int,
    prefilter_top_m: int = 1000,
) -> Tuple[np.ndarray, pd.DataFrame]:
    n_original = X_train.shape[1]
    prefilter_top_m = min(int(prefilter_top_m), n_original)
    k = min(int(k), prefilter_top_m)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        anova_scores, _ = f_classif(X_scaled, y_train)

    anova_scores = sanitize_scores(anova_scores)
    prefilter_ranked = top_ranked_indices(anova_scores, prefilter_top_m)
    X_prefiltered = X_scaled[:, prefilter_ranked]

    estimator = make_rfe_estimator(selector, random_state=random_state)

    rfe = RFE(
        estimator=estimator,
        n_features_to_select=k,
        step=0.20,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rfe.fit(X_prefiltered, y_train)

    selected_positions = np.where(rfe.support_)[0]
    selected_original_indices = prefilter_ranked[selected_positions].astype(int)

    selected_importance = coefficient_importance(rfe.estimator_)

    info = pd.DataFrame(
        {
            "feature_index": selected_original_indices,
            "prefilter_anova_score": anova_scores[selected_original_indices],
            "rfe_importance": selected_importance,
        }
    )

    info = info.sort_values(
        ["rfe_importance", "prefilter_anova_score"],
        ascending=[False, False],
    ).reset_index(drop=True)
    info.insert(0, "rank", np.arange(1, len(info) + 1))

    ranked_indices = info["feature_index"].to_numpy(dtype=int)

    return matrix_indices_from_ranked(ranked_indices), info


def compute_l1_logistic_scores(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int,
) -> np.ndarray:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    estimator = OneVsRestClassifier(
        LogisticRegression(
            penalty="l1",
            solver="liblinear",
            class_weight="balanced",
            C=1.0,
            max_iter=5000,
            random_state=random_state,
        ),
        n_jobs=-1,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        estimator.fit(X_scaled, y_train)

    coefs = []
    for binary_estimator in estimator.estimators_:
        coef = np.asarray(binary_estimator.coef_).reshape(-1)
        coefs.append(coef)

    coef_matrix = np.vstack(coefs)
    scores = np.mean(np.abs(coef_matrix), axis=0)

    return sanitize_scores(scores)


def compute_tree_importance_scores(
    selector: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int,
) -> np.ndarray:
    if selector == "random_forest_importance":
        estimator = RandomForestClassifier(
            n_estimators=300,
            criterion="gini",
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
    elif selector == "extra_trees_importance":
        estimator = ExtraTreesClassifier(
            n_estimators=300,
            criterion="gini",
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown embedded selector: {selector}")

    estimator.fit(X_train, y_train)
    return sanitize_scores(estimator.feature_importances_)


def compute_embedded_scores(
    selector: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int,
) -> np.ndarray:
    if selector == "l1_logistic":
        return compute_l1_logistic_scores(
            X_train=X_train,
            y_train=y_train,
            random_state=random_state,
        )

    if selector in {"random_forest_importance", "extra_trees_importance"}:
        return compute_tree_importance_scores(
            selector=selector,
            X_train=X_train,
            y_train=y_train,
            random_state=random_state,
        )

    raise ValueError(f"Unknown embedded selector: {selector}")


def select_features_for_config(
    method_family: str,
    selector: str,
    k: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int,
) -> Tuple[np.ndarray, pd.DataFrame, float]:
    start = time.time()
    n_features = X_train.shape[1]

    if method_family == "Baseline" or selector == "none":
        selected_indices = np.arange(n_features, dtype=int)
        info = pd.DataFrame(
            {
                "rank": np.arange(1, n_features + 1),
                "feature_index": selected_indices,
                "score": np.ones(n_features),
            }
        )
        elapsed = time.time() - start
        return selected_indices, info, float(elapsed)

    if method_family == "Filter":
        scores = compute_filter_scores(
            selector=selector,
            X_train=X_train,
            y_train=y_train,
            random_state=random_state,
        )
        ranked = top_ranked_indices(scores, k)
        selected_indices = matrix_indices_from_ranked(ranked)

        info = pd.DataFrame(
            {
                "rank": np.arange(1, len(ranked) + 1),
                "feature_index": ranked,
                "score": scores[ranked],
            }
        )

        elapsed = time.time() - start
        return selected_indices, info, float(elapsed)

    if method_family == "Wrapper":
        selected_indices, info = select_wrapper_rfe(
            selector=selector,
            X_train=X_train,
            y_train=y_train,
            k=k,
            random_state=random_state,
            prefilter_top_m=1000,
        )
        elapsed = time.time() - start
        return selected_indices, info, float(elapsed)

    if method_family == "Embedded":
        scores = compute_embedded_scores(
            selector=selector,
            X_train=X_train,
            y_train=y_train,
            random_state=random_state,
        )
        ranked = top_ranked_indices(scores, k)
        selected_indices = matrix_indices_from_ranked(ranked)

        info = pd.DataFrame(
            {
                "rank": np.arange(1, len(ranked) + 1),
                "feature_index": ranked,
                "score": scores[ranked],
            }
        )

        elapsed = time.time() - start
        return selected_indices, info, float(elapsed)

    raise ValueError(f"Unknown method family: {method_family}")


def build_mlp_classifier(
    hidden_layer_sizes: Tuple[int, ...],
    alpha: float,
    max_iter: int,
    random_state: int,
) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                MLPClassifier(
                    hidden_layer_sizes=hidden_layer_sizes,
                    activation="relu",
                    solver="adam",
                    alpha=alpha,
                    batch_size="auto",
                    learning_rate_init=0.001,
                    max_iter=max_iter,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=20,
                    random_state=random_state,
                    verbose=False,
                ),
            ),
        ]
    )


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


def make_config_id(dataset: str, method_family: str, selector: str, k: int) -> str:
    return f"{dataset}__{method_family.lower()}__{selector}__k{k}"


def build_config_plan(repo_root: Path, datasets: List[str], families: List[str]) -> pd.DataFrame:
    rows = []

    for dataset in datasets:
        X, _ = load_dataset_matrix(repo_root, dataset)
        n_features = len(get_feature_names(X))

        if "Baseline" in families:
            rows.append(
                {
                    "dataset": dataset,
                    "method_family": "Baseline",
                    "selector": "none",
                    "k": n_features,
                    "source_model": "baseline",
                    "source_cv_macro_f1_mean": np.nan,
                    "source_cv_accuracy_mean": np.nan,
                    "n_original_features": n_features,
                }
            )

        for family in ["Filter", "Wrapper", "Embedded"]:
            if family not in families:
                continue

            exp_dir = repo_root / "results" / "experiments" / FAMILY_TO_DIR[family]
            best_path = exp_dir / "best_configs_by_dataset.csv"
            best_df = read_csv_required(best_path)

            matched = best_df[best_df["dataset"] == dataset]
            if matched.empty:
                raise ValueError(f"No best config found for {dataset} in {best_path}")

            row = matched.iloc[0]

            rows.append(
                {
                    "dataset": dataset,
                    "method_family": family,
                    "selector": row["selector"],
                    "k": int(row["k"]),
                    "source_model": row["model"],
                    "source_cv_macro_f1_mean": float(row["cv_macro_f1_mean"]),
                    "source_cv_accuracy_mean": float(row["cv_accuracy_mean"]),
                    "n_original_features": n_features,
                }
            )

    plan = pd.DataFrame(rows)
    plan["config_id"] = plan.apply(
        lambda row: make_config_id(
            row["dataset"],
            row["method_family"],
            row["selector"],
            int(row["k"]),
        ),
        axis=1,
    )

    return plan[
        [
            "config_id",
            "dataset",
            "method_family",
            "selector",
            "k",
            "source_model",
            "source_cv_macro_f1_mean",
            "source_cv_accuracy_mean",
            "n_original_features",
        ]
    ]


def save_selected_features(
    out_dir: Path,
    config_id: str,
    dataset: str,
    method_family: str,
    selector: str,
    k: int,
    selected_info: pd.DataFrame,
    feature_names: List[str],
) -> None:
    if method_family == "Baseline":
        return

    selected_dir = out_dir / "selected_features"
    selected_dir.mkdir(parents=True, exist_ok=True)

    df = selected_info.copy()
    df["feature_name"] = df["feature_index"].apply(lambda idx: feature_names[int(idx)])

    df.insert(0, "config_id", config_id)
    df.insert(1, "dataset", dataset)
    df.insert(2, "method_family", method_family)
    df.insert(3, "selector", selector)
    df.insert(4, "k", int(k))

    ordered_cols = [
        "config_id",
        "dataset",
        "method_family",
        "selector",
        "k",
        "rank",
        "feature_index",
        "feature_name",
    ]

    for col in ["score", "rfe_importance", "prefilter_anova_score"]:
        if col in df.columns:
            ordered_cols.append(col)

    df = df[ordered_cols]

    df.to_csv(selected_dir / f"{config_id}.csv", index=False)


def run_cv_for_config(
    config: pd.Series,
    X: pd.DataFrame,
    y: pd.DataFrame,
    cv_folds: pd.DataFrame,
    hidden_layer_sizes: Tuple[int, ...],
    alpha: float,
    max_iter: int,
    random_state: int,
) -> List[Dict[str, object]]:
    rows = []

    dataset = config["dataset"]
    method_family = config["method_family"]
    selector = config["selector"]
    k = int(config["k"])
    config_id = config["config_id"]

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

        selected_indices, _, fs_elapsed = select_features_for_config(
            method_family=method_family,
            selector=selector,
            k=k,
            X_train=X_train,
            y_train=y_train,
            random_state=random_state,
        )

        X_train_selected = X_train[:, selected_indices]
        X_val_selected = X_val[:, selected_indices]

        model = build_mlp_classifier(
            hidden_layer_sizes=hidden_layer_sizes,
            alpha=alpha,
            max_iter=max_iter,
            random_state=random_state,
        )

        fit_start = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train_selected, y_train)
        y_pred = model.predict(X_val_selected)
        fit_elapsed = time.time() - fit_start

        metrics = compute_metrics(y_val, y_pred)

        classifier = model.named_steps["classifier"]
        n_iter = getattr(classifier, "n_iter_", None)
        best_validation_score = getattr(classifier, "best_validation_score_", None)

        row = {
            "config_id": config_id,
            "dataset": dataset,
            "method_family": method_family,
            "selector": selector,
            "k": k,
            "model": "mlp",
            "fold": int(fold),
            "n_train": int(len(y_train)),
            "n_val": int(len(y_val)),
            "n_original_features": int(X_train.shape[1]),
            "n_selected_features": int(len(selected_indices)),
            "hidden_layer_sizes": str(hidden_layer_sizes),
            "alpha": float(alpha),
            "max_iter": int(max_iter),
            "mlp_n_iter": int(n_iter) if n_iter is not None else None,
            "mlp_best_internal_validation_score": (
                float(best_validation_score)
                if best_validation_score is not None
                else None
            ),
            "feature_selection_seconds": round(float(fs_elapsed), 4),
            "model_runtime_seconds": round(float(fit_elapsed), 4),
            "total_runtime_seconds": round(float(fs_elapsed + fit_elapsed), 4),
        }
        row.update(metrics)
        rows.append(row)

    return rows


def run_final_test_for_config(
    config: pd.Series,
    X: pd.DataFrame,
    y: pd.DataFrame,
    train_val_ids: List[str],
    test_ids: List[str],
    label_ids_sorted: List[int],
    feature_names: List[str],
    hidden_layer_sizes: Tuple[int, ...],
    alpha: float,
    max_iter: int,
    random_state: int,
    out_dir: Path,
) -> Dict[str, object]:
    dataset = config["dataset"]
    method_family = config["method_family"]
    selector = config["selector"]
    k = int(config["k"])
    config_id = config["config_id"]

    X_train_val, y_train_val, _ = align_by_sample_ids(X, y, train_val_ids)
    X_test, y_test, _ = align_by_sample_ids(X, y, test_ids)

    selected_indices, selected_info, fs_elapsed = select_features_for_config(
        method_family=method_family,
        selector=selector,
        k=k,
        X_train=X_train_val,
        y_train=y_train_val,
        random_state=random_state,
    )

    save_selected_features(
        out_dir=out_dir,
        config_id=config_id,
        dataset=dataset,
        method_family=method_family,
        selector=selector,
        k=k,
        selected_info=selected_info,
        feature_names=feature_names,
    )

    X_train_val_selected = X_train_val[:, selected_indices]
    X_test_selected = X_test[:, selected_indices]

    model = build_mlp_classifier(
        hidden_layer_sizes=hidden_layer_sizes,
        alpha=alpha,
        max_iter=max_iter,
        random_state=random_state,
    )

    fit_start = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train_val_selected, y_train_val)
    y_pred = model.predict(X_test_selected)
    fit_elapsed = time.time() - fit_start

    metrics = compute_metrics(y_test, y_pred)

    label_mapping = get_label_mapping(y)
    id_to_label = {
        int(row["label_id"]): str(row["label"])
        for _, row in label_mapping.iterrows()
    }
    labels_for_cm = [id_to_label[i] for i in label_ids_sorted]

    cm = confusion_matrix(y_test, y_pred, labels=label_ids_sorted)

    cm_dir = out_dir / "confusion_matrices"
    cm_dir.mkdir(parents=True, exist_ok=True)

    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{label}" for label in labels_for_cm],
        columns=[f"pred_{label}" for label in labels_for_cm],
    )
    cm_df.to_csv(cm_dir / f"{config_id}__mlp.csv")

    classifier = model.named_steps["classifier"]
    n_iter = getattr(classifier, "n_iter_", None)
    best_validation_score = getattr(classifier, "best_validation_score_", None)

    row = {
        "config_id": config_id,
        "dataset": dataset,
        "method_family": method_family,
        "selector": selector,
        "k": k,
        "model": "mlp",
        "n_train_val": int(len(y_train_val)),
        "n_test": int(len(y_test)),
        "n_original_features": int(X_train_val.shape[1]),
        "n_selected_features": int(len(selected_indices)),
        "hidden_layer_sizes": str(hidden_layer_sizes),
        "alpha": float(alpha),
        "max_iter": int(max_iter),
        "mlp_n_iter": int(n_iter) if n_iter is not None else None,
        "mlp_best_internal_validation_score": (
            float(best_validation_score)
            if best_validation_score is not None
            else None
        ),
        "feature_selection_seconds": round(float(fs_elapsed), 4),
        "model_runtime_seconds": round(float(fit_elapsed), 4),
        "total_runtime_seconds": round(float(fs_elapsed + fit_elapsed), 4),
    }
    row.update(metrics)

    return row


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
        "mlp_n_iter",
        "mlp_best_internal_validation_score",
    ]

    available_metric_cols = [col for col in metric_cols if col in cv_results.columns]

    summary = (
        cv_results.groupby(
            [
                "config_id",
                "dataset",
                "method_family",
                "selector",
                "k",
                "model",
                "n_original_features",
                "n_selected_features",
                "hidden_layer_sizes",
                "alpha",
                "max_iter",
            ]
        )[available_metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )

    summary.columns = [
        "_".join(col).rstrip("_") if isinstance(col, tuple) else col
        for col in summary.columns
    ]

    rename_map = {}
    for metric in available_metric_cols:
        rename_map[f"{metric}_mean"] = f"cv_{metric}_mean"
        rename_map[f"{metric}_std"] = f"cv_{metric}_std"

    summary = summary.rename(columns=rename_map)

    return summary.sort_values(
        ["dataset", "cv_macro_f1_mean", "cv_accuracy_mean"],
        ascending=[True, False, False],
    ).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=DATASETS)
    parser.add_argument("--families", nargs="*", default=DEFAULT_FAMILIES)
    parser.add_argument("--hidden-layer-sizes", nargs="*", type=int, default=[64])
    parser.add_argument("--alpha", type=float, default=0.001)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    hidden_layer_sizes = tuple(args.hidden_layer_sizes)

    repo_root = find_repo_root()
    out_dir = repo_root / "results" / "experiments" / "mlp_classifier"
    out_dir.mkdir(parents=True, exist_ok=True)

    config_plan = build_config_plan(
        repo_root=repo_root,
        datasets=args.datasets,
        families=args.families,
    )
    config_plan.to_csv(out_dir / "config_plan.csv", index=False)

    all_cv_rows: List[Dict[str, object]] = []
    all_test_rows: List[Dict[str, object]] = []
    all_label_mappings: List[pd.DataFrame] = []

    print("Running MLP Neural Network classifier experiments.")
    print(f"Datasets: {args.datasets}")
    print(f"Families: {args.families}")
    print(f"Hidden layer sizes: {hidden_layer_sizes}")
    print(f"Alpha: {args.alpha}")
    print(f"Max iter: {args.max_iter}")
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

        dataset_configs = config_plan[config_plan["dataset"] == dataset]

        print(
            f"  samples={len(y)}, features={len(feature_names)}, "
            f"train_val={len(train_val_ids)}, test={len(test_ids)}, "
            f"configs={len(dataset_configs)}"
        )

        for _, config in dataset_configs.iterrows():
            print(
                f"  MLP config: {config['method_family']} / "
                f"{config['selector']} / k={int(config['k'])}"
            )

            cv_rows = run_cv_for_config(
                config=config,
                X=X,
                y=y,
                cv_folds=cv_folds,
                hidden_layer_sizes=hidden_layer_sizes,
                alpha=args.alpha,
                max_iter=args.max_iter,
                random_state=args.random_state,
            )
            all_cv_rows.extend(cv_rows)

            test_row = run_final_test_for_config(
                config=config,
                X=X,
                y=y,
                train_val_ids=train_val_ids,
                test_ids=test_ids,
                label_ids_sorted=label_ids_sorted,
                feature_names=feature_names,
                hidden_layer_sizes=hidden_layer_sizes,
                alpha=args.alpha,
                max_iter=args.max_iter,
                random_state=args.random_state,
                out_dir=out_dir,
            )
            all_test_rows.append(test_row)

        print("")

    cv_results = pd.DataFrame(all_cv_rows)
    cv_summary = summarize_cv(cv_results)
    test_results = pd.DataFrame(all_test_rows)

    merge_cols = [
        "config_id",
        "dataset",
        "method_family",
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

    test_results = test_results.merge(
        cv_summary[merge_cols],
        on=["config_id", "dataset", "method_family", "selector", "k", "model"],
        how="left",
    )

    best_mlp_by_dataset = (
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
    best_mlp_by_dataset.to_csv(out_dir / "best_mlp_by_dataset.csv", index=False)
    label_mappings.to_csv(out_dir / "label_mappings.csv", index=False)

    metadata = {
        "experiment_name": "mlp_classifier",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "random_state": args.random_state,
        "datasets": args.datasets,
        "families": args.families,
        "hidden_layer_sizes": hidden_layer_sizes,
        "alpha": args.alpha,
        "max_iter": args.max_iter,
        "notes": [
            "This experiment evaluates MLP Neural Network as an additional classifier.",
            "MLP is evaluated on no-feature-selection baseline and on the best Filter, Wrapper, and Embedded configurations from previous CV results.",
            "Feature selection is refitted inside each CV training fold before MLP training.",
            "For final test, feature selection is fitted on train_val and evaluated once on the held-out test set.",
            "Test labels are not used for selecting feature selection method, k, or model.",
        ],
    }

    with open(out_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("MLP classifier experiment completed.")
    print(f"Saved results to: {out_dir}")


if __name__ == "__main__":
    main()
