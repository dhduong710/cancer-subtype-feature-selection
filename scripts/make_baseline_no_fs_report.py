#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd


DATASET_DISPLAY_NAMES = {
    "mogcn_brca": "MoGCN BRCA",
    "cancersd_stad": "CancerSD STAD",
    "mogonet_brca_optional": "MOGONET BRCA Optional",
}

MODEL_DISPLAY_NAMES = {
    "logistic_regression": "One-vs-Rest Logistic Regression",
    "linear_svm": "Linear SVM",
    "random_forest": "Random Forest",
    "gaussian_nb": "Gaussian Naive Bayes",
}


def find_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def read_json_required(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No data available._"

    df = df.copy().fillna("")
    headers = list(df.columns)

    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for _, row in df.iterrows():
        values = [str(row[col]) for col in headers]
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def round_float_columns(df: pd.DataFrame, digits: int = 4) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].round(digits)
    return df


def add_display_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "dataset" in df.columns:
        df["Dataset"] = df["dataset"].map(DATASET_DISPLAY_NAMES).fillna(df["dataset"])
    if "model" in df.columns:
        df["Model"] = df["model"].map(MODEL_DISPLAY_NAMES).fillna(df["model"])
    return df


def compact_cv_summary(cv_summary: pd.DataFrame) -> pd.DataFrame:
    df = add_display_columns(cv_summary)
    keep_cols = [
        "Dataset",
        "Model",
        "cv_accuracy_mean",
        "cv_accuracy_std",
        "cv_balanced_accuracy_mean",
        "cv_macro_f1_mean",
        "cv_macro_f1_std",
        "cv_weighted_f1_mean",
        "cv_runtime_seconds_mean",
    ]
    df = df[keep_cols].sort_values(
        ["Dataset", "cv_macro_f1_mean", "cv_accuracy_mean"],
        ascending=[True, False, False],
    )
    df = df.rename(
        columns={
            "cv_accuracy_mean": "CV Accuracy Mean",
            "cv_accuracy_std": "CV Accuracy Std",
            "cv_balanced_accuracy_mean": "CV Balanced Accuracy Mean",
            "cv_macro_f1_mean": "CV Macro F1 Mean",
            "cv_macro_f1_std": "CV Macro F1 Std",
            "cv_weighted_f1_mean": "CV Weighted F1 Mean",
            "cv_runtime_seconds_mean": "CV Runtime Mean (s)",
        }
    )
    return round_float_columns(df)


def compact_best_models(best_models: pd.DataFrame) -> pd.DataFrame:
    df = add_display_columns(best_models)
    keep_cols = [
        "Dataset",
        "Model",
        "cv_accuracy_mean",
        "cv_accuracy_std",
        "cv_macro_f1_mean",
        "cv_macro_f1_std",
        "cv_weighted_f1_mean",
    ]
    df = df[keep_cols].sort_values("Dataset")
    df = df.rename(
        columns={
            "cv_accuracy_mean": "CV Accuracy Mean",
            "cv_accuracy_std": "CV Accuracy Std",
            "cv_macro_f1_mean": "CV Macro F1 Mean",
            "cv_macro_f1_std": "CV Macro F1 Std",
            "cv_weighted_f1_mean": "CV Weighted F1 Mean",
        }
    )
    return round_float_columns(df)


def compact_test_results(test_results: pd.DataFrame) -> pd.DataFrame:
    df = add_display_columns(test_results)
    keep_cols = [
        "Dataset",
        "Model",
        "n_train_val",
        "n_test",
        "n_features",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "cv_macro_f1_mean",
        "cv_macro_f1_std",
        "runtime_seconds",
    ]
    df = df[keep_cols].sort_values(["Dataset", "macro_f1"], ascending=[True, False])
    df = df.rename(
        columns={
            "n_train_val": "Train+Validation Samples",
            "n_test": "Test Samples",
            "n_features": "Features",
            "accuracy": "Test Accuracy",
            "balanced_accuracy": "Test Balanced Accuracy",
            "macro_f1": "Test Macro F1",
            "weighted_f1": "Test Weighted F1",
            "cv_macro_f1_mean": "CV Macro F1 Mean",
            "cv_macro_f1_std": "CV Macro F1 Std",
            "runtime_seconds": "Test Runtime (s)",
        }
    )
    return round_float_columns(df)


def per_dataset_best_test(test_results: pd.DataFrame) -> pd.DataFrame:
    df = add_display_columns(test_results)
    df = df.sort_values(["dataset", "macro_f1"], ascending=[True, False])
    df = df.groupby("dataset", as_index=False).head(1)
    keep_cols = [
        "Dataset",
        "Model",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
    ]
    df = df[keep_cols].sort_values("Dataset")
    df = df.rename(
        columns={
            "accuracy": "Test Accuracy",
            "balanced_accuracy": "Test Balanced Accuracy",
            "macro_f1": "Test Macro F1",
            "weighted_f1": "Test Weighted F1",
        }
    )
    return round_float_columns(df)


def model_description_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Model": "One-vs-Rest Logistic Regression",
                "Pipeline": "StandardScaler + OneVsRestClassifier(LogisticRegression)",
                "Reason": "Linear baseline suitable for high-dimensional gene expression data.",
            },
            {
                "Model": "Linear SVM",
                "Pipeline": "StandardScaler + LinearSVC",
                "Reason": "Strong linear classifier for high-dimensional classification problems.",
            },
            {
                "Model": "Random Forest",
                "Pipeline": "RandomForestClassifier",
                "Reason": "Tree-based ensemble baseline and useful reference before feature importance experiments.",
            },
            {
                "Model": "Gaussian Naive Bayes",
                "Pipeline": "StandardScaler + GaussianNB",
                "Reason": "Simple probabilistic baseline for multiclass classification.",
            },
        ]
    )


def dataset_feature_table(test_results: pd.DataFrame) -> pd.DataFrame:
    df = add_display_columns(test_results)
    df = (
        df[["Dataset", "n_train_val", "n_test", "n_features"]]
        .drop_duplicates()
        .sort_values("Dataset")
        .rename(
            columns={
                "n_train_val": "Train+Validation Samples",
                "n_test": "Test Samples",
                "n_features": "Features Used",
            }
        )
    )
    return df


def write_report(repo_root: Path) -> Path:
    result_dir = repo_root / "results" / "experiments" / "baseline_no_fs"
    report_dir = repo_root / "reports" / "baseline_no_fs"
    report_dir.mkdir(parents=True, exist_ok=True)

    metadata = read_json_required(result_dir / "run_metadata.json")
    cv_results = read_csv_required(result_dir / "cv_results.csv")
    cv_summary = read_csv_required(result_dir / "cv_summary.csv")
    test_results = read_csv_required(result_dir / "test_results.csv")
    best_models = read_csv_required(result_dir / "best_models_by_cv.csv")
    label_mappings = read_csv_required(result_dir / "label_mappings.csv")

    lines: List[str] = []

    lines.append("# Baseline Classification Without Feature Selection")
    lines.append("")
    lines.append(f"Generated at: `{datetime.now().isoformat(timespec='seconds')}`")
    lines.append("")
    lines.append("## 1. Purpose")
    lines.append("")
    lines.append(
        "This report documents the baseline classification experiment without feature "
        "selection for the project **A comparative study of feature selection method for "
        "cancer classification using gene expression data**."
    )
    lines.append("")
    lines.append(
        "The purpose of this step is to establish a reference performance level using "
        "all available gene expression features. Later feature selection methods will be "
        "compared against this baseline to evaluate whether they can preserve or improve "
        "classification performance while using fewer genes."
    )
    lines.append("")

    lines.append("## 2. Experimental protocol")
    lines.append("")
    lines.append(
        "For each dataset, 5-fold stratified cross-validation was performed on the "
        "`train_val` subset. The held-out test set was not used during cross-validation."
    )
    lines.append("")
    lines.append(
        "After cross-validation, each baseline classifier was trained once on the full "
        "`train_val` subset and evaluated on the held-out test set. The test results are "
        "reported for analysis, but model selection should be based on cross-validation "
        "performance, especially CV Macro F1."
    )
    lines.append("")
    lines.append(
        "No feature selection was applied in this experiment. Therefore, each model used "
        "the full feature set of the corresponding dataset."
    )
    lines.append("")
    lines.append("**Leakage prevention:**")
    lines.append("")
    lines.append(
        "Standardization was placed inside an `sklearn` Pipeline, so the scaler was fitted "
        "only on the training portion of each CV fold and then applied to the validation "
        "portion. For final test evaluation, the scaler was fitted only on `train_val` and "
        "then applied to the test set."
    )
    lines.append("")

    lines.append("## 3. Datasets and number of features")
    lines.append("")
    lines.append(markdown_table(dataset_feature_table(test_results)))
    lines.append("")

    lines.append("## 4. Baseline models")
    lines.append("")
    lines.append(markdown_table(model_description_table()))
    lines.append("")

    lines.append("## 5. Cross-validation results")
    lines.append("")
    lines.append(
        "The following table summarizes the 5-fold cross-validation results. The main "
        "selection metric is **CV Macro F1**, because the datasets are multiclass and "
        "some classes are smaller than others."
    )
    lines.append("")
    lines.append(markdown_table(compact_cv_summary(cv_summary)))
    lines.append("")

    lines.append("## 6. Best baseline model by cross-validation")
    lines.append("")
    lines.append(
        "For each dataset, the best baseline model was selected according to the highest "
        "mean CV Macro F1. If two models were close, CV Accuracy was used as an additional "
        "reference."
    )
    lines.append("")
    lines.append(markdown_table(compact_best_models(best_models)))
    lines.append("")

    lines.append("## 7. Final held-out test results")
    lines.append("")
    lines.append(
        "The table below shows the final test performance after training each model on "
        "`train_val` and evaluating it once on the held-out test set."
    )
    lines.append("")
    lines.append(markdown_table(compact_test_results(test_results)))
    lines.append("")

    lines.append("## 8. Best test result per dataset")
    lines.append("")
    lines.append(
        "This table is descriptive only. It should not be used to choose models, because "
        "model selection should be based on cross-validation rather than test performance."
    )
    lines.append("")
    lines.append(markdown_table(per_dataset_best_test(test_results)))
    lines.append("")

    lines.append("## 9. Main observations")
    lines.append("")
    lines.append(
        "- Random Forest achieved the best mean CV Macro F1 on all three datasets, making "
        "it the strongest no-feature-selection baseline according to the predefined "
        "cross-validation selection protocol."
    )
    lines.append(
        "- On MoGCN BRCA and CancerSD STAD, Random Forest also achieved the best held-out "
        "test Macro F1 among the evaluated baseline models."
    )
    lines.append(
        "- On MOGONET BRCA Optional, Gaussian Naive Bayes achieved the highest held-out "
        "test Macro F1, but Random Forest remained the best model according to CV Macro F1. "
        "This difference should be reported carefully because the test set should not drive "
        "model selection."
    )
    lines.append(
        "- Gaussian Naive Bayes performed weakly on MoGCN BRCA but strongly on MOGONET BRCA "
        "Optional, suggesting that the already feature-reduced MOGONET representation may be "
        "more compatible with simple probabilistic assumptions than the raw high-dimensional "
        "MoGCN BRCA gene expression matrix."
    )
    lines.append(
        "- These results provide the control baseline for later feature selection experiments. "
        "A feature selection method will be considered useful if it can maintain or improve "
        "Macro F1 while reducing the number of selected genes."
    )
    lines.append("")

    lines.append("## 10. Generated result files")
    lines.append("")
    lines.append("The baseline experiment generated the following files:")
    lines.append("")
    lines.append("- `results/experiments/baseline_no_fs/cv_results.csv`")
    lines.append("- `results/experiments/baseline_no_fs/cv_summary.csv`")
    lines.append("- `results/experiments/baseline_no_fs/test_results.csv`")
    lines.append("- `results/experiments/baseline_no_fs/best_models_by_cv.csv`")
    lines.append("- `results/experiments/baseline_no_fs/label_mappings.csv`")
    lines.append("- `results/experiments/baseline_no_fs/run_metadata.json`")
    lines.append("- `results/experiments/baseline_no_fs/confusion_matrices/*.csv`")
    lines.append("")

    lines.append("## 11. Reproducibility metadata")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(metadata, indent=2, ensure_ascii=False))
    lines.append("```")
    lines.append("")

    lines.append("## 12. Label mappings")
    lines.append("")
    label_mappings_display = label_mappings.copy()
    label_mappings_display["Dataset"] = (
        label_mappings_display["dataset"]
        .map(DATASET_DISPLAY_NAMES)
        .fillna(label_mappings_display["dataset"])
    )
    label_mappings_display = label_mappings_display[
        ["Dataset", "label_id", "label"]
    ].rename(
        columns={
            "label_id": "Label ID",
            "label": "Class",
        }
    )
    lines.append(markdown_table(label_mappings_display))
    lines.append("")

    report_path = report_dir / "baseline_no_fs_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    return report_path


def main() -> None:
    repo_root = find_repo_root()
    report_path = write_report(repo_root)

    print("Baseline no-feature-selection report created successfully.")
    print(f"Report path: {report_path}")


if __name__ == "__main__":
    main()
