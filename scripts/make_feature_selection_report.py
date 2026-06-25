#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

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

SELECTOR_DISPLAY_NAMES = {
    "none": "No feature selection",
    "variance_top_k": "Variance Top-K",
    "anova_f": "ANOVA F-test",
    "mutual_info": "Mutual Information",
    "rfe_logistic": "RFE Logistic",
    "rfe_linear_svm": "RFE Linear SVM",
    "l1_logistic": "L1 Logistic",
    "random_forest_importance": "Random Forest Importance",
    "extra_trees_importance": "Extra Trees Importance",
}

EXPERIMENTS = [
    {
        "family": "Filter",
        "directory": "filter_feature_selection",
    },
    {
        "family": "Wrapper",
        "directory": "wrapper_rfe",
    },
    {
        "family": "Embedded",
        "directory": "embedded_feature_selection",
    },
]


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

    if "selector" in df.columns:
        df["Selector"] = df["selector"].map(SELECTOR_DISPLAY_NAMES).fillna(df["selector"])

    return df


def load_baseline_best(repo_root: Path) -> pd.DataFrame:
    baseline_dir = repo_root / "results" / "experiments" / "baseline_no_fs"

    best_cv = read_csv_required(baseline_dir / "best_models_by_cv.csv")
    test_results = read_csv_required(baseline_dir / "test_results.csv")

    rows = []

    for _, row in best_cv.iterrows():
        dataset = row["dataset"]
        model = row["model"]

        matched_test = test_results[
            (test_results["dataset"] == dataset)
            & (test_results["model"] == model)
        ]

        if matched_test.empty:
            raise ValueError(f"Missing baseline test result for {dataset}/{model}")

        test_row = matched_test.iloc[0]

        n_features = int(test_row["n_features"])

        rows.append(
            {
                "dataset": dataset,
                "method_family": "Baseline",
                "selector": "none",
                "k": n_features,
                "model": model,
                "n_original_features": n_features,
                "n_selected_features": n_features,
                "feature_reduction_pct": 0.0,
                "cv_accuracy_mean": float(row["cv_accuracy_mean"]),
                "cv_accuracy_std": float(row["cv_accuracy_std"]),
                "cv_macro_f1_mean": float(row["cv_macro_f1_mean"]),
                "cv_macro_f1_std": float(row["cv_macro_f1_std"]),
                "cv_weighted_f1_mean": float(row["cv_weighted_f1_mean"]),
                "cv_weighted_f1_std": float(row["cv_weighted_f1_std"]),
                "test_accuracy": float(test_row["accuracy"]),
                "test_balanced_accuracy": float(test_row["balanced_accuracy"]),
                "test_macro_f1": float(test_row["macro_f1"]),
                "test_weighted_f1": float(test_row["weighted_f1"]),
            }
        )

    return pd.DataFrame(rows)


def load_family_best(repo_root: Path) -> pd.DataFrame:
    rows = []

    for exp in EXPERIMENTS:
        family = exp["family"]
        exp_dir = repo_root / "results" / "experiments" / exp["directory"]

        best_cv = read_csv_required(exp_dir / "best_configs_by_dataset.csv")
        test_results = read_csv_required(exp_dir / "test_results.csv")

        for _, row in best_cv.iterrows():
            dataset = row["dataset"]
            selector = row["selector"]
            k = int(row["k"])
            model = row["model"]

            matched_test = test_results[
                (test_results["dataset"] == dataset)
                & (test_results["selector"] == selector)
                & (test_results["k"].astype(int) == k)
                & (test_results["model"] == model)
            ]

            if matched_test.empty:
                raise ValueError(
                    f"Missing test result for {family}: "
                    f"{dataset}/{selector}/k={k}/{model}"
                )

            test_row = matched_test.iloc[0]

            n_original = int(test_row["n_original_features"])
            n_selected = int(test_row["n_selected_features"])
            reduction = 100.0 * (1.0 - n_selected / n_original)

            rows.append(
                {
                    "dataset": dataset,
                    "method_family": family,
                    "selector": selector,
                    "k": k,
                    "model": model,
                    "n_original_features": n_original,
                    "n_selected_features": n_selected,
                    "feature_reduction_pct": reduction,
                    "cv_accuracy_mean": float(row["cv_accuracy_mean"]),
                    "cv_accuracy_std": float(row["cv_accuracy_std"]),
                    "cv_macro_f1_mean": float(row["cv_macro_f1_mean"]),
                    "cv_macro_f1_std": float(row["cv_macro_f1_std"]),
                    "cv_weighted_f1_mean": float(row["cv_weighted_f1_mean"]),
                    "cv_weighted_f1_std": float(row["cv_weighted_f1_std"]),
                    "test_accuracy": float(test_row["accuracy"]),
                    "test_balanced_accuracy": float(test_row["balanced_accuracy"]),
                    "test_macro_f1": float(test_row["macro_f1"]),
                    "test_weighted_f1": float(test_row["weighted_f1"]),
                }
            )

    return pd.DataFrame(rows)


def load_best_by_dataset_selector(repo_root: Path) -> pd.DataFrame:
    rows = []

    for exp in EXPERIMENTS:
        family = exp["family"]
        exp_dir = repo_root / "results" / "experiments" / exp["directory"]

        best_cv = read_csv_required(exp_dir / "best_configs_by_dataset_selector.csv")
        test_results = read_csv_required(exp_dir / "test_results.csv")

        for _, row in best_cv.iterrows():
            dataset = row["dataset"]
            selector = row["selector"]
            k = int(row["k"])
            model = row["model"]

            matched_test = test_results[
                (test_results["dataset"] == dataset)
                & (test_results["selector"] == selector)
                & (test_results["k"].astype(int) == k)
                & (test_results["model"] == model)
            ]

            if matched_test.empty:
                raise ValueError(
                    f"Missing selector-level test result for {family}: "
                    f"{dataset}/{selector}/k={k}/{model}"
                )

            test_row = matched_test.iloc[0]

            n_original = int(test_row["n_original_features"])
            n_selected = int(test_row["n_selected_features"])
            reduction = 100.0 * (1.0 - n_selected / n_original)

            rows.append(
                {
                    "dataset": dataset,
                    "method_family": family,
                    "selector": selector,
                    "k": k,
                    "model": model,
                    "n_original_features": n_original,
                    "n_selected_features": n_selected,
                    "feature_reduction_pct": reduction,
                    "cv_accuracy_mean": float(row["cv_accuracy_mean"]),
                    "cv_accuracy_std": float(row["cv_accuracy_std"]),
                    "cv_macro_f1_mean": float(row["cv_macro_f1_mean"]),
                    "cv_macro_f1_std": float(row["cv_macro_f1_std"]),
                    "cv_weighted_f1_mean": float(row["cv_weighted_f1_mean"]),
                    "cv_weighted_f1_std": float(row["cv_weighted_f1_std"]),
                    "test_accuracy": float(test_row["accuracy"]),
                    "test_balanced_accuracy": float(test_row["balanced_accuracy"]),
                    "test_macro_f1": float(test_row["macro_f1"]),
                    "test_weighted_f1": float(test_row["weighted_f1"]),
                }
            )

    return pd.DataFrame(rows)


def choose_best_feature_selection(family_best: pd.DataFrame) -> pd.DataFrame:
    return (
        family_best.sort_values(
            ["dataset", "cv_macro_f1_mean", "cv_accuracy_mean"],
            ascending=[True, False, False],
        )
        .groupby("dataset", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )


def compare_baseline_vs_best_fs(
    baseline_best: pd.DataFrame,
    best_fs: pd.DataFrame,
) -> pd.DataFrame:
    base = baseline_best.copy()
    fs = best_fs.copy()

    base = base.add_prefix("baseline_")
    fs = fs.add_prefix("fs_")

    merged = base.merge(
        fs,
        left_on="baseline_dataset",
        right_on="fs_dataset",
        how="inner",
    )

    rows = []

    for _, row in merged.iterrows():
        rows.append(
            {
                "dataset": row["baseline_dataset"],
                "baseline_model": row["baseline_model"],
                "baseline_features": int(row["baseline_n_selected_features"]),
                "baseline_cv_macro_f1": float(row["baseline_cv_macro_f1_mean"]),
                "baseline_test_macro_f1": float(row["baseline_test_macro_f1"]),
                "fs_family": row["fs_method_family"],
                "fs_selector": row["fs_selector"],
                "fs_k": int(row["fs_k"]),
                "fs_model": row["fs_model"],
                "fs_selected_features": int(row["fs_n_selected_features"]),
                "feature_reduction_pct": float(row["fs_feature_reduction_pct"]),
                "fs_cv_macro_f1": float(row["fs_cv_macro_f1_mean"]),
                "fs_test_macro_f1": float(row["fs_test_macro_f1"]),
                "delta_cv_macro_f1": float(
                    row["fs_cv_macro_f1_mean"] - row["baseline_cv_macro_f1_mean"]
                ),
                "delta_test_macro_f1": float(
                    row["fs_test_macro_f1"] - row["baseline_test_macro_f1"]
                ),
            }
        )

    return pd.DataFrame(rows)


def pretty_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    out = add_display_columns(df)

    keep_cols = [
        "Dataset",
        "method_family",
        "Selector",
        "k",
        "Model",
        "n_original_features",
        "n_selected_features",
        "feature_reduction_pct",
        "cv_accuracy_mean",
        "cv_macro_f1_mean",
        "cv_macro_f1_std",
        "test_accuracy",
        "test_macro_f1",
    ]

    out = out[keep_cols].rename(
        columns={
            "method_family": "Family",
            "k": "k",
            "n_original_features": "Original Features",
            "n_selected_features": "Selected Features",
            "feature_reduction_pct": "Reduction (%)",
            "cv_accuracy_mean": "CV Accuracy Mean",
            "cv_macro_f1_mean": "CV Macro F1 Mean",
            "cv_macro_f1_std": "CV Macro F1 Std",
            "test_accuracy": "Test Accuracy",
            "test_macro_f1": "Test Macro F1",
        }
    )

    return round_float_columns(out)


def pretty_selector_table(df: pd.DataFrame) -> pd.DataFrame:
    out = add_display_columns(df)

    keep_cols = [
        "Dataset",
        "method_family",
        "Selector",
        "k",
        "Model",
        "n_selected_features",
        "feature_reduction_pct",
        "cv_macro_f1_mean",
        "cv_macro_f1_std",
        "test_macro_f1",
    ]

    out = out[keep_cols].rename(
        columns={
            "method_family": "Family",
            "n_selected_features": "Selected Features",
            "feature_reduction_pct": "Reduction (%)",
            "cv_macro_f1_mean": "CV Macro F1 Mean",
            "cv_macro_f1_std": "CV Macro F1 Std",
            "test_macro_f1": "Test Macro F1",
        }
    )

    return round_float_columns(out)


def pretty_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["Dataset"] = out["dataset"].map(DATASET_DISPLAY_NAMES).fillna(out["dataset"])
    out["Baseline Model"] = out["baseline_model"].map(MODEL_DISPLAY_NAMES).fillna(out["baseline_model"])
    out["FS Selector"] = out["fs_selector"].map(SELECTOR_DISPLAY_NAMES).fillna(out["fs_selector"])
    out["FS Model"] = out["fs_model"].map(MODEL_DISPLAY_NAMES).fillna(out["fs_model"])

    keep_cols = [
        "Dataset",
        "Baseline Model",
        "baseline_features",
        "baseline_cv_macro_f1",
        "baseline_test_macro_f1",
        "fs_family",
        "FS Selector",
        "fs_k",
        "FS Model",
        "fs_selected_features",
        "feature_reduction_pct",
        "fs_cv_macro_f1",
        "fs_test_macro_f1",
        "delta_cv_macro_f1",
        "delta_test_macro_f1",
    ]

    out = out[keep_cols].rename(
        columns={
            "baseline_features": "Baseline Features",
            "baseline_cv_macro_f1": "Baseline CV Macro F1",
            "baseline_test_macro_f1": "Baseline Test Macro F1",
            "fs_family": "Best FS Family",
            "fs_k": "FS k",
            "fs_selected_features": "FS Selected Features",
            "feature_reduction_pct": "Reduction (%)",
            "fs_cv_macro_f1": "FS CV Macro F1",
            "fs_test_macro_f1": "FS Test Macro F1",
            "delta_cv_macro_f1": "Δ CV Macro F1",
            "delta_test_macro_f1": "Δ Test Macro F1",
        }
    )

    return round_float_columns(out)


def get_experiment_dir_by_family(repo_root: Path, family: str) -> Path:
    for exp in EXPERIMENTS:
        if exp["family"] == family:
            return repo_root / "results" / "experiments" / exp["directory"]
    raise ValueError(f"Unknown family: {family}")


def selected_features_path(
    repo_root: Path,
    family: str,
    dataset: str,
    selector: str,
    k: int,
) -> Path:
    exp_dir = get_experiment_dir_by_family(repo_root, family)
    return exp_dir / "selected_features" / f"{dataset}__{selector}__k{k}.csv"


def normalize_selected_features_table(
    df: pd.DataFrame,
    max_rows: int = 20,
) -> pd.DataFrame:
    df = df.head(max_rows).copy()

    possible_score_cols = [
        "score",
        "rfe_importance",
        "prefilter_anova_score",
    ]

    keep_cols = ["rank", "feature_name"]

    for col in possible_score_cols:
        if col in df.columns:
            keep_cols.append(col)

    df = df[keep_cols]

    rename_map = {
        "rank": "Rank",
        "feature_name": "Feature",
        "score": "Score",
        "rfe_importance": "RFE Importance",
        "prefilter_anova_score": "Prefilter ANOVA Score",
    }

    df = df.rename(columns=rename_map)
    return round_float_columns(df, digits=6)


def write_top_feature_files(
    repo_root: Path,
    report_dir: Path,
    best_fs: pd.DataFrame,
) -> List[Dict[str, object]]:
    top_dir = report_dir / "top_selected_features"
    top_dir.mkdir(parents=True, exist_ok=True)

    records = []

    for _, row in best_fs.iterrows():
        dataset = row["dataset"]
        family = row["method_family"]
        selector = row["selector"]
        k = int(row["k"])

        src = selected_features_path(
            repo_root=repo_root,
            family=family,
            dataset=dataset,
            selector=selector,
            k=k,
        )

        if not src.exists():
            records.append(
                {
                    "dataset": dataset,
                    "family": family,
                    "selector": selector,
                    "k": k,
                    "path": str(src),
                    "status": "MISSING",
                }
            )
            continue

        selected = read_csv_required(src)
        top20 = normalize_selected_features_table(selected, max_rows=20)

        out_path = top_dir / f"{dataset}__best_fs_top20.csv"
        top20.to_csv(out_path, index=False)

        records.append(
            {
                "dataset": dataset,
                "family": family,
                "selector": selector,
                "k": k,
                "path": str(out_path),
                "status": "OK",
            }
        )

    return records


def generate_observations(comparison: pd.DataFrame) -> List[str]:
    lines = []

    for _, row in comparison.iterrows():
        dataset_display = DATASET_DISPLAY_NAMES.get(row["dataset"], row["dataset"])
        selector_display = SELECTOR_DISPLAY_NAMES.get(row["fs_selector"], row["fs_selector"])
        model_display = MODEL_DISPLAY_NAMES.get(row["fs_model"], row["fs_model"])

        delta_cv = row["delta_cv_macro_f1"]
        delta_test = row["delta_test_macro_f1"]

        cv_direction = "improved" if delta_cv >= 0 else "decreased"
        test_direction = "improved" if delta_test >= 0 else "decreased"

        lines.append(
            f"- **{dataset_display}**: best feature-selection configuration was "
            f"{row['fs_family']} / {selector_display} / k={int(row['fs_k'])} / "
            f"{model_display}. It reduced the feature count from "
            f"{int(row['baseline_features'])} to {int(row['fs_selected_features'])} "
            f"genes/features ({row['feature_reduction_pct']:.2f}% reduction). "
            f"CV Macro F1 {cv_direction} by {delta_cv:.4f}, and test Macro F1 "
            f"{test_direction} by {delta_test:.4f} compared with the no-feature-selection "
            f"baseline."
        )

    return lines


def write_report(repo_root: Path) -> Path:
    report_dir = repo_root / "reports" / "feature_selection"
    report_dir.mkdir(parents=True, exist_ok=True)

    baseline_best = load_baseline_best(repo_root)
    family_best = load_family_best(repo_root)
    selector_best = load_best_by_dataset_selector(repo_root)
    best_fs = choose_best_feature_selection(family_best)
    comparison = compare_baseline_vs_best_fs(baseline_best, best_fs)

    baseline_best.to_csv(report_dir / "baseline_best_summary.csv", index=False)
    family_best.to_csv(report_dir / "family_best_summary.csv", index=False)
    selector_best.to_csv(report_dir / "selector_best_summary.csv", index=False)
    best_fs.to_csv(report_dir / "best_feature_selection_by_dataset.csv", index=False)
    comparison.to_csv(report_dir / "baseline_vs_best_feature_selection.csv", index=False)

    top_feature_records = write_top_feature_files(repo_root, report_dir, best_fs)

    lines: List[str] = []

    lines.append("# Comparative Feature Selection Report")
    lines.append("")
    lines.append(f"Generated at: `{datetime.now().isoformat(timespec='seconds')}`")
    lines.append("")
    lines.append("## 1. Purpose")
    lines.append("")
    lines.append(
        "This report summarizes the feature selection experiments for the project "
        "**A comparative study of feature selection method for cancer classification "
        "using gene expression data**."
    )
    lines.append("")
    lines.append(
        "The goal is to compare whether different feature selection strategies can "
        "reduce the number of genes/features while preserving or improving cancer "
        "classification performance."
    )
    lines.append("")

    lines.append("## 2. Experimental design")
    lines.append("")
    lines.append(
        "The experiments compare four groups: no feature selection baseline, filter "
        "methods, wrapper methods, and embedded methods."
    )
    lines.append("")
    lines.append("- **Baseline:** all available features were used.")
    lines.append("- **Filter methods:** Variance Top-K, ANOVA F-test, Mutual Information.")
    lines.append("- **Wrapper methods:** RFE Logistic and RFE Linear SVM with ANOVA prefiltering.")
    lines.append("- **Embedded methods:** L1 Logistic, Random Forest Importance, Extra Trees Importance.")
    lines.append("")
    lines.append(
        "For each feature selection configuration, the selected features were evaluated "
        "using the same downstream classifiers as the baseline: Logistic Regression, "
        "Linear SVM, Random Forest, and Gaussian Naive Bayes."
    )
    lines.append("")
    lines.append(
        "The main selection criterion is **5-fold CV Macro F1** on `train_val`. The "
        "held-out test set is used only for final evaluation and descriptive comparison."
    )
    lines.append("")

    lines.append("## 3. Leakage prevention")
    lines.append("")
    lines.append(
        "Feature selection was fitted only on the training portion of each split. During "
        "cross-validation, feature ranking and selection were fitted on the CV training "
        "fold and then applied to the CV validation fold. During final testing, feature "
        "selection was fitted on `train_val` and evaluated once on the held-out test set."
    )
    lines.append("")
    lines.append(
        "Therefore, test labels were not used for selecting the feature selection method, "
        "the number of selected features, or the classifier."
    )
    lines.append("")

    lines.append("## 4. Best no-feature-selection baseline")
    lines.append("")
    lines.append(markdown_table(pretty_summary_table(baseline_best)))
    lines.append("")

    lines.append("## 5. Best configuration from each feature-selection family")
    lines.append("")
    lines.append(
        "For each dataset and method family, the table below reports the best "
        "configuration according to CV Macro F1."
    )
    lines.append("")
    lines.append(markdown_table(pretty_summary_table(family_best)))
    lines.append("")

    lines.append("## 6. Best configuration for each selector")
    lines.append("")
    lines.append(
        "This table gives a more detailed view by showing the best configuration for "
        "each individual selector."
    )
    lines.append("")
    lines.append(markdown_table(pretty_selector_table(selector_best)))
    lines.append("")

    lines.append("## 7. Overall best feature-selection configuration per dataset")
    lines.append("")
    lines.append(
        "The table below selects the single best feature-selection configuration for "
        "each dataset based on CV Macro F1."
    )
    lines.append("")
    lines.append(markdown_table(pretty_summary_table(best_fs)))
    lines.append("")

    lines.append("## 8. Comparison with no-feature-selection baseline")
    lines.append("")
    lines.append(
        "Positive Δ values mean that the selected feature-selection configuration "
        "outperformed the no-feature-selection baseline. Negative Δ values mean that "
        "performance decreased compared with using all features."
    )
    lines.append("")
    lines.append(markdown_table(pretty_comparison_table(comparison)))
    lines.append("")

    lines.append("## 9. Main observations")
    lines.append("")
    lines.extend(generate_observations(comparison))
    lines.append("")
    lines.append(
        "Overall, these experiments show the trade-off between performance and the "
        "number of selected genes/features. In a feature selection project, a method is "
        "especially useful when it achieves similar or better Macro F1 with substantially "
        "fewer features."
    )
    lines.append("")

    lines.append("## 10. Top selected features for the best configuration")
    lines.append("")
    lines.append(
        "The following subsections list the top selected features from the best "
        "feature-selection configuration for each dataset. Full selected feature files "
        "are available under each experiment directory."
    )
    lines.append("")

    for record in top_feature_records:
        dataset = record["dataset"]
        dataset_display = DATASET_DISPLAY_NAMES.get(dataset, dataset)
        family = record["family"]
        selector = record["selector"]
        selector_display = SELECTOR_DISPLAY_NAMES.get(selector, selector)
        k = int(record["k"])

        lines.append(f"### 10.{len([r for r in top_feature_records if r['dataset'] <= dataset])}. {dataset_display}")
        lines.append("")
        lines.append(f"Best selector: **{family} / {selector_display} / k={k}**")
        lines.append("")

        if record["status"] != "OK":
            lines.append(f"_Selected feature file missing: `{record['path']}`_")
            lines.append("")
            continue

        top_features = read_csv_required(Path(record["path"]))
        lines.append(markdown_table(top_features))
        lines.append("")

    lines.append("## 11. Generated report files")
    lines.append("")
    lines.append("This script generated the following report-level summary files:")
    lines.append("")
    lines.append("- `reports/feature_selection/feature_selection_report.md`")
    lines.append("- `reports/feature_selection/baseline_best_summary.csv`")
    lines.append("- `reports/feature_selection/family_best_summary.csv`")
    lines.append("- `reports/feature_selection/selector_best_summary.csv`")
    lines.append("- `reports/feature_selection/best_feature_selection_by_dataset.csv`")
    lines.append("- `reports/feature_selection/baseline_vs_best_feature_selection.csv`")
    lines.append("- `reports/feature_selection/top_selected_features/*.csv`")
    lines.append("")

    lines.append("## 12. Source experiment directories")
    lines.append("")
    lines.append("The report was built from:")
    lines.append("")
    lines.append("- `results/experiments/baseline_no_fs/`")
    lines.append("- `results/experiments/filter_feature_selection/`")
    lines.append("- `results/experiments/wrapper_rfe/`")
    lines.append("- `results/experiments/embedded_feature_selection/`")
    lines.append("")

    report_path = report_dir / "feature_selection_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    return report_path


def main() -> None:
    repo_root = find_repo_root()
    report_path = write_report(repo_root)

    print("Combined feature selection report created successfully.")
    print(f"Report path: {report_path}")


if __name__ == "__main__":
    main()
