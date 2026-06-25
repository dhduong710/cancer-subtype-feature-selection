#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd


DATASET_DISPLAY = {
    "mogcn_brca": "MoGCN BRCA",
    "cancersd_stad": "CancerSD STAD",
    "mogonet_brca_optional": "MOGONET BRCA Optional",
}

MODEL_DISPLAY = {
    "logistic_regression": "Logistic Regression",
    "linear_svm": "Linear SVM",
    "random_forest": "Random Forest",
    "gaussian_nb": "GaussianNB",
    "mlp": "MLP Neural Network",
}

SELECTOR_DISPLAY = {
    "none": "No FS",
    "variance_top_k": "Variance Top-K",
    "anova_f": "ANOVA F-test",
    "mutual_info": "Mutual Information",
    "rfe_logistic": "RFE Logistic",
    "rfe_linear_svm": "RFE Linear SVM",
    "l1_logistic": "L1 Logistic",
    "random_forest_importance": "RF Importance",
    "extra_trees_importance": "Extra Trees Importance",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No data available._"

    df = df.copy().fillna("")
    lines = []
    cols = list(df.columns)

    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")

    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")

    return "\n".join(lines)


def round_df(df: pd.DataFrame, digits: int = 4) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].round(digits)
    return df


def add_display(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Dataset"] = df["dataset"].map(DATASET_DISPLAY).fillna(df["dataset"])
    df["Selector"] = df["selector"].map(SELECTOR_DISPLAY).fillna(df["selector"])
    df["Model"] = df["model"].map(MODEL_DISPLAY).fillna(df["model"])
    return df


def load_baseline(root: Path) -> pd.DataFrame:
    d = root / "results" / "experiments" / "baseline_no_fs"

    best = read_csv(d / "best_models_by_cv.csv")
    test = read_csv(d / "test_results.csv")

    rows = []

    for _, r in best.iterrows():
        dataset = r["dataset"]
        model = r["model"]

        t = test[(test["dataset"] == dataset) & (test["model"] == model)].iloc[0]
        n_features = int(t["n_features"])

        rows.append(
            {
                "dataset": dataset,
                "experiment_group": "Classical baseline",
                "method_family": "Baseline",
                "selector": "none",
                "k": n_features,
                "model": model,
                "n_original_features": n_features,
                "n_selected_features": n_features,
                "feature_reduction_pct": 0.0,
                "cv_accuracy_mean": r["cv_accuracy_mean"],
                "cv_accuracy_std": r["cv_accuracy_std"],
                "cv_macro_f1_mean": r["cv_macro_f1_mean"],
                "cv_macro_f1_std": r["cv_macro_f1_std"],
                "test_accuracy": t["accuracy"],
                "test_balanced_accuracy": t["balanced_accuracy"],
                "test_macro_f1": t["macro_f1"],
                "test_weighted_f1": t["weighted_f1"],
            }
        )

    return pd.DataFrame(rows)


def load_classical_fs(root: Path, family: str, dirname: str) -> pd.DataFrame:
    d = root / "results" / "experiments" / dirname

    best = read_csv(d / "best_configs_by_dataset.csv")
    test = read_csv(d / "test_results.csv")

    rows = []

    for _, r in best.iterrows():
        dataset = r["dataset"]
        selector = r["selector"]
        k = int(r["k"])
        model = r["model"]

        t = test[
            (test["dataset"] == dataset)
            & (test["selector"] == selector)
            & (test["k"].astype(int) == k)
            & (test["model"] == model)
        ].iloc[0]

        n_original = int(t["n_original_features"])
        n_selected = int(t["n_selected_features"])

        rows.append(
            {
                "dataset": dataset,
                "experiment_group": f"Classical {family}",
                "method_family": family,
                "selector": selector,
                "k": k,
                "model": model,
                "n_original_features": n_original,
                "n_selected_features": n_selected,
                "feature_reduction_pct": 100.0 * (1.0 - n_selected / n_original),
                "cv_accuracy_mean": r["cv_accuracy_mean"],
                "cv_accuracy_std": r["cv_accuracy_std"],
                "cv_macro_f1_mean": r["cv_macro_f1_mean"],
                "cv_macro_f1_std": r["cv_macro_f1_std"],
                "test_accuracy": t["accuracy"],
                "test_balanced_accuracy": t["balanced_accuracy"],
                "test_macro_f1": t["macro_f1"],
                "test_weighted_f1": t["weighted_f1"],
            }
        )

    return pd.DataFrame(rows)


def load_mlp(root: Path) -> pd.DataFrame:
    d = root / "results" / "experiments" / "mlp_classifier"

    best = read_csv(d / "best_mlp_by_dataset.csv")
    test = read_csv(d / "test_results.csv")

    rows = []

    for _, r in best.iterrows():
        config_id = r["config_id"]
        t = test[test["config_id"] == config_id].iloc[0]

        n_original = int(r["n_original_features"])
        n_selected = int(r["n_selected_features"])

        rows.append(
            {
                "dataset": r["dataset"],
                "experiment_group": "MLP best",
                "method_family": r["method_family"],
                "selector": r["selector"],
                "k": int(r["k"]),
                "model": "mlp",
                "n_original_features": n_original,
                "n_selected_features": n_selected,
                "feature_reduction_pct": 100.0 * (1.0 - n_selected / n_original),
                "cv_accuracy_mean": r["cv_accuracy_mean"],
                "cv_accuracy_std": r["cv_accuracy_std"],
                "cv_macro_f1_mean": r["cv_macro_f1_mean"],
                "cv_macro_f1_std": r["cv_macro_f1_std"],
                "test_accuracy": t["accuracy"],
                "test_balanced_accuracy": t["balanced_accuracy"],
                "test_macro_f1": t["macro_f1"],
                "test_weighted_f1": t["weighted_f1"],
            }
        )

    return pd.DataFrame(rows)


def build_final_table(root: Path) -> pd.DataFrame:
    parts = [
        load_baseline(root),
        load_classical_fs(root, "Filter", "filter_feature_selection"),
        load_classical_fs(root, "Wrapper", "wrapper_rfe"),
        load_classical_fs(root, "Embedded", "embedded_feature_selection"),
        load_mlp(root),
    ]

    final = pd.concat(parts, ignore_index=True)

    final = final.sort_values(
        ["dataset", "cv_macro_f1_mean", "cv_accuracy_mean"],
        ascending=[True, False, False],
    ).reset_index(drop=True)

    return final


def best_overall_by_dataset(final: pd.DataFrame) -> pd.DataFrame:
    return (
        final.sort_values(
            ["dataset", "cv_macro_f1_mean", "cv_accuracy_mean"],
            ascending=[True, False, False],
        )
        .groupby("dataset", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )


def best_test_by_dataset(final: pd.DataFrame) -> pd.DataFrame:
    return (
        final.sort_values(
            ["dataset", "test_macro_f1", "test_accuracy"],
            ascending=[True, False, False],
        )
        .groupby("dataset", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )


def mlp_vs_best_classical(final: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for dataset in sorted(final["dataset"].unique()):
        dataset_df = final[final["dataset"] == dataset]

        classical = dataset_df[dataset_df["model"] != "mlp"].sort_values(
            ["cv_macro_f1_mean", "cv_accuracy_mean"],
            ascending=[False, False],
        ).iloc[0]

        mlp = dataset_df[dataset_df["model"] == "mlp"].sort_values(
            ["cv_macro_f1_mean", "cv_accuracy_mean"],
            ascending=[False, False],
        ).iloc[0]

        rows.append(
            {
                "dataset": dataset,
                "best_classical_group": classical["experiment_group"],
                "best_classical_selector": classical["selector"],
                "best_classical_model": classical["model"],
                "best_classical_cv_macro_f1": classical["cv_macro_f1_mean"],
                "best_classical_test_macro_f1": classical["test_macro_f1"],
                "mlp_group": mlp["experiment_group"],
                "mlp_selector": mlp["selector"],
                "mlp_cv_macro_f1": mlp["cv_macro_f1_mean"],
                "mlp_test_macro_f1": mlp["test_macro_f1"],
                "delta_mlp_minus_classical_cv_macro_f1": (
                    mlp["cv_macro_f1_mean"] - classical["cv_macro_f1_mean"]
                ),
                "delta_mlp_minus_classical_test_macro_f1": (
                    mlp["test_macro_f1"] - classical["test_macro_f1"]
                ),
            }
        )

    return pd.DataFrame(rows)


def pretty_main(df: pd.DataFrame) -> pd.DataFrame:
    out = add_display(df)

    out = out[
        [
            "Dataset",
            "experiment_group",
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
    ].rename(
        columns={
            "experiment_group": "Experiment",
            "method_family": "FS Family",
            "n_original_features": "Original Features",
            "n_selected_features": "Selected Features",
            "feature_reduction_pct": "Reduction (%)",
            "cv_accuracy_mean": "CV Accuracy",
            "cv_macro_f1_mean": "CV Macro F1",
            "cv_macro_f1_std": "CV Macro F1 Std",
            "test_accuracy": "Test Accuracy",
            "test_macro_f1": "Test Macro F1",
        }
    )

    return round_df(out)


def pretty_mlp_compare(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["Dataset"] = out["dataset"].map(DATASET_DISPLAY).fillna(out["dataset"])
    out["Best Classical Selector"] = (
        out["best_classical_selector"].map(SELECTOR_DISPLAY).fillna(out["best_classical_selector"])
    )
    out["Best Classical Model"] = (
        out["best_classical_model"].map(MODEL_DISPLAY).fillna(out["best_classical_model"])
    )
    out["MLP Selector"] = out["mlp_selector"].map(SELECTOR_DISPLAY).fillna(out["mlp_selector"])

    out = out[
        [
            "Dataset",
            "best_classical_group",
            "Best Classical Selector",
            "Best Classical Model",
            "best_classical_cv_macro_f1",
            "best_classical_test_macro_f1",
            "MLP Selector",
            "mlp_cv_macro_f1",
            "mlp_test_macro_f1",
            "delta_mlp_minus_classical_cv_macro_f1",
            "delta_mlp_minus_classical_test_macro_f1",
        ]
    ].rename(
        columns={
            "best_classical_group": "Best Classical Experiment",
            "best_classical_cv_macro_f1": "Classical CV Macro F1",
            "best_classical_test_macro_f1": "Classical Test Macro F1",
            "mlp_cv_macro_f1": "MLP CV Macro F1",
            "mlp_test_macro_f1": "MLP Test Macro F1",
            "delta_mlp_minus_classical_cv_macro_f1": "Δ MLP-Classical CV",
            "delta_mlp_minus_classical_test_macro_f1": "Δ MLP-Classical Test",
        }
    )

    return round_df(out)


def make_bar_chart(
    df: pd.DataFrame,
    value_col: str,
    title: str,
    ylabel: str,
    out_path: Path,
) -> None:
    plot_df = df.copy()
    plot_df["Dataset"] = plot_df["dataset"].map(DATASET_DISPLAY).fillna(plot_df["dataset"])
    plot_df["Experiment"] = plot_df["experiment_group"]

    pivot = plot_df.pivot(index="Dataset", columns="Experiment", values=value_col)

    ax = pivot.plot(kind="bar", figsize=(12, 6))
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Dataset")
    ax.legend(title="Experiment", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def make_feature_chart(df: pd.DataFrame, out_path: Path) -> None:
    plot_df = df.copy()
    plot_df["Dataset"] = plot_df["dataset"].map(DATASET_DISPLAY).fillna(plot_df["dataset"])
    plot_df["Experiment"] = plot_df["experiment_group"]

    pivot = plot_df.pivot(index="Dataset", columns="Experiment", values="n_selected_features")

    ax = pivot.plot(kind="bar", figsize=(12, 6))
    ax.set_title("Number of selected features")
    ax.set_ylabel("Number of features")
    ax.set_xlabel("Dataset")
    ax.legend(title="Experiment", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def observations(final: pd.DataFrame, mlp_compare: pd.DataFrame) -> List[str]:
    lines = []

    best_cv = best_overall_by_dataset(final)

    for _, r in best_cv.iterrows():
        dataset = DATASET_DISPLAY.get(r["dataset"], r["dataset"])
        selector = SELECTOR_DISPLAY.get(r["selector"], r["selector"])
        model = MODEL_DISPLAY.get(r["model"], r["model"])

        lines.append(
            f"- **{dataset}**: best configuration by CV Macro F1 is "
            f"{r['experiment_group']} / {selector} / k={int(r['k'])} / {model}, "
            f"with CV Macro F1 = {r['cv_macro_f1_mean']:.4f} and test Macro F1 = "
            f"{r['test_macro_f1']:.4f}."
        )

    for _, r in mlp_compare.iterrows():
        dataset = DATASET_DISPLAY.get(r["dataset"], r["dataset"])
        delta = r["delta_mlp_minus_classical_cv_macro_f1"]

        if delta >= 0:
            msg = "MLP was competitive or better than the best classical configuration by CV Macro F1"
        else:
            msg = "the best classical configuration remained stronger than MLP by CV Macro F1"

        lines.append(
            f"- **{dataset}**: {msg} "
            f"(Δ CV Macro F1 = {delta:.4f})."
        )

    return lines


def write_report(root: Path, report_dir: Path, final: pd.DataFrame) -> Path:
    best_cv = best_overall_by_dataset(final)
    best_test = best_test_by_dataset(final)
    mlp_compare = mlp_vs_best_classical(final)

    figure_dir = report_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    make_bar_chart(
        final,
        value_col="cv_macro_f1_mean",
        title="CV Macro F1 comparison",
        ylabel="CV Macro F1",
        out_path=figure_dir / "cv_macro_f1_comparison.png",
    )

    make_bar_chart(
        final,
        value_col="test_macro_f1",
        title="Held-out Test Macro F1 comparison",
        ylabel="Test Macro F1",
        out_path=figure_dir / "test_macro_f1_comparison.png",
    )

    make_feature_chart(
        final,
        out_path=figure_dir / "selected_features_comparison.png",
    )

    lines: List[str] = []

    lines.append("# Final Experiment Report")
    lines.append("")
    lines.append(f"Generated at: `{datetime.now().isoformat(timespec='seconds')}`")
    lines.append("")
    lines.append("## 1. Purpose")
    lines.append("")
    lines.append(
        "This report summarizes the final experimental comparison for the project "
        "**A comparative study of feature selection method for cancer classification "
        "using gene expression data**."
    )
    lines.append("")
    lines.append(
        "It combines the no-feature-selection baseline, filter feature selection, wrapper "
        "RFE, embedded feature selection, and the additional MLP Neural Network classifier."
    )
    lines.append("")

    lines.append("## 2. Experimental setting")
    lines.append("")
    lines.append(
        "All configurations were evaluated using 5-fold stratified cross-validation on "
        "`train_val`. The held-out test set was used only once for final evaluation."
    )
    lines.append("")
    lines.append(
        "The main model-selection metric is **CV Macro F1**, because the datasets are "
        "multiclass and class distributions are imbalanced."
    )
    lines.append("")
    lines.append(
        "Feature selection was always fitted inside the training part of each split to "
        "avoid data leakage."
    )
    lines.append("")

    lines.append("## 3. Final comparison table")
    lines.append("")
    lines.append(md_table(pretty_main(final)))
    lines.append("")

    lines.append("## 4. Best overall configuration by CV Macro F1")
    lines.append("")
    lines.append(md_table(pretty_main(best_cv)))
    lines.append("")

    lines.append("## 5. Best descriptive test result")
    lines.append("")
    lines.append(
        "This table is descriptive only. Test results should not be used for model selection."
    )
    lines.append("")
    lines.append(md_table(pretty_main(best_test)))
    lines.append("")

    lines.append("## 6. MLP versus best classical configuration")
    lines.append("")
    lines.append(md_table(pretty_mlp_compare(mlp_compare)))
    lines.append("")

    lines.append("## 7. Figures")
    lines.append("")
    lines.append("![CV Macro F1](figures/cv_macro_f1_comparison.png)")
    lines.append("")
    lines.append("![Test Macro F1](figures/test_macro_f1_comparison.png)")
    lines.append("")
    lines.append("![Selected Features](figures/selected_features_comparison.png)")
    lines.append("")

    lines.append("## 8. Main observations")
    lines.append("")
    lines.extend(observations(final, mlp_compare))
    lines.append("")
    lines.append(
        "Overall, the results show that feature selection can greatly reduce the number "
        "of genes/features while preserving or improving classification performance. "
        "The MLP classifier adds a neural-network baseline, but it does not consistently "
        "outperform the best classical machine learning configurations on these small "
        "high-dimensional datasets."
    )
    lines.append("")

    lines.append("## 9. Generated files")
    lines.append("")
    lines.append("- `reports/final_experiments/final_experiment_report.md`")
    lines.append("- `reports/final_experiments/final_comparison_table.csv`")
    lines.append("- `reports/final_experiments/best_overall_by_cv.csv`")
    lines.append("- `reports/final_experiments/best_descriptive_test.csv`")
    lines.append("- `reports/final_experiments/mlp_vs_best_classical.csv`")
    lines.append("- `reports/final_experiments/figures/*.png`")
    lines.append("")

    report_path = report_dir / "final_experiment_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    return report_path


def main() -> None:
    root = repo_root()
    report_dir = root / "reports" / "final_experiments"
    report_dir.mkdir(parents=True, exist_ok=True)

    final = build_final_table(root)
    best_cv = best_overall_by_dataset(final)
    best_test = best_test_by_dataset(final)
    mlp_compare = mlp_vs_best_classical(final)

    final.to_csv(report_dir / "final_comparison_table.csv", index=False)
    best_cv.to_csv(report_dir / "best_overall_by_cv.csv", index=False)
    best_test.to_csv(report_dir / "best_descriptive_test.csv", index=False)
    mlp_compare.to_csv(report_dir / "mlp_vs_best_classical.csv", index=False)

    report_path = write_report(root, report_dir, final)

    print("Final experiment report created successfully.")
    print(f"Report path: {report_path}")


if __name__ == "__main__":
    main()
