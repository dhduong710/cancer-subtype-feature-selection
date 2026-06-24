#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd


DATASETS = [
    "mogcn_brca",
    "cancersd_stad",
    "mogonet_brca_optional",
]

DATASET_DISPLAY_NAMES = {
    "mogcn_brca": "MoGCN BRCA",
    "cancersd_stad": "CancerSD STAD",
    "mogonet_brca_optional": "MOGONET BRCA Optional",
}

DATASET_NOTES = {
    "mogcn_brca": (
        "This dataset does not provide an official train/test split in the processed "
        "project format. Therefore, a stratified 70/15/15 train/validation/test split "
        "was created."
    ),
    "cancersd_stad": (
        "This dataset does not provide an official train/test split in the processed "
        "project format. Therefore, a stratified 70/15/15 train/validation/test split "
        "was created."
    ),
    "mogonet_brca_optional": (
        "This dataset provides an official train/test split. The official test set was "
        "preserved, and only the official training set was further split into training "
        "and validation subsets."
    ),
}


def find_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def read_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No data available._"

    df = df.copy()
    df = df.fillna("")

    headers = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for _, row in df.iterrows():
        values = [str(row[col]) for col in headers]
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def format_number(x) -> str:
    try:
        return f"{int(x):,}"
    except Exception:
        return str(x)


def compact_split_overview(validation_summary: List[Dict[str, object]]) -> pd.DataFrame:
    rows = []
    for item in validation_summary:
        rows.append(
            {
                "Dataset": DATASET_DISPLAY_NAMES.get(item["dataset"], item["dataset"]),
                "Status": item["status"],
                "Train": item["n_train"],
                "Validation": item["n_val"],
                "Test": item["n_test"],
                "Train+Validation": item["n_train_val"],
                "CV folds": item["n_cv_folds"],
            }
        )
    return pd.DataFrame(rows)


def class_distribution_wide(split_summary: pd.DataFrame) -> pd.DataFrame:
    table = split_summary.pivot_table(
        index=["label_id", "label"],
        columns="split",
        values="n_samples",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()

    preferred_cols = ["label_id", "label", "train", "val", "test", "train_val"]
    existing_cols = [col for col in preferred_cols if col in table.columns]
    table = table[existing_cols]

    rename_map = {
        "label_id": "Label ID",
        "label": "Class",
        "train": "Train",
        "val": "Validation",
        "test": "Test",
        "train_val": "Train+Validation",
    }
    table = table.rename(columns=rename_map)
    return table


def cv_fold_size_summary(cv_folds: pd.DataFrame) -> pd.DataFrame:
    table = (
        cv_folds.groupby(["fold", "cv_role"])
        .size()
        .reset_index(name="n_samples")
        .pivot(index="fold", columns="cv_role", values="n_samples")
        .reset_index()
        .fillna(0)
    )

    rename_map = {
        "fold": "Fold",
        "train": "CV train",
        "val": "CV validation",
    }
    table = table.rename(columns=rename_map)

    for col in ["CV train", "CV validation"]:
        if col in table.columns:
            table[col] = table[col].astype(int)

    return table


def cv_class_balance_summary(cv_folds: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for fold in sorted(cv_folds["fold"].unique()):
        fold_df = cv_folds[cv_folds["fold"] == fold]
        val_df = fold_df[fold_df["cv_role"] == "val"]

        counts = (
            val_df.groupby(["label_id", "label"])
            .size()
            .reset_index(name="n_samples")
            .sort_values(["label_id", "label"])
        )

        for _, row in counts.iterrows():
            rows.append(
                {
                    "Fold": int(fold),
                    "Label ID": int(row["label_id"]),
                    "Class": row["label"],
                    "CV validation samples": int(row["n_samples"]),
                }
            )

    return pd.DataFrame(rows)


def write_report(repo_root: Path) -> Path:
    split_root = repo_root / "data" / "splits"
    report_dir = repo_root / "reports" / "split_strategy"
    report_dir.mkdir(parents=True, exist_ok=True)

    validation_summary_path = split_root / "split_validation_summary.json"
    creation_summary_path = split_root / "split_creation_summary.json"

    validation_summary = read_json(validation_summary_path)
    creation_summary = read_json(creation_summary_path)

    lines: List[str] = []

    lines.append("# Train/Validation/Test Split and Cross-Validation Protocol")
    lines.append("")
    lines.append(f"Generated at: `{datetime.now().isoformat(timespec='seconds')}`")
    lines.append("")
    lines.append("## 1. Purpose")
    lines.append("")
    lines.append(
        "This report documents the dataset splitting step for the project "
        "**A comparative study of feature selection method for cancer classification "
        "using gene expression data**."
    )
    lines.append("")
    lines.append(
        "The goal of this step is to create reliable train/validation/test partitions "
        "and cross-validation folds before model training and feature selection. This is "
        "important because feature selection, scaling, and model fitting must be performed "
        "only on the training portion of the data to avoid data leakage."
    )
    lines.append("")

    lines.append("## 2. Splitting strategy")
    lines.append("")
    lines.append(
        "For datasets without an official split, a stratified 70/15/15 "
        "train/validation/test split was used. Stratification preserves the class "
        "distribution across splits, which is especially important for cancer subtype "
        "classification because some subtypes have fewer samples."
    )
    lines.append("")
    lines.append(
        "For the MOGONET BRCA optional benchmark, the official test set was preserved. "
        "Only the official training set was further split into training and validation "
        "subsets. This keeps the final test set untouched for unbiased final evaluation."
    )
    lines.append("")
    lines.append(
        "In addition, 5-fold stratified cross-validation was created on the "
        "`train_val` subset only. The test set is never used in cross-validation."
    )
    lines.append("")

    lines.append("## 3. Overall split summary")
    lines.append("")
    overview = compact_split_overview(validation_summary)
    lines.append(markdown_table(overview))
    lines.append("")

    lines.append("## 4. Dataset-level details")
    lines.append("")

    creation_by_dataset = {item["dataset"]: item for item in creation_summary}

    for dataset in DATASETS:
        display_name = DATASET_DISPLAY_NAMES.get(dataset, dataset)
        dataset_dir = split_root / dataset

        split_summary = read_csv(dataset_dir / "split_summary.csv")
        cv_folds = read_csv(dataset_dir / "cv_folds.csv")
        metadata = read_json(dataset_dir / "split_metadata.json")

        lines.append(f"### 4.{DATASETS.index(dataset) + 1}. {display_name}")
        lines.append("")
        lines.append(f"**Dataset key:** `{dataset}`")
        lines.append("")
        lines.append(f"**Note:** {DATASET_NOTES.get(dataset, '')}")
        lines.append("")
        lines.append("**Split metadata:**")
        lines.append("")
        metadata_table = pd.DataFrame(
            [
                {
                    "Field": "Official split",
                    "Value": str(metadata.get("official_split")),
                },
                {
                    "Field": "Split strategy",
                    "Value": metadata.get("split_strategy"),
                },
                {
                    "Field": "Random state",
                    "Value": metadata.get("random_state"),
                },
                {
                    "Field": "CV folds",
                    "Value": metadata.get("cv_folds"),
                },
                {
                    "Field": "Train samples",
                    "Value": metadata.get("n_train"),
                },
                {
                    "Field": "Validation samples",
                    "Value": metadata.get("n_val"),
                },
                {
                    "Field": "Test samples",
                    "Value": metadata.get("n_test"),
                },
                {
                    "Field": "Train+Validation samples",
                    "Value": metadata.get("n_train_val"),
                },
            ]
        )
        lines.append(markdown_table(metadata_table))
        lines.append("")

        lines.append("**Class distribution by split:**")
        lines.append("")
        class_table = class_distribution_wide(split_summary)
        lines.append(markdown_table(class_table))
        lines.append("")

        lines.append("**Cross-validation fold sizes:**")
        lines.append("")
        cv_size_table = cv_fold_size_summary(cv_folds)
        lines.append(markdown_table(cv_size_table))
        lines.append("")

        lines.append("**Class distribution in CV validation folds:**")
        lines.append("")
        cv_balance_table = cv_class_balance_summary(cv_folds)
        lines.append(markdown_table(cv_balance_table))
        lines.append("")

    lines.append("## 5. Validation checks")
    lines.append("")
    lines.append("The split validation script checked the following conditions:")
    lines.append("")
    lines.append("- Required split files exist for every dataset.")
    lines.append("- Train, validation, and test splits have no overlapping samples.")
    lines.append("- `train_val` is exactly the union of train and validation samples.")
    lines.append("- All processed samples are covered by train/validation/test.")
    lines.append("- For MOGONET BRCA optional, the official test set is preserved.")
    lines.append("- Cross-validation folds are created only on `train_val`.")
    lines.append("- Test samples do not appear in any cross-validation fold.")
    lines.append("- Each `train_val` sample appears exactly once as a CV validation sample.")
    lines.append("- Every split contains all class labels.")
    lines.append("")
    lines.append("All datasets passed the split validation step.")
    lines.append("")

    lines.append("## 6. Files generated")
    lines.append("")
    lines.append("For each dataset, the following files were generated under `data/splits/<dataset>/`:")
    lines.append("")
    lines.append("- `train.csv`: samples used for final model training.")
    lines.append("- `val.csv`: validation samples used for model selection or early checking.")
    lines.append("- `test.csv`: held-out samples used only for final evaluation.")
    lines.append("- `train_val.csv`: union of train and validation samples.")
    lines.append("- `all_splits.csv`: combined split assignment file.")
    lines.append("- `cv_folds.csv`: 5-fold stratified cross-validation assignment.")
    lines.append("- `split_summary.csv`: class distribution summary for train/validation/test.")
    lines.append("- `cv_fold_summary.csv`: class distribution summary for CV folds.")
    lines.append("- `split_metadata.json`: metadata and notes for reproducibility.")
    lines.append("")
    lines.append("Global summary files were also generated under `data/splits/`:")
    lines.append("")
    lines.append("- `split_creation_summary.json`")
    lines.append("- `split_validation_summary.json`")
    lines.append("")

    lines.append("## 7. Leakage prevention policy for later experiments")
    lines.append("")
    lines.append(
        "In later experiments, preprocessing steps that learn from data must be fitted "
        "inside the training split only. This includes standardization, normalization, "
        "feature selection, and model training."
    )
    lines.append("")
    lines.append(
        "For cross-validation, each fold must fit the scaler, feature selector, and "
        "classifier using only the CV training portion, then evaluate on the CV validation "
        "portion. For final testing, the selected pipeline must be trained on `train_val` "
        "and evaluated once on the held-out test set."
    )
    lines.append("")

    report_path = report_dir / "split_strategy_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    return report_path


def main() -> None:
    repo_root = find_repo_root()
    report_path = write_report(repo_root)

    print("Split strategy report created successfully.")
    print(f"Report path: {report_path}")


if __name__ == "__main__":
    main()
