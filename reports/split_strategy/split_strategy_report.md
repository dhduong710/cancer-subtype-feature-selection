# Train/Validation/Test Split and Cross-Validation Protocol

Generated at: `2026-06-25T00:17:21`

## 1. Purpose

This report documents the dataset splitting step for the project **A comparative study of feature selection method for cancer classification using gene expression data**.

The goal of this step is to create reliable train/validation/test partitions and cross-validation folds before model training and feature selection. This is important because feature selection, scaling, and model fitting must be performed only on the training portion of the data to avoid data leakage.

## 2. Splitting strategy

For datasets without an official split, a stratified 70/15/15 train/validation/test split was used. Stratification preserves the class distribution across splits, which is especially important for cancer subtype classification because some subtypes have fewer samples.

For the MOGONET BRCA optional benchmark, the official test set was preserved. Only the official training set was further split into training and validation subsets. This keeps the final test set untouched for unbiased final evaluation.

In addition, 5-fold stratified cross-validation was created on the `train_val` subset only. The test set is never used in cross-validation.

## 3. Overall split summary

| Dataset | Status | Train | Validation | Test | Train+Validation | CV folds |
| --- | --- | --- | --- | --- | --- | --- |
| MoGCN BRCA | PASS | 357 | 77 | 77 | 434 | 5 |
| CancerSD STAD | PASS | 252 | 55 | 55 | 307 | 5 |
| MOGONET BRCA Optional | PASS | 489 | 123 | 263 | 612 | 5 |

## 4. Dataset-level details

### 4.1. MoGCN BRCA

**Dataset key:** `mogcn_brca`

**Note:** This dataset does not provide an official train/test split in the processed project format. Therefore, a stratified 70/15/15 train/validation/test split was created.

**Split metadata:**

| Field | Value |
| --- | --- |
| Official split | False |
| Split strategy | stratified_70_15_15_holdout |
| Random state | 42 |
| CV folds | 5 |
| Train samples | 357 |
| Validation samples | 77 |
| Test samples | 77 |
| Train+Validation samples | 434 |

**Class distribution by split:**

| Label ID | Class | Train | Validation | Test | Train+Validation |
| --- | --- | --- | --- | --- | --- |
| 0 | LumA | 174 | 37 | 37 | 211 |
| 1 | LumB | 68 | 15 | 15 | 83 |
| 2 | Basal | 78 | 17 | 17 | 95 |
| 3 | Her2 | 37 | 8 | 8 | 45 |

**Cross-validation fold sizes:**

| Fold | CV train | CV validation |
| --- | --- | --- |
| 1 | 347 | 87 |
| 2 | 347 | 87 |
| 3 | 347 | 87 |
| 4 | 347 | 87 |
| 5 | 348 | 86 |

**Class distribution in CV validation folds:**

| Fold | Label ID | Class | CV validation samples |
| --- | --- | --- | --- |
| 1 | 0 | LumA | 43 |
| 1 | 1 | LumB | 16 |
| 1 | 2 | Basal | 19 |
| 1 | 3 | Her2 | 9 |
| 2 | 0 | LumA | 42 |
| 2 | 1 | LumB | 17 |
| 2 | 2 | Basal | 19 |
| 2 | 3 | Her2 | 9 |
| 3 | 0 | LumA | 42 |
| 3 | 1 | LumB | 17 |
| 3 | 2 | Basal | 19 |
| 3 | 3 | Her2 | 9 |
| 4 | 0 | LumA | 42 |
| 4 | 1 | LumB | 17 |
| 4 | 2 | Basal | 19 |
| 4 | 3 | Her2 | 9 |
| 5 | 0 | LumA | 42 |
| 5 | 1 | LumB | 16 |
| 5 | 2 | Basal | 19 |
| 5 | 3 | Her2 | 9 |

### 4.2. CancerSD STAD

**Dataset key:** `cancersd_stad`

**Note:** This dataset does not provide an official train/test split in the processed project format. Therefore, a stratified 70/15/15 train/validation/test split was created.

**Split metadata:**

| Field | Value |
| --- | --- |
| Official split | False |
| Split strategy | stratified_70_15_15_holdout |
| Random state | 42 |
| CV folds | 5 |
| Train samples | 252 |
| Validation samples | 55 |
| Test samples | 55 |
| Train+Validation samples | 307 |

**Class distribution by split:**

| Label ID | Class | Train | Validation | Test | Train+Validation |
| --- | --- | --- | --- | --- | --- |
| 0 | CIN | 142 | 31 | 31 | 173 |
| 1 | EBV | 20 | 4 | 4 | 24 |
| 2 | GS | 43 | 9 | 10 | 52 |
| 3 | MSI | 47 | 11 | 10 | 58 |

**Cross-validation fold sizes:**

| Fold | CV train | CV validation |
| --- | --- | --- |
| 1 | 245 | 62 |
| 2 | 245 | 62 |
| 3 | 246 | 61 |
| 4 | 246 | 61 |
| 5 | 246 | 61 |

**Class distribution in CV validation folds:**

| Fold | Label ID | Class | CV validation samples |
| --- | --- | --- | --- |
| 1 | 0 | CIN | 35 |
| 1 | 1 | EBV | 5 |
| 1 | 2 | GS | 11 |
| 1 | 3 | MSI | 11 |
| 2 | 0 | CIN | 35 |
| 2 | 1 | EBV | 4 |
| 2 | 2 | GS | 11 |
| 2 | 3 | MSI | 12 |
| 3 | 0 | CIN | 34 |
| 3 | 1 | EBV | 5 |
| 3 | 2 | GS | 10 |
| 3 | 3 | MSI | 12 |
| 4 | 0 | CIN | 34 |
| 4 | 1 | EBV | 5 |
| 4 | 2 | GS | 10 |
| 4 | 3 | MSI | 12 |
| 5 | 0 | CIN | 35 |
| 5 | 1 | EBV | 5 |
| 5 | 2 | GS | 10 |
| 5 | 3 | MSI | 11 |

### 4.3. MOGONET BRCA Optional

**Dataset key:** `mogonet_brca_optional`

**Note:** This dataset provides an official train/test split. The official test set was preserved, and only the official training set was further split into training and validation subsets.

**Split metadata:**

| Field | Value |
| --- | --- |
| Official split | True |
| Split strategy | official_test_preserved_train_val_split_from_official_train |
| Random state | 42 |
| CV folds | 5 |
| Train samples | 489 |
| Validation samples | 123 |
| Test samples | 263 |
| Train+Validation samples | 612 |

**Class distribution by split:**

| Label ID | Class | Train | Validation | Test | Train+Validation |
| --- | --- | --- | --- | --- | --- |
| 0 | Normal-like | 64 | 16 | 35 | 80 |
| 1 | Basal-like | 73 | 19 | 39 | 92 |
| 2 | HER2-enriched | 26 | 6 | 14 | 32 |
| 3 | Luminal A | 244 | 61 | 131 | 305 |
| 4 | Luminal B | 82 | 21 | 44 | 103 |

**Cross-validation fold sizes:**

| Fold | CV train | CV validation |
| --- | --- | --- |
| 1 | 489 | 123 |
| 2 | 489 | 123 |
| 3 | 490 | 122 |
| 4 | 490 | 122 |
| 5 | 490 | 122 |

**Class distribution in CV validation folds:**

| Fold | Label ID | Class | CV validation samples |
| --- | --- | --- | --- |
| 1 | 0 | Normal-like | 16 |
| 1 | 1 | Basal-like | 19 |
| 1 | 2 | HER2-enriched | 7 |
| 1 | 3 | Luminal A | 61 |
| 1 | 4 | Luminal B | 20 |
| 2 | 0 | Normal-like | 16 |
| 2 | 1 | Basal-like | 19 |
| 2 | 2 | HER2-enriched | 7 |
| 2 | 3 | Luminal A | 61 |
| 2 | 4 | Luminal B | 20 |
| 3 | 0 | Normal-like | 16 |
| 3 | 1 | Basal-like | 18 |
| 3 | 2 | HER2-enriched | 6 |
| 3 | 3 | Luminal A | 61 |
| 3 | 4 | Luminal B | 21 |
| 4 | 0 | Normal-like | 16 |
| 4 | 1 | Basal-like | 18 |
| 4 | 2 | HER2-enriched | 6 |
| 4 | 3 | Luminal A | 61 |
| 4 | 4 | Luminal B | 21 |
| 5 | 0 | Normal-like | 16 |
| 5 | 1 | Basal-like | 18 |
| 5 | 2 | HER2-enriched | 6 |
| 5 | 3 | Luminal A | 61 |
| 5 | 4 | Luminal B | 21 |

## 5. Validation checks

The split validation script checked the following conditions:

- Required split files exist for every dataset.
- Train, validation, and test splits have no overlapping samples.
- `train_val` is exactly the union of train and validation samples.
- All processed samples are covered by train/validation/test.
- For MOGONET BRCA optional, the official test set is preserved.
- Cross-validation folds are created only on `train_val`.
- Test samples do not appear in any cross-validation fold.
- Each `train_val` sample appears exactly once as a CV validation sample.
- Every split contains all class labels.

All datasets passed the split validation step.

## 6. Files generated

For each dataset, the following files were generated under `data/splits/<dataset>/`:

- `train.csv`: samples used for final model training.
- `val.csv`: validation samples used for model selection or early checking.
- `test.csv`: held-out samples used only for final evaluation.
- `train_val.csv`: union of train and validation samples.
- `all_splits.csv`: combined split assignment file.
- `cv_folds.csv`: 5-fold stratified cross-validation assignment.
- `split_summary.csv`: class distribution summary for train/validation/test.
- `cv_fold_summary.csv`: class distribution summary for CV folds.
- `split_metadata.json`: metadata and notes for reproducibility.

Global summary files were also generated under `data/splits/`:

- `split_creation_summary.json`
- `split_validation_summary.json`

## 7. Leakage prevention policy for later experiments

In later experiments, preprocessing steps that learn from data must be fitted inside the training split only. This includes standardization, normalization, feature selection, and model training.

For cross-validation, each fold must fit the scaler, feature selector, and classifier using only the CV training portion, then evaluate on the CV validation portion. For final testing, the selected pipeline must be trained on `train_val` and evaluated once on the held-out test set.
