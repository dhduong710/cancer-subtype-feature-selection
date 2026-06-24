# Data

Raw and processed data files are not committed to this repository because of file size and licensing constraints.

## Expected raw data structure

The local raw data folder should be organized as follows:

    data/raw/mogcn_brca/
    ├── fpkm_data.csv
    └── sample_classes.csv

    data/raw/cancersd_stad/
    ├── mRNA.zip
    └── patient_diagnose.csv

    data/raw/mogonet_brca_optional/
    ├── 1_tr.csv
    ├── 1_te.csv
    ├── 1_featname.csv
    ├── labels_tr.csv
    └── labels_te.csv

The raw data files are ignored by Git.

## Processed data structure

Each dataset is converted into a unified machine learning format:

    data/processed/<dataset_name>/
    ├── X.csv
    ├── y.csv
    ├── metadata.csv
    └── feature_names.csv

Where:

- `X.csv`: samples x genes/features matrix.
- `y.csv`: sample IDs and class labels.
- `metadata.csv`: sample-level metadata.
- `feature_names.csv`: retained gene or feature names.

For MOGONET BRCA, the original train/test split is also preserved:

    data/processed/mogonet_brca_optional/
    ├── X_train.csv
    ├── X_test.csv
    ├── y_train.csv
    └── y_test.csv

## Dataset summary

| Dataset | Cancer type | Samples | Features | Classes | Role |
|---|---:|---:|---:|---:|---|
| MoGCN BRCA | BRCA | 511 | 19,454 genes | 4 | Main dataset |
| CancerSD STAD | STAD | 362 | 4,089 genes | 4 | Main dataset |
| MOGONET BRCA | BRCA | 875 | 1,000 preprocessed features | 5 | Optional benchmark |

## Preprocessing policy

The preprocessing scripts only perform structural cleaning:

- sample-label matching,
- numeric conversion,
- missing-value checking,
- removal of all-missing or constant features,
- conversion into a unified machine learning format.

To avoid data leakage, the following operations are not performed during preprocessing:

- global standardization,
- global normalization,
- feature selection,
- model training.

These operations must be fitted only on training data during downstream machine learning experiments.

## Notes on MOGONET BRCA

MOGONET BRCA omics-1 is already preprocessed and feature-reduced by the original MOGONET dataset. Therefore, it is treated as an optional benchmark dataset, not as a raw gene expression dataset.

Original sample IDs are not included in the MOGONET CSV files, so synthetic sample IDs are generated during preprocessing while preserving the original train/test split.
