# Baseline Classification Without Feature Selection

Generated at: `2026-06-25T00:37:33`

## 1. Purpose

This report documents the baseline classification experiment without feature selection for the project **A comparative study of feature selection method for cancer classification using gene expression data**.

The purpose of this step is to establish a reference performance level using all available gene expression features. Later feature selection methods will be compared against this baseline to evaluate whether they can preserve or improve classification performance while using fewer genes.

## 2. Experimental protocol

For each dataset, 5-fold stratified cross-validation was performed on the `train_val` subset. The held-out test set was not used during cross-validation.

After cross-validation, each baseline classifier was trained once on the full `train_val` subset and evaluated on the held-out test set. The test results are reported for analysis, but model selection should be based on cross-validation performance, especially CV Macro F1.

No feature selection was applied in this experiment. Therefore, each model used the full feature set of the corresponding dataset.

**Leakage prevention:**

Standardization was placed inside an `sklearn` Pipeline, so the scaler was fitted only on the training portion of each CV fold and then applied to the validation portion. For final test evaluation, the scaler was fitted only on `train_val` and then applied to the test set.

## 3. Datasets and number of features

| Dataset | Train+Validation Samples | Test Samples | Features Used |
| --- | --- | --- | --- |
| CancerSD STAD | 307 | 55 | 4089 |
| MOGONET BRCA Optional | 612 | 263 | 1000 |
| MoGCN BRCA | 434 | 77 | 19454 |

## 4. Baseline models

| Model | Pipeline | Reason |
| --- | --- | --- |
| One-vs-Rest Logistic Regression | StandardScaler + OneVsRestClassifier(LogisticRegression) | Linear baseline suitable for high-dimensional gene expression data. |
| Linear SVM | StandardScaler + LinearSVC | Strong linear classifier for high-dimensional classification problems. |
| Random Forest | RandomForestClassifier | Tree-based ensemble baseline and useful reference before feature importance experiments. |
| Gaussian Naive Bayes | StandardScaler + GaussianNB | Simple probabilistic baseline for multiclass classification. |

## 5. Cross-validation results

The following table summarizes the 5-fold cross-validation results. The main selection metric is **CV Macro F1**, because the datasets are multiclass and some classes are smaller than others.

| Dataset | Model | CV Accuracy Mean | CV Accuracy Std | CV Balanced Accuracy Mean | CV Macro F1 Mean | CV Macro F1 Std | CV Weighted F1 Mean | CV Runtime Mean (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CancerSD STAD | Random Forest | 0.8242 | 0.0387 | 0.8163 | 0.8197 | 0.0211 | 0.8264 | 0.3758 |
| CancerSD STAD | One-vs-Rest Logistic Regression | 0.8208 | 0.0434 | 0.8576 | 0.8157 | 0.052 | 0.8257 | 0.4095 |
| CancerSD STAD | Linear SVM | 0.8143 | 0.0378 | 0.8609 | 0.8109 | 0.0459 | 0.8198 | 0.9027 |
| CancerSD STAD | Gaussian Naive Bayes | 0.7199 | 0.0313 | 0.6397 | 0.6512 | 0.0638 | 0.7118 | 0.0117 |
| MOGONET BRCA Optional | Random Forest | 0.7566 | 0.0209 | 0.739 | 0.7255 | 0.017 | 0.7637 | 0.3943 |
| MOGONET BRCA Optional | Gaussian Naive Bayes | 0.7223 | 0.0421 | 0.7767 | 0.7129 | 0.0387 | 0.7369 | 0.0059 |
| MOGONET BRCA Optional | One-vs-Rest Logistic Regression | 0.6912 | 0.0252 | 0.6293 | 0.6307 | 0.0322 | 0.6899 | 0.2586 |
| MOGONET BRCA Optional | Linear SVM | 0.6749 | 0.0111 | 0.5834 | 0.5933 | 0.0401 | 0.6712 | 23.4901 |
| MoGCN BRCA | Random Forest | 0.9055 | 0.0274 | 0.8884 | 0.8878 | 0.0348 | 0.9075 | 0.4576 |
| MoGCN BRCA | One-vs-Rest Logistic Regression | 0.8457 | 0.0349 | 0.8789 | 0.8359 | 0.0441 | 0.8517 | 3.7706 |
| MoGCN BRCA | Linear SVM | 0.8341 | 0.0339 | 0.8719 | 0.8254 | 0.0411 | 0.8417 | 1.5133 |
| MoGCN BRCA | Gaussian Naive Bayes | 0.5068 | 0.0625 | 0.3354 | 0.3273 | 0.0766 | 0.4464 | 0.1033 |

## 6. Best baseline model by cross-validation

For each dataset, the best baseline model was selected according to the highest mean CV Macro F1. If two models were close, CV Accuracy was used as an additional reference.

| Dataset | Model | CV Accuracy Mean | CV Accuracy Std | CV Macro F1 Mean | CV Macro F1 Std | CV Weighted F1 Mean |
| --- | --- | --- | --- | --- | --- | --- |
| CancerSD STAD | Random Forest | 0.8242 | 0.0387 | 0.8197 | 0.0211 | 0.8264 |
| MOGONET BRCA Optional | Random Forest | 0.7566 | 0.0209 | 0.7255 | 0.017 | 0.7637 |
| MoGCN BRCA | Random Forest | 0.9055 | 0.0274 | 0.8878 | 0.0348 | 0.9075 |

## 7. Final held-out test results

The table below shows the final test performance after training each model on `train_val` and evaluating it once on the held-out test set.

| Dataset | Model | Train+Validation Samples | Test Samples | Features | Test Accuracy | Test Balanced Accuracy | Test Macro F1 | Test Weighted F1 | CV Macro F1 Mean | CV Macro F1 Std | Test Runtime (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CancerSD STAD | Random Forest | 307 | 55 | 4089 | 0.8182 | 0.8177 | 0.7874 | 0.8252 | 0.8197 | 0.0211 | 0.4272 |
| CancerSD STAD | One-vs-Rest Logistic Regression | 307 | 55 | 4089 | 0.7455 | 0.8024 | 0.7439 | 0.7648 | 0.8157 | 0.052 | 0.4864 |
| CancerSD STAD | Linear SVM | 307 | 55 | 4089 | 0.7455 | 0.8024 | 0.7307 | 0.7632 | 0.8109 | 0.0459 | 0.4828 |
| CancerSD STAD | Gaussian Naive Bayes | 307 | 55 | 4089 | 0.7455 | 0.6766 | 0.6992 | 0.754 | 0.6512 | 0.0638 | 0.0139 |
| MOGONET BRCA Optional | Gaussian Naive Bayes | 612 | 263 | 1000 | 0.8175 | 0.8502 | 0.8123 | 0.8279 | 0.7129 | 0.0387 | 0.0083 |
| MOGONET BRCA Optional | Random Forest | 612 | 263 | 1000 | 0.8099 | 0.7444 | 0.7491 | 0.8066 | 0.7255 | 0.017 | 0.6051 |
| MOGONET BRCA Optional | One-vs-Rest Logistic Regression | 612 | 263 | 1000 | 0.7719 | 0.6905 | 0.699 | 0.7651 | 0.6307 | 0.0322 | 0.3212 |
| MOGONET BRCA Optional | Linear SVM | 612 | 263 | 1000 | 0.7224 | 0.6054 | 0.6188 | 0.7133 | 0.5933 | 0.0401 | 29.3875 |
| MoGCN BRCA | Random Forest | 434 | 77 | 19454 | 0.9091 | 0.8693 | 0.8784 | 0.9104 | 0.8878 | 0.0348 | 0.5184 |
| MoGCN BRCA | One-vs-Rest Logistic Regression | 434 | 77 | 19454 | 0.7792 | 0.7833 | 0.7427 | 0.7887 | 0.8359 | 0.0441 | 3.0887 |
| MoGCN BRCA | Linear SVM | 434 | 77 | 19454 | 0.7532 | 0.7619 | 0.7194 | 0.7672 | 0.8254 | 0.0411 | 1.8718 |
| MoGCN BRCA | Gaussian Naive Bayes | 434 | 77 | 19454 | 0.6234 | 0.4316 | 0.4075 | 0.5499 | 0.3273 | 0.0766 | 0.1242 |

## 8. Best test result per dataset

This table is descriptive only. It should not be used to choose models, because model selection should be based on cross-validation rather than test performance.

| Dataset | Model | Test Accuracy | Test Balanced Accuracy | Test Macro F1 | Test Weighted F1 |
| --- | --- | --- | --- | --- | --- |
| CancerSD STAD | Random Forest | 0.8182 | 0.8177 | 0.7874 | 0.8252 |
| MOGONET BRCA Optional | Gaussian Naive Bayes | 0.8175 | 0.8502 | 0.8123 | 0.8279 |
| MoGCN BRCA | Random Forest | 0.9091 | 0.8693 | 0.8784 | 0.9104 |

## 9. Main observations

- Random Forest achieved the best mean CV Macro F1 on all three datasets, making it the strongest no-feature-selection baseline according to the predefined cross-validation selection protocol.
- On MoGCN BRCA and CancerSD STAD, Random Forest also achieved the best held-out test Macro F1 among the evaluated baseline models.
- On MOGONET BRCA Optional, Gaussian Naive Bayes achieved the highest held-out test Macro F1, but Random Forest remained the best model according to CV Macro F1. This difference should be reported carefully because the test set should not drive model selection.
- Gaussian Naive Bayes performed weakly on MoGCN BRCA but strongly on MOGONET BRCA Optional, suggesting that the already feature-reduced MOGONET representation may be more compatible with simple probabilistic assumptions than the raw high-dimensional MoGCN BRCA gene expression matrix.
- These results provide the control baseline for later feature selection experiments. A feature selection method will be considered useful if it can maintain or improve Macro F1 while reducing the number of selected genes.

## 10. Generated result files

The baseline experiment generated the following files:

- `results/experiments/baseline_no_fs/cv_results.csv`
- `results/experiments/baseline_no_fs/cv_summary.csv`
- `results/experiments/baseline_no_fs/test_results.csv`
- `results/experiments/baseline_no_fs/best_models_by_cv.csv`
- `results/experiments/baseline_no_fs/label_mappings.csv`
- `results/experiments/baseline_no_fs/run_metadata.json`
- `results/experiments/baseline_no_fs/confusion_matrices/*.csv`

## 11. Reproducibility metadata

```json
{
  "experiment_name": "baseline_no_feature_selection",
  "created_at": "2026-06-25T00:29:29",
  "random_state": 42,
  "datasets": [
    "mogcn_brca",
    "cancersd_stad",
    "mogonet_brca_optional"
  ],
  "models": [
    "logistic_regression",
    "linear_svm",
    "random_forest",
    "gaussian_nb"
  ],
  "notes": [
    "This experiment uses no feature selection.",
    "Standardization is fitted inside each training fold through sklearn Pipeline.",
    "Cross-validation is performed only on train_val.",
    "Final test evaluation trains on train_val and evaluates once on test.",
    "These results are baseline controls for later feature selection experiments."
  ]
}
```

## 12. Label mappings

| Dataset | Label ID | Class |
| --- | --- | --- |
| MoGCN BRCA | 0 | LumA |
| MoGCN BRCA | 1 | LumB |
| MoGCN BRCA | 2 | Basal |
| MoGCN BRCA | 3 | Her2 |
| CancerSD STAD | 0 | CIN |
| CancerSD STAD | 1 | EBV |
| CancerSD STAD | 2 | GS |
| CancerSD STAD | 3 | MSI |
| MOGONET BRCA Optional | 0 | Normal-like |
| MOGONET BRCA Optional | 1 | Basal-like |
| MOGONET BRCA Optional | 2 | HER2-enriched |
| MOGONET BRCA Optional | 3 | Luminal A |
| MOGONET BRCA Optional | 4 | Luminal B |
