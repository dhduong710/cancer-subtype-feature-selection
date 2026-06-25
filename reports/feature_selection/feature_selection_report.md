# Comparative Feature Selection Report

Generated at: `2026-06-25T09:00:51`

## 1. Purpose

This report summarizes the feature selection experiments for the project **A comparative study of feature selection method for cancer classification using gene expression data**.

The goal is to compare whether different feature selection strategies can reduce the number of genes/features while preserving or improving cancer classification performance.

## 2. Experimental design

The experiments compare four groups: no feature selection baseline, filter methods, wrapper methods, and embedded methods.

- **Baseline:** all available features were used.
- **Filter methods:** Variance Top-K, ANOVA F-test, Mutual Information.
- **Wrapper methods:** RFE Logistic and RFE Linear SVM with ANOVA prefiltering.
- **Embedded methods:** L1 Logistic, Random Forest Importance, Extra Trees Importance.

For each feature selection configuration, the selected features were evaluated using the same downstream classifiers as the baseline: Logistic Regression, Linear SVM, Random Forest, and Gaussian Naive Bayes.

The main selection criterion is **5-fold CV Macro F1** on `train_val`. The held-out test set is used only for final evaluation and descriptive comparison.

## 3. Leakage prevention

Feature selection was fitted only on the training portion of each split. During cross-validation, feature ranking and selection were fitted on the CV training fold and then applied to the CV validation fold. During final testing, feature selection was fitted on `train_val` and evaluated once on the held-out test set.

Therefore, test labels were not used for selecting the feature selection method, the number of selected features, or the classifier.

## 4. Best no-feature-selection baseline

| Dataset | Family | Selector | k | Model | Original Features | Selected Features | Reduction (%) | CV Accuracy Mean | CV Macro F1 Mean | CV Macro F1 Std | Test Accuracy | Test Macro F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CancerSD STAD | Baseline | No feature selection | 4089 | Random Forest | 4089 | 4089 | 0.0 | 0.8242 | 0.8197 | 0.0211 | 0.8182 | 0.7874 |
| MoGCN BRCA | Baseline | No feature selection | 19454 | Random Forest | 19454 | 19454 | 0.0 | 0.9055 | 0.8878 | 0.0348 | 0.9091 | 0.8784 |
| MOGONET BRCA Optional | Baseline | No feature selection | 1000 | Random Forest | 1000 | 1000 | 0.0 | 0.7566 | 0.7255 | 0.017 | 0.8099 | 0.7491 |

## 5. Best configuration from each feature-selection family

For each dataset and method family, the table below reports the best configuration according to CV Macro F1.

| Dataset | Family | Selector | k | Model | Original Features | Selected Features | Reduction (%) | CV Accuracy Mean | CV Macro F1 Mean | CV Macro F1 Std | Test Accuracy | Test Macro F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CancerSD STAD | Filter | ANOVA F-test | 200 | One-vs-Rest Logistic Regression | 4089 | 200 | 95.1088 | 0.8534 | 0.8545 | 0.0411 | 0.8182 | 0.7947 |
| MoGCN BRCA | Filter | Mutual Information | 100 | One-vs-Rest Logistic Regression | 19454 | 100 | 99.486 | 0.924 | 0.9113 | 0.0568 | 0.9091 | 0.8952 |
| MOGONET BRCA Optional | Filter | Variance Top-K | 200 | Random Forest | 1000 | 200 | 80.0 | 0.7664 | 0.7329 | 0.0349 | 0.8251 | 0.7733 |
| CancerSD STAD | Wrapper | RFE Logistic | 100 | One-vs-Rest Logistic Regression | 4089 | 100 | 97.5544 | 0.8731 | 0.8787 | 0.0324 | 0.8182 | 0.7868 |
| MoGCN BRCA | Wrapper | RFE Linear SVM | 100 | Random Forest | 19454 | 100 | 99.486 | 0.9263 | 0.9119 | 0.0241 | 0.9351 | 0.9311 |
| MOGONET BRCA Optional | Wrapper | RFE Logistic | 200 | Gaussian Naive Bayes | 1000 | 200 | 80.0 | 0.7354 | 0.7275 | 0.0378 | 0.8327 | 0.8269 |
| CancerSD STAD | Embedded | L1 Logistic | 200 | One-vs-Rest Logistic Regression | 4089 | 200 | 95.1088 | 0.8698 | 0.8716 | 0.0617 | 0.8182 | 0.82 |
| MoGCN BRCA | Embedded | Extra Trees Importance | 500 | One-vs-Rest Logistic Regression | 19454 | 500 | 97.4298 | 0.9309 | 0.925 | 0.0283 | 0.9091 | 0.8907 |
| MOGONET BRCA Optional | Embedded | Extra Trees Importance | 500 | Random Forest | 1000 | 500 | 50.0 | 0.7582 | 0.7295 | 0.0223 | 0.8175 | 0.7743 |

## 6. Best configuration for each selector

This table gives a more detailed view by showing the best configuration for each individual selector.

| Dataset | Family | Selector | k | Model | Selected Features | Reduction (%) | CV Macro F1 Mean | CV Macro F1 Std | Test Macro F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CancerSD STAD | Filter | ANOVA F-test | 200 | One-vs-Rest Logistic Regression | 200 | 95.1088 | 0.8545 | 0.0411 | 0.7947 |
| CancerSD STAD | Filter | Mutual Information | 200 | One-vs-Rest Logistic Regression | 200 | 95.1088 | 0.8523 | 0.0395 | 0.8063 |
| CancerSD STAD | Filter | Variance Top-K | 500 | One-vs-Rest Logistic Regression | 500 | 87.7721 | 0.846 | 0.021 | 0.8377 |
| MoGCN BRCA | Filter | ANOVA F-test | 200 | One-vs-Rest Logistic Regression | 200 | 98.9719 | 0.9058 | 0.0327 | 0.9287 |
| MoGCN BRCA | Filter | Mutual Information | 100 | One-vs-Rest Logistic Regression | 100 | 99.486 | 0.9113 | 0.0568 | 0.8952 |
| MoGCN BRCA | Filter | Variance Top-K | 500 | Random Forest | 500 | 97.4298 | 0.8877 | 0.0241 | 0.875 |
| MOGONET BRCA Optional | Filter | ANOVA F-test | 500 | Random Forest | 500 | 50.0 | 0.7272 | 0.0243 | 0.7497 |
| MOGONET BRCA Optional | Filter | Mutual Information | 500 | Random Forest | 500 | 50.0 | 0.7247 | 0.0233 | 0.7466 |
| MOGONET BRCA Optional | Filter | Variance Top-K | 200 | Random Forest | 200 | 80.0 | 0.7329 | 0.0349 | 0.7733 |
| CancerSD STAD | Wrapper | RFE Linear SVM | 100 | One-vs-Rest Logistic Regression | 100 | 97.5544 | 0.8673 | 0.0251 | 0.7682 |
| CancerSD STAD | Wrapper | RFE Logistic | 100 | One-vs-Rest Logistic Regression | 100 | 97.5544 | 0.8787 | 0.0324 | 0.7868 |
| MoGCN BRCA | Wrapper | RFE Linear SVM | 100 | Random Forest | 100 | 99.486 | 0.9119 | 0.0241 | 0.9311 |
| MoGCN BRCA | Wrapper | RFE Logistic | 200 | Random Forest | 200 | 98.9719 | 0.91 | 0.0362 | 0.9197 |
| MOGONET BRCA Optional | Wrapper | RFE Linear SVM | 100 | Gaussian Naive Bayes | 100 | 90.0 | 0.7229 | 0.0411 | 0.8179 |
| MOGONET BRCA Optional | Wrapper | RFE Logistic | 200 | Gaussian Naive Bayes | 200 | 80.0 | 0.7275 | 0.0378 | 0.8269 |
| CancerSD STAD | Embedded | Extra Trees Importance | 500 | One-vs-Rest Logistic Regression | 500 | 87.7721 | 0.8438 | 0.0399 | 0.7728 |
| CancerSD STAD | Embedded | L1 Logistic | 200 | One-vs-Rest Logistic Regression | 200 | 95.1088 | 0.8716 | 0.0617 | 0.82 |
| CancerSD STAD | Embedded | Random Forest Importance | 200 | Linear SVM | 200 | 95.1088 | 0.8652 | 0.0202 | 0.7833 |
| MoGCN BRCA | Embedded | Extra Trees Importance | 500 | One-vs-Rest Logistic Regression | 500 | 97.4298 | 0.925 | 0.0283 | 0.8907 |
| MoGCN BRCA | Embedded | L1 Logistic | 500 | Random Forest | 500 | 97.4298 | 0.9194 | 0.0437 | 0.9203 |
| MoGCN BRCA | Embedded | Random Forest Importance | 500 | Random Forest | 500 | 97.4298 | 0.9142 | 0.033 | 0.9203 |
| MOGONET BRCA Optional | Embedded | Extra Trees Importance | 500 | Random Forest | 500 | 50.0 | 0.7295 | 0.0223 | 0.7743 |
| MOGONET BRCA Optional | Embedded | L1 Logistic | 200 | Gaussian Naive Bayes | 200 | 80.0 | 0.7251 | 0.0383 | 0.7974 |
| MOGONET BRCA Optional | Embedded | Random Forest Importance | 50 | Gaussian Naive Bayes | 50 | 95.0 | 0.7286 | 0.036 | 0.8013 |

## 7. Overall best feature-selection configuration per dataset

The table below selects the single best feature-selection configuration for each dataset based on CV Macro F1.

| Dataset | Family | Selector | k | Model | Original Features | Selected Features | Reduction (%) | CV Accuracy Mean | CV Macro F1 Mean | CV Macro F1 Std | Test Accuracy | Test Macro F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CancerSD STAD | Wrapper | RFE Logistic | 100 | One-vs-Rest Logistic Regression | 4089 | 100 | 97.5544 | 0.8731 | 0.8787 | 0.0324 | 0.8182 | 0.7868 |
| MoGCN BRCA | Embedded | Extra Trees Importance | 500 | One-vs-Rest Logistic Regression | 19454 | 500 | 97.4298 | 0.9309 | 0.925 | 0.0283 | 0.9091 | 0.8907 |
| MOGONET BRCA Optional | Filter | Variance Top-K | 200 | Random Forest | 1000 | 200 | 80.0 | 0.7664 | 0.7329 | 0.0349 | 0.8251 | 0.7733 |

## 8. Comparison with no-feature-selection baseline

Positive Δ values mean that the selected feature-selection configuration outperformed the no-feature-selection baseline. Negative Δ values mean that performance decreased compared with using all features.

| Dataset | Baseline Model | Baseline Features | Baseline CV Macro F1 | Baseline Test Macro F1 | Best FS Family | FS Selector | FS k | FS Model | FS Selected Features | Reduction (%) | FS CV Macro F1 | FS Test Macro F1 | Δ CV Macro F1 | Δ Test Macro F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CancerSD STAD | Random Forest | 4089 | 0.8197 | 0.7874 | Wrapper | RFE Logistic | 100 | One-vs-Rest Logistic Regression | 100 | 97.5544 | 0.8787 | 0.7868 | 0.059 | -0.0006 |
| MoGCN BRCA | Random Forest | 19454 | 0.8878 | 0.8784 | Embedded | Extra Trees Importance | 500 | One-vs-Rest Logistic Regression | 500 | 97.4298 | 0.925 | 0.8907 | 0.0372 | 0.0123 |
| MOGONET BRCA Optional | Random Forest | 1000 | 0.7255 | 0.7491 | Filter | Variance Top-K | 200 | Random Forest | 200 | 80.0 | 0.7329 | 0.7733 | 0.0074 | 0.0243 |

## 9. Main observations

- **CancerSD STAD**: best feature-selection configuration was Wrapper / RFE Logistic / k=100 / One-vs-Rest Logistic Regression. It reduced the feature count from 4089 to 100 genes/features (97.55% reduction). CV Macro F1 improved by 0.0590, and test Macro F1 decreased by -0.0006 compared with the no-feature-selection baseline.
- **MoGCN BRCA**: best feature-selection configuration was Embedded / Extra Trees Importance / k=500 / One-vs-Rest Logistic Regression. It reduced the feature count from 19454 to 500 genes/features (97.43% reduction). CV Macro F1 improved by 0.0372, and test Macro F1 improved by 0.0123 compared with the no-feature-selection baseline.
- **MOGONET BRCA Optional**: best feature-selection configuration was Filter / Variance Top-K / k=200 / Random Forest. It reduced the feature count from 1000 to 200 genes/features (80.00% reduction). CV Macro F1 improved by 0.0074, and test Macro F1 improved by 0.0243 compared with the no-feature-selection baseline.

Overall, these experiments show the trade-off between performance and the number of selected genes/features. In a feature selection project, a method is especially useful when it achieves similar or better Macro F1 with substantially fewer features.

## 10. Top selected features for the best configuration

The following subsections list the top selected features from the best feature-selection configuration for each dataset. Full selected feature files are available under each experiment directory.

### 10.1. CancerSD STAD

Best selector: **Wrapper / RFE Logistic / k=100**

| Rank | Feature | RFE Importance | Prefilter ANOVA Score |
| --- | --- | --- | --- |
| 1 | SLC14A1 | 0.237799 | 16.957165 |
| 2 | CRABP2 | 0.235444 | 18.249296 |
| 3 | SPATA18 | 0.225581 | 22.988781 |
| 4 | PROC | 0.220115 | 31.207773 |
| 5 | RPL22L1 | 0.21992 | 71.650971 |
| 6 | GXYLT2 | 0.21418 | 13.347538 |
| 7 | DIRAS3 | 0.211547 | 13.874597 |
| 8 | CRAT | 0.209385 | 32.327991 |
| 9 | SLC22A3 | 0.207981 | 32.061516 |
| 10 | HSPA1B | 0.207534 | 16.422404 |
| 11 | MADCAM1 | 0.201896 | 23.640438 |
| 12 | PCLO | 0.198146 | 30.895437 |
| 13 | SYNE4 | 0.194891 | 21.01789 |
| 14 | MFSD6L | 0.194503 | 13.68146 |
| 15 | RNF208 | 0.194086 | 26.20558 |
| 16 | PTPRT | 0.19307 | 13.927633 |
| 17 | ZIC2 | 0.192242 | 18.901661 |
| 18 | GLB1L2 | 0.191398 | 14.352084 |
| 19 | FUT6 | 0.189894 | 21.265685 |
| 20 | EPDR1 | 0.189574 | 15.349464 |

### 10.2. MoGCN BRCA

Best selector: **Embedded / Extra Trees Importance / k=500**

| Rank | Feature | Score |
| --- | --- | --- |
| 1 | C5AR2 | 0.003717 |
| 2 | FOXA1 | 0.003681 |
| 3 | ESR1 | 0.003082 |
| 4 | GABRP | 0.00265 |
| 5 | CDKN3 | 0.002638 |
| 6 | FOXC1 | 0.002563 |
| 7 | EN1 | 0.002557 |
| 8 | TSPAN1 | 0.002255 |
| 9 | AR | 0.002206 |
| 10 | HORMAD1 | 0.00217 |
| 11 | REEP6 | 0.002074 |
| 12 | SRSF12 | 0.002053 |
| 13 | SLC44A4 | 0.002009 |
| 14 | ROPN1B | 0.001952 |
| 15 | RERG | 0.001864 |
| 16 | GTSE1 | 0.001856 |
| 17 | BCL2 | 0.001834 |
| 18 | BCAS1 | 0.001817 |
| 19 | PRR15 | 0.001793 |
| 20 | HID1 | 0.001704 |

### 10.3. MOGONET BRCA Optional

Best selector: **Filter / Variance Top-K / k=200**

| Rank | Feature | Score |
| --- | --- | --- |
| 1 | TFF1|7031 | 0.057239 |
| 2 | AGR3|155465 | 0.046312 |
| 3 | GABRP|2568 | 0.045082 |
| 4 | SOX10|6663 | 0.044088 |
| 5 | ANKRD30A|91074 | 0.042957 |
| 6 | CYP2B7P1|1556 | 0.041963 |
| 7 | FABP7|2173 | 0.04087 |
| 8 | KRT6B|3854 | 0.038481 |
| 9 | C1orf64|149563 | 0.037972 |
| 10 | GP2|2813 | 0.037487 |
| 11 | AGR2|10551 | 0.037301 |
| 12 | ABCC11|85320 | 0.034896 |
| 13 | C20orf114|92747 | 0.034779 |
| 14 | TFF3|7033 | 0.034635 |
| 15 | KLK7|5650 | 0.033419 |
| 16 | SLC6A14|11254 | 0.032323 |
| 17 | KLK6|5653 | 0.03175 |
| 18 | SFRP1|6422 | 0.031152 |
| 19 | VGLL1|51442 | 0.031024 |
| 20 | KRT16|3868 | 0.030678 |

## 11. Generated report files

This script generated the following report-level summary files:

- `reports/feature_selection/feature_selection_report.md`
- `reports/feature_selection/baseline_best_summary.csv`
- `reports/feature_selection/family_best_summary.csv`
- `reports/feature_selection/selector_best_summary.csv`
- `reports/feature_selection/best_feature_selection_by_dataset.csv`
- `reports/feature_selection/baseline_vs_best_feature_selection.csv`
- `reports/feature_selection/top_selected_features/*.csv`

## 12. Source experiment directories

The report was built from:

- `results/experiments/baseline_no_fs/`
- `results/experiments/filter_feature_selection/`
- `results/experiments/wrapper_rfe/`
- `results/experiments/embedded_feature_selection/`
