# Wine Quality Classification — Technical Report
### Data Mining: Model Tuning | Logistic Regression & SVM
**Author:** Faus7679 | **Dataset:** Red Wine Quality (UCI ML Repository)

---

## a) Problem Statement

Wine quality assessment is traditionally performed by human experts, an expensive and inconsistent process. The goal of this project is to build two automated classifiers — **Logistic Regression** and **Support Vector Machine (SVM)** — that predict whether a red wine is of *good quality* (score ≥ 6) or *poor quality* (score < 6) based on 11 physicochemical features (acidity, sulfur dioxide, alcohol content, etc.). Each model is tuned with grid search, validated with k-fold cross-validation, and evaluated on held-out data to provide precision, recall, F1-score, and AUC metrics.

---

## b) Algorithm of the Solution

### Step 1 — Load Packages
`numpy`, `pandas`, `scikit-learn`, `imbalanced-learn`, `matplotlib`, `seaborn`.

### Step 2 — Preprocessing

**a) Missing data (median imputation)**  
`SimpleImputer(strategy='median')` replaces any missing values with the column median. Median is robust to the right-skewed distributions characteristic of wine chemistry measurements.

```python
imputer = SimpleImputer(strategy='median')
df_imp  = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
# Missing values BEFORE imputation: 0
# Missing values AFTER  imputation: 0
```

**b) Feature engineering — two techniques**

*Technique 1: Log transformation* — `np.log1p()` is applied to five skewed features (`residual sugar`, `chlorides`, `free sulfur dioxide`, `total sulfur dioxide`, `sulphates`) to normalise their distributions.

*Technique 2: Interaction features* — Three domain-driven features are derived:
- `alcohol_x_sulphates` — multiplicative interaction
- `volatile_to_citric` — acidity ratio
- `total_acidity` — sum of all acid measurements

Total features after engineering: **20**.

**c) Feature selection — SelectKBest (ANOVA F-test)**  
The top-10 features are selected by univariate ANOVA F-scores. The dominant features are `volatile acidity` (F = 2082), `alcohol_x_sulphates` (F = 72), `alcohol` (F = 48), and `sulphates` (F = 43).

```
Selected: ['fixed acidity', 'volatile acidity', 'total sulfur dioxide',
           'density', 'sulphates', 'alcohol', 'log_sulphates',
           'alcohol_x_sulphates', 'volatile_to_citric', 'total_acidity']
```

### Step 3 — Subset & Split
All 1 599 instances are retained. A stratified 80/20 split yields **1 279 training** and **320 test** samples.

```
Training set: (1279, 10) | {Good: 684, Poor: 595}
Test set:     (320, 10)  | {Good: 171, Poor: 149}
```

### Step 4 — Build & Run Classifiers

#### Logistic Regression
- **Class imbalance → SMOTE:** Synthetic minority over-sampling balances the training set (595 → 684 poor samples) without touching the test set.
- **K-Fold CV (k=5):** Mean accuracy = **0.9459** (±0.028).
- **Grid Search:** Best `C=10`, `penalty='l1'`, `solver='liblinear'`; best CV F1 = **0.9457**.
- **Overfitting:** Train–validation gap = **0.0045** → no significant overfitting.

#### Support Vector Machine
- **Class imbalance → `class_weight='balanced'`:** SVM penalty is weighted inversely proportional to class frequency.
- **K-Fold CV (k=5):** Mean accuracy = **0.9296** (±0.027).
- **Grid Search:** Best `C=10`, `gamma='scale'`, `kernel='linear'`; best CV F1 = **0.9467**.
- **Overfitting:** Train–validation gap = **0.0068** → no significant overfitting.

---

## c) Analysis of Findings

### Confusion Matrices

| | **LR — Predicted Poor** | **LR — Predicted Good** |
|---|---|---|
| **Actual Poor** | 132 (TN) | 17 (FP) |
| **Actual Good** | 8 (FN) | 163 (TP) |

| | **SVM — Predicted Poor** | **SVM — Predicted Good** |
|---|---|---|
| **Actual Poor** | 134 (TN) | 15 (FP) |
| **Actual Good** | 8 (FN) | 163 (TP) |

### Precision, Recall, and F-Measure

| Metric | Logistic Regression | SVM |
|---|---|---|
| **Accuracy** | 0.9219 | 0.9281 |
| **Precision (macro)** | 0.9242 | 0.9297 |
| **Recall (macro)** | 0.9196 | 0.9263 |
| **F1-Score (macro)** | 0.9211 | 0.9275 |
| **ROC-AUC** | 0.9810 | 0.9817 |

*Generated plots:* `eda_plots.png`, `feature_importance.png`, `classification_results.png`, `comparison_chart.png`

### Interpretation

Both classifiers achieve high and nearly identical performance, with ROC-AUC exceeding **0.98**, indicating excellent discriminative ability. The SVM marginally outperforms Logistic Regression on all metrics (e.g., F1 = 0.9275 vs. 0.9211), suggesting the feature space has a near-linear structure — confirmed by grid search selecting `kernel='linear'` for SVM.

`volatile acidity` dominates feature importance with an ANOVA F-score of 2082 (next highest: 72), reflecting the well-documented negative effect of acetic acid on wine taste. The engineered `alcohol_x_sulphates` interaction (F = 72) outperforms either feature alone, validating the feature engineering step.

Learning curves for both models show the training and validation accuracy curves converging and stabilising with gaps below 0.01, confirming **no overfitting**. The narrow cross-validation confidence intervals (±0.03) indicate stable generalisation.

SMOTE improved LR's recall on the minority (Poor) class from an unbalanced baseline, reducing false negatives. The `class_weight='balanced'` approach for SVM achieved a similar effect with slightly better precision retention.

---

## d) References

1. Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). *Modeling wine preferences by data mining from physicochemical properties*. Decision Support Systems, 47(4), 547–553.
2. Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *JAIR*, 16, 321–357.
3. Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825–2830.
4. UCI Machine Learning Repository. (2009). Wine Quality Data Set. https://archive.ics.uci.edu/ml/datasets/wine+quality
5. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.
