import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import (
    train_test_split,
    KFold,
    cross_val_score,
    GridSearchCV,
    learning_curve,
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    accuracy_score,
)

try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    SMOTE = None

warnings.filterwarnings('ignore')

def load_dataset() -> pd.DataFrame:
    data_paths = [
        'wine-quality-red.csv',
        os.path.expanduser('~/Downloads/wine-quality-red.csv'),
    ]

    df = None
    for path in data_paths:
        if os.path.exists(path):
            try:
                tmp = pd.read_csv(path, sep=';')
                if tmp.shape[1] >= 12:
                    df = tmp
                    print(f"Loaded (semicolon-delimited): {path}")
                else:
                    df = pd.read_csv(path)
                    print(f"Loaded (comma-delimited): {path}")
            except Exception:
                df = pd.read_csv(path)
                print(f"Loaded: {path}")
            break

    if df is None:
        raise FileNotFoundError(
            "wine-quality-red.csv not found. "
            "Place the file in the repository root or in ~/Downloads/"
        )

    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print("\n=== Descriptive Statistics ===")
    print(df.describe().to_string())

    return df


def create_eda_plots(df: pd.DataFrame) -> None:
    print("\n=== Quality Score Distribution ===")
    print(df['quality'].value_counts().sort_index())

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    df['quality'].value_counts().sort_index().plot(
        kind='bar', ax=axes[0], color='steelblue', edgecolor='black'
    )
    axes[0].set_title('Wine Quality Score Distribution')
    axes[0].set_xlabel('Quality Score')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=0)

    corr = df.corr(numeric_only=True)
    sns.heatmap(
        corr,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        ax=axes[1],
        linewidths=0.5,
        annot_kws={'size': 7},
    )
    axes[1].set_title('Feature Correlation Matrix')

    plt.tight_layout()
    plt.savefig('eda_plots.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Plot saved → eda_plots.png")


def preprocess_and_engineer(df: pd.DataFrame):
    print("\n=== a) Missing Data Handling ===")
    print(f"Missing values BEFORE imputation: {df.isnull().sum().sum()}")

    imputer = SimpleImputer(strategy='median')
    df_imp = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    print(f"Missing values AFTER  imputation: {df_imp.isnull().sum().sum()}")

    print("\n=== b) Feature Engineering ===")
    df_eng = df_imp.copy()

    skewed = [
        'residual sugar',
        'chlorides',
        'free sulfur dioxide',
        'total sulfur dioxide',
        'sulphates',
    ]
    for feat in skewed:
        df_eng[f'log_{feat.replace(" ", "_")}'] = np.log1p(df_imp[feat])
    print(f"Technique 1 – log-transformed {len(skewed)} skewed features.")

    df_eng['alcohol_x_sulphates'] = df_eng['alcohol'] * df_eng['sulphates']
    df_eng['volatile_to_citric'] = (
        df_eng['volatile acidity'] / (df_eng['citric acid'] + 1e-3)
    )
    df_eng['total_acidity'] = (
        df_eng['fixed acidity']
        + df_eng['volatile acidity']
        + df_eng['citric acid']
    )
    print(
        'Technique 2 – added interaction features: '
        'alcohol_x_sulphates, volatile_to_citric, total_acidity'
    )
    print(f"Total features after engineering: {df_eng.shape[1]}")

    df_eng['quality_binary'] = (df_eng['quality'] >= 6).astype(int)
    X_all = df_eng.drop(columns=['quality', 'quality_binary'])
    y = df_eng['quality_binary']

    print("\nBinary target class distribution:")
    print(y.value_counts().rename({0: 'Poor (0)', 1: 'Good (1)'}))
    print(f"Class ratio (Good/Poor): {y.mean():.2%}")

    return df_eng, X_all, y


def select_features(X_all: pd.DataFrame, y: pd.Series):
    print("\n=== c) Feature Selection ===")

    scaler_fs = StandardScaler()
    X_scaled_all = scaler_fs.fit_transform(X_all)

    selector = SelectKBest(score_func=f_classif, k=10)
    selector.fit(X_scaled_all, y)

    feat_scores = pd.DataFrame(
        {
            'Feature': X_all.columns,
            'F-Score': selector.scores_,
            'P-value': selector.pvalues_,
        }
    ).sort_values('F-Score', ascending=False)

    print("\nFeature Ranking (top 10 selected):")
    print(feat_scores.to_string(index=False))

    selected_features = X_all.columns[selector.get_support()].tolist()
    print(f"\nSelected features: {selected_features}")

    plt.figure(figsize=(10, 5))
    top15 = feat_scores.head(15).sort_values('F-Score')
    colors = [
        'teal' if f in selected_features else 'lightgray'
        for f in top15['Feature']
    ]
    plt.barh(top15['Feature'], top15['F-Score'], color=colors)
    plt.xlabel('ANOVA F-Score')
    plt.title(
        'Feature Importance (SelectKBest – ANOVA F-test)\n'
        'Teal = selected, Gray = excluded'
    )
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Plot saved → feature_importance.png")

    X = selector.transform(X_scaled_all)
    print(f"\nFinal feature matrix: {X.shape}")
    print(f"Class distribution  : {pd.Series(y.values).value_counts().to_dict()}")

    return X, selector, selected_features, feat_scores


def train_logistic_regression(X_train, y_train, X_test, y_test, kfold):
    print("\n=== Logistic Regression ===")

    smote = SMOTE(random_state=42)
    X_tr_sm, y_tr_sm = smote.fit_resample(X_train, y_train)

    print("Class distribution BEFORE SMOTE:", pd.Series(y_train).value_counts().to_dict())
    print("Class distribution AFTER  SMOTE:", pd.Series(y_tr_sm).value_counts().to_dict())

    print("\n=== a) 5-Fold Cross-Validation – Logistic Regression ===")
    lr_base = LogisticRegression(random_state=42, max_iter=1000)
    cv_scores_lr = cross_val_score(
        lr_base, X_tr_sm, y_tr_sm, cv=kfold, scoring='accuracy'
    )
    print(f"CV fold scores : {np.round(cv_scores_lr, 4)}")
    print(f"Mean accuracy  : {cv_scores_lr.mean():.4f}  (±{cv_scores_lr.std() * 2:.4f})")

    print("\n=== b) Grid Search – Logistic Regression ===")
    lr_param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear'],
    }
    lr_gs = GridSearchCV(
        LogisticRegression(random_state=42, max_iter=1000),
        lr_param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1,
    )
    lr_gs.fit(X_tr_sm, y_tr_sm)
    print(f"Best parameters : {lr_gs.best_params_}")
    print(f"Best CV F1-score: {lr_gs.best_score_:.4f}")

    lr_best = lr_gs.best_estimator_

    print("\n=== d) Overfitting Check – Logistic Regression ===")
    train_sz_lr, tr_scores_lr, val_scores_lr = learning_curve(
        lr_best,
        X_tr_sm,
        y_tr_sm,
        cv=5,
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy',
    )
    tr_mean_lr = tr_scores_lr.mean(axis=1)
    val_mean_lr = val_scores_lr.mean(axis=1)
    gap_lr = tr_mean_lr[-1] - val_mean_lr[-1]

    print(f"Final train accuracy     : {tr_mean_lr[-1]:.4f}")
    print(f"Final validation accuracy: {val_mean_lr[-1]:.4f}")
    print(
        f"Train–Val gap            : {gap_lr:.4f}  "
        f"({'⚠️ possible overfit' if gap_lr > 0.05 else '✅ no significant overfit'})"
    )

    y_pred_lr = lr_best.predict(X_test)
    y_prob_lr = lr_best.predict_proba(X_test)[:, 1]

    print("\n=== Logistic Regression – Classification Report ===")
    print(classification_report(y_test, y_pred_lr, target_names=['Poor', 'Good']))

    return {
        'model': lr_best,
        'cv_scores': cv_scores_lr,
        'y_pred': y_pred_lr,
        'y_prob': y_prob_lr,
        'train_curve': tr_mean_lr,
        'val_curve': val_mean_lr,
    }


def train_svm(X_train, y_train, X_test, y_test, kfold):
    print("\n=== Support Vector Machine (SVM) ===")

    print("\n=== a) 5-Fold Cross-Validation – SVM ===")
    svm_base = SVC(
        kernel='rbf',
        random_state=42,
        class_weight='balanced',
        probability=True,
    )
    cv_scores_svm = cross_val_score(
        svm_base, X_train, y_train, cv=kfold, scoring='accuracy'
    )
    print(f"CV fold scores : {np.round(cv_scores_svm, 4)}")
    print(f"Mean accuracy  : {cv_scores_svm.mean():.4f}  (±{cv_scores_svm.std() * 2:.4f})")

    print("\n=== b) Grid Search – SVM ===")
    svm_param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto'],
        'kernel': ['linear', 'rbf'],
    }
    svm_gs = GridSearchCV(
        SVC(random_state=42, class_weight='balanced', probability=True),
        svm_param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1,
    )
    svm_gs.fit(X_train, y_train)
    print(f"Best parameters : {svm_gs.best_params_}")
    print(f"Best CV F1-score: {svm_gs.best_score_:.4f}")

    svm_best = svm_gs.best_estimator_

    print("\n=== d) Overfitting Check – SVM ===")
    train_sz_svm, tr_scores_svm, val_scores_svm = learning_curve(
        svm_best,
        X_train,
        y_train,
        cv=5,
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy',
    )
    tr_mean_svm = tr_scores_svm.mean(axis=1)
    val_mean_svm = val_scores_svm.mean(axis=1)
    gap_svm = tr_mean_svm[-1] - val_mean_svm[-1]

    print(f"Final train accuracy     : {tr_mean_svm[-1]:.4f}")
    print(f"Final validation accuracy: {val_mean_svm[-1]:.4f}")
    print(
        f"Train–Val gap            : {gap_svm:.4f}  "
        f"({'⚠️ possible overfit' if gap_svm > 0.05 else '✅ no significant overfit'})"
    )

    y_pred_svm = svm_best.predict(X_test)
    y_prob_svm = svm_best.predict_proba(X_test)[:, 1]

    print("\n=== SVM – Classification Report ===")
    print(classification_report(y_test, y_pred_svm, target_names=['Poor', 'Good']))

    return {
        'model': svm_best,
        'cv_scores': cv_scores_svm,
        'y_pred': y_pred_svm,
        'y_prob': y_prob_svm,
        'train_curve': tr_mean_svm,
        'val_curve': val_mean_svm,
    }


def evaluate_model(y_test, y_pred, y_prob):
    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    return {
        'confusion_matrix': cm,
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'fpr': fpr,
        'tpr': tpr,
        'roc_auc': roc_auc,
    }


def create_result_plots(y_test, lr_eval, svm_eval, lr_results, svm_results):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ConfusionMatrixDisplay(
        confusion_matrix=lr_eval['confusion_matrix'],
        display_labels=['Poor', 'Good'],
    ).plot(ax=axes[0, 0], colorbar=False)
    axes[0, 0].set_title('Logistic Regression Confusion Matrix')

    ConfusionMatrixDisplay(
        confusion_matrix=svm_eval['confusion_matrix'],
        display_labels=['Poor', 'Good'],
    ).plot(ax=axes[0, 1], colorbar=False)
    axes[0, 1].set_title('SVM Confusion Matrix')

    axes[1, 0].plot(lr_eval['fpr'], lr_eval['tpr'], label=f"LR (AUC = {lr_eval['roc_auc']:.4f})")
    axes[1, 0].plot(svm_eval['fpr'], svm_eval['tpr'], label=f"SVM (AUC = {svm_eval['roc_auc']:.4f})")
    axes[1, 0].plot([0, 1], [0, 1], 'k--')
    axes[1, 0].set_title('ROC Curves')
    axes[1, 0].set_xlabel('False Positive Rate')
    axes[1, 0].set_ylabel('True Positive Rate')
    axes[1, 0].legend()

    x_lr = np.linspace(0.1, 1.0, 10)
    axes[1, 1].plot(x_lr, lr_results['train_curve'], label='LR Train')
    axes[1, 1].plot(x_lr, lr_results['val_curve'], label='LR Validation')
    axes[1, 1].plot(x_lr, svm_results['train_curve'], label='SVM Train')
    axes[1, 1].plot(x_lr, svm_results['val_curve'], label='SVM Validation')
    axes[1, 1].set_title('Learning Curves')
    axes[1, 1].set_xlabel('Training Size Fraction')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig('classification_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Plot saved → classification_results.png")

    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    lr_vals = [
        lr_eval['accuracy'],
        lr_eval['precision'],
        lr_eval['recall'],
        lr_eval['f1'],
        lr_eval['roc_auc'],
    ]
    svm_vals = [
        svm_eval['accuracy'],
        svm_eval['precision'],
        svm_eval['recall'],
        svm_eval['f1'],
        svm_eval['roc_auc'],
    ]

    x = np.arange(len(metrics))
    width = 0.35
    plt.figure(figsize=(10, 5))
    plt.bar(x - width / 2, lr_vals, width, label='Logistic Regression')
    plt.bar(x + width / 2, svm_vals, width, label='SVM')
    plt.xticks(x, metrics)
    plt.ylim(0.85, 1.0)
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.legend()
    plt.tight_layout()
    plt.savefig('comparison_chart.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Plot saved → comparison_chart.png")


def print_summary(lr_eval, svm_eval):
    print("\n=== Final Model Comparison ===")
    summary_df = pd.DataFrame(
        {
            'Metric': ['Accuracy', 'Precision (macro)', 'Recall (macro)', 'F1-Score (macro)', 'ROC-AUC'],
            'Logistic Regression': [
                lr_eval['accuracy'],
                lr_eval['precision'],
                lr_eval['recall'],
                lr_eval['f1'],
                lr_eval['roc_auc'],
            ],
            'SVM': [
                svm_eval['accuracy'],
                svm_eval['precision'],
                svm_eval['recall'],
                svm_eval['f1'],
                svm_eval['roc_auc'],
            ],
        }
    )
    print(summary_df.to_string(index=False))


if __name__ == '__main__':
    print('✅ All packages loaded successfully.')

    df = load_dataset()
    create_eda_plots(df)

    _, X_all, y = preprocess_and_engineer(df)
    X, selector, selected_features, feat_scores = select_features(X_all, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTraining set : {X_train.shape}  | class dist → {pd.Series(y_train).value_counts().to_dict()}")
    print(f"Test set     : {X_test.shape}  | class dist → {pd.Series(y_test).value_counts().to_dict()}")

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    lr_results = train_logistic_regression(X_train, y_train, X_test, y_test, kfold)
    svm_results = train_svm(X_train, y_train, X_test, y_test, kfold)

    lr_eval = evaluate_model(y_test, lr_results['y_pred'], lr_results['y_prob'])
    svm_eval = evaluate_model(y_test, svm_results['y_pred'], svm_results['y_prob'])

    create_result_plots(y_test, lr_eval, svm_eval, lr_results, svm_results)
    print_summary(lr_eval, svm_eval)

    print("\nGenerated files:")
    print("- eda_plots.png")
    print("- feature_importance.png")
    print("- classification_results.png")
    print("- comparison_chart.png")
