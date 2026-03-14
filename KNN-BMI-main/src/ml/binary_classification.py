# ================================================================================================
# IMPORTS
# ================================================================================================
from collections import Counter
import os
import pandas as pd
import numpy as np
import yaml
import joblib
import pickle

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, label_binarize, OneHotEncoder, OrdinalEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_recall_curve, auc, 
    roc_curve, confusion_matrix, average_precision_score, precision_score, 
    recall_score, classification_report, balanced_accuracy_score
)
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier
from catboost import CatBoostClassifier 

import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imbens.ensemble import SelfPacedEnsembleClassifier

# ================================================================================================
# GLOBAL SETTINGS
# ================================================================================================
PREFIX = "normalhigh"
INIT = "rfe_ml"  # "multivariate_columns"
SEED = 2025

# ================================================================================================
# UTILITY FUNCTIONS - DATA LOADING
# ================================================================================================

def load_config(config_path="./conf/data/features.yaml"):
    """설정 파일 로딩"""
    return yaml.safe_load(open(config_path, "r", encoding="utf-8"))


def load_and_prepare_data(train_file, test_file, config, init, labels_to_keep=[0, 2]):
    """데이터 로딩 및 전처리"""
    print("=" * 80)
    print("📂 데이터 로딩 중...")
    print("=" * 80)
    
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)

    print(f"\n원본 레이블 분포:")
    print(f"Train: {train_data['label'].value_counts().to_dict()}")
    print(f"Test: {test_data['label'].value_counts().to_dict()}")
    
    # 레이블 필터링
    train_data = train_data[train_data["label"].isin(labels_to_keep)]
    test_data = test_data[test_data["label"].isin(labels_to_keep)]
    
    print(f"\n필터링 후 레이블 분포:")
    print(f"Train: {train_data['label'].value_counts().to_dict()}")
    print(f"Test: {test_data['label'].value_counts().to_dict()}")
    print(f"\n데이터 Shape: Train={train_data.shape}, Test={test_data.shape}")
    
    # Feature 정의
    numeric_columns = config[init]['continuous']
    categorical_vars = config[init]['category']
    target = config['derived_columns']["label"]
    features = numeric_columns + categorical_vars + target

    # X, y 분리
    x_train = train_data[features].copy()
    y_train = x_train[target].copy().replace(2, 1).squeeze()
    x_train = x_train.drop(columns=target)

    x_test = test_data[features].copy()
    y_test = x_test[target].copy().replace(2, 1).squeeze()
    x_test = x_test.drop(columns=target)

        # 스케일 조정 (bwei를 kg 단위로)
    if "bwei" in x_train.columns:
        x_train["bwei"] = x_train["bwei"] / 1000
    if "bwei" in x_test.columns:
        x_test["bwei"] = x_test["bwei"] / 1000

        print(f"\n최종 데이터 Shape: X_train={x_train.shape}, X_test={x_test.shape}")
        print(f"Null 값 확인:\n{x_train.isnull().sum().to_dict()}")
    
    return x_train, x_test, y_train, y_test, numeric_columns, categorical_vars

# ================================================================================================
# MODEL DEFINITION
# ================================================================================================

def get_models(y_train, seed=2025, smote_strategy=0.3, scale_pos_weight=3):
    """모델 파이프라인 정의
    
    Args:
        y_train: 학습 레이블
        seed: 랜덤 시드
        smote_strategy: SMOTE 샘플링 비율
        scale_pos_weight: XGBoost/CatBoost용 클래스 가중치
    """
    smote_sampler = SMOTE(sampling_strategy=smote_strategy, random_state=seed)
    
    # 클래스 분포 확인
    class_counts = Counter(y_train)
    print(f"\n클래스 분포: {dict(class_counts)}")
    print(f"scale_pos_weight: {scale_pos_weight}")
    
    models_params = {
        "LogReg_L2": {
                'model': ImbPipeline(steps=[
                    ('smote', smote_sampler),
                ('classifier', LogisticRegression(
                    penalty='l2', 
                    solver='sag', 
                        random_state=seed, 
                        class_weight='balanced',
                    max_iter=500
                ))
            ])
        },
        
        "SVM_RBF": {
                'model': ImbPipeline(steps=[
                ('smote', smote_sampler),
                ('classifier', SVC(
                    kernel='rbf', 
                        random_state=seed, 
                        class_weight='balanced',
                    probability=True, 
                    C=50
                ))
            ])
        },
        
        "RandomForest": {
                'model': ImbPipeline(steps=[
                ('smote', smote_sampler),
                ('classifier', RandomForestClassifier(
                        random_state=seed, 
                        class_weight='balanced',
                    criterion='entropy', 
                    n_estimators=2000, 
                    max_depth=8, 
                    max_features='log2', 
                    bootstrap=True
                ))
            ])
        },
        
        'XGBoost': {
                'model': ImbPipeline(steps=[
                ('smote', smote_sampler),
                ('classifier', XGBClassifier(
                        random_state=seed, 
                        scale_pos_weight=scale_pos_weight,
                    n_jobs=-1, 
                    eval_metric='logloss', 
                    n_estimators=2000, 
                    eta=0.001
                ))
            ])
        },
        
        'CatBoost': {
                'model': ImbPipeline(steps=[
                ('smote', smote_sampler),
                ('classifier', CatBoostClassifier(
                        random_state=seed, 
                        scale_pos_weight=scale_pos_weight,
                    verbose=0, 
                    n_estimators=2000, 
                    bootstrap_type='MVS', 
                    learning_rate=0.001, 
                    thread_count=-1
                ))
            ])
        },
        
        'HistGradBoost': {
                'model': ImbPipeline(steps=[
                ('smote', smote_sampler),
                ('classifier', HistGradientBoostingClassifier(
                        random_state=seed, 
                        class_weight='balanced',
                    max_iter=2000, 
                    learning_rate=0.001, 
                    max_depth=8
                ))
            ])
            },
        
        'SelfPacedEnsemble': {
            'model': SelfPacedEnsembleClassifier(
                n_estimators=50,
                k_bins=5,
                random_state=seed,
                n_jobs=-1
            )
        }
        }
    
    return models_params

# ================================================================================================
# THRESHOLD OPTIMIZATION
# ================================================================================================

def find_optimal_thresholds(y_true, y_prob):
    """여러 방법으로 최적 threshold 찾기
    
    Returns:
        dict: Youden, F1, Balanced Accuracy, Clinical (Sensitivity≥80%) threshold
    """
    fpr, tpr, thresholds_roc = roc_curve(y_true, y_prob)
    precision, recall, thresholds_pr = precision_recall_curve(y_true, y_prob)
    
    # 1. Youden's Index (Sensitivity + Specificity - 1 최대화)
    youden_idx = np.argmax(tpr - fpr)
    threshold_youden = thresholds_roc[youden_idx]
    
    # 2. F1-score 최대화
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    f1_idx = np.argmax(f1_scores)
    threshold_f1 = thresholds_pr[f1_idx]
    
    # 3. Balanced Accuracy 최대화
    best_ba = 0
    threshold_ba = 0.5
    for threshold in np.arange(0.1, 0.9, 0.01):
        y_pred = (y_prob >= threshold).astype(int)
        ba = balanced_accuracy_score(y_true, y_pred)
        if ba > best_ba:
            best_ba = ba
            threshold_ba = threshold
    
    # 4. 임상 목표: Sensitivity >= 0.80
    target_sensitivity = 0.80
    idx_sens = np.where(tpr >= target_sensitivity)[0]
    if len(idx_sens) > 0:
        threshold_clinical = thresholds_roc[idx_sens[0]]
        clinical_specificity = 1 - fpr[idx_sens[0]]
    else:
        threshold_clinical = 0.5
        clinical_specificity = 0
    
    return {
        'youden': threshold_youden,
        'f1': threshold_f1,
        'balanced_accuracy': threshold_ba,
        'clinical_sens80': threshold_clinical,
        'clinical_specificity': clinical_specificity
    }


def evaluate_with_threshold(y_true, y_prob, threshold, threshold_name="Custom"):
    """특정 threshold로 평가"""
    y_pred = (y_prob >= threshold).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    sensitivity = recall_score(y_true, y_pred, pos_label=1)  # TPR
    specificity = recall_score(y_true, y_pred, pos_label=0)  # TNR
    ppv = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    return {
        'name': threshold_name,
        'threshold': threshold,
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
        'sensitivity': sensitivity,
        'specificity': specificity,
        'ppv': ppv,
        'npv': npv,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn
    }


# ================================================================================================
# EVALUATION FUNCTIONS
# ================================================================================================

def evaluate_model(model, model_name, x_test, y_test):
    """모델 평가 (통합 버전)
    
    Args:
        model: 학습된 모델 또는 파이프라인
        model_name: 모델 이름
        x_test: 테스트 피처
        y_test: 테스트 레이블
        
    Returns:
        dict: 평가 메트릭
    """
    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)
    
    # 이진 분류용 확률
    if len(np.unique(y_test)) == 2 and y_prob.shape[1] == 2:
        y_prob_pos = y_prob[:, 1]
    else:
        y_prob_pos = y_prob

    accuracy = accuracy_score(y_test, y_pred)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    micro_f1 = f1_score(y_test, y_pred, average='micro')

    precision_macro = precision_score(y_test, y_pred, average="macro")
    recall_macro = recall_score(y_test, y_pred, average="macro")
    precision_micro = precision_score(y_test, y_pred, average="micro")
    recall_micro = recall_score(y_test, y_pred, average="micro")
    precision_weighted = precision_score(y_test, y_pred, average="weighted")
    recall_weighted = recall_score(y_test, y_pred, average="weighted")

    precision, recall, _ = precision_recall_curve(y_test, y_prob_pos)
    auprc = auc(recall, precision)
    auroc = roc_auc_score(y_test, y_prob_pos)
    
    conf_matrix = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob_pos)

    print(f"\n📊 Model: {model_name}")
    print(f"✔ Accuracy: {accuracy:.4f}")
    print(f"✔ Weighted F1: {weighted_f1:.4f}")
    print(f"✔ AUROC: {auroc:.4f}")
    print(f"✔ AUPRC: {auprc:.4f}")

    return {
        "model": model_name,
        "accuracy": accuracy,
        "weighted_f1_score": weighted_f1,
        "macro_f1_score": macro_f1,
        "micro_f1_score": micro_f1,
        "precision_macro": precision_macro,
        "precision_micro": precision_micro,
        "precision_weighted": precision_weighted,
        "recall_macro": recall_macro,
        "recall_micro": recall_micro,
        "recall_weighted": recall_weighted,
        "auroc": auroc,
        "auprc": auprc,
        "fpr": fpr,
        "tpr": tpr,
        "precision": precision,
        "recall": recall,
        "conf_matrix": conf_matrix
    }




# ================================================================================================
# CALIBRATION METRICS
# ================================================================================================

def compute_binary_calibration_errors(y_true, y_prob, n_bins=10):
    """이진 분류 ECE, MCE 계산"""
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    n_samples = len(y_true)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        bin_indices = np.where((y_prob > bin_lower) & (y_prob <= bin_upper))[0]
        if len(bin_indices) > 0:
            bin_accuracy = np.mean(y_true[bin_indices])
            bin_confidence = np.mean(y_prob[bin_indices])
            abs_diff = abs(bin_accuracy - bin_confidence)
            ece += (len(bin_indices) / n_samples) * abs_diff
            mce = max(mce, abs_diff)
    return ece, mce


def compute_brier_score(y_true, y_prob):
    """이진 분류 Brier Score"""
    return np.mean((y_true - y_prob) ** 2)


# ================================================================================================
# VISUALIZATION FUNCTIONS
# ================================================================================================

def plot_roc_pr_curves(evaluation_results, title_prefix=""):
    """ROC 및 Precision-Recall Curve 그리기"""
    plt.figure(figsize=(14, 6))
    
    # ROC Curve
    ax1 = plt.subplot(1, 2, 1)
    for res in evaluation_results:
        ax1.plot(res["fpr"], res["tpr"],
                label=f"{res['model']} (AUROC={res['auroc']:.2f})")
    ax1.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax1.set_title(f"ROC Curve {title_prefix}")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.legend(loc="lower right")
        
    # PR Curve
    ax2 = plt.subplot(1, 2, 2)
    for res in evaluation_results:
        ax2.plot(res["recall"], res["precision"],
                label=f"{res['model']} (AUPRC={res['auprc']:.2f})")
        ax2.set_title(f"Precision-Recall Curve {title_prefix}")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.legend(loc="lower left")
        
    plt.tight_layout()
    plt.show()


def plot_calibration_curves(models_dict, x_test, y_test, title_prefix=""):
    """Calibration Curve 그리기
    
    Args:
        models_dict: {model_name: model} 딕셔너리
        x_test: 테스트 피처
        y_test: 테스트 레이블
    """
    plt.figure(figsize=(8, 6))

    for model_name, model in models_dict.items():
        try:
            prob_pos = model.predict_proba(x_test)[:, 1]
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_test, prob_pos, n_bins=10, strategy='quantile'
            )
            plt.plot(mean_predicted_value, fraction_of_positives, 'o-', label=model_name)
        except Exception as e:
            print(f"⚠️ {model_name} calibration curve 오류: {e}")

    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    plt.title(f"Calibration Curves {title_prefix}")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()


def plot_confusion_matrices(conf_matrix_dict, title="Confusion Matrices"):
    """Confusion Matrix 시각화
    
    Args:
        conf_matrix_dict: {model_name: confusion_matrix} 딕셔너리
    """
    model_names = list(conf_matrix_dict.keys())
    n_models = len(model_names)

    if n_models == 0:
        print("No models found in conf_matrix_dict.")
        return

    # 서브플롯 레이아웃
    ncols = min(n_models, 3)
    nrows = (n_models - 1) // ncols + 1

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                                figsize=(5 * ncols + 2, 5 * nrows + 2), dpi=300)

    if n_models == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()

    for ax, model_name in zip(axes, model_names):
        matrix = np.array(conf_matrix_dict[model_name])

        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            print(f"⚠️ {model_name}: 잘못된 matrix shape {matrix.shape}")
            continue
        
        n_classes = matrix.shape[0]
        labels = [str(i) for i in range(n_classes)]
            
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".0f",
            cmap="Reds",
                xticklabels=labels,
                yticklabels=labels,
            annot_kws={"size": 20},
            cbar=True,
            ax=ax
        )

        ax.set_title(model_name, fontsize=14, fontweight="bold")
        ax.set_xlabel("Predicted", fontsize=14)
        ax.set_ylabel("Actual", fontsize=14)

        # 남는 축 제거
    for i in range(len(model_names), len(axes)):
        fig.delaxes(axes[i])

        plt.suptitle(title, fontsize=18, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.show()


def plot_feature_importance(importance_file, config, top_n=10):
    """Feature Importance 시각화
    
    Args:
        importance_file: feature importance CSV 파일 경로
        config: YAML 설정
        top_n: 상위 몇 개 feature 표시
    """
    df = pd.read_csv(importance_file)
    column_map = config.get('column_map', {})
    
    df = df.sort_values(by=["Model", "Importance"], ascending=[True, False])
    df['feature_name'] = df["Feature"].map(column_map).fillna(df["Feature"])
    df['Importance_norm'] = df.groupby('Model')['Importance'].transform(
        lambda x: x / x.sum()
    )

    models = df["Model"].unique()

    fig, axes = plt.subplots(len(models), 1,
                            figsize=(15, 5 * len(models)),
                                sharex=False, dpi=300)

    if len(models) == 1:
        axes = [axes]

    for i, model in enumerate(models):
        ax = axes[i]
        subset = df[df["Model"] == model].head(top_n)

        sns.barplot(data=subset,
                    y="feature_name", x="Importance_norm",
                    ax=ax, palette="viridis", dodge=False)

        ax.set_xlabel("")
        ax.set_ylabel(model, fontsize=14, fontweight="bold", rotation=90)
        ax.tick_params(axis='y', labelsize=12)

    plt.suptitle("Feature Importance by Model", fontsize=18, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


# ================================================================================================
# MAIN EXECUTION
# ================================================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("🚀 이진 분류 모델 학습 및 평가 시작")
    print("="*80)
    
    # === 1. 설정 및 데이터 로딩 ===
    config = load_config()
    train_file = "./data/split/train_ml_dataset.csv"
    test_file = "./data/split/test_ml_dataset.csv"
    
    x_train, x_test, y_train, y_test, numeric_cols, cat_cols = load_and_prepare_data(
        train_file, test_file, config, INIT
    )
    
    # === 2. 모델 정의 ===
    models_params = get_models(y_train, seed=SEED)
    
    # === 3. 모델 학습 및 평가 ===
    results = []
    models_best = []
    models_best_name = []
    models_calibrated = []  # Calibrated 모델 저장
    models_calibrated_name = []
    threshold_results = []
    
    for name, params in models_params.items():
        print(f"\n{'='*60}")
        print(f"🔹 Training {name}")
        print(f"{'='*60}")
        
        # 모델 학습
        model = params['model']
        model.fit(x_train, y_train)
        joblib.dump(model, f'./model/pipeline_{name}_{PREFIX}.pkl')
        
        # Calibration 적용 (Platt Scaling)
        print(f"🔧 Applying Calibration (Platt Scaling)...")
        calibrated_model_platt = CalibratedClassifierCV(
            model, 
            method='sigmoid',  # Platt scaling
            cv='prefit'  # 이미 학습된 모델 사용
        )
        calibrated_model_platt.fit(x_train, y_train)
        joblib.dump(calibrated_model_platt, f'./model/pipeline_{name}_calibrated_platt_{PREFIX}.pkl')
        
        # Calibration 적용 (Isotonic Regression)
        print(f"🔧 Applying Calibration (Isotonic Regression)...")
        calibrated_model_isotonic = CalibratedClassifierCV(
            model, 
            method='isotonic',  # Isotonic regression (비선형)
            cv='prefit'
        )
        calibrated_model_isotonic.fit(x_train, y_train)
        joblib.dump(calibrated_model_isotonic, f'./model/pipeline_{name}_calibrated_isotonic_{PREFIX}.pkl')
        
        # Calibrated 모델 저장
        models_calibrated.extend([calibrated_model_platt, calibrated_model_isotonic])
        models_calibrated_name.extend([f"{name}_Calibrated_Platt", f"{name}_Calibrated_Isotonic"])
        
        # 예측
        y_pred_default = model.predict(x_test)
        y_pred_proba = model.predict_proba(x_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        print(f"\n📊 {name} - ROC AUC: {auc_score:.4f}")
        print(f"📋 Classification Report (Default threshold=0.5):")
        print(classification_report(y_test, y_pred_default))
        
        # Threshold 최적화
        print(f"\n🔍 Finding Optimal Thresholds...")
        optimal_thresholds = find_optimal_thresholds(y_test, y_pred_proba)
        
        print(f"\n📌 Optimal Thresholds:")
        print(f"  Youden's Index:     {optimal_thresholds['youden']:.3f}")
        print(f"  F1-score Max:       {optimal_thresholds['f1']:.3f}")
        print(f"  Balanced Accuracy:  {optimal_thresholds['balanced_accuracy']:.3f}")
        print(f"  Clinical (Sens≥80%): {optimal_thresholds['clinical_sens80']:.3f}")
        
        # 각 threshold로 평가
        threshold_strategies = [
            ('Default (0.5)', 0.5),
            ('Youden', optimal_thresholds['youden']),
            ('F1-Max', optimal_thresholds['f1']),
            ('Balanced-Acc', optimal_thresholds['balanced_accuracy']),
            ('Clinical-Sens80', optimal_thresholds['clinical_sens80'])
        ]
        
        print(f"\n📊 Performance with Different Thresholds:")
        print(f"{'Strategy':<20} {'Threshold':<10} {'Accuracy':<10} {'Bal-Acc':<10} {'Sens':<10} {'Spec':<10} {'F1':<10}")
        print("-" * 90)
        
        for strategy_name, threshold in threshold_strategies:
            result = evaluate_with_threshold(y_test, y_pred_proba, threshold, strategy_name)
            
            print(f"{result['name']:<20} {result['threshold']:<10.3f} "
                  f"{result['accuracy']:<10.3f} {result['balanced_accuracy']:<10.3f} "
                  f"{result['sensitivity']:<10.3f} {result['specificity']:<10.3f} "
                  f"{result['f1_score']:<10.3f}")
            
            result['model'] = name
            result['auroc'] = auc_score
            threshold_results.append(result)
        
        # 최적 threshold 저장
        best_threshold = optimal_thresholds['f1']
        threshold_info = {
            'model_name': name,
            'optimal_threshold': best_threshold,
            'all_thresholds': optimal_thresholds
        }
        
        with open(f'./model/threshold_{name}_{PREFIX}.pkl', 'wb') as f:
            pickle.dump(threshold_info, f)
        
        models_best.append(model)
        models_best_name.append(name)
        results.append({
            'model_name': name, 
            'pipeline': model, 
            'auc': auc_score,
            'optimal_threshold': best_threshold
        })
        
        print("-" * 60)
    
    # === 4. Threshold 결과 저장 ===
    df_threshold_results = pd.DataFrame(threshold_results)
    df_threshold_results.to_csv(f'threshold_comparison_{PREFIX}.csv', index=False)
    
    print("\n" + "="*80)
    print("📊 THRESHOLD OPTIMIZATION SUMMARY")
    print("="*80)
    
    for model in df_threshold_results['model'].unique():
        model_data = df_threshold_results[
            (df_threshold_results['model'] == model) & 
            (df_threshold_results['name'] == 'F1-Max')
        ].iloc[0]
        
        print(f"\n  {model}:")
        print(f"    Threshold: {model_data['threshold']:.3f}")
        print(f"    Sensitivity: {model_data['sensitivity']:.3f}")
        print(f"    Specificity: {model_data['specificity']:.3f}")
        print(f"    F1-Score: {model_data['f1_score']:.3f}")
    
    # Best 모델 저장
    if results:
        best_result = max(results, key=lambda x: x['auc'])
        joblib.dump(best_result['pipeline'], f'./model/best_pipeline_{PREFIX}.pkl')
        print(f"\n✅ Best Model: {best_result['model_name']} (AUC: {best_result['auc']:.4f})")
    
    # === 5. 모델 평가 및 Feature Importance 추출 ===
    print("\n" + "="*80)
    print("📊 모델 상세 평가 중 (원본 + Calibrated)...")
    print("="*80)
    
    evaluation_results = []
    conf_matrix_dict = {}
    feature_importance_list = []
    
    # 원본 + Calibrated 모델 통합
    all_models = models_best + models_calibrated
    all_model_names = models_best_name + models_calibrated_name
    models_dict = {name: model for name, model in zip(all_model_names, all_models)}
    
    for model, model_name in zip(all_models, all_model_names):
        res = evaluate_model(model, model_name, x_test, y_test)
        evaluation_results.append(res)
        conf_matrix_dict[model_name] = res["conf_matrix"]
        
        # Feature Importance 추출 (원본 모델만)
        if "Calibrated" not in model_name:
            try:
                classifier = model.named_steps['classifier']
                
                if hasattr(classifier, "feature_importances_"):
                    feature_names = x_train.columns.tolist()
                    feature_importances = classifier.feature_importances_
                    
                    for feature, importance in zip(feature_names, feature_importances):
                        feature_importance_list.append({
                            "Model": model_name,
                            "Feature": feature,
                            "Importance": importance
                        })
                
                elif hasattr(classifier, "coef_"):
                    feature_names = x_train.columns.tolist()
                    coef = np.abs(classifier.coef_).flatten()
                    
                    for feature, importance in zip(feature_names, coef):
                        feature_importance_list.append({
                            "Model": model_name,
                            "Feature": feature,
                            "Importance": importance
                        })
            except Exception as e:
                print(f"⚠️ {model_name} feature importance 추출 실패: {e}")
    
    # === 6. 평가 메트릭 저장 ===
    scalar_metrics = []
    for res in evaluation_results:
        scalar_metrics.append({
            "model": res["model"],
            "auroc": res["auroc"],
            "auprc": res["auprc"],
            "accuracy": res["accuracy"],
            "weighted_f1_score": res["weighted_f1_score"],
            "macro_f1_score": res["macro_f1_score"],
            "micro_f1_score": res["micro_f1_score"],
            "precision_macro": res["precision_macro"],
            "precision_micro": res["precision_micro"],
            "precision_weighted": res["precision_weighted"],
            "recall_macro": res["recall_macro"],
            "recall_micro": res["recall_micro"],
            "recall_weighted": res["recall_weighted"],
        })
    
    df_metrics = pd.DataFrame(scalar_metrics)
    df_metrics.to_csv(f"evaluation_metrics_{PREFIX}.csv", index=False)
    
    with open(f"conf_matrix_{PREFIX}.pkl", "wb") as f:
        pickle.dump(conf_matrix_dict, f)
    
    if feature_importance_list:
        df_feature_importance = pd.DataFrame(feature_importance_list)
        df_feature_importance.to_csv(f"feature_importance_{PREFIX}.csv", index=False)
        print("✅ Feature importances saved.")
    
    print("✅ Evaluation metrics saved.")
    
    # === 7. 시각화 ===
    print("\n" + "="*80)
    print("📈 시각화 생성 중...")
    print("="*80)
    
    # ROC & PR Curves
    plot_roc_pr_curves(evaluation_results, title_prefix="(Internal Validation)")
    
    # Calibration Curves
    plot_calibration_curves(models_dict, x_test, y_test, title_prefix="(Internal Validation)")
    
    # Confusion Matrices
    plot_confusion_matrices(conf_matrix_dict, title="Confusion Matrices (Internal Validation)")
    
    # Feature Importance
    if os.path.exists(f"feature_importance_{PREFIX}.csv"):
        plot_feature_importance(f"feature_importance_{PREFIX}.csv", config, top_n=10)
    
    # === 8. Calibration 메트릭 계산 (원본 + Calibrated) ===
    print("\n" + "="*80)
    print("🎯 Calibration 메트릭 계산 중 (원본 vs Calibrated 비교)...")
    print("="*80)
    
    calibration_results = []
    for model, model_name in zip(all_models, all_model_names):
        prob_pred = model.predict_proba(x_test)[:, 1]
        ece, mce = compute_binary_calibration_errors(y_test, prob_pred, n_bins=10)
        brier = compute_brier_score(y_test, prob_pred)
        
        calibration_results.append({
            'Model': model_name,
            'ECE': ece,
            'MCE': mce,
            'Brier Score': brier
        })
        
        # Calibrated 모델과 원본 비교
        model_type = "📊 Original" if "Calibrated" not in model_name else "🔧 Calibrated"
        print(f"{model_type} {model_name}:")
        print(f"  ECE: {ece:.4f}, MCE: {mce:.4f}, Brier: {brier:.4f}")
    
    df_calibration = pd.DataFrame(calibration_results)
    df_calibration.to_csv(f"calibration_metrics_{PREFIX}.csv", index=False)
    print("\n✅ Calibration metrics saved.")
    
    # Calibration 개선 효과 분석
    print("\n" + "="*80)
    print("📈 Calibration 개선 효과")
    print("="*80)
    
    for base_name in models_best_name:
        original = df_calibration[df_calibration['Model'] == base_name]
        platt = df_calibration[df_calibration['Model'] == f"{base_name}_Calibrated_Platt"]
        isotonic = df_calibration[df_calibration['Model'] == f"{base_name}_Calibrated_Isotonic"]
        
        if not original.empty and not platt.empty:
            print(f"\n🔹 {base_name}:")
            print(f"  Original ECE: {original['ECE'].values[0]:.4f}")
            print(f"  Platt ECE:    {platt['ECE'].values[0]:.4f} (Δ {platt['ECE'].values[0] - original['ECE'].values[0]:+.4f})")
            if not isotonic.empty:
                print(f"  Isotonic ECE: {isotonic['ECE'].values[0]:.4f} (Δ {isotonic['ECE'].values[0] - original['ECE'].values[0]:+.4f})")
    
    # === 9. 외부 검증 (옵션) ===
    external_file = "./data/external/external_validation_dataset.csv"
    if os.path.exists(external_file):
        print("\n" + "="*80)
        print("🌐 외부 검증 시작...")
        print("="*80)
        
        # 외부 데이터 로딩
        external_data = pd.read_csv(external_file)
        external_data = external_data[numeric_cols + cat_cols + [config['derived_columns']["label"][0]]]
        external_data = external_data.dropna()
        external_data = external_data[external_data['label'].isin([0, 2])]
        external_data['label'] = external_data['label'].replace(2, 1)
        
        x_external = external_data[numeric_cols + cat_cols]
        y_external = external_data['label']
        
        # bwei 스케일 조정
        if "bwei" in x_external.columns:
            x_external["bwei"] = x_external["bwei"] / 1000
        
        print(f"외부 검증 데이터 Shape: {x_external.shape}")
        
        # 모델 평가 (원본 + Calibrated)
        external_evaluation_results = []
        external_conf_matrix_dict = {}
        external_calibration_results = []
        
        for model, model_name in zip(all_models, all_model_names):
            res = evaluate_model(model, model_name, x_external, y_external)
            external_evaluation_results.append(res)
            external_conf_matrix_dict[model_name] = res["conf_matrix"]
            
            # External Calibration 메트릭
            prob_pred = model.predict_proba(x_external)[:, 1]
            ece, mce = compute_binary_calibration_errors(y_external, prob_pred, n_bins=10)
            brier = compute_brier_score(y_external, prob_pred)
            external_calibration_results.append({
                'Model': model_name,
                'ECE': ece,
                'MCE': mce,
                'Brier Score': brier
            })
        
        # 결과 저장
        external_scalar_metrics = []
        for res in external_evaluation_results:
            external_scalar_metrics.append({
                "model": res["model"],
                "auroc": res["auroc"],
                "auprc": res["auprc"],
                "accuracy": res["accuracy"],
                "weighted_f1_score": res["weighted_f1_score"],
                "macro_f1_score": res["macro_f1_score"],
                "micro_f1_score": res["micro_f1_score"],
                "precision_macro": res["precision_macro"],
                "recall_macro": res["recall_macro"],
            })
        
        df_external_metrics = pd.DataFrame(external_scalar_metrics)
        df_external_metrics.to_csv(f"evaluation_metrics_external_{PREFIX}.csv", index=False)
        
        # External Calibration 결과 저장
        df_external_calibration = pd.DataFrame(external_calibration_results)
        df_external_calibration.to_csv(f"calibration_metrics_external_{PREFIX}.csv", index=False)
        
        with open(f"conf_matrix_external_{PREFIX}.pkl", "wb") as f:
            pickle.dump(external_conf_matrix_dict, f)
        
        print("✅ External validation metrics saved.")
        
        # External Calibration 개선 효과 분석
        print("\n" + "="*80)
        print("📈 External Validation - Calibration 개선 효과")
        print("="*80)
        
        for base_name in models_best_name:
            original = df_external_calibration[df_external_calibration['Model'] == base_name]
            platt = df_external_calibration[df_external_calibration['Model'] == f"{base_name}_Calibrated_Platt"]
            isotonic = df_external_calibration[df_external_calibration['Model'] == f"{base_name}_Calibrated_Isotonic"]
            
            if not original.empty and not platt.empty:
                print(f"\n🔹 {base_name}:")
                print(f"  Original ECE: {original['ECE'].values[0]:.4f}")
                print(f"  Platt ECE:    {platt['ECE'].values[0]:.4f} (Δ {platt['ECE'].values[0] - original['ECE'].values[0]:+.4f})")
                if not isotonic.empty:
                    print(f"  Isotonic ECE: {isotonic['ECE'].values[0]:.4f} (Δ {isotonic['ECE'].values[0] - original['ECE'].values[0]:+.4f})")
                
                # 가장 개선된 방법 표시
                improvements = {
                    'Platt': original['ECE'].values[0] - platt['ECE'].values[0],
                }
                if not isotonic.empty:
                    improvements['Isotonic'] = original['ECE'].values[0] - isotonic['ECE'].values[0]
                
                best_method = max(improvements, key=improvements.get)
                best_improvement = improvements[best_method]
                
                if best_improvement > 0:
                    print(f"  ✅ Best: {best_method} (ECE 개선 {best_improvement:.4f})")
                else:
                    print(f"  ⚠️ Calibration이 성능을 악화시킴")
        
        # 외부 검증 시각화
        plot_roc_pr_curves(external_evaluation_results, title_prefix="(External Validation)")
        plot_calibration_curves(models_dict, x_external, y_external, title_prefix="(External Validation)")
        plot_confusion_matrices(external_conf_matrix_dict, title="Confusion Matrices (External Validation)")
    else:
        print(f"\n⚠️ 외부 검증 파일이 없습니다: {external_file}")
    
    print("\n" + "="*80)
    print("✅ 모든 작업 완료!")
    print("="*80)
