"""
Class-Balanced Loss for GBDT - 이진 분류 실험
불균형 데이터에 대한 Class-Balanced Loss 적용 및 성능 비교

참고: https://github.com/Luojiaqimath/ClassbalancedLoss4GBDT
"""

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
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_recall_curve, auc, 
    roc_curve, confusion_matrix, average_precision_score, precision_score, 
    recall_score, classification_report, balanced_accuracy_score
)
from sklearn.model_selection import StratifiedKFold

import lightgbm as lgb
import xgboost as xgb
from xgboost import XGBClassifier

import matplotlib.pyplot as plt
import seaborn as sns

# Class-Balanced Loss Functions
try:
    from gbdtCBL.binary import (
        ACELoss,    # Adaptive Cross-Entropy Loss
        AWELoss,    # Adaptive Weighted Cross-Entropy Loss
        FLLoss,     # Focal Loss
        CBELoss,    # Class-Balanced Effective Loss
        ASLLoss,    # Asymmetric Loss
        WCELoss     # Weighted Cross-Entropy Loss
    )
    CBL_AVAILABLE = True
    print("✅ gbdtCBL 로드 성공")
except ImportError as e:
    print(f"⚠️ gbdtCBL 패키지 import 실패: {e}")
    print("   설치 명령: pip install gbdtCBL==0.1")
    CBL_AVAILABLE = False

# ================================================================================================
# GLOBAL SETTINGS
# ================================================================================================
PREFIX = "normalhigh_cbl"
INIT = "rfe_ml"
SEED = 2025

# ================================================================================================
# UTILITY FUNCTIONS
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

    # 스케일 조정
    if "bwei" in x_train.columns:
        x_train["bwei"] = x_train["bwei"] / 1000
    if "bwei" in x_test.columns:
        x_test["bwei"] = x_test["bwei"] / 1000

    print(f"\n최종 데이터 Shape: X_train={x_train.shape}, X_test={x_test.shape}")
    print(f"클래스 불균형 비율: {Counter(y_train)}")
    
    return x_train, x_test, y_train, y_test


def sigmoid(x):
    """Sigmoid 함수 (overflow 방지)"""
    kEps = 1e-16
    x = np.minimum(-x, 88.7)  # exp overflow 방지
    return 1 / (1 + np.exp(x) + kEps)


def predict_proba_lgb(model, X):
    """LightGBM custom objective용 확률 예측"""
    prediction = model.predict(X)
    prediction_probabilities = sigmoid(prediction).reshape(-1, 1)
    prediction_probabilities = np.concatenate(
        (1 - prediction_probabilities, prediction_probabilities), 1
    )
    return prediction_probabilities


def eval_auc(labels, preds):
    """AUC 평가 함수 (LightGBM용)"""
    p = sigmoid(preds)
    return 'auc', roc_auc_score(labels, p), True


# ================================================================================================
# MODEL TRAINING FUNCTIONS
# ================================================================================================

def train_lightgbm_baseline(x_train, y_train, x_test, y_test, params=None):
    """LightGBM 기본 (class_weight='balanced')"""
    if params is None:
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'n_estimators': 1000,
            'class_weight': 'balanced',
            'random_state': SEED,
            'verbose': -1
        }
    
    model = lgb.LGBMClassifier(**params)
    model.fit(
        x_train, y_train,
        eval_set=[(x_test, y_test)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    
    return model


def train_lightgbm_cbl(x_train, y_train, x_test, y_test, loss_fn, loss_name, params=None):
    """LightGBM + Class-Balanced Loss"""
    if params is None:
        params = {
            'num_leaves': 31,
            'learning_rate': 0.05,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': SEED,
            'verbose': -1
        }
    
    train_data = lgb.Dataset(x_train, label=y_train)
    val_data = lgb.Dataset(x_test, label=y_test, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        fobj=loss_fn,
        feval=eval_auc,
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    
    return model


def train_xgboost_baseline(x_train, y_train, x_test, y_test, params=None):
    """XGBoost 기본 (scale_pos_weight 설정)"""
    if params is None:
        class_counts = Counter(y_train)
        scale_pos_weight = class_counts[0] / class_counts[1]
        
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 6,
            'learning_rate': 0.05,
            'n_estimators': 1000,
            'scale_pos_weight': scale_pos_weight,
            'random_state': SEED,
            'tree_method': 'hist',
            'early_stopping_rounds': 50
        }
    
    model = XGBClassifier(**params)
    model.fit(
        x_train, y_train,
        eval_set=[(x_test, y_test)],
        verbose=False
    )
    
    return model


def train_xgboost_cbl(x_train, y_train, x_test, y_test, loss_fn, loss_name, params=None):
    """XGBoost + Class-Balanced Loss
    
    Note: XGBoost의 custom objective는 (gradient, hessian)을 반환해야 함
    gbdtCBL의 loss function들은 이를 자동으로 처리
    """
    if params is None:
        params = {
            'max_depth': 6,
            'learning_rate': 0.05,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': SEED,
            'tree_method': 'hist',
        }
    
    # XGBoost는 custom objective를 직접 전달
    model = xgb.XGBClassifier(
        n_estimators=1000,
        objective=loss_fn,
        eval_metric='auc',
        early_stopping_rounds=50,
        **params
    )
    
    model.fit(
        x_train, y_train,
        eval_set=[(x_test, y_test)],
        verbose=False
    )
    
    return model


def train_catboost_baseline(x_train, y_train, x_test, y_test, params=None):
    """CatBoost 기본 (class_weights='Balanced')"""
    from catboost import CatBoostClassifier, Pool
    
    if params is None:
        class_counts = Counter(y_train)
        class_weights = {0: 1.0, 1: class_counts[0] / class_counts[1]}
        
        params = {
            'iterations': 1000,
            'learning_rate': 0.05,
            'depth': 6,
            'loss_function': 'Logloss',
            'eval_metric': 'AUC',
            'class_weights': class_weights,
            'random_seed': SEED,
            'verbose': False,
            'early_stopping_rounds': 50
        }
    
    model = CatBoostClassifier(**params)
    
    model.fit(
        x_train, y_train,
        eval_set=(x_test, y_test),
        verbose=False
    )
    
    return model


def train_catboost_focal(x_train, y_train, x_test, y_test, gamma=2.0, params=None):
    """CatBoost + Focal Loss (내장 지원)"""
    from catboost import CatBoostClassifier
    
    if params is None:
        params = {
            'iterations': 1000,
            'learning_rate': 0.05,
            'depth': 6,
            'loss_function': f'Focal:focal_alpha=0.25;focal_gamma={gamma}',
            'eval_metric': 'AUC',
            'random_seed': SEED,
            'verbose': False,
            'early_stopping_rounds': 50
        }
    
    model = CatBoostClassifier(**params)
    
    model.fit(
        x_train, y_train,
        eval_set=(x_test, y_test),
        verbose=False
    )
    
    return model


# ================================================================================================
# EVALUATION FUNCTIONS
# ================================================================================================

def evaluate_model(model, x_test, y_test, model_name, is_lgb_custom=False):
    """모델 평가"""
    if is_lgb_custom:
        # LightGBM custom objective의 경우
        y_pred_proba = predict_proba_lgb(model, x_test)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)
    else:
        y_pred = model.predict(x_test)
        y_pred_proba = model.predict_proba(x_test)[:, 1]
    
    # 메트릭 계산
    accuracy = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    auroc = roc_auc_score(y_test, y_pred_proba)
    
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
    auprc = auc(recall_curve, precision_curve)
    
    conf_matrix = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = conf_matrix.ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print(f"\n📊 {model_name}")
    print(f"  AUROC: {auroc:.4f}")
    print(f"  AUPRC: {auprc:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Balanced Acc: {balanced_acc:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  Sensitivity: {sensitivity:.4f}")
    print(f"  Specificity: {specificity:.4f}")
    
    return {
        'model': model_name,
        'auroc': auroc,
        'auprc': auprc,
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'confusion_matrix': conf_matrix,
        'y_pred_proba': y_pred_proba
    }


# ================================================================================================
# MAIN EXPERIMENT
# ================================================================================================

def run_experiment():
    """Class-Balanced Loss 실험 실행"""
    
    if not CBL_AVAILABLE:
        print("❌ gbdtCBL 패키지를 먼저 설치해주세요.")
        return None, None, None
    
    print("\n" + "="*80)
    print("🚀 Class-Balanced Loss 실험 시작")
    print("="*80)
    print()
    print("📌 기존 모델 성능 (SMOTE + class_weight 사용):")
    print("   - 최고 AUROC: 0.735 (HistGradBoost)")
    print("   - Precision: 0.56~0.57 (낮음 ⚠️)")
    print("   - Recall: 0.60~0.67 (중간)")
    print("   - 문제: False Positive 많고 불균형")
    print()
    print("🎯 목표: Class-Balanced Loss로 균형잡힌 성능 개선")
    
    # 데이터 로딩
    config = load_config()
    train_file = "./data/split/train_ml_dataset.csv"
    test_file = "./data/split/test_ml_dataset.csv"
    
    x_train, x_test, y_train, y_test = load_and_prepare_data(
        train_file, test_file, config, INIT
    )
    
    # 실험할 Loss Functions 정의
    loss_configs = [
        # === Baseline Models ===
        {'type': 'baseline', 'library': 'lightgbm', 'name': 'LightGBM_Baseline'},
        {'type': 'baseline', 'library': 'xgboost', 'name': 'XGBoost_Baseline'},
        {'type': 'baseline', 'library': 'catboost', 'name': 'CatBoost_Baseline'},
        
        # === LightGBM + CBL ===
        {'type': 'cbl', 'library': 'lightgbm', 'loss': ACELoss(m=0.1), 'name': 'LightGBM_ACELoss'},
        {'type': 'cbl', 'library': 'lightgbm', 'loss': AWELoss(r1=2.0, m=0.1), 'name': 'LightGBM_AWELoss'},
        {'type': 'cbl', 'library': 'lightgbm', 'loss': FLLoss(r1=2.0), 'name': 'LightGBM_FocalLoss'},
        {'type': 'cbl', 'library': 'lightgbm', 'loss': CBELoss(b=0.999), 'name': 'LightGBM_CBELoss'},
        {'type': 'cbl', 'library': 'lightgbm', 'loss': ASLLoss(r1=4.0, r2=1.0, m=0.1), 'name': 'LightGBM_ASLLoss'},
        
        # === XGBoost + CBL ===
        {'type': 'cbl', 'library': 'xgboost', 'loss': ACELoss(m=0.1), 'name': 'XGBoost_ACELoss'},
        {'type': 'cbl', 'library': 'xgboost', 'loss': AWELoss(r1=2.0, m=0.1), 'name': 'XGBoost_AWELoss'},
        {'type': 'cbl', 'library': 'xgboost', 'loss': FLLoss(r1=2.0), 'name': 'XGBoost_FocalLoss'},
        
        # === CatBoost (내장 Focal Loss) ===
        {'type': 'catboost_focal', 'library': 'catboost', 'gamma': 2.0, 'name': 'CatBoost_FocalLoss'},
        {'type': 'catboost_focal', 'library': 'catboost', 'gamma': 3.0, 'name': 'CatBoost_FocalLoss_g3'},
    ]
    
    results = []
    models_dict = {}
    
    print("\n" + "="*80)
    print("📈 모델 학습 및 평가")
    print("="*80)
    
    for config in loss_configs:
        print(f"\n{'='*60}")
        print(f"🔹 Training: {config['name']}")
        print(f"{'='*60}")
        
        try:
            if config['type'] == 'baseline':
                if config['library'] == 'lightgbm':
                    model = train_lightgbm_baseline(x_train, y_train, x_test, y_test)
                    result = evaluate_model(model, x_test, y_test, config['name'], is_lgb_custom=False)
                elif config['library'] == 'xgboost':
                    model = train_xgboost_baseline(x_train, y_train, x_test, y_test)
                    result = evaluate_model(model, x_test, y_test, config['name'], is_lgb_custom=False)
                else:  # catboost
                    model = train_catboost_baseline(x_train, y_train, x_test, y_test)
                    result = evaluate_model(model, x_test, y_test, config['name'], is_lgb_custom=False)
            
            elif config['type'] == 'cbl':
                if config['library'] == 'lightgbm':
                    model = train_lightgbm_cbl(
                        x_train, y_train, x_test, y_test, 
                        config['loss'], config['name']
                    )
                    result = evaluate_model(model, x_test, y_test, config['name'], is_lgb_custom=True)
                else:  # xgboost
                    model = train_xgboost_cbl(
                        x_train, y_train, x_test, y_test,
                        config['loss'], config['name']
                    )
                    result = evaluate_model(model, x_test, y_test, config['name'], is_lgb_custom=False)
            
            else:  # catboost_focal
                model = train_catboost_focal(
                    x_train, y_train, x_test, y_test,
                    gamma=config['gamma']
                )
                result = evaluate_model(model, x_test, y_test, config['name'], is_lgb_custom=False)
            
            results.append(result)
            models_dict[config['name']] = model
            
            # 모델 저장
            joblib.dump(model, f"./model/{config['name']}_{PREFIX}.pkl")
            
        except Exception as e:
            print(f"❌ Error training {config['name']}: {e}")
            continue
    
    # 결과 저장
    df_results = pd.DataFrame([{
        'model': r['model'],
        'auroc': r['auroc'],
        'auprc': r['auprc'],
        'accuracy': r['accuracy'],
        'balanced_accuracy': r['balanced_accuracy'],
        'f1_score': r['f1_score'],
        'precision': r['precision'],
        'recall': r['recall'],
        'sensitivity': r['sensitivity'],
        'specificity': r['specificity']
    } for r in results])
    
    df_results = df_results.sort_values('auroc', ascending=False)
    df_results.to_csv(f'cbl_comparison_{PREFIX}.csv', index=False)
    
    print("\n" + "="*80)
    print("📊 실험 결과 요약")
    print("="*80)
    print()
    
    # 전체 결과 출력
    print(df_results.to_string(index=False))
    
    print()
    print("="*80)
    print("📈 Boosting 모델별 비교")
    print("="*80)
    
    # LightGBM 모델들
    lgb_models = df_results[df_results['model'].str.contains('LightGBM')]
    if not lgb_models.empty:
        print("\n🟢 LightGBM 모델:")
        print(lgb_models[['model', 'auroc', 'precision', 'recall', 'f1_score']].to_string(index=False))
    
    # XGBoost 모델들
    xgb_models = df_results[df_results['model'].str.contains('XGBoost')]
    if not xgb_models.empty:
        print("\n🔵 XGBoost 모델:")
        print(xgb_models[['model', 'auroc', 'precision', 'recall', 'f1_score']].to_string(index=False))
    
    # CatBoost 모델들
    cat_models = df_results[df_results['model'].str.contains('CatBoost')]
    if not cat_models.empty:
        print("\n🟠 CatBoost 모델:")
        print(cat_models[['model', 'auroc', 'precision', 'recall', 'f1_score']].to_string(index=False))
    
    print()
    print("="*80)
    print("🎯 성능 개선 분석")
    print("="*80)
    
    # Baseline 찾기
    baseline_rows = df_results[df_results['model'].str.contains('Baseline')]
    if not baseline_rows.empty:
        baseline_auroc = baseline_rows['auroc'].iloc[0]
        baseline_precision = baseline_rows['precision'].iloc[0]
        baseline_recall = baseline_rows['recall'].iloc[0]
        
        # 최고 성능 모델
        best_model = df_results.iloc[0]
        
        auroc_improve = (best_model['auroc'] - baseline_auroc) * 100
        precision_improve = (best_model['precision'] - baseline_precision) * 100
        recall_improve = (best_model['recall'] - baseline_recall) * 100
        
        print(f"\n최고 성능 모델: {best_model['model']}")
        print(f"  AUROC: {baseline_auroc:.3f} → {best_model['auroc']:.3f} ({auroc_improve:+.1f}%)")
        print(f"  Precision: {baseline_precision:.3f} → {best_model['precision']:.3f} ({precision_improve:+.1f}%)")
        print(f"  Recall: {baseline_recall:.3f} → {best_model['recall']:.3f} ({recall_improve:+.1f}%)")
        print(f"  F1 Score: {best_model['f1_score']:.3f}")
        print(f"  Balanced Acc: {best_model['balanced_accuracy']:.3f}")
    
    print()
    print("="*80)
    print("🏆 Boosting 라이브러리별 최고 성능")
    print("="*80)
    
    # LightGBM 최고
    if not lgb_models.empty:
        best_lgb = lgb_models.iloc[0]
        print(f"\n🟢 LightGBM 최고: {best_lgb['model']}")
        print(f"   AUROC: {best_lgb['auroc']:.3f} | Precision: {best_lgb['precision']:.3f} | Recall: {best_lgb['recall']:.3f}")
    
    # XGBoost 최고
    if not xgb_models.empty:
        best_xgb = xgb_models.iloc[0]
        print(f"\n🔵 XGBoost 최고: {best_xgb['model']}")
        print(f"   AUROC: {best_xgb['auroc']:.3f} | Precision: {best_xgb['precision']:.3f} | Recall: {best_xgb['recall']:.3f}")
    
    # CatBoost 최고
    if not cat_models.empty:
        best_cat = cat_models.iloc[0]
        print(f"\n🟠 CatBoost 최고: {best_cat['model']}")
        print(f"   AUROC: {best_cat['auroc']:.3f} | Precision: {best_cat['precision']:.3f} | Recall: {best_cat['recall']:.3f}")
    
    print()
    print("="*80)
    print("💡 목적별 추천 모델")
    print("="*80)
    
    # Precision이 가장 높은 모델
    best_precision_model = df_results.loc[df_results['precision'].idxmax()]
    print(f"\n🎯 Precision 최우선 (False Positive 최소화)")
    print(f"   모델: {best_precision_model['model']}")
    print(f"   Precision: {best_precision_model['precision']:.3f} | Recall: {best_precision_model['recall']:.3f}")
    
    # Recall이 가장 높은 모델
    best_recall_model = df_results.loc[df_results['recall'].idxmax()]
    print(f"\n🎯 Recall 최우선 (고위험자 놓치지 않기)")
    print(f"   모델: {best_recall_model['model']}")
    print(f"   Precision: {best_recall_model['precision']:.3f} | Recall: {best_recall_model['recall']:.3f}")
    
    # Balanced Accuracy가 가장 높은 모델
    best_balanced_model = df_results.loc[df_results['balanced_accuracy'].idxmax()]
    print(f"\n🎯 균형잡힌 성능")
    print(f"   모델: {best_balanced_model['model']}")
    print(f"   Balanced Acc: {best_balanced_model['balanced_accuracy']:.3f} | F1: {best_balanced_model['f1_score']:.3f}")
    
    # 시각화
    plot_comparison(df_results)
    
    return results, models_dict, df_results


def plot_comparison(df_results):
    """결과 비교 시각화"""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12), dpi=150)
    
    metrics = ['auroc', 'auprc', 'balanced_accuracy', 'f1_score', 'precision', 'recall']
    titles = ['AUROC', 'AUPRC', 'Balanced Accuracy', 'F1 Score', 'Precision', 'Recall']
    
    for ax, metric, title in zip(axes.flatten(), metrics, titles):
        df_sorted = df_results.sort_values(metric)
        
        colors = ['#e74c3c' if 'Baseline' in m else '#3498db' for m in df_sorted['model']]
        
        ax.barh(df_sorted['model'], df_sorted[metric], color=colors)
        ax.set_xlabel(title, fontsize=12, fontweight='bold')
        ax.set_title(f'{title} Comparison', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # 값 표시
        for i, (idx, row) in enumerate(df_sorted.iterrows()):
            ax.text(row[metric], i, f' {row[metric]:.3f}', 
                   va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'cbl_comparison_{PREFIX}.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n✅ 비교 그래프 저장: cbl_comparison_{PREFIX}.png")
    
    # Precision-Recall Trade-off 그래프 추가
    fig2, ax2 = plt.subplots(figsize=(10, 8), dpi=150)
    
    for _, row in df_results.iterrows():
        color = '#e74c3c' if 'Baseline' in row['model'] else '#3498db'
        marker = 'o' if 'Baseline' in row['model'] else 's'
        size = 150 if 'Baseline' in row['model'] else 100
        
        ax2.scatter(row['recall'], row['precision'], 
                   s=size, c=color, marker=marker, 
                   alpha=0.7, edgecolors='black', linewidth=1.5,
                   label=row['model'])
    
    ax2.set_xlabel('Recall (Sensitivity)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Precision (PPV)', fontsize=14, fontweight='bold')
    ax2.set_title('Precision-Recall Trade-off', fontsize=16, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # 대각선 (이상적인 균형선)
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Perfect Balance')
    
    plt.tight_layout()
    plt.savefig(f'cbl_precision_recall_tradeoff_{PREFIX}.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Precision-Recall 그래프 저장: cbl_precision_recall_tradeoff_{PREFIX}.png")


# ================================================================================================
# MAIN EXECUTION
# ================================================================================================

if __name__ == "__main__":
    results, models_dict, df_results = run_experiment()
    
    if results is not None:
        print("\n" + "="*80)
        print("✅ 실험 완료!")
        print("="*80)
        print(f"\n저장된 파일:")
        print(f"  - 결과 CSV: cbl_comparison_{PREFIX}.csv")
        print(f"  - 비교 그래프: cbl_comparison_{PREFIX}.png")
        print(f"  - P-R 그래프: cbl_precision_recall_tradeoff_{PREFIX}.png")
        print(f"  - 모델 파일: ./model/*_{PREFIX}.pkl")
    else:
        print("\n❌ 실험 실행 실패")
        print("gbdtCBL 패키지 설치 후 다시 시도해주세요.")

