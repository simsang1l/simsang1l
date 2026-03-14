import sys, os
import argparse
import pandas as pd
import numpy as np
import json
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, f1_score, roc_curve, precision_recall_curve
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from collections import Counter
import joblib
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# Optional imports for advanced models
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

try:
    from skopt import BayesSearchCV
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.utils.utils import load_data, load_config


def loader(config_path, feature_config_path, ml_config_path, train_data_type, test_data_type, column_category):
    """
    데이터 불러오기, 원하는 컬럼만 가져오기
    데이터 전처리 (원핫인코딩, normalization 등)
    """
    feature_config = load_config(feature_config_path)
    ml_config = load_config(ml_config_path)
    
    train_data = load_data(train_data_type, "train", config_path)
    test_data = load_data(test_data_type, "test", config_path)
    # 임시 조건걸기
    train_data = train_data[train_data['label'].isin([0, 1])]
    test_data = test_data[test_data['label'].isin([0, 1])]

    numeric_columns = feature_config[column_category]['continuous']
    categorical_vars = feature_config[column_category]['category']
    target_vars = feature_config['derived_columns']["target"]
    target = feature_config['derived_columns']["label"]

    features = numeric_columns + categorical_vars + target_vars
    x_train = train_data[features]
    y_train = train_data[target].values.ravel()  # 1차원 배열로 변환
    x_test = test_data[features]
    y_test = test_data[target].values.ravel()    # 1차원 배열로 변환


    """
    데이터 전처리
    """
    preprocessing_config = ml_config.get('preprocessing', {})
    
    # 연속형 변수 단위 변경 (설정 파일에서 가져오기)
    scale_vars = preprocessing_config.get('scale_vars', [])
    scale_factor = preprocessing_config.get('scale_factor', 7)
    weight_vars = preprocessing_config.get('weight_vars', [])
    weight_factor = preprocessing_config.get('weight_factor', 1000)
    
    # 스케일 변수 처리
    for v in scale_vars:
        if v in x_train.columns:
            x_train[v] = x_train[v] / scale_factor
        if v in x_test.columns:
            x_test[v] = x_test[v] / scale_factor
    
    # 가중치 변수 처리
    for v in weight_vars:
        if v in x_train.columns:
            x_train[v] = x_train[v] / weight_factor
        if v in x_test.columns:
            x_test[v] = x_test[v] / weight_factor

    # 원-핫 인코딩
    x_train = pd.get_dummies(x_train, columns=categorical_vars, drop_first=True)
    x_test = pd.get_dummies(x_test, columns=categorical_vars, drop_first=True)

    # normalization
    scaler = StandardScaler()
    x_train[numeric_columns] = scaler.fit_transform(x_train[numeric_columns])
    x_test[numeric_columns] = scaler.transform(x_test[numeric_columns])

    return x_train, y_train, x_test, y_test, ml_config

def load_external_data(config_path, feature_config_path, ml_config_path, external_data_path, column_category):
    """
    외부 데이터(시간적 검증용) 로딩 및 전처리
    
    Args:
        config_path: 메인 설정 파일 경로
        feature_config_path: 특성 설정 파일 경로
        ml_config_path: ML 설정 파일 경로
        external_data_path: 외부 데이터 파일 경로
        column_category: 컬럼 카테고리
    
    Returns:
        x_external, y_external: 전처리된 외부 데이터
    """
    # 설정 파일 로드
    feature_config = load_config(feature_config_path)
    ml_config = load_config(ml_config_path)
    
    # 외부 데이터 로드
    external_data = pd.read_csv(external_data_path)
    
    # 동일한 조건 적용 (임시 조건)
    external_data = external_data[external_data['label'].isin([0, 1])]
    
    # 특성 및 타겟 설정
    numeric_columns = feature_config[column_category]['continuous']
    categorical_vars = feature_config[column_category]['category']
    target_vars = feature_config['derived_columns']["target"]
    target = feature_config['derived_columns']["label"]
    
    features = numeric_columns + categorical_vars + target_vars
    x_external = external_data[features]
    y_external = external_data[target].values.ravel()
    
    # 전처리 설정 가져오기
    preprocessing_config = ml_config.get('preprocessing', {})
    
    # 연속형 변수 단위 변경 (설정 파일에서 가져오기)
    scale_vars = preprocessing_config.get('scale_vars', [])
    scale_factor = preprocessing_config.get('scale_factor', 7)
    weight_vars = preprocessing_config.get('weight_vars', [])
    weight_factor = preprocessing_config.get('weight_factor', 1000)
    
    # 스케일 변수 처리
    for v in scale_vars:
        if v in x_external.columns:
            x_external[v] = x_external[v] / scale_factor
    
    # 가중치 변수 처리
    for v in weight_vars:
        if v in x_external.columns:
            x_external[v] = x_external[v] / weight_factor
    
    # 원-핫 인코딩
    x_external = pd.get_dummies(x_external, columns=categorical_vars, drop_first=True)
    
    # normalization (훈련 데이터와 동일한 스케일러 사용)
    scaler = StandardScaler()
    x_external[numeric_columns] = scaler.fit_transform(x_external[numeric_columns])
    
    return x_external, y_external

def evaluate_external_performance(results, x_external, y_external):
    """
    외부 데이터로 모델 성능 평가
    
    Args:
        results: train 함수의 반환값
        x_external, y_external: 외부 데이터
    
    Returns:
        dict: 외부 데이터 평가 결과
    """
    external_results = {}
    
    print(f"\n=== 외부 데이터 성능 평가 ===")
    print(f"외부 데이터 형태: {x_external.shape}")
    print(f"외부 데이터 클래스 분포: {Counter(y_external)}")
    
    for model_name, result in results['all_results'].items():
        print(f"\n--- {model_name} 외부 데이터 평가 ---")
        
        try:
            # 모델로 예측
            model = result['model']
            y_pred_external = model.predict(x_external)
            y_prob_external = model.predict_proba(x_external)[:, 1] if len(np.unique(y_external)) == 2 else None
            
            # 성능 평가
            accuracy_external = accuracy_score(y_external, y_pred_external)
            f1_external = f1_score(y_external, y_pred_external, average='weighted')
            auc_external = roc_auc_score(y_external, y_prob_external) if y_prob_external is not None else None
            
            # 결과 저장
            external_results[model_name] = {
                'external_accuracy': accuracy_external,
                'external_f1': f1_external,
                'external_auc': auc_external,
                'external_predictions': y_pred_external,
                'external_probabilities': y_prob_external,
                'external_classification_report': classification_report(y_external, y_pred_external),
                'external_confusion_matrix': confusion_matrix(y_external, y_pred_external)
            }
            
            print(f"외부 데이터 정확도: {accuracy_external:.4f}")
            print(f"외부 데이터 F1: {f1_external:.4f}")
            if auc_external:
                print(f"외부 데이터 AUC: {auc_external:.4f}")
                
        except Exception as e:
            print(f"{model_name} 외부 데이터 평가 중 오류: {e}")
            external_results[model_name] = None
    
    return external_results

def get_model_instance(model_name, model_config, random_state):
    """모델 인스턴스를 동적으로 생성"""
    if model_name == 'LogisticRegression':
        max_iter = model_config.get('max_iter', 2000)
        return LogisticRegression(random_state=random_state, max_iter=max_iter)
    
    elif model_name == 'RandomForest':
        return RandomForestClassifier(random_state=random_state)
    
    elif model_name == 'SVM':
        probability = model_config.get('probability', True)
        return SVC(random_state=random_state, probability=probability)
    
    elif model_name == 'XGBoost':
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not installed. Install with: pip install xgboost")
        return XGBClassifier(random_state=random_state, eval_metric='logloss', verbosity=0)
    
    elif model_name == 'LightGBM':
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is not installed. Install with: pip install lightgbm")
        return LGBMClassifier(random_state=random_state, verbose=-1)
    
    elif model_name == 'CatBoost':
        if not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost is not installed. Install with: pip install catboost")
        return CatBoostClassifier(random_state=random_state, verbose=False)
    
    elif model_name == 'GradientBoosting':
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(random_state=random_state)
    
    else:
        raise ValueError(f"지원하지 않는 모델: {model_name}")

def get_search_cv(method, estimator, param_grid, cv, scoring, ml_config):
    """최적화 방법에 따른 SearchCV 객체 반환"""
    opt_config = ml_config.get('optimization', {})
    n_jobs = opt_config.get('n_jobs', -1)
    verbose = opt_config.get('verbose', 0)
    
    if method == 'grid':
        return GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=verbose
        )
    
    elif method == 'random':
        n_iter = opt_config.get('n_iter', 50)
        return RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=ml_config.get('random_state', 42)
        )
    
    elif method == 'bayes':
        if not SKOPT_AVAILABLE:
            print("scikit-optimize가 설치되지 않음. Grid Search로 대체합니다.")
            return get_search_cv('grid', estimator, param_grid, cv, scoring, ml_config)
        
        n_calls = opt_config.get('n_calls', 50)
        return BayesSearchCV(
            estimator=estimator,
            search_spaces=param_grid,
            n_iter=n_calls,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=ml_config.get('random_state', 42)
        )
    
    else:
        raise ValueError(f"지원하지 않는 최적화 방법: {method}")

def train(x_train, y_train, x_test, y_test, ml_config):
    """
    머신러닝 모델 학습 및 평가 - 설정 파일 기반
    
    Args:
        x_train, y_train: 훈련 데이터
        x_test, y_test: 테스트 데이터
        ml_config: ML 설정 딕셔너리
    
    Returns:
        dict: 최고 성능 모델과 결과들
    """
    # 설정 값들 가져오기
    random_state = ml_config.get('random_state', 42)
    cv_folds = ml_config.get('cv_folds', 5)
    scoring_metric = ml_config.get('scoring_metric', 'roc_auc')
    
    print(f"학습 시작 - 훈련 데이터: {x_train.shape}, 테스트 데이터: {x_test.shape}")
    print(f"훈련 데이터 클래스 분포: {Counter(y_train)}")
    print(f"테스트 데이터 클래스 분포: {Counter(y_test)}")
    
    # 데이터 검증
    print(f"\n=== 데이터 검증 ===")
    print(f"훈련 데이터 고유값: {np.unique(y_train)}")
    print(f"테스트 데이터 고유값: {np.unique(y_test)}")
    print(f"훈련 데이터 타입: {type(y_train)}, 형태: {y_train.shape}")
    print(f"테스트 데이터 타입: {type(y_test)}, 형태: {y_test.shape}")
    
    # 클래스가 하나만 있는지 확인
    if len(np.unique(y_train)) < 2:
        raise ValueError(f"훈련 데이터에 클래스가 하나만 있습니다: {np.unique(y_train)}")
    if len(np.unique(y_test)) < 2:
        raise ValueError(f"테스트 데이터에 클래스가 하나만 있습니다: {np.unique(y_test)}")
    
    # 1. 클래스 불균형 처리 (SMOTE)
    smote_config = ml_config.get('smote', {})
    use_smote = smote_config.get('enabled', True)
    
    if use_smote:
        print("\n=== SMOTE 오버샘플링 적용 ===")
        try:
            smote_random_state = smote_config.get('random_state', random_state)
            smote = SMOTE(random_state=smote_random_state)
            x_train_balanced, y_train_balanced = smote.fit_resample(x_train, y_train)
            print(f"SMOTE 후 클래스 분포: {Counter(y_train_balanced)}")
            x_train, y_train = x_train_balanced, y_train_balanced
        except Exception as e:
            print(f"SMOTE 적용 실패: {e}")
            print("SMOTE 없이 진행합니다.")
    
    # 2. 교차 검증 설정
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    # 3. 스코어링 메트릭 설정
    n_classes = len(np.unique(y_train))
    if scoring_metric == 'roc_auc' and n_classes > 2:
        scoring = 'roc_auc_ovr'  # 다중 클래스의 경우
    else:
        scoring = scoring_metric
    
    # 4. 모델 설정 가져오기
    models_config = ml_config.get('models', {})
    optimization_method = ml_config.get('optimization', {}).get('method', 'grid')
    
    # 5. 모델 학습 및 하이퍼파라미터 최적화
    results = {}
    best_score = 0
    best_model_name = None
    best_model = None
    
    print(f"\n=== 모델 학습 및 하이퍼파라미터 최적화 (방법: {optimization_method}) ===")
    
    for model_name, model_config in models_config.items():
        if not model_config.get('enabled', False):
            print(f"{model_name}: 비활성화됨")
            continue
            
        print(f"\n--- {model_name} 최적화 중 ---")
        
        try:
            # 모델 인스턴스 생성
            model = get_model_instance(model_name, model_config, random_state)
            param_grid = model_config.get('params', {})
            
            # SearchCV 객체 생성
            search_cv = get_search_cv(optimization_method, model, param_grid, cv, scoring, ml_config)
            
            # 학습
            search_cv.fit(x_train, y_train)
            
            # 최적 모델로 예측
            best_estimator = search_cv.best_estimator_
            y_pred = best_estimator.predict(x_test)
            y_prob = best_estimator.predict_proba(x_test)[:, 1] if n_classes == 2 else None
            
            # 성능 평가
            accuracy = accuracy_score(y_test, y_pred)
            auc_score = roc_auc_score(y_test, y_prob) if y_prob is not None else None
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # 예측 결과 검증
            print(f"예측값 고유값: {np.unique(y_pred)}")
            print(f"예측값 분포: {Counter(y_pred)}")
            
            # 결과 저장
            results[model_name] = {
                'best_params': search_cv.best_params_,
                'best_cv_score': search_cv.best_score_,
                'test_accuracy': accuracy,
                'test_auc': auc_score,
                'test_f1': f1,
                'model': best_estimator,
                'predictions': y_pred,
                'probabilities': y_prob,
                'classification_report': classification_report(y_test, y_pred),
                'confusion_matrix': confusion_matrix(y_test, y_pred)
            }
            
            print(f"최적 파라미터: {search_cv.best_params_}")
            print(f"CV 점수: {search_cv.best_score_:.4f}")
            print(f"테스트 정확도: {accuracy:.4f}")
            print(f"테스트 F1: {f1:.4f}")
            if auc_score:
                print(f"테스트 AUC: {auc_score:.4f}")
            
            # 최고 성능 모델 추적
            current_score = auc_score if auc_score else accuracy
            if current_score > best_score:
                best_score = current_score
                best_model_name = model_name
                best_model = best_estimator
                
        except Exception as e:
            print(f"{model_name} 학습 중 오류 발생: {e}")
            continue
    
    if not results:
        raise ValueError("활성화된 모델이 없거나 모든 모델 학습이 실패했습니다.")
    
    # 6. 결과 요약
    print(f"\n=== 최종 결과 ===")
    print(f"최고 성능 모델: {best_model_name}")
    print(f"최고 점수: {best_score:.4f}")
    
    # 7. 최고 모델 상세 평가
    print(f"\n=== {best_model_name} 상세 평가 ===")
    best_result = results[best_model_name]
    print("분류 리포트:")
    print(best_result['classification_report'])
    print("\n혼동 행렬:")
    print(best_result['confusion_matrix'])
    
    # 8. 모델 저장
    save_config = ml_config.get('model_save', {})
    save_dir = save_config.get('directory', 'model')
    save_prefix = save_config.get('prefix', 'best_model')
    
    model_save_path = f"{save_dir}/{save_prefix}_{best_model_name.lower()}.pkl"
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(best_model, model_save_path)
    print(f"\n최고 모델 저장: {model_save_path}")
    
    return {
        'best_model_name': best_model_name,
        'best_model': best_model,
        'best_score': best_score,
        'all_results': results,
        'model_save_path': model_save_path,
        'ml_config': ml_config,
        'y_test': y_test  # y_test 추가
    }

def save_results(results, ml_config, x_train_shape, x_test_shape, y_test=None, external_results=None, x_external_shape=None, output_dir=None):
    """
    머신러닝 실험 결과를 다양한 형태로 저장
    
    Args:
        results: train 함수의 반환값
        ml_config: ML 설정
        x_train_shape, x_test_shape: 데이터 형태 정보
        y_test: 테스트 데이터의 실제 값 (그림 생성용)
        external_results: 외부 데이터 평가 결과
        x_external_shape: 외부 데이터 형태
        output_dir: 결과 저장 디렉토리 (None이면 설정 파일에서 가져옴)
    """
    # 결과 저장 설정 가져오기
    results_config = ml_config.get('results_save', {})
    if not results_config.get('enabled', True):
        print("결과 저장이 비활성화되어 있습니다.")
        return None
    
    # 결과 저장 디렉토리 설정
    if output_dir is None:
        output_dir = results_config.get('directory', 'results')
    
    # 결과 저장 디렉토리 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = Path(output_dir) / f"ml_experiment_{timestamp}"
    result_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 요약 결과 JSON 저장 (항상 저장)
    summary_results = {
        "experiment_info": {
            "timestamp": timestamp,
            "train_data_shape": x_train_shape,
            "test_data_shape": x_test_shape,
            "external_data_shape": x_external_shape if x_external_shape else None,
            "best_model": results['best_model_name'],
            "best_score": float(results['best_score']),
            "ml_config_used": ml_config
        },
        "model_comparison": {}
    }
    
    # 2. 모델별 상세 결과 수집
    model_comparison_data = []
    for model_name, result in results['all_results'].items():
        model_info = {
            "model_name": model_name,
            "cv_score": float(result['best_cv_score']),
            "test_accuracy": float(result['test_accuracy']),
            "test_f1": float(result['test_f1']),
            "test_auc": float(result['test_auc']) if result['test_auc'] else None,
            "best_params": result['best_params']
        }
        
        # 외부 데이터 결과 추가
        if external_results and model_name in external_results and external_results[model_name]:
            ext_result = external_results[model_name]
            model_info.update({
                "external_accuracy": float(ext_result['external_accuracy']),
                "external_f1": float(ext_result['external_f1']),
                "external_auc": float(ext_result['external_auc']) if ext_result['external_auc'] else None
            })
        
        summary_results["model_comparison"][model_name] = model_info
        model_comparison_data.append(model_info)
    
    # JSON 저장
    with open(result_dir / "summary_results.json", 'w', encoding='utf-8') as f:
        json.dump(summary_results, f, indent=2, ensure_ascii=False)
    
    # 3. 모델 비교 CSV 저장 (항상 저장)
    comparison_df = pd.DataFrame(model_comparison_data)
    comparison_df.to_csv(result_dir / "model_comparison.csv", index=False)
    
    saved_files = ["summary_results.json", "model_comparison.csv"]
    
    # 4. 상세 분류 리포트 저장 (설정에 따라)
    if results_config.get('save_detailed_reports', True):
        with open(result_dir / "detailed_reports.txt", 'w', encoding='utf-8') as f:
            f.write(f"=== 머신러닝 실험 상세 리포트 ===\n")
            f.write(f"실험 시간: {timestamp}\n")
            f.write(f"훈련 데이터 형태: {x_train_shape}\n")
            f.write(f"테스트 데이터 형태: {x_test_shape}\n")
            if x_external_shape:
                f.write(f"외부 데이터 형태: {x_external_shape}\n")
            f.write(f"최고 성능 모델: {results['best_model_name']}\n")
            f.write(f"최고 점수: {results['best_score']:.4f}\n\n")
            
            for model_name, result in results['all_results'].items():
                f.write(f"\n{'='*50}\n")
                f.write(f"모델: {model_name}\n")
                f.write(f"{'='*50}\n")
                f.write(f"최적 파라미터: {result['best_params']}\n")
                f.write(f"CV 점수: {result['best_cv_score']:.4f}\n")
                f.write(f"테스트 정확도: {result['test_accuracy']:.4f}\n")
                f.write(f"테스트 F1: {result['test_f1']:.4f}\n")
                if result['test_auc']:
                    f.write(f"테스트 AUC: {result['test_auc']:.4f}\n")
                f.write(f"\n분류 리포트:\n{result['classification_report']}\n")
                f.write(f"\n혼동 행렬:\n{result['confusion_matrix']}\n")
                
                # 외부 데이터 결과 추가
                if external_results and model_name in external_results and external_results[model_name]:
                    ext_result = external_results[model_name]
                    f.write(f"\n--- 외부 데이터 평가 결과 ---\n")
                    f.write(f"외부 데이터 정확도: {ext_result['external_accuracy']:.4f}\n")
                    f.write(f"외부 데이터 F1: {ext_result['external_f1']:.4f}\n")
                    if ext_result['external_auc']:
                        f.write(f"외부 데이터 AUC: {ext_result['external_auc']:.4f}\n")
                    f.write(f"\n외부 데이터 분류 리포트:\n{ext_result['external_classification_report']}\n")
                    f.write(f"\n외부 데이터 혼동 행렬:\n{ext_result['external_confusion_matrix']}\n")
        saved_files.append("detailed_reports.txt")
    
    # 5. 최고 모델의 예측 결과 저장 (설정에 따라)
    if results_config.get('save_predictions', True):
        best_result = results['all_results'][results['best_model_name']]
        predictions_df = pd.DataFrame({
            'predictions': best_result['predictions'],
            'probabilities': best_result['probabilities'] if best_result['probabilities'] is not None else [None] * len(best_result['predictions'])
        })
        predictions_df.to_csv(result_dir / "best_model_predictions.csv", index=False)
        saved_files.append("best_model_predictions.csv")
        
        # 외부 데이터 예측 결과 저장
        if external_results and results['best_model_name'] in external_results and external_results[results['best_model_name']]:
            ext_result = external_results[results['best_model_name']]
            external_predictions_df = pd.DataFrame({
                'external_predictions': ext_result['external_predictions'],
                'external_probabilities': ext_result['external_probabilities'] if ext_result['external_probabilities'] is not None else [None] * len(ext_result['external_predictions'])
            })
            external_predictions_df.to_csv(result_dir / "best_model_external_predictions.csv", index=False)
            saved_files.append("best_model_external_predictions.csv")
    
    # 6. 혼동 행렬을 CSV로 저장 (설정에 따라)
    if results_config.get('save_confusion_matrices', True):
        for model_name, result in results['all_results'].items():
            cm_df = pd.DataFrame(result['confusion_matrix'])
            cm_df.to_csv(result_dir / f"confusion_matrix_{model_name.lower()}.csv", index=False)
            saved_files.append(f"confusion_matrix_{model_name.lower()}.csv")
            
            # 외부 데이터 혼동 행렬 저장
            if external_results and model_name in external_results and external_results[model_name]:
                ext_result = external_results[model_name]
                ext_cm_df = pd.DataFrame(ext_result['external_confusion_matrix'])
                ext_cm_df.to_csv(result_dir / f"external_confusion_matrix_{model_name.lower()}.csv", index=False)
                saved_files.append(f"external_confusion_matrix_{model_name.lower()}.csv")
    
    # 7. 그림 저장 (설정에 따라)
    if results_config.get('save_plots', True):
        saved_plots = create_and_save_plots(results, y_test, result_dir, save_plots=True)
        saved_files.extend(saved_plots)
        
        # 외부 데이터 그림 저장
        if external_results:
            saved_external_plots = create_external_plots(results, external_results, result_dir, save_plots=True)
            saved_files.extend(saved_external_plots)
    
    return result_dir, saved_files

def print_and_log_results(results, ml_config, x_train_shape, x_test_shape, save_results_flag=True, external_results=None, x_external_shape=None):
    """
    결과를 출력하고 필요시 저장
    
    Args:
        results: train 함수의 반환값
        ml_config: ML 설정
        x_train_shape, x_test_shape: 데이터 형태
        save_results_flag: 결과 저장 여부
        external_results: 외부 데이터 평가 결과
        x_external_shape: 외부 데이터 형태
    """
    # 콘솔 출력
    print(f"\n=== 최종 실행 결과 ===")
    print(f"최고 성능 모델: {results['best_model_name']}")
    print(f"최고 점수: {results['best_score']:.4f}")
    print(f"모델 저장 경로: {results['model_save_path']}")
    
    # 모든 모델 성능 비교
    print(f"\n=== 모든 모델 성능 비교 ===")
    for model_name, result in results['all_results'].items():
        print(f"{model_name}:")
        print(f"  - CV 점수: {result['best_cv_score']:.4f}")
        print(f"  - 테스트 정확도: {result['test_accuracy']:.4f}")
        print(f"  - 테스트 F1: {result['test_f1']:.4f}")
        if result['test_auc']:
            print(f"  - 테스트 AUC: {result['test_auc']:.4f}")
        print(f"  - 최적 파라미터: {result['best_params']}")
        
        # 외부 데이터 결과 출력
        if external_results and model_name in external_results and external_results[model_name]:
            ext_result = external_results[model_name]
            print(f"  - 외부 데이터 정확도: {ext_result['external_accuracy']:.4f}")
            print(f"  - 외부 데이터 F1: {ext_result['external_f1']:.4f}")
            if ext_result['external_auc']:
                print(f"  - 외부 데이터 AUC: {ext_result['external_auc']:.4f}")
        print()
    
    # 결과 저장
    result_dir = None
    if save_results_flag:
        print(f"\n=== 결과 저장 중... ===")
        result_dir, saved_files = save_results(results, ml_config, x_train_shape, x_test_shape, results['y_test'], external_results, x_external_shape)
        print(f"결과 저장 완료: {result_dir}")
        print(f"저장된 파일들:")
        for file in saved_files:
            print(f"  - {file}")
    
    return result_dir

def create_and_save_plots(results, y_test, result_dir, save_plots=True):
    """
    머신러닝 결과를 시각화하고 저장
    
    Args:
        results: train 함수의 반환값
        y_test: 테스트 데이터의 실제 값
        result_dir: 결과 저장 디렉토리
        save_plots: 그림 저장 여부
    """
    if not save_plots:
        return []
    
    saved_plots = []
    
    # 1. 모델 성능 비교 막대 그래프
    plt.figure(figsize=(12, 6))
    models = list(results['all_results'].keys())
    accuracies = [results['all_results'][model]['test_accuracy'] for model in models]
    f1_scores = [results['all_results'][model]['test_f1'] for model in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    plt.bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8)
    plt.bar(x + width/2, f1_scores, width, label='F1 Score', alpha=0.8)
    
    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks(x, models, rotation=45)
    plt.legend()
    plt.tight_layout()
    
    plot_path = result_dir / "model_performance_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    saved_plots.append("model_performance_comparison.png")
    
    # 2. 각 모델별 혼동 행렬 히트맵
    for model_name, result in results['all_results'].items():
        plt.figure(figsize=(8, 6))
        cm = result['confusion_matrix']
        
        # 클래스 레이블 생성 (실제 클래스 수에 따라)
        n_classes = cm.shape[0]
        class_labels = [f'Class {i}' for i in range(n_classes)]
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_labels, yticklabels=class_labels)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        plot_path = result_dir / f"confusion_matrix_{model_name.lower()}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        saved_plots.append(f"confusion_matrix_{model_name.lower()}.png")
    
    # 3. ROC 곡선 (이진 분류인 경우)
    best_model_name = results['best_model_name']
    best_result = results['all_results'][best_model_name]
    
    if best_result['probabilities'] is not None and len(np.unique(y_test)) == 2:
        plt.figure(figsize=(10, 8))
        
        # 모든 모델의 ROC 곡선
        for model_name, result in results['all_results'].items():
            if result['probabilities'] is not None:
                fpr, tpr, _ = roc_curve(y_test, result['probabilities'])
                auc_score = result['test_auc']
                plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_path = result_dir / "roc_curves_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        saved_plots.append("roc_curves_comparison.png")
        
        # Precision-Recall 곡선
        plt.figure(figsize=(10, 8))
        for model_name, result in results['all_results'].items():
            if result['probabilities'] is not None:
                precision, recall, _ = precision_recall_curve(y_test, result['probabilities'])
                f1 = result['test_f1']
                plt.plot(recall, precision, label=f'{model_name} (F1 = {f1:.3f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_path = result_dir / "precision_recall_curves.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        saved_plots.append("precision_recall_curves.png")
    
    # 4. 특성 중요도 (트리 기반 모델들)
    for model_name, result in results['all_results'].items():
        model = result['model']
        
        # 특성 중요도가 있는 모델들 확인
        if hasattr(model, 'feature_importances_'):
            plt.figure(figsize=(12, 8))
            
            # 특성 중요도 상위 20개만 표시
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:20]
            
            plt.title(f'Feature Importance - {model_name}')
            plt.bar(range(len(indices)), importances[indices])
            plt.xticks(range(len(indices)), [f'Feature {i}' for i in indices], rotation=45)
            plt.xlabel('Features')
            plt.ylabel('Importance')
            plt.tight_layout()
            
            plot_path = result_dir / f"feature_importance_{model_name.lower()}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            saved_plots.append(f"feature_importance_{model_name.lower()}.png")
        
        # CatBoost의 경우 특별 처리
        elif model_name == 'CatBoost' and hasattr(model, 'get_feature_importance'):
            plt.figure(figsize=(12, 8))
            
            try:
                importances = model.get_feature_importance()
                indices = np.argsort(importances)[::-1][:20]
                
                plt.title(f'Feature Importance - {model_name}')
                plt.bar(range(len(indices)), importances[indices])
                plt.xticks(range(len(indices)), [f'Feature {i}' for i in indices], rotation=45)
                plt.xlabel('Features')
                plt.ylabel('Importance')
                plt.tight_layout()
                
                plot_path = result_dir / f"feature_importance_{model_name.lower()}.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                saved_plots.append(f"feature_importance_{model_name.lower()}.png")
            except:
                print(f"CatBoost 특성 중요도 추출 실패")
    
    return saved_plots

def create_external_plots(results, external_results, result_dir, save_plots=True):
    """
    외부 데이터 평가 결과를 시각화하고 저장
    
    Args:
        results: train 함수의 반환값
        external_results: 외부 데이터 평가 결과
        result_dir: 결과 저장 디렉토리
        save_plots: 그림 저장 여부
    
    Returns:
        list: 저장된 그림 파일명 리스트
    """
    if not save_plots or not external_results:
        return []
    
    saved_plots = []
    
    # 1. 테스트 데이터 vs 외부 데이터 성능 비교
    plt.figure(figsize=(15, 6))
    
    models = list(results['all_results'].keys())
    test_accuracies = [results['all_results'][model]['test_accuracy'] for model in models]
    external_accuracies = []
    
    for model in models:
        if model in external_results and external_results[model]:
            external_accuracies.append(external_results[model]['external_accuracy'])
        else:
            external_accuracies.append(0)
    
    x = np.arange(len(models))
    width = 0.35
    
    plt.subplot(1, 2, 1)
    plt.bar(x - width/2, test_accuracies, width, label='Test Data', alpha=0.8)
    plt.bar(x + width/2, external_accuracies, width, label='External Data', alpha=0.8)
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.title('Test vs External Data Accuracy Comparison')
    plt.xticks(x, models, rotation=45)
    plt.legend()
    
    # F1 점수 비교
    test_f1s = [results['all_results'][model]['test_f1'] for model in models]
    external_f1s = []
    
    for model in models:
        if model in external_results and external_results[model]:
            external_f1s.append(external_results[model]['external_f1'])
        else:
            external_f1s.append(0)
    
    plt.subplot(1, 2, 2)
    plt.bar(x - width/2, test_f1s, width, label='Test Data', alpha=0.8)
    plt.bar(x + width/2, external_f1s, width, label='External Data', alpha=0.8)
    plt.xlabel('Models')
    plt.ylabel('F1 Score')
    plt.title('Test vs External Data F1 Comparison')
    plt.xticks(x, models, rotation=45)
    plt.legend()
    
    plt.tight_layout()
    plot_path = result_dir / "test_vs_external_performance.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    saved_plots.append("test_vs_external_performance.png")
    
    # 2. 외부 데이터 혼동 행렬 히트맵
    for model_name, ext_result in external_results.items():
        if ext_result is None:
            continue
            
        plt.figure(figsize=(8, 6))
        cm = ext_result['external_confusion_matrix']
        
        # 클래스 레이블 생성
        n_classes = cm.shape[0]
        class_labels = [f'Class {i}' for i in range(n_classes)]
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', 
                   xticklabels=class_labels, yticklabels=class_labels)
        plt.title(f'External Data Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        plot_path = result_dir / f"external_confusion_matrix_{model_name.lower()}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        saved_plots.append(f"external_confusion_matrix_{model_name.lower()}.png")
    
    # 3. 성능 변화율 계산 및 시각화
    performance_changes = []
    model_names = []
    
    for model_name in models:
        if model_name in external_results and external_results[model_name]:
            test_acc = results['all_results'][model_name]['test_accuracy']
            ext_acc = external_results[model_name]['external_accuracy']
            change_rate = ((ext_acc - test_acc) / test_acc) * 100
            performance_changes.append(change_rate)
            model_names.append(model_name)
    
    if performance_changes:
        plt.figure(figsize=(10, 6))
        colors = ['red' if x < 0 else 'green' for x in performance_changes]
        plt.bar(model_names, performance_changes, color=colors, alpha=0.7)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.xlabel('Models')
        plt.ylabel('Performance Change (%)')
        plt.title('Performance Change: External vs Test Data')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_path = result_dir / "performance_change_external_vs_test.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        saved_plots.append("performance_change_external_vs_test.png")
    
    return saved_plots

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='conf/config.yaml', help='설정 파일 경로')
    parser.add_argument('--feature_config_path', type=str, default='conf/data/features.yaml', help='특성 설정 파일 경로')
    parser.add_argument('--ml_config_path', type=str, default='conf/ml_config.yaml', help='ML 설정 파일 경로')
    parser.add_argument('--train_data_type', type=str, default='split', help='훈련 데이터 경로')
    parser.add_argument('--test_data_type', type=str, default='split', help='테스트 데이터 경로')
    parser.add_argument('--column_category', type=str, default='elastic_significant_20250610', help='yaml에 있는 컬럼 분류')
    parser.add_argument('--save_results', action='store_true', default=True, help='결과를 파일로 저장할지 여부')
    parser.add_argument('--no_save_results', dest='save_results', action='store_false', help='결과 저장 안함')
    parser.add_argument('--output_dir', type=str, default='results', help='결과 저장 디렉토리')
    parser.add_argument('--external_data_path', type=str, default='data/external/external_dataset.csv', help='외부 데이터 파일 경로')
    parser.add_argument('--evaluate_external', action='store_true', default=True, help='외부 데이터로 평가할지 여부')
    parser.add_argument('--no_evaluate_external', dest='evaluate_external', action='store_false', help='외부 데이터 평가 안함')
    args = parser.parse_args()

    config_path = args.config_path
    feature_config_path = args.feature_config_path
    ml_config_path = args.ml_config_path
    train_data_type = args.train_data_type
    test_data_type = args.test_data_type
    column_category = args.column_category
    external_data_path = args.external_data_path
    evaluate_external = args.evaluate_external

    # 데이터 불러오기, 컬럼 목록 설정
    x_train, y_train, x_test, y_test, ml_config = loader(config_path, feature_config_path, ml_config_path, train_data_type, test_data_type, column_category)
    print(f"훈련 데이터 형태: {x_train.shape}")
    print(f"테스트 데이터 형태: {x_test.shape}")
    print(f"훈련 타겟 형태: {y_train.shape}")
    print(f"테스트 타겟 형태: {y_test.shape}")
    print(f"사용된 ML 설정: {ml_config_path}")

    # 머신러닝 모델 학습
    results = train(x_train, y_train, x_test, y_test, ml_config)

    # 외부 데이터 평가
    external_results = True
    x_external_shape = None
    if evaluate_external:
        try:
            print(f"\n=== 외부 데이터 로딩 및 평가 ===")
            x_external, y_external = load_external_data(config_path, feature_config_path, ml_config_path, external_data_path, column_category)
            x_external_shape = x_external.shape
            print(f"외부 데이터 로딩 완료: {x_external.shape}")
            
            external_results = evaluate_external_performance(results, x_external, y_external)
            
        except Exception as e:
            print(f"외부 데이터 평가 중 오류 발생: {e}")
            print("외부 데이터 평가를 건너뜁니다.")

    # 평가 결과 출력 및 저장
    print_and_log_results(results, ml_config, x_train.shape, x_test.shape, 
                         save_results_flag=args.save_results, 
                         external_results=external_results, 
                         x_external_shape=x_external_shape)