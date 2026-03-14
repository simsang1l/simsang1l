# NICU 퇴원 BMI 예측 파이프라인

한국신생아네트워크(KNN) 다기관 임상 데이터를 활용해 신생아중환자실(NICU) 퇴원 시 BMI 위험 인자를 분석하고, 퇴원 BMI를 예측하는 머신러닝 파이프라인입니다.

## 목차

1. [주요 수치](#주요-수치)
2. [전체 흐름](#전체-흐름)
3. [핵심 설계 결정](#핵심-설계-결정)
   - [BMI z-score — 절대값 대신 보정값](#bmi-z-score--절대값-대신-보정값)
   - [시간 기반 외부 검증 분할](#시간-기반-외부-검증-분할)
   - [다단계 변수 선택 파이프라인](#다단계-변수-선택-파이프라인)
   - [Config 기반 변수·실험 관리](#config-기반-변수실험-관리)
   - [Step Registry — 단계별 실행 및 의존성 관리](#step-registry--단계별-실행-및-의존성-관리)
4. [프로젝트 구조](#프로젝트-구조)
5. [Tech Stack](#tech-stack)
6. [Lessons Learned](#lessons-learned)

---

### 주요 수치

| 항목 | 내용 |
|------|------|
| 데이터 출처 | 한국신생아네트워크(KNN) 다기관 공개 데이터 |
| 분류 목표 | 퇴원 BMI z-score → **Normal / Low / High** (3분류) |
| 입력 변수 | 산모·임신·신생아·질환 정보 **100+ 임상 변수** |
| 외부 검증 | 2020년 기준 시간 분할 (derivation / external validation) |
| ML 모델 | Logistic Regression · RandomForest · XGBoost · LightGBM · CatBoost |

---

## 전체 흐름

```
KNN 공개 데이터 (다기관 NICU 임상 데이터)
      │
      ▼
1. 전처리 (preprocess)
   ├── 100+ 임상 변수 타입 분류 (category / numeric / base)
   ├── BMI z-score 계산 (재태주수·성별 보정, LMS method)
   └── 파생 변수 생성 (NRP grade, 28일 이내 처치 여부 등)
      │
      ▼
2. 선정·제외 조건 적용 (filter)
   └── 재태주수 서브그룹 분리 (< 28주 / 28~32주 / 32~36주 / 36~42주)
      │
      ▼
3. 데이터 분할 (split)
   └── 2020년 기준 시간 분할 → derivation(학습) / external validation(검증)
      │
      ▼
4. 통계 분석 (stats)
   ├── Table 1 (기저 특성 비교)
   ├── 단변량 분석: 상관계수·p-value·Odds Ratio
   ├── 다중공선성 검사: VIF
   └── 다변량 분석: 로지스틱 회귀 · Elastic Net · Backward Elimination
      │
      ▼
5. 머신러닝 모델 학습 및 외부 검증
   └── LR · RF · XGBoost · LightGBM · CatBoost (SMOTE 불균형 처리)
```

---

## 핵심 설계 결정

### BMI z-score — 절대값 대신 보정값

퇴원 시 BMI 절대값은 재태주수와 성별에 따라 기준이 달라 의미 있는 비교가 어렵습니다. 미숙아 전용 성장 참조값(LMS method)을 코드로 직접 구현해 재태주수·성별 보정 z-score를 계산하고, WHO 기준 ±1.5 SD를 기준으로 3분류했습니다.

```python
def bmi_zscore(weight, length, ga, gender):
    """
    미숙아 BMI Curves 논문의 LMS 참조값 (재태주수 24~41주, 성별 분리)
    z-score = ((BMI / M) ** L - 1) / (L * S)
    """
    ref = BMI_REFERENCE[gender][ga]
    L, M, S = ref["L"], ref["M"], ref["S"]
    bmi = (weight / (length ** 2)) * 10
    return ((bmi / M) ** L - 1) / (L * S)
```

일반 성인·소아 BMI z-score 라이브러리는 미숙아 재태주수를 지원하지 않아 논문 기반으로 직접 구현했습니다.

---

### 시간 기반 외부 검증 분할

Random split 대신 **2020년을 기준으로** 이전 데이터는 derivation cohort(모델 개발), 이후 데이터는 external validation cohort(성능 검증)로 분리했습니다.

```python
# config.yaml
split_year: 2020

# 시간 기반 분할 — 미래 데이터로 과거 모델 평가
derivation = df[df["birth_year"] < split_year]
validation = df[df["birth_year"] >= split_year]
```

시간 순서를 무시한 random split은 데이터 누수(data leakage) 위험이 있고, 실제 임상 환경(미래 환자에게 적용)을 반영하지 못합니다.

---

### 다단계 변수 선택 파이프라인

100+ 임상 변수를 그대로 모델에 투입하면 다중공선성·과적합 문제가 생깁니다. 단계별로 변수를 줄여나갔습니다.

```
① 상관계수 필터     |r| ≥ 0.1 (Pearson)
       ↓
② 유의성 검정       p < 0.05 (Chi-square / Kruskal-Wallis)
       ↓
③ 효과크기 확인     OR 95% CI에 1 미포함
       ↓
④ 다중공선성 제거   VIF < 5
       ↓
⑤ 단계적 제거       Backward Elimination
       ↓
⑥ 정규화 선택       Elastic Net (L1+L2 페널티)
```

각 단계를 독립 모듈로 구현해 `python main.py --step corr` 처럼 단계별로 실행하고 결과를 확인할 수 있습니다.

---

### Config 기반 변수·실험 관리

100+ 변수의 타입(category/numeric)과 분석 목적별 변수 셋을 코드 밖에서 관리합니다.

```yaml
# conf/data/features.yaml
features_preprocessing:
  gagew: numeric    # 임신 나이(Gestational age) 주
  rds:   category   # 신생아 호흡곤란증후군(RDS) 유무
  bwei:  numeric    # 출생체중
  ...

corr_columns_bmi_exp:   # 분석 목적별 변수 셋
  categorical: [resui, rds, sft, lbp, ...]
  numerical:   [gagew, apgs1, bwei, ...]
```

변수를 추가하거나 분석 셋을 변경할 때 코드를 수정하지 않고 YAML만 편집합니다.

---

### Step Registry — 단계별 실행 및 의존성 관리

`main.py`에 모든 분석 단계를 registry로 등록하고, 의존성을 자동으로 해결합니다.

```python
STEPS = [
    ("preprocess", "src.data.make_dataset", "make_preprocessed_data", []),
    ("filter",     "src.data.make_dataset", "make_filtered_data",     ["preprocess"]),
    ("split",      "src.data.make_dataset", "make_split_data",        ["filter", "preprocess"]),
    ("corr",       "src.stats.stats_runner", "run_corr",              []),
    ...
]

# split 실행 시 의존 단계(preprocess → filter)를 자동으로 먼저 실행
python main.py --step split
```

분석 순서가 바뀌거나 특정 단계만 재실행할 때 의존성을 수동으로 추적하지 않아도 됩니다.

---

## 프로젝트 구조

```
KNN-BMI-main/
├── README.md
├── main.py                    # CLI 진입점 · Step registry · 의존성 해결
├── conf/
│   ├── config.yaml            # 실험 설정 (cutoff, 경로, 파일명 등)
│   └── data/
│       └── features.yaml      # 변수 정의 · 분석 목적별 변수 셋
├── src/
│   ├── data/
│   │   ├── make_dataset.py    # 전처리 · 필터링 · 분할 실행 함수
│   │   └── dataset_utils.py   # BMI z-score · 전처리 · 분할 로직
│   ├── stats/
│   │   ├── stats_runner.py    # 통계 분석 단계별 실행 함수
│   │   ├── stats_utils.py     # 상관분석 · VIF · 로지스틱 회귀 · Elastic Net
│   │   ├── backward_elimination.py
│   │   └── linear.py
│   ├── ml/
│   │   ├── ml_main.py         # ML 모델 학습 · 평가 · 외부 검증
│   │   └── binary_classification.py
│   └── utils/
│       └── utils.py           # 데이터 로드 · 저장 · 로깅 공통 유틸
└── ref/                       # WHO BMI 참조값 테이블 (xlsx)
```

---

## Tech Stack

| 구분 | 기술 |
|------|------|
| 데이터 처리 | pandas, numpy |
| 통계 분석 | scipy, statsmodels, pingouin |
| ML 모델 | scikit-learn, XGBoost, LightGBM, CatBoost |
| 불균형 처리 | imbalanced-learn (SMOTE) |
| 설정 관리 | PyYAML |
| 로깅 | Python logging (KST 기준 일별 로테이션) |

---

## Lessons Learned

**BMI z-score 직접 구현의 필요성**
미숙아(24~41주)의 성장 기준은 일반 소아와 달라 범용 라이브러리로 처리할 수 없었습니다. 논문(BMI Curves for Preterm Infants)의 LMS 참조값을 직접 코드로 구현해 재태주수·성별별 z-score를 계산했습니다.

**시간 기반 분할의 중요성**
Random split은 미래 데이터가 학습에 섞여 성능이 과대평가될 위험이 있습니다. 연도 기준으로 분리해 실제 임상 적용(과거 데이터로 학습 → 미래 환자에게 적용) 시나리오를 반영했습니다.

**변수 선택을 단계적으로 해야 하는 이유**
처음에는 상관계수만으로 변수를 선택했으나, VIF 검사를 하지 않으면 다중공선성이 높은 변수들이 함께 모델에 들어가 계수 해석이 불안정해집니다. 상관 → p-value → OR → VIF → backward 순서로 단계를 나눠 각 기준이 다른 정보를 제공함을 확인했습니다.

**Config 기반 관리가 필요한 규모**
100+ 변수를 코드 안에 하드코딩하면 분석 목적이 바뀔 때마다 코드를 수정해야 합니다. YAML로 변수 셋을 분리하자 `corr_columns_bmi`, `corr_columns_zscore` 등 목적별 변수 셋을 독립적으로 관리할 수 있었습니다.

---

작성자: 심상일 · [simsang1db@gmail.com](mailto:simsang1db@gmail.com)
