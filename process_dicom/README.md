# 의료 영상 큐레이션 파이프라인

외부 기관에서 개발한 AI 분류 모델(X-Ray 촬영 방향, MRI 시퀀스)을 본원 데이터로 검증하기 위한 파이프라인입니다. 가명화 처리된 DICOM 파일을 전처리하고, CDM(병원 데이터 표준화 DB)의 환자 ID와 연결해 멀티모달 데이터셋을 구축한 뒤 딥러닝 모델 성능을 검증합니다.

---

### 주요 수치

| 항목 | 내용 |
|------|------|
| 처리 데이터 규모 | **2500만 건** DICOM 영상 (2023.09 ~ 2024.05) |
| AI 모델 성능 향상 | 전처리(역상 보정) 적용으로 Accuracy **40% → 87%** |
| 처리 속도 개선 | 파일 rename·저장 프로세스 **1주 → 2일** (멀티프로세싱) |
| 지원 모달리티 | X-Ray (CR) · MRI (MR) · CT |
| 버전 이력 | v0.1 → **v0.8** (8회 반복 개선) |

---

## 전체 흐름

```
병원 서버 DICOM 파일
      │
      ▼
1. DICOM 태그 추출
   ├── 핵심 태그 → PostgreSQL (메타데이터 관리)
   └── 전체 태그 → Parquet 파일 (snappy 압축, 분석용)
      │
      ▼
2. CDM 연계 및 파일 리네이밍
   └── 가명화된 DICOM의 원내 환자번호 → CDM patient_id 매핑 → DB 저장
       (CDM: 병원 EMR을 표준 형식으로 변환한 DB. 영상-임상 데이터 통합에 필요)
      │
      ▼
3. 모달리티별 전처리
   ├── X-Ray: 역상 보정 (배경 흰색 이미지 자동 감지·반전)
   └── MRI:   DICOM → NIfTI 변환 (병렬 처리)
      │
      ▼
4. 자동 레이블링 (DICOM 태그 기반 규칙)
   ├── X-Ray: AP / PA / Lateral / Others
   └── MRI:   T1ce / bb / Others
      │
      ▼
5. 외부 AI 분류 모델 적용 (Docker 이미지로 제공받아 본원 환경에 배포·실행)
      │
      ▼
6. 예측 결과 검증
   └── Accuracy / Precision / Recall / Specificity
```

---

## 아키텍처 진화

### 핵심 병목: 수백만 건의 DICOM 파일 처리 속도

초기에는 단일 프로세스로 순차 처리했고, 수백만 건 규모에서 처리 시간이 병목이 됐습니다.
속도 개선을 위해 **멀티프로세싱**과 **비동기 I/O** 두 방식을 직접 구현·비교했습니다.

**① 순차 처리 (기준점)**

```python
for file in files:
    extracted_tags = process_dicom_file(file, ...)
```

**② 비동기 I/O (`asyncio` + `aiofiles`) — 실험**

```python
async def process_file_async(path):
    async with aiofiles.open(path, 'rb') as f:
        content = await f.read()          # I/O는 비동기
    img = pydicom.dcmread(BytesIO(content))  # 파싱은 동기·CPU bound → 이벤트 루프 블로킹
    ...

results = await asyncio.gather(*[process_file_async(f) for f in files])
```

asyncio는 I/O 대기 시간을 재활용하는 방식입니다. 그러나 `pydicom`의 DICOM 파싱은 C 확장 기반의 동기 함수이므로, 파일 읽기를 아무리 비동기로 처리해도 **파싱 단계에서 이벤트 루프가 블로킹**되어 사실상 순차 처리와 차이가 없었습니다.

**③ 멀티프로세싱 (v0.8 채택) — 1주 → 2일**

```python
# 프로세스를 분리하면 GIL 제약 없이 CPU 코어를 병렬 활용 가능
with Pool(processes=cpu_count()) as pool:
    results = list(tqdm(pool.imap(process_dicom_file, files), total=len(files)))
```

DICOM 파싱은 CPU bound이므로, GIL 제약이 없는 멀티프로세싱이 적합했습니다. CPU 코어 수만큼 프로세스를 생성해 파일을 병렬 처리하여 **처리 시간을 1주에서 2일로 단축**했습니다.

---

## 역상 보정 — Accuracy 40% → 87%

### 문제

병원 장비마다 X-Ray DICOM의 배경이 흰색 또는 검은색으로 다릅니다. 외부 AI 모델은 **검은 배경(흉부가 밝음)** 기준으로 학습되어 있어, 흰 배경 이미지를 그대로 입력하면 예측이 크게 틀렸습니다.

```
흰 배경 이미지 (보정 전)    →    AI 모델 예측 Accuracy 40%
검은 배경 이미지 (보정 후)  →    AI 모델 예측 Accuracy 87%
```

### 해결

픽셀 4개 모서리의 평균 밝기를 샘플링해 배경색을 자동 판별하고, 흰 배경이면 색상을 반전해 저장합니다.

```python
# 배경 판별: 4개 모서리 픽셀의 평균 밝기와 동적 임계값 비교
avg_corner = np.mean([pixel[0,0], pixel[0,-1], pixel[-1,0], pixel[-1,-1]])
threshold  = np.max(pixel) / 2   # 비트 깊이(8·16bit)에 무관하게 동작

if avg_corner > threshold:       # 밝은 배경 → 역상 처리
    inverted = np.invert(pixel_array).astype(ds.pixel_array.dtype)
    ds.PixelData = inverted.tobytes()
    ds.save_as(output_path)
```

비트 깊이가 8bit/16bit로 병원마다 달라, 고정 임계값(127) 대신 `max/2` 방식으로 동적 임계값을 사용했습니다.

---

## 레이블링 규칙

AI 모델의 예측을 검증하려면 정답(ground truth) 레이블이 필요합니다. 수만 건을 수동 레이블링하는 대신, DICOM 태그에 이미 촬영 방향·시퀀스 정보가 담겨 있다는 점을 활용해 **자동 레이블링**을 구현했습니다.

태그가 없거나 병원 간 표기가 달라(`ViewPosition`이 비어 있고 `SeriesDescription`에만 방향이 적힌 경우 등) 우선순위 기반 규칙으로 설계했습니다.

### X-Ray (촬영 방향 분류)

> AP: 정면 전후방 촬영 · PA: 정면 후전방 촬영 · Lateral: 측면 촬영

| 레이블 | 우선순위 1 | 우선순위 2 | 우선순위 3 |
|--------|-----------|-----------|-----------|
| AP | ViewPosition = AP | SeriesDescription / ProtocolName에 AP | 위 둘이 없고 StudyDescription에 AP |
| PA | ViewPosition = PA | SeriesDescription / ProtocolName에 PA | 위 둘이 없고 StudyDescription에 PA |
| Lateral | ViewPosition에 LAT | SeriesDescription / ProtocolName에 LAT | 위 둘이 없고 StudyDescription에 LAT |
| Others | 위 조건 모두 미해당 | | |

### MRI 시퀀스 분류 (국립암센터 기준)

> T1ce: 조영증강 T1 시퀀스 (혈관·종양 강조) · bb: Black Blood 시퀀스 (혈관벽 강조)

| 레이블 | 조건 |
|--------|------|
| T1ce | ScanningSequence=GR AND RepetitionTime≤1000 (TOF·mIP·reformat 제외) OR SeriesDescription에 t1ce 포함 |
| bb | ScanningSequence=IR AND 2000≤RepetitionTime≤3000 OR SeriesDescription에 bb 포함 |
| Others | 위 조건 미해당 |

---

## 주요 기술 설계

### 이중 저장 구조

전체 DICOM 태그를 DB에 저장하면 파일 1개당 수백 행이 생성됩니다. 용도에 따라 저장 방식을 분리했습니다.

```
핵심 태그 + CDM 연계 → PostgreSQL
  ├── 조회·필터·조인에 최적화
  └── 파일 리네이밍 이력, 레이블, CDM patient_id 포함

전체 태그 (수백 개/파일) → Parquet (snappy 압축)
  └── 태그 분포 분석, 보유율 계산 등 사후 분석용
```

### 성능 검증

예측 결과(AI 모델 output)와 자동 생성 레이블을 매핑해 4가지 지표를 계산합니다.

```python
accuracy  = accuracy_score(label, pred)
precision = precision_score(label, pred, average='weighted')
recall    = recall_score(label, pred, average='weighted')

# sklearn에 없는 Specificity를 클래스별로 직접 계산 후 평균
for i in range(num_classes):
    tn = np.sum((actual != i) & (predicted != i))
    fp = np.sum((actual != i) & (predicted == i))
    specificity = tn / (tn + fp)
```

---

## 프로젝트 구조

```
process_dicom/
├── README.md
└── examples/
    ├── process_dicom.py         # DICOM 태그 추출 · 파일 리네이밍 · DB 저장 (메인)
    ├── process_dicom_ct.py      # CT 전용 처리
    ├── dicom_key.py             # DB 스키마 정의 (SQLAlchemy ORM)
    ├── xray_util.py             # X-Ray 레이블링 · 역상 보정 · Metric 계산
    ├── mri_util.py              # MRI NIfTI 변환 · 레이블링 · Metric 계산
    ├── calc_metric.py           # X-Ray / MRI Metric 통합 계산
    ├── display_images.py        # DICOM 이미지 시각화 (단일 / 배치 / 원본-보정 비교)
    └── utils/
        ├── calc_parquet_org.py          # 코호트·모달리티·시리즈별 태그 수 집계
        └── calc_parquet_series_rate.py  # 태그 보유율 계산
```

---

## Tech Stack

| 구분 | 기술 |
|------|------|
| DICOM 처리 | pydicom, dicom2nifti |
| 병렬 처리 | multiprocessing (Pool), asyncio + aiofiles |
| DB | PostgreSQL (psycopg2, SQLAlchemy) |
| 파일 저장 | Parquet (pyarrow, snappy 압축) |
| 이미지 처리 | PIL/Pillow, matplotlib |
| 성능 평가 | scikit-learn |
| 환경 관리 | python-dotenv, Docker |

---

## Lessons Learned

**CPU bound vs I/O bound — 병렬화 방식 선택**
asyncio는 I/O 대기 시간을 재활용하는 방식이라, DICOM 파싱처럼 CPU가 병목인 작업에는 맞지 않았습니다. `pydicom`은 C 확장 기반 동기 함수이기 때문에, 파일 읽기를 비동기로 처리해도 파싱 단계에서 이벤트 루프가 블로킹됩니다. 두 방식을 실제 구현해 비교한 결과, GIL 제약 없이 CPU 코어를 활용하는 멀티프로세싱이 이 작업에 적합하다는 것을 확인했습니다.

**태그 기반 레이블링의 한계와 개선**
병원마다 동일한 촬영 방식을 다른 태그에 기록하는 경우가 있었습니다. `ViewPosition`이 비어 있고 `SeriesDescription`에만 방향이 적혀 있는 경우 등, 예외 케이스를 운영 중 발견할 때마다 우선순위 규칙을 추가하며 v0.1 → v0.8까지 반복 개선했습니다.

**동적 임계값으로 비트 깊이 차이 흡수**
역상 판별에 고정 임계값(127)을 쓰면 16bit DICOM에서 오작동합니다. `max_pixel / 2`로 동적 임계값을 계산하면 비트 깊이에 무관하게 동일한 로직으로 처리할 수 있었습니다.

**저장 전략 분리**
전체 DICOM 태그를 DB에 저장하면 2500만 건 기준 수십억 행이 생성됩니다. 조회·매핑용은 PostgreSQL, 분석·집계용은 Parquet으로 분리해 각 용도에 맞는 성능을 확보했습니다.

---

작성자: 심상일 · [simsang1db@gmail.com](mailto:simsang1db@gmail.com)
