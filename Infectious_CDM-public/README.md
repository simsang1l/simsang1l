# 감염병 CDM 구축 — 다기관 ETL 프레임워크

3종 감염병(코로나19, 쯔쯔가무시, SFTS) 데이터를 OMOP-CDM 표준 모델로 변환하는 ETL 프레임워크.
4개 대학병원(전북대·전남대·경북대·계명대)에 동일 코드로 적용했습니다.

---

### 주요 수치

| 항목 | 내용 |
|------|------|
| 적용 기관 | **4개** 대학병원 (전북대, 전남대, 경북대, 계명대) |
| 대상 감염병 | **3종** (코로나19, 쯔쯔가무시, SFTS) |
| 구축 CDM 테이블 | **15개** (OMOP-CDM 표준 + 로컬 매핑 테이블) |
| 용어 매핑률 | **80% 이상** (KCD7 / EDI / ATC / LOINC / SNOMED-CT) |
| 데이터 일치율 | **90% 이상** (원본 대비 CDM 적재 건수) |
| 품질 검증 | ISO/IEC 25024 기반 자체 QC 프레임워크 구축 |
| 발표 | [OMOP-CDM 데이터 구축 방법 강의](https://youtu.be/5OuzHSFHGtI) |

---

## Architecture

```
병원 EMR (CSV)
      │
      ▼
config.yaml ──── 병원별 컬럼명, 경로, 인코딩, 변환 규칙
      │
      ▼
DataTransformer (Python OOP)
 ├── 원천 데이터 읽기 → 코호트 필터링
 ├── 표준 용어 매핑 (KCD7 / EDI / ATC / LOINC / SNOMED-CT)
 └── CDM 스키마 변환 후 저장
      │
      ▼
QC Framework
 ├── 테이블 건수 비교 (원본 vs CDM, 10개 테이블)
 ├── 필드별 분포 요약 (null 비율, 유니크 값, 기초통계)
 ├── DQ 검사 (완전성 / 유효성 / 일관성 / 정확성)
 └── 결과 리포트 자동 생성 (Excel)
```

---

## 아키텍처 진화

### 과거 — 테이블별 개별 스크립트

```
care_site.py / person.py / visit_occurrence.py / ...
```

테이블마다 독립된 스크립트. 공통 로직(CSV I/O, 로깅, config 로딩)이 매 파일에 중복되어 유지보수가 어려웠습니다.

### 현재 — OOP DataTransformer + Config 기반 다기관 적용

```python
class DataTransformer:                        # 공통 로직 (config, CSV I/O, logging)
class PersonTransformer(DataTransformer):     # 테이블별 변환 로직만 구현
class DrugexposureChunkTransformer(...):      # 대용량 테이블은 청크 처리 (100,000건 단위)
```

`config.yaml`에 병원별 차이(컬럼명, 경로, 인코딩)만 정의하면 **코드 수정 없이** 새 기관에 적용 가능합니다.
3·4번째 병원은 설정 파일 작성만으로 적용했습니다.

```yaml
# config.yaml — 병원마다 이 파일만 교체
source_path: "./source"
CDM_path: "./cdm"
hospital_code: "JBUH"          # 기관 코드
diag_condition: "A9380"        # 감염병 진단코드 (코호트 필터 조건)
chunksize: 100000

person:
  columns:
    gender_source_value: "SEX"      # 병원 EMR 컬럼명을 여기서 정의
    death_datetime: "DIEDATE"
```

> 전체 코드: [`examples/DataTransformer.py`](./examples/DataTransformer.py)

---

## 품질 검증 (QC)

OHDSI DQD만으로는 감염병 코호트 특성에 맞는 검증이 부족해 ISO/IEC 25024 기반 자체 프레임워크를 설계했습니다.

| 기준 | 주요 검증 내용 |
|------|--------------|
| **완전성** | NOT NULL 컬럼 값 존재 여부, 원본 대비 CDM 레코드 수 일치 |
| **유효성** | 데이터 타입, 허용 코드 리스트, 수치 범위 |
| **일관성** | FK 참조 무결성, 날짜 포맷, 공통 어휘 일관성 |
| **정확성** | 메타데이터 정확성, 원본 대비 값 정확성 |

검증 결과는 Excel 리포트(원본 비교 / 필드 분포 / DQ 점검사항 / 메타데이터)로 자동 생성됩니다.

> 상세 코드: [`examples/QC/`](./examples/QC/)

---

## Troubleshooting

**코드 사용 기간으로 인한 데이터 누락**

원내 검사 코드가 특정 기간에만 유효한 경우, 해당 기간 밖 처방 데이터가 매핑에서 누락되는 문제가 발생했습니다.

**해결**: 코드 유효 기간을 고려한 매핑 로직 추가 + 만료 코드는 후속 코드로 연결하는 매핑 테이블을 별도 관리.

---

## Lessons Learned

**Config 기반 다기관 확장**
두 번째 병원 적용 시점에서 공통 로직과 병원별 차이를 분리하는 설계로 전환했습니다. 이후 기관 추가 시 설정 파일 작성만으로 적용이 가능해졌고, 코드 중복 없이 일관된 ETL 품질을 유지할 수 있었습니다.

**자체 QC 프레임워크**
외부 도구로 커버되지 않는 도메인 특화 검증 항목(코호트 건수 정합성, 검사 단위 매핑 정확성 등)을 직접 설계하고 자동화했습니다.

**용어 매핑의 한계와 보완**
80% 이상 매핑을 달성했지만, 코드 유효 기간·병원별 표기 차이 등으로 일부 미매핑이 발생했습니다. 미매핑 항목을 추적해 원인을 분석하고 매핑률을 지속 개선하는 워크플로를 구축했습니다.

---

## Tech Stack

Python · pandas · PyYAML · OMOP-CDM v5.4 · KCD7 / EDI / ATC / LOINC / SNOMED-CT

---

작성자: 심상일 · [simsang1db@gmail.com](mailto:simsang1db@gmail.com)
