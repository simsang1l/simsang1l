## Hi, I'm Sang-Il Sim 👋

2TB DW 구축과 2,500만 건 대용량 데이터 처리를 위한 ETL 파이프라인을 설계·운영하고, 다기관 데이터 표준화 프레임워크를 4개 기관에 적용했습니다.

- 다기관 ETL 프레임워크를 설계하여 **4개 대학병원**에 config 전환만으로 적용
- 의료 AI 증강 인터페이스 **특허 등록** · K-CURE 경진대회 **최우수상**

<p>
  ✉️ <a href="mailto:simsang1db@gmail.com">simsang1db@gmail.com</a>
</p>

> ⚠️ 의료 데이터를 다루는 프로젝트 특성상, 실행 가능한 코드가 아닌 **설계·구현 방식을 보여주는 예시 코드**를 포함하고 있습니다.

---

### Tech Stack

**Languages**&ensp; Python · SQL
**Database**&ensp; PostgreSQL · Oracle · MariaDB
**Pipeline**&ensp; Apache Airflow · Docker · Linux
**Big Data**&ensp; Apache Spark · Apache Kafka
**Monitoring**&ensp; ElasticSearch · Logstash · Grafana · Apache Superset
**Tools**&ensp; Git

---

### Projects

| 프로젝트 | 설명 | 규모 / 성과 | 주요 기술 |
|---------|------|------------|----------|
| [OMOP-CDM 구축](./JBUH-CDM-public) | 병원 EMR → OMOP-CDM 표준 변환 ETL 파이프라인 | **2TB** 변환, 쿼리 24h→2h | Airflow, PostgreSQL, Oracle, Docker |
| [감염병 CDM 구축](./Infectious_CDM-public) | 3종 감염병 데이터 다기관 OMOP-CDM 변환 프레임워크 | **4개 병원** 적용, 매핑률 80%+ | Python, PostgreSQL |
| [의료 영상 큐레이션](./process_dicom) | DICOM 메타데이터 추출·전처리·CDM 연계 파이프라인 | **2,500만 건**, Accuracy 40%→87% | Python, PostgreSQL, Docker |
| [K-CURE 경진대회](./K-CURE) | 20GB+ 의료 데이터 통합 → 폐암 생존 분석 | 🏆 **최우수상** | Python, pandas, lifelines |
| [NICU BMI 예측모델](./KNN-BMI-main) | 신생아 퇴원 시 비정상 BMI 위험 예측 ML 모델 | **논문 게재** | Python, scikit-learn, R |
| [로그 모니터](https://github.com/simsang1l/log_monitor) | 서버 SSH 로그 실시간 수집·분석·시각화 파이프라인 | — | Kafka, Spark, Airflow, ELK, Grafana |
| [질병 진단 증강 UI](./AugmentedUI) | AI 예측 결과를 의료 영상 위에 오버레이하는 데스크톱 앱 | **특허 등록** · 학회 논문 | Python, PyTorch, PyQt |
