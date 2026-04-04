## Hi, I'm Sang-Il Sim 👋

2TB DW 구축, 2,500만 건 대용량 데이터 처리를 위한 ETL 파이프라인을 설계·운영하고, 다기관 데이터 표준화 ETL 구조를 설계하여 4개 기관에 데이터마트를 구축했습니다.

- 다기관 ETL 구조를 설계하여 **4개 기관**에 구축
- K-CURE 경진대회 **최우수상** · AI 증강 인터페이스 **특허 등록**

<p>
  ✉️ <a href="mailto:simsang1db@gmail.com">simsang1db@gmail.com</a>
</p>

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
| [대규모 데이터 표준화 ETL 파이프라인 (OMOP-CDM)](./JBUH-CDM-public) | 2TB 규모 이기종 DB 간 데이터 표준화 파이프라인 설계 및 고도화 (Oracle → PostgreSQL 이관 포함) | **2TB** DW, 쿼리 24h→3h, SCI 논문 기여 | Airflow, Spark, PostgreSQL, Oracle, Docker |
| [4개 기관 데이터 표준화 ETL 구조 설계](./Infectious_CDM-public) | 기관별 상이한 용어·형식·테이블 구조를 분석하여 표준화 데이터 구축 | **4개 기관** 적용, 매핑률 80%+, 10개 기관 교육 | Python, PostgreSQL |
| [실시간 서버 로그 모니터링 파이프라인](https://github.com/simsang1l/log_monitor) | 실시간 SSH 로그 수집·분석 및 이상 패턴 탐지 파이프라인 | 개인 프로젝트 | Kafka, Spark Streaming, ELK, Grafana |
| [대규모 데이터 통합 분석 — K-CURE 경진대회](./K-CURE) | 20GB+ 8개 이종 테이블 데이터 통합 파이프라인 구축 및 분석 코호트 생성 | 🏆 **최우수상** | Python, pandas, scipy, statsmodels |
| [위험 인자 분석 및 ML 기반 예측 모델 개발](./KNN-BMI-main) | 위험 인자 탐색 및 ML 기반 예측 모델 개발 | **논문 under review** | Python, scikit-learn, XGBoost, R |
| [이종 데이터 통합 파이프라인 (2,500만건 영상-DB 매칭)](./process_dicom) | 2,500만건 영상 파일과 RDBMS 간 매핑 로직 설계 및 데이터셋 구축 | **2,500만 건**, 매칭률 80%+ | Python, PostgreSQL, Docker |
| [AI 증강 인터페이스](./AugmentedUI) | AI 예측 결과를 영상 위에 오버레이하는 데스크톱 앱 | **특허 등록** · 학회 논문 | Python, PyTorch, PyQt |
