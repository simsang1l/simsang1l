## Hi, I'm Sang-Il Sim 👋

2TB DW 구축과 2,500만 건 대용량 데이터 처리를 위한 ETL 파이프라인을 설계·운영하고, 다기관 데이터 표준화 프레임워크를 4개 기관에 적용했습니다.

- 다기관 ETL 프레임워크를 설계하여 **4개 대학병원**에 config 전환만으로 적용
- 의료 AI 증강 인터페이스 **특허 등록** · K-CURE 경진대회 **최우수상**

<p>
  ✉️ <a href="mailto:simsang1db@gmail.com">simsang1db@gmail.com</a>
</p>

> ⚠️ 의료 데이터 보안 규정상 실데이터 및 실행 환경은 포함되어 있지 않으며, **설계 구조와 구현 방식을 확인할 수 있는 코드**로 구성되어 있습니다.

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
| [OMOP-CDM 신규 구축 및 ETL 파이프라인 고도화](./JBUH-CDM-public) | 2TB 의료 데이터 OMOP-CDM 신규 구축 및 지속적 고도화 (Oracle → PostgreSQL 이관 포함) | **2TB** DW, 쿼리 24h→2h, SCI 논문 기여 | Airflow, PostgreSQL, Oracle, Docker |
| [다기관 데이터 표준화 프레임워크 설계 및 기술 지원](./Infectious_CDM-public) | 기관별 상이한 용어·형식·테이블 구조를 분석하여 다기관 공동 연구를 위한 표준화 데이터 구축 | **4개 병원** 적용, 매핑률 80%+, 10개 기관 교육 | Python, PostgreSQL |
| [영상·DB 이종 데이터 통합 및 멀티모달 데이터셋 구축](./process_dicom) | 2500만건 DICOM 영상과 RDBMS 간 식별자 없이 매핑 로직 설계 및 멀티모달 데이터셋 구축 | **2,500만 건**, 매칭률 80%+, Accuracy 40%→87% | Python, PostgreSQL, Docker |
| [K-CURE 경진대회](./K-CURE) | 20GB+ 의료 데이터 통합 → 폐암 생존 분석 | 🏆 **최우수상** | Python, pandas, lifelines |
| [신생아 NICU 퇴원 시 비정상 BMI 위험요인 분석 및 예측](./KNN-BMI-main) | 퇴원 시 비정상 BMI에 대한 위험요인 탐색 및 ML 기반 예측 모델 개발 | **논문 게재** | Python, scikit-learn, R |
| [실시간 서버 로그 모니터링 파이프라인](https://github.com/simsang1l/log_monitor) | 서버 보안 위협 감지를 위한 실시간 SSH 로그 수집·분석 파이프라인 | 개인 프로젝트 | Kafka, Spark, ELK, Grafana |
| [질병 진단 증강 UI](./AugmentedUI) | AI 예측 결과를 의료 영상 위에 오버레이하는 데스크톱 앱 | **특허 등록** · 학회 논문 | Python, PyTorch, PyQt |
