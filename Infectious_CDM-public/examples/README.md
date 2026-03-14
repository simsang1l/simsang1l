# Raw data to CDM 변환 코드
<img src="https://img.shields.io/badge/Python-3776AB?style=flat&logo=Python&logoColor=FFFFFF"/>  

----
csv로 저장된 raw data를 OMOP-CDM구조와 유사하게 변환하는 코드입니다.  
python으로 작성되어 있어 python 3.8이상을 추천드립니다.

----

## 실행 가이드
1. EMR데이터가 저장된 폴더에 `location.csv`, `concept_EDI_KDC.csv`, `concept_etc.csv`, `concept_KCD7.csv`, `concept_unit.csv`, `edi_atc.csv`, `unit_concept_synonym.csv` 파일이 있는지 확인합니다.  
2. 본인의 데이터 환경에 맞게 ETL정의서를 작성합니다.
3. 작성된 ETL정의서를 기반으로 config.yaml파일에 변환시 필요한 사용되는 변수를 정의해야 합니다.  
3-1. 특정 상병에 해당하는 환자의 데이터셋을 구축하고 싶다면 config.yaml에 있는 diag_condition에 상병코드를 입력해주세요.(ex. A9380)  
4. main.py를 실행합니다.  
5. 실행완료 후 `log 폴더에 있는 파일`과 QC폴더에 있는 `품질진단지표.xlsx파일`을 전달해주시면 됩니다.  


## config.yaml구조
```
|── 공통으로 사용하는 값
└── CDM테이블
    ├── data
            └── 해당 테이블의 원천, 실행 결과 파일명 변수 정의
    └── columns
            └── 해당 테이블에 필요한 컬럼명 변수 정의
```

**config.yaml에서 공통으로 사용하는 값**

`source_path`: raw data가 저장된 경로  
`CDM_path`: CDM데이터가 저장된 경로  
`encoding`: csv파일 읽을 때 encoding설정  
`source_dtype`: csv파일 읽을 때 data type설정  
`target_zip`: 해당 기관의 우편번호 앞 3자리  
`data_range`: 변환할 데이터의 마지막 시점  
`care_site_data`: care_site 데이터가 저장된 파일명  
`person_data`: person 데이터가 저장된 파일명  
`provider_data`: provider 데이터가 저장된 파일명  
`visit_data`: visit_occurrence 데이터가 저장된 파일명  
`visit_detail_data`: visit_detail 데이터가 저장된 파일명  
`local_edi_data`: local_edi 데이터가 저장된 파일명  
`person_source_value`: 원천 데이터의 환자등록번호 컬럼명  
`location_data`: location 데이터가 저장된 파일명  
`concept_unit`: unit_concept_id가 저장된 파일명  
`concept_etc`: type_concept_id등 concept_id로 표현하기 위한 값들이 저장된 파일명  
`unit_concept_synonym`: 동일한 unit_concept_id 매핑을 위한 동의어가 정의된 파일명  

**CDM테이블명**  

**<u>data</u>**

`source_data`: 해당 CDM 테이블로 변환할 raw data 파일명  
`output_filename`: 변환될 CDM테이블의 파일명

**<u>columns</u>**

테이블별 CDM컬럼에 매칭되는 raw data의 컬럼명 입력 (ETL정의서 참고)


## 데이터 파일
`location`:  [링크](https://github.com/ohdsi-korea/ThemisKorea/tree/master/location)에서 다운받은 location데이터  
`concept_EDI_KDC`:  vocabulary_id가 EDI, KDC인 concept 데이터  
`concept_etc`: domain_id가 `Type Concept`, `Provider`, `Meas Value`, `Meas Value Operator`인 concept 데이터  
`concept_KCD7`: vocabulary_id가 KCD7인 concept데이터  
`concept_unit`: domain_id가 `Unit`인 경우  
`edi_atc`: [의약품관리종합포털](https://biz.kpis.or.kr/kpis_biz/index.jsp?sso=ok) -> 의약품정보검색 -> 코드매핑조회 창에서 `전체 다운로드`한 파일과 의약품정보검색 -> ATC정보조회에서 하단의 `ATC Index(WHO ATC 및 K-ATC)`를 이용하여 연결  
`unit_concept_synonym`: unit에 대한 동의어 정의 데이터  

## 실행 결과 구조
```
└── cdm_path
    └── 상병코드
    |── care_site.csv
    |── ...
    └── visit_occurrence.csv
```
