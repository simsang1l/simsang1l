import pandas as pd
import numpy as np
import yaml
import os
from datetime import datetime
import logging
import warnings
import inspect
import re

# 숫자 값을 유지하고, 문자가 포함된 값을 NaN으로 대체하는 함수 정의
def convert_to_numeric(value):
    try:
        # pd.to_numeric을 사용하여 숫자로 변환 시도
        return pd.to_numeric(value)
    except ValueError:
        # 변환이 불가능한 경우 NaN 반환
        return np.nan
    
class DataTransformer:
    """
    기본 데이터 변환 클래스.
    설정 파일을 로드하고, CSV 파일 읽기 및 쓰기를 담당합니다.
    """
    def __init__(self, config_path):
        self.config = self.load_config(config_path)
        self.setup_logging()
        warnings.showwarning = self.custom_warning_handler

        # 공통 변수 정의
        self.cdm_path = self.config["CDM_path"]
        self.person_data = self.config["person_data"]
        self.provider_data = self.config["provider_data"]
        self.care_site_data = self.config["care_site_data"]
        self.visit_data = self.config["visit_data"]
        self.visit_detail = self.config["visit_detail_data"]
        self.measurement_edi_data = self.config["measurement_edi_data"]
        self.procedure_edi_data = self.config["procedure_edi_data"]
        self.source_flag = "source"
        self.cdm_flag = "CDM"
        self.source_dtype = self.config["source_dtype"]
        self.source_encoding = self.config["source_encoding"]
        self.cdm_encoding = self.config["cdm_encoding"]
        self.person_source_value = self.config["person_source_value"]
        self.data_range = self.config["data_range"]
        self.target_zip = self.config["target_zip"]
        self.location_data = self.config["location_data"]
        self.concept_unit = self.config["concept_unit"]
        self.hospital = self.config["hospital"]
        self.drug_edi_data = self.config["drug_edi_data"]
        # self.sugacode = self.config["sugacode"]
        self.edicode = self.config["edicode"]
        self.fromdate = self.config["fromdate"]
        self.todate = self.config["todate"]
        self.frstrgstdt = self.config["frstrgstdt"]
        self.concept_etc = self.config["concept_etc"]
        self.unit_concept_synonym = self.config["unit_concept_synonym"]
        self.visit_no = self.config["visit_no"]
        self.diag_condition = self.config["diag_condition"]
        self.no_matching_concept = self.config["no_matching_concept"]
        self.no_matching_concept_id = self.no_matching_concept[0]
        self.no_matching_concept_name = self.no_matching_concept[1]
        self.concept_kcd = self.config["concept_kcd"]
        self.local_kcd_data = self.config["local_kcd_data"]
        self.hospital_code = self.config["hospital_code"]
        self.care_site_fromdate = self.config["care_site_fromdate"]
        self.care_site_todate = self.config["care_site_todate"]

        # 의료기관이 여러개인 경우 의료기관 코드 폴더 생성
        os.makedirs(os.path.join(self.cdm_path, self.hospital_code), exist_ok = True)
        # 상병조건이 있다면 조건에 맞는 폴더 생성
        os.makedirs(os.path.join(self.cdm_path, self.hospital_code, str(self.diag_condition)), exist_ok = True)

    def load_config(self, config_path):
        """
        YAML 설정 파일을 로드합니다.
        """
        with open(config_path, 'r', encoding="utf-8") as file:
            return yaml.safe_load(file)
        
    def read_csv(self, file_name, path_type = 'source', encoding = None, dtype = None):
        """
        CSV 파일을 읽어 DataFrame으로 반환합니다.
        path_type에 따라 'source' 또는 'CDM' 경로에서 파일을 읽습니다.
        """
        hospital_code = self.hospital_code
        if path_type == "source":
            full_path = os.path.join(self.config["source_path"], file_name + ".csv")
            default_encoding = self.source_encoding
            
        elif path_type == "CDM":
            if hospital_code :
                full_path = os.path.join(self.config["CDM_path"], hospital_code, str(self.diag_condition), file_name + ".csv")
            elif self.diag_condition:
                full_path = os.path.join(self.config["CDM_path"], str(self.diag_condition), file_name + ".csv")
            else :
                full_path = os.path.join(self.config["CDM_path"], file_name + ".csv")
            default_encoding = self.cdm_encoding
        else :
            raise ValueError(f"Invalid path type: {path_type}")
        
        encoding = encoding if encoding else default_encoding

        return pd.read_csv(full_path, dtype = dtype, encoding = encoding)

    def write_csv(self, df, file_path, filename, encoding = 'utf-8', hospital_code = None):
        """
        DataFrame을 CSV 파일로 저장합니다.
        """
        encoding = self.cdm_encoding
        hospital_code = self.hospital_code
        if self.diag_condition:
            df.to_csv(os.path.join(file_path, hospital_code, str(self.diag_condition), filename + ".csv"), encoding = encoding, index = False)
        else:
            df.to_csv(os.path.join(file_path, hospital_code, filename + ".csv"), encoding = encoding, index = False)

    def transform(self):
        """
        데이터 변환을 수행하는 메소드. 하위 클래스에서 구현해야 합니다.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
            
    def setup_logging(self):
        """
        실행 시 로그에 기록하는 메소드입니다.
        """
        log_path = "./log"
        os.makedirs(log_path, exist_ok = True)
        log_filename = datetime.now().strftime('log_%Y-%m-%d_%H%M%S.log')
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s\n'

        filename = os.path.join(log_path, log_filename)
        logging.basicConfig(filename = filename, level = logging.DEBUG, format = log_format, encoding = "utf-8")

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.ERROR)
        console_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(console_handler)

    def custom_warning_handler(self, message, category, filename, lineno, file=None, line=None):
        """
        실행 시 로그에 warning 항목을 기록하는 메소드입니다.
        """
        calling_frame = inspect.currentframe().f_back
        calling_code = calling_frame.f_code
        calling_function_name = calling_code.co_name
        logging.warning(f"{category.__name__} in {calling_function_name} (Line {lineno}): {message}")
                                 

         
class PersonTransformer(DataTransformer):
    def __init__(self, config_path):
        super().__init__(config_path)
        self.table = "person"
        self.cdm_config = self.config[self.table]

        self.source_data = self.cdm_config["data"]["source_data"]
        self.output_filename = self.cdm_config["data"]["output_filename"]
        self.location_source_value = self.cdm_config["columns"]["location_source_value"]
        self.gender_source_value = self.cdm_config["columns"]["gender_source_value"]
        self.death_datetime = self.cdm_config["columns"]["death_datetime"]
        self.birth_datetime = self.cdm_config["columns"]["birth_datetime"]
        self.race_source_value = self.cdm_config["columns"]["race_source_value"]
        self.person_name = self.cdm_config["columns"]["person_name"]
        self.abotyp = self.cdm_config["columns"]["abotyp"]
        self.rhtyp = self.cdm_config["columns"]["rhtyp"]
        self.source_condition = self.cdm_config["data"]["source_condition"]
        self.diagcode = self.cdm_config["columns"]["diagcode"]
        self.ruleout = self.cdm_config["columns"]["ruleout"]


    def transform(self):
        """
        소스 데이터를 읽어들여 CDM 형식으로 변환하고 결과를 CSV 파일로 저장하는 메소드.
        """
        try:
            source_data = self.process_source()
            transformed_data = self.transform_cdm(source_data)

            # save_path = os.path.join(self.cdm_path, self.output_filename)
            self.write_csv(transformed_data, self.cdm_path, self.output_filename)

            logging.info(f"{self.table} 테이블 변환 완료")
            logging.info(f"============================")

        except Exception as e:
            logging.error(f"{self.table} 테이블 변환 중 오류:\n {e}", exc_info=True)
            raise

    def process_source(self):
        """
        소스 데이터와 care site 데이터를 읽어들이고 병합하는 메소드.
        """
        try:
            source_data = self.read_csv(self.source_data, path_type = self.source_flag, dtype = self.source_dtype)
            location_data = self.read_csv(self.location_data, path_type = self.source_flag, dtype = self.source_dtype, encoding=self.cdm_encoding)
            logging.debug(f"원천 데이터 row수: {len(source_data)}")
            
            source_data = pd.merge(source_data, location_data, left_on = self.location_source_value, right_on="LOCATION_SOURCE_VALUE", how = "left")
            source_data.loc[source_data["LOCATION_ID"].isna(), "LOCATION_ID"] = None
            logging.debug(f"location 테이블과 결합 후 원천 데이터1 row수: {len(source_data)}")

            # # 상병조건이 있는 경우 (2025.01.07 기준 다시 사용..) (2024.11기준 사용하지 않음)
            if self.diag_condition:
                condition = self.read_csv(self.source_condition, path_type=self.source_flag, dtype=self.source_dtype)
                # 상병 조건 적용
                pattern_regex = f"^({'|'.join(re.escape(p) for p in self.diag_condition)})"
                
                # self.ruleout 컬럼이 R인경우 RULEOUT
                condition = condition[condition[self.diagcode].str.contains(pattern_regex, regex = True, na=False) & (condition[self.ruleout] == 'C')] 
                condition = condition[self.person_source_value].drop_duplicates()

                source_data = pd.merge(source_data, condition, on=self.person_source_value, how = "inner", suffixes=('', '_diag'))
                
            logging.debug(f"CDM테이블과 결합 후 원천 데이터 row수: source: {len(source_data)}")

            return source_data
        
        except Exception as e :
            logging.error(f"{self.table} 테이블 소스 데이터 처리 중 오류: {e}", exc_info=True)
            raise

    def transform_cdm(self, source):
        """
        주어진 소스 데이터를 CDM 형식에 맞게 변환하는 메소드.
        변환된 데이터는 새로운 DataFrame으로 구성됩니다.
        """
        try : 
            race_conditions = [
                source[self.race_source_value] == 'N',
                source[self.race_source_value] == 'Y'
            ]
            race_concept_id = [38003585, 8552]

            gender_conditions = [
                source[self.gender_source_value].isin(['M']),
                source[self.gender_source_value].isin(['F'])
            ]
            gender_concept_id = [8507, 8532]

            cdm = pd.DataFrame({
                "person_id" : source.index + 1,
                "gender_concept_id": np.select(gender_conditions, gender_concept_id, default = self.no_matching_concept_id),
                "year_of_birth": source[self.birth_datetime].str[:4],
                "month_of_birth": source[self.birth_datetime].str[4:6],
                "day_of_birth": source[self.birth_datetime].str[6:8],
                "birth_datetime": pd.to_datetime(source[self.birth_datetime], format = "%Y%m%d", errors='coerce'),
                "death_datetime": pd.to_datetime(source[self.death_datetime], format = "%Y%m%d", errors='coerce'),
                "race_concept_id": np.select(race_conditions, race_concept_id, default = self.no_matching_concept_id),
                "ethnicity_concept_id": self.no_matching_concept_id,
                "location_id": source["LOCATION_ID"],
                "provider_id": None,
                "care_site_id": None, 
                "person_source_value": source[self.person_source_value],
                "환자명": source[self.person_name],
                "gender_source_value": source[self.gender_source_value],
                "gender_source_concept_id": np.select(gender_conditions, gender_concept_id, default = self.no_matching_concept_id),
                "race_source_value": source[self.race_source_value],
                "race_source_concept_id": self.no_matching_concept_id,
                "ethnicity_source_value": None,
                "ethnicity_source_concept_id": self.no_matching_concept_id,
                "혈액형(ABO)": source[self.abotyp],
                "혈액형(RH)": source[self.rhtyp]
                })
            
            cdm["birth_datetime"] = cdm["birth_datetime"].dt.strftime('%Y-%m-%d %H:%M:%S')
            cdm["death_datetime"] = cdm["death_datetime"].dt.strftime('%Y-%m-%d %H:%M:%S')

            logging.debug(f"CDM 데이터 row수: {len(cdm)}")
            logging.debug(f"요약:\n{cdm.describe(include = 'all').T.to_string()}")
            logging.debug(f"컬럼별 null 개수:\n{cdm.isnull().sum().to_string()}")

            return cdm   

        except Exception as e :
            logging.error(f"{self.table} 테이블 CDM 데이터 변환 중 오류: {e}", exc_info = True)


class MeasurementDiagTransformer(DataTransformer):
    def __init__(self, config_path):
        super().__init__(config_path)
        self.table = "measurement_diag"
        self.cdm_config = self.config[self.table]

        # 컬럼 변수 재정의   
        self.source_data1 = self.cdm_config["data"]["source_data1"]
        self.source_data2 = self.cdm_config["data"]["source_data2"]
        self.source_data3 = self.cdm_config["data"]["source_data3"]
        self.source_data4 = self.cdm_config["data"]["source_data4"]
        self.output_filename = self.cdm_config["data"]["output_filename"]
        self.meddept = self.cdm_config["columns"]["meddept"]
        self.provider = self.cdm_config["columns"]["provider"]
        self.orddate = self.cdm_config["columns"]["orddate"]
        self.measurement_date = self.cdm_config["columns"]["measurement_date"]
        self.measurement_source_value = self.cdm_config["columns"]["measurement_source_value"]
        self.value_source_value = self.cdm_config["columns"]["value_source_value"]
        self.range_low = self.cdm_config["columns"]["range_low"]
        self.range_high = self.cdm_config["columns"]["range_high"]
        self.unit_source_value = self.cdm_config["columns"]["unit_source_value"]
        self.ordcode = self.cdm_config["columns"]["ordcode"]
        self.orddd = self.cdm_config["columns"]["orddd"]
        self.spccd = self.cdm_config["columns"]["spccd"]
                
    def transform(self):
        """
        소스 데이터를 읽어들여 CDM 형식으로 변환하고 결과를 CSV 파일로 저장하는 메소드입니다.
        """
        try : 
            source_data = self.process_source()
            transformed_data = self.transform_cdm(source_data)


            # save_path = os.path.join(self.cdm_path, self.output_filename)
            self.write_csv(transformed_data, self.cdm_path, self.output_filename)

            logging.info(f"{self.table} 테이블 변환 완료")
            logging.info(f"============================")
        
        except Exception as e :
            logging.error(f"{self.table} 테이블 변환 중 오류:\n {e}", exc_info=True)
            raise

    def process_source(self):
        """
        소스 데이터를 로드하고 전처리 작업을 수행하는 메소드입니다.
        """
        try:
            source1 = self.read_csv(self.source_data1, path_type = self.source_flag, dtype = self.source_dtype)
            source2 = self.read_csv(self.source_data2, path_type = self.source_flag, dtype = self.source_dtype)
            source3 = self.read_csv(self.source_data3, path_type = self.source_flag, dtype = self.source_dtype)
            source4 = self.read_csv(self.source_data4, path_type = self.source_flag, dtype = self.source_dtype)
            local_edi = self.read_csv(self.measurement_edi_data, path_type = self.cdm_flag, dtype = self.source_dtype)
            person_data = self.read_csv(self.person_data, path_type = self.cdm_flag, dtype = self.source_dtype)
            provider_data = self.read_csv(self.provider_data, path_type = self.cdm_flag, dtype = self.source_dtype)
            care_site_data = self.read_csv(self.care_site_data, path_type = self.cdm_flag, dtype = self.source_dtype)
            visit_data = self.read_csv(self.visit_data, path_type = self.cdm_flag, dtype = self.source_dtype)
            visit_detail = self.read_csv(self.visit_detail, path_type = self.cdm_flag, dtype = self.source_dtype)
            unit_data = self.read_csv(self.concept_unit, path_type = self.source_flag , dtype = self.source_dtype)#, encoding=self.cdm_encoding)
            concept_etc = self.read_csv(self.concept_etc, path_type = self.source_flag, dtype = self.source_dtype)#, encoding=self.cdm_encoding)
            unit_concept_synonym = self.read_csv(self.unit_concept_synonym, path_type = self.source_flag, dtype = self.source_dtype)#, encoding=self.cdm_encoding)
            logging.debug(f'원천 데이터 row수: {len(source1)}, {len(source2)}, {len(source3)}, {len(source4)}')

            # 원천에서 조건걸기
            source1 = source1[[self.hospital, self.orddate, self.person_source_value, "PRCPHISTNO", "ORDDD", "CRETNO", "PRCPCLSCD", "LASTUPDTDT", "ORDDRID", "PRCPNM", "PRCPCD", "PRCPHISTCD", "PRCPNO", "ORDDEPTCD"]]
            source1[self.orddate] = pd.to_datetime(source1[self.orddate])
            source1["ORDDD"] = pd.to_datetime(source1["ORDDD"])
            source1 = source1[(source1[self.orddate] <= self.data_range)]
            source1 = source1[(source1["PRCPHISTCD"] == "O") & (source1[self.hospital] == self.hospital_code) ]
            
            source2 = source2[[self.hospital, self.orddate, "PRCPNO", "PRCPHISTNO", "EXECPRCPUNIQNO", "ORDDD", self.unit_source_value, "EXECDD", "EXECTM"]]
            source2[self.orddate] = pd.to_datetime(source2[self.orddate])
            source2 = source2[(source2[self.orddate] <= self.data_range)]

            source3 = source3[[self.hospital, self.orddate, "EXECPRCPUNIQNO", "BCNO", "TCLSCD", "SPCCD", "ORDDD"]]
            source3[self.orddate] = pd.to_datetime(source3[self.orddate])
            source3 = source3[(source3[self.orddate] <= self.data_range)]

            source4 = source4[[self.hospital, "BCNO", "TCLSCD", self.spccd, "RSLTFLAG", self.measurement_source_value, self.measurement_date, self.range_low, self.range_high, self.value_source_value, "RSLTSTAT", "LASTREPTDT", self.frstrgstdt]]
            source4 = source4[(source4["RSLTFLAG"] == "O") & (source4["RSLTSTAT"].isin(["4", "5"]))]

            logging.debug(f'조건적용 후 원천 데이터 row수: {len(source1)}, {len(source2)}, {len(source3)}, {len(source4)}')

            source = pd.merge(source2, source1, left_on=[self.hospital, self.orddate, "PRCPNO", "PRCPHISTNO"], right_on=[self.hospital, self.orddate, "PRCPNO", "PRCPHISTNO"], how="inner", suffixes=("", "_diag1"))
            logging.debug(f'source1, source2 병합 후 데이터 개수:, {len(source)}')
            del source1
            del source2

            source = pd.merge(source, source3, left_on=[self.hospital, self.orddate, "EXECPRCPUNIQNO"], right_on=[self.hospital, self.orddate, "EXECPRCPUNIQNO"], how="inner", suffixes=("", "_diag3"))
            logging.debug(f'source, source3 병합 후 데이터 개수:, {len(source)}')

            source = pd.merge(source, source4, left_on=[self.hospital, "BCNO", "TCLSCD", self.spccd], right_on=[self.hospital, "BCNO", "TCLSCD", self.spccd], how="inner", suffixes=("", "_diag4"))
            logging.debug(f'source, source4 병합 후 데이터 개수:, {len(source)}')
            del source3
            del source4

            # visit_source_key 생성
            source["진료일시"] = source[self.orddd]
            source["처방일"] = source[self.orddate]
            source["보고일시"] = source["LASTREPTDT"]
            source["실시일시"] = source["EXECDD"] + source["EXECTM"]
            source["접수일시"] = source[self.measurement_date]

            source[self.orddd] = pd.to_datetime(source[self.orddd])
            source["visit_source_key"] = source[self.person_source_value] + ';' + source[self.orddd].dt.strftime("%Y%m%d") + ';' + source[self.visit_no] + ';' + source[self.hospital]
            source[self.measurement_date] = pd.to_datetime(source[self.measurement_date])

            # value_as_number float형태로 저장되게 값 변경
            # source["value_as_number"] = source[self.value_source_value].str.extract('(-?\d+\.\d+|\d+)')
            source["value_as_number"] = source[self.value_source_value].apply(convert_to_numeric)
            source["value_as_number"].astype(float)
            # source[self.range_low] = source[self.range_low].str.extract('(-?\d+\.\d+|\d+)')
            source[self.range_low] = source[self.range_low].apply(convert_to_numeric)
            source[self.range_low].astype(float)
            # source[self.range_high] = source[self.range_high].str.extract('(-?\d+\.\d+|\d+)')
            source[self.range_high] = source[self.range_high].apply(convert_to_numeric)
            source[self.range_high].astype(float)
            source[self.frstrgstdt] = pd.to_datetime(source[self.frstrgstdt])
            
            person_data = person_data[["person_id", "person_source_value", "환자명"]]
            # person table과 병합
            source = pd.merge(source, person_data, left_on=self.person_source_value, right_on="person_source_value", how="inner")
            del person_data
            logging.debug(f'person 테이블과 결합 후 데이터 row수: {len(source)}')
            
            # local_edi 전처리
            local_edi = local_edi[[self.ordcode, self.fromdate, self.todate, self.edicode, "concept_id", self.hospital, "TCLSNM", self.spccd, "ORDNM"]]
            local_edi[self.fromdate] = pd.to_datetime(local_edi[self.fromdate] , format="%Y%m%d", errors="coerce")
            local_edi[self.todate] = pd.to_datetime(local_edi[self.todate] , format="%Y%m%d", errors="coerce")

            source = pd.merge(source, local_edi, left_on=[self.measurement_source_value, self.spccd, self.hospital], right_on=[self.ordcode, self.spccd, self.hospital], how="left", suffixes=('', '_testcd'))
            logging.debug(f"EDI코드 사용기간별 필터 적용 전 데이터 row수: {len(source)}")

            source[self.fromdate] = source[self.fromdate].fillna(pd.to_datetime('1900-01-01'))
            source[self.todate] = source[self.todate].fillna(pd.to_datetime('2099-12-31'))

            # source = pd.merge(source, local_edi, left_on=[self.ordcode, self.spccd, self.hospital], right_on=[self.ordcode, self.spccd, self.hospital], how="left", suffixes=('', '_order'))
            source = source[(source[self.orddate] >= source[self.fromdate]) & (source[self.orddate] <= source[self.todate])]
            del local_edi
            logging.debug(f'EDI코드 테이블과 병합 후 데이터 row수:, {len(source)}')
    
            # 데이터 컬럼 줄이기
            care_site_data = care_site_data[["care_site_id", "care_site_source_value", "place_of_service_source_value", "care_site_name", self.care_site_fromdate, self.care_site_todate]]
            provider_data = provider_data[["provider_id", "provider_source_value", "provider_name"]]
            visit_data = visit_data[["visit_occurrence_id", "visit_start_date", "care_site_id", "visit_source_value", "person_id", "visit_source_key"]]

            # care_site table과 병합
            source = pd.merge(source, care_site_data, left_on=[self.meddept, self.hospital], right_on=["care_site_source_value", "place_of_service_source_value"], how="left")
            del care_site_data
            logging.debug(f'care_site 테이블과 결합 후 데이터 row수: {len(source)}')

            source[self.care_site_fromdate] = pd.to_datetime(source[self.care_site_fromdate], errors = "coerce")
            source[self.care_site_todate] = pd.to_datetime(source[self.care_site_todate], errors = "coerce")
            source[self.care_site_fromdate] = source[self.care_site_fromdate].fillna(pd.to_datetime("1900-01-01"))
            source[self.care_site_todate] = source[self.care_site_todate].fillna(pd.to_datetime("2099-12-31"))
            source= source[(source[self.frstrgstdt].dt.date >= source[self.care_site_fromdate]) & (source[self.frstrgstdt].dt.date <= source[self.care_site_todate])]
            logging.debug(f"care_site 사용 기간 조건 설정 후 원천 데이터 row수: {len(source)}")

            # provider table과 병합
            source = pd.merge(source, provider_data, left_on=self.provider, right_on="provider_source_value", how="left", suffixes=('', '_y'))
            logging.debug(f'provider 테이블과 결합 후 데이터 row수: {len(source)}')

            # visit_start_datetime 형태 변경
            # source["ORDDD"] = pd.to_datetime(source["ORDDD"])
            visit_data["visit_start_date"] = pd.to_datetime(visit_data["visit_start_date"])

            # visit_occurrence table과 병합
            source = pd.merge(source, visit_data, left_on=["visit_source_key"], right_on=["visit_source_key"], how="left", suffixes=('', '_y'))
            del visit_data
            logging.debug(f'visit_occurrence 테이블과 결합 후 데이터 row수: {len(source)}')

            # visit_detail = visit_detail[["visit_detail_id", "visit_source_key"]]
            visit_detail = visit_detail[["visit_detail_id", "visit_detail_start_datetime", "visit_detail_end_datetime", "visit_occurrence_id", "visit_source_key"]]
            visit_detail["visit_detail_start_datetime"] = pd.to_datetime(visit_detail["visit_detail_start_datetime"])
            visit_detail["visit_detail_end_datetime"] = pd.to_datetime(visit_detail["visit_detail_end_datetime"])
            source = pd.merge(source, visit_detail, left_on=["visit_source_key"], right_on=["visit_source_key"], how="left", suffixes=('', '_detail'))
            logging.debug(f"visit_detail 테이블과 결합 후 원천 데이터 row수: {len(source)}")

                        # 날짜 조건을 만족하는 경우와 아닌 경우로 나눠 최종적으로는 합친다
            # 날짜 조건에 해당하지 않는 경우, 조인해도 어차피 visit_detail관련 데이터가 null 
            source_notjoin_condition = source[source["visit_detail_id"].isna()]
            logging.info(f"조인에 영향없는 경우: {len(source_notjoin_condition)}")
            
            # 날짜 조건에 해당하는 경우, 조인하면 visit_detail관련 데이터가 null이 아님
            source_join_condition = source[~source["visit_detail_id"].isna()]
            source_date_condition = source_join_condition[(source_join_condition[self.orddate] >= source_join_condition["visit_detail_start_datetime"]) 
                                                          & (source_join_condition[self.orddate] <= source_join_condition["visit_detail_end_datetime"])]
            source_notdate_condition = source[(~source.index.isin(source_date_condition.index)) & (~source.index.isin(source_notjoin_condition.index))]
            logging.info(f"조인에 영향있는 경우 : {len(source_join_condition)}")
            logging.info(f"날짜 조건에 해당하는 경우 : {len(source_date_condition)}")
            logging.info(f"날짜 조건에 해당하지 않는 경우 : {len(source_notdate_condition)}")
            source = pd.concat([source_notjoin_condition, source_date_condition, source_notdate_condition], ignore_index=True).copy()
            
            
            # source["visit_detail_start_datetime"] = source["visit_detail_start_datetime"].fillna(pd.to_datetime('1900-01-01'))
            # source["visit_detail_end_datetime"] = source["visit_detail_end_datetime"].fillna(pd.to_datetime('2099-12-31'))
            # source = source[(source[self.orddate] >= source["visit_detail_start_datetime"]) & (source[self.orddate] <= source["visit_detail_end_datetime"])]
            # # source["visit_detail_id"] = source.apply(lambda row: row['visit_detail_id'] if pd.notna(row['visit_detail_start_datetime']) and row['visit_detail_start_datetime'] <= row[self.orddate] <= row['visit_detail_end_datetime'] else pd.NA, axis=1)
            # source = source.drop(columns = ["visit_detail_start_datetime", "visit_detail_end_datetime"])
            logging.debug(f"visit_detail 테이블과 결합 후 조건 적용 후 원천 데이터 row수: {len(source)}")
            ## visit_detail로 인해 중복되는 항목 제거를 위함.
            source = source.drop_duplicates(subset=[self.hospital, self.person_source_value, self.orddd, self.orddate, self.visit_no, self.measurement_source_value, self.ordcode, self.spccd, "PRCPNO"])
            logging.debug(f"visit_detail 테이블과 결합 후 조건 적용 및 중복제거 후 원천 데이터 row수: {len(source)}")

            # 값이 없는 경우 0으로 값 입력
            # source.loc[source["care_site_id"].isna(), "care_site_id"] = 0
            source.loc[source["concept_id"].isna(), "concept_id"] = 0
            
            ### unit매핑 작업 ###
            # concept_unit과 병합
            unit_data = unit_data[["concept_id", "concept_name", "concept_code"]]
            source = pd.merge(source, unit_data, left_on=self.unit_source_value, right_on="concept_code", how="left", suffixes=["", "_unit"])
            logging.debug(f'unit 테이블과 결합 후 데이터 row수: {len(source)}')
            # unit 동의어 적용
            source = pd.merge(source, unit_concept_synonym, left_on = self.unit_source_value, right_on = "concept_synonym_name", how = "left", suffixes=["", "_synonym"])
            logging.debug(f'unit synonym 테이블과 결합 후 데이터 row수: {len(source)}')
            

            ### concept_etc테이블과 병합 ###
            concept_etc = concept_etc[["concept_id", "concept_name"]]
            concept_etc["concept_id"] = concept_etc["concept_id"].astype(int)            

            # type_concept_id 만들고 type_concept_id_name 기반 만들기
            source["measurement_type_concept_id"] = 44818702
            source = pd.merge(source, concept_etc, left_on = "measurement_type_concept_id", right_on="concept_id", how="left", suffixes=('', '_measurement_type'))
            logging.debug(f'concept_etc: type_concept_id 테이블과 결합 후 데이터 row수: {len(source)}')

            # operator_concept_id 만들고 operator_concept_id_name 기반 만들기
            operator_condition = [
                    source[self.value_source_value].isin([">"])
                    , source[self.value_source_value].isin([">="])
                    , source[self.value_source_value].isin(["="])
                    , source[self.value_source_value].isin(["<="])
                    , source[self.value_source_value].isin(["<"])
            ]
            operator_value = [
                    4172704
                    , 4171755
                    , 4172703
                    , 4171754
                    , 4171756
            ]

            source["operator_concept_id"] = np.select(operator_condition, operator_value)
            source = pd.merge(source, concept_etc, left_on = "operator_concept_id", right_on="concept_id", how="left", suffixes=('', '_operator'))
            logging.debug(f'concept_etc: operator_concept_id 테이블과 결합 후 데이터 row수: {len(source)}')

            # value_as_concept_id 만들고 value_as_concept_id_name 기반 만들기
            value_concept_condition = [
                source[self.value_source_value] == "+"
                , source[self.value_source_value] == "++"
                , source[self.value_source_value] == "+++"
                , source[self.value_source_value] == "++++"
                , source[self.value_source_value].str.lower() == "negative"
                , source[self.value_source_value].str.lower() == "positive"
            ]
            value_concept_value = [
                4123508
                , 4126673
                , 4125547
                , 4126674
                , 9189
                , 9191
            ]
            source["value_as_concept_id"] = np.select(value_concept_condition, value_concept_value)
            source = pd.merge(source, concept_etc, left_on = "value_as_concept_id", right_on="concept_id", how="left", suffixes=('', '_value_as_concept'))
            logging.debug(f'concept_etc: value_as_concept_id 테이블과 결합 후 데이터 row수: {len(source)}')

            logging.debug(f'CDM 테이블과 결합 후 데이터 row수: {len(source)}')

            return source
        
        except Exception as e :
            logging.error(f"{self.table} 테이블 소스 데이터 처리 중 오류: {e}", exc_info = True)

    def transform_cdm(self, source):
        """
        주어진 소스 데이터를 CDM 형식에 맞게 변환하는 메소드.
        변환된 데이터는 새로운 DataFrame으로 구성됩니다.
        """
        try :
            measurement_date_condition = [source[self.measurement_date].notna()]
            measurement_date_value = [source[self.measurement_date].dt.date]
            measurement_datetime_value = [source[self.measurement_date]]
            measurement_time_value = [source[self.measurement_date].dt.time]

            unit_id_condition = [
                source["concept_id_unit"].notnull(),
                source["concept_id_synonym"].notnull()
            ]

            unit_id_value = [
                source["concept_id_unit"],
                source["concept_id_synonym"]
            ]

            unit_name_condition = [
                source["concept_id_unit"].notnull(),
                source["concept_id_synonym"].notnull()
            ]

            unit_name_value = [
                source["concept_name"],
                source["concept_name_synonym"]
            ]

            cdm = pd.DataFrame({
                "measurement_id": source.index + 1,
                "person_id": source["person_id"],
                "환자명": source["환자명"],
                "measurement_concept_id": np.select([source["concept_id"].notna()], [source["concept_id"]], default=self.no_matching_concept_id),
                "measurement_date": np.select(measurement_date_condition, measurement_date_value, default=source[self.orddate].dt.date),
                "measurement_datetime": np.select(measurement_date_condition, measurement_datetime_value, default=source[self.orddate]),
                "measurement_time": np.select(measurement_date_condition, measurement_time_value, default=source[self.orddate].dt.time),
                # "measurement_date_type": np.select(measurement_date_condition, ["보고일"], default="처방일"),
                "measurement_type_concept_id": np.select([source["measurement_type_concept_id"].notna()], [source["measurement_type_concept_id"]], default=self.no_matching_concept_id),
                "measurement_type_concept_id_name": np.select([source["concept_name_measurement_type"].notna()], [source["concept_name_measurement_type"]], default=self.no_matching_concept_name),
                "operator_concept_id": np.select([source["operator_concept_id"].notna()], [source["operator_concept_id"]], default=self.no_matching_concept_id),
                "operator_concept_id_name": np.select([source["concept_name_operator"].notna()], [source["concept_name_operator"]], default=self.no_matching_concept_name) ,
                "value_as_number": source["value_as_number"],
                "value_as_concept_id": np.select([source["value_as_concept_id"].notna()], [source["value_as_concept_id"]], default=self.no_matching_concept_id),
                "value_as_concept_id_name": np.select([source["concept_name_value_as_concept"].notna()], [source["concept_name_value_as_concept"]], default=self.no_matching_concept_name),
                "unit_concept_id": np.select(unit_id_condition, unit_id_value, default=self.no_matching_concept_id),
                "unit_concept_id_name": np.select(unit_name_condition, unit_name_value, default=self.no_matching_concept_name),
                "range_low": source[self.range_low],
                "range_high": source[self.range_high],
                "provider_id": source["provider_id"],
                "provider_name": source["provider_name"],
                "visit_occurrence_id": source["visit_occurrence_id"],
                "visit_detail_id": source["visit_detail_id"],
                "measurement_source_value": source[self.measurement_source_value],
                "measurement_source_value_name": source["TCLSNM"],
                "measurement_source_concept_id": np.select([source["concept_id"].notna()], [source["concept_id"]], default=self.no_matching_concept_id),
                "EDI코드": source[self.edicode],
                "unit_source_value": source[self.unit_source_value],
                "value_source_value": source[self.value_source_value].str[:50],
                "vocabulary_id": "EDI",
                "visit_source_key": source["visit_source_key"]
                })

            logging.debug(f'CDM 데이터 row수: {len(cdm)}')
            logging.debug(f"요약:\n{cdm.describe(include = 'all').T.to_string()}")
            logging.debug(f"컬럼별 null 개수:\n{cdm.isnull().sum().to_string()}")

            return cdm   

        except Exception as e :
            logging.error(f"{self.table} 테이블 CDM 데이터 변환 중 오류:\n {e}", exc_info = True)

