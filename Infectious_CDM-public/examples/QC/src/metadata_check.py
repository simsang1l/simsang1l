import pandas as pd
import numpy as np
import os, sys
from datetime import datetime
import inspect
# 상위 디렉토리 (my_project 폴더)를 sys.path에 추가
current_dir = os.path.dirname(__file__)  # 현재 파일의 디렉토리
parent_dir = os.path.dirname(current_dir)  # 현재 디렉토리의 부모 디렉토리 (QC)
project_dir = os.path.dirname(parent_dir)  # QC의 부모 디렉토리 (my_project)
sys.path.insert(0, project_dir)
from QC.src.excel_util import find_excel_row_and_write_metadata_count

def META_0001(config, cdm_path, excel_path, sheetname):
    """
    WBC Count 데이터 수 확인
    """
    table_name = "measurement"
    column_name = "measurement_source_value"
    code_list = config["metadata"]["wbc_count"]
    id = inspect.currentframe().f_code.co_name
    cdm = pd.read_csv(os.path.join(cdm_path, table_name + ".csv"), dtype=str)
    feature = cdm[cdm[column_name].isin(code_list)]

    feature_count = len(feature)
    row_count = len(cdm)
    feature_patient_count = feature["person_id"].nunique()
    patient_count = cdm["person_id"].nunique()
    patient_ratio = feature_patient_count / patient_count

    find_excel_row_and_write_metadata_count(excel_path, sheetname, id, feature_count, row_count, feature_patient_count, patient_count, patient_ratio)

def META_0002(config, cdm_path, excel_path, sheetname):
    """
    hemogrobin(Hb) 데이터 수 확인
    """
    table_name = "measurement"
    column_name = "measurement_source_value"
    code_list = config["metadata"]["hb"]
    id = inspect.currentframe().f_code.co_name
    cdm = pd.read_csv(os.path.join(cdm_path, table_name + ".csv"), dtype=str)
    feature = cdm[cdm[column_name].isin(code_list)]

    feature_count = len(feature)
    row_count = len(cdm)
    feature_patient_count = feature["person_id"].nunique()
    patient_count = cdm["person_id"].nunique()
    patient_ratio = feature_patient_count / patient_count

    find_excel_row_and_write_metadata_count(excel_path, sheetname, id, feature_count, row_count, feature_patient_count, patient_count, patient_ratio)

def META_0003(config, cdm_path, excel_path, sheetname):
    """
    Hematocrit(Hct) 데이터 수 확인
    """
    table_name = "measurement"
    column_name = "measurement_source_value"
    code_list = config["metadata"]["hematocrit"]
    id = inspect.currentframe().f_code.co_name
    cdm = pd.read_csv(os.path.join(cdm_path, table_name + ".csv"), dtype=str)
    feature = cdm[cdm[column_name].isin(code_list)]

    feature_count = len(feature)
    row_count = len(cdm)
    feature_patient_count = feature["person_id"].nunique()
    patient_count = cdm["person_id"].nunique()
    patient_ratio = feature_patient_count / patient_count

    find_excel_row_and_write_metadata_count(excel_path, sheetname, id, feature_count, row_count, feature_patient_count, patient_count, patient_ratio)

def META_0004(config, cdm_path, excel_path, sheetname):
    """
    platelet_count 데이터 수 확인
    """
    table_name = "measurement"
    column_name = "measurement_source_value"
    code_list = config["metadata"]["platelet_count"]
    id = inspect.currentframe().f_code.co_name
    cdm = pd.read_csv(os.path.join(cdm_path, table_name + ".csv"), dtype=str)
    feature = cdm[cdm[column_name].isin(code_list)]

    feature_count = len(feature)
    row_count = len(cdm)
    feature_patient_count = feature["person_id"].nunique()
    patient_count = cdm["person_id"].nunique()
    patient_ratio = feature_patient_count / patient_count

    find_excel_row_and_write_metadata_count(excel_path, sheetname, id, feature_count, row_count, feature_patient_count, patient_count, patient_ratio)

def META_0005(config, cdm_path, excel_path, sheetname):
    """
    lymphocyte_count 데이터 수 확인
    """
    table_name = "measurement"
    column_name = "measurement_source_value"
    code_list = config["metadata"]["lymphocyte_count"]
    id = inspect.currentframe().f_code.co_name
    cdm = pd.read_csv(os.path.join(cdm_path, table_name + ".csv"), dtype=str)
    feature = cdm[cdm[column_name].isin(code_list)]

    feature_count = len(feature)
    row_count = len(cdm)
    feature_patient_count = feature["person_id"].nunique()
    patient_count = cdm["person_id"].nunique()
    patient_ratio = feature_patient_count / patient_count

    find_excel_row_and_write_metadata_count(excel_path, sheetname, id, feature_count, row_count, feature_patient_count, patient_count, patient_ratio)


def META_0006(config, cdm_path, excel_path, sheetname):
    """
    monocyte_count 데이터 수 확인
    """
    table_name = "measurement"
    column_name = "measurement_source_value"
    code_list = config["metadata"]["monocyte_count"]
    id = inspect.currentframe().f_code.co_name
    cdm = pd.read_csv(os.path.join(cdm_path, table_name + ".csv"), dtype=str)
    feature = cdm[cdm[column_name].isin(code_list)]

    feature_count = len(feature)
    row_count = len(cdm)
    feature_patient_count = feature["person_id"].nunique()
    patient_count = cdm["person_id"].nunique()
    patient_ratio = feature_patient_count / patient_count

    find_excel_row_and_write_metadata_count(excel_path, sheetname, id, feature_count, row_count, feature_patient_count, patient_count, patient_ratio)


def META_0007(config, cdm_path, excel_path, sheetname):
    """
    neurophil_count 데이터 수 확인
    """
    table_name = "measurement"
    column_name = "measurement_source_value"
    code_list = config["metadata"]["neurophil_count"]
    id = inspect.currentframe().f_code.co_name
    cdm = pd.read_csv(os.path.join(cdm_path, table_name + ".csv"), dtype=str)
    feature = cdm[cdm[column_name].isin(code_list)]

    feature_count = len(feature)
    row_count = len(cdm)
    feature_patient_count = feature["person_id"].nunique()
    patient_count = cdm["person_id"].nunique()
    patient_ratio = feature_patient_count / patient_count

    find_excel_row_and_write_metadata_count(excel_path, sheetname, id, feature_count, row_count, feature_patient_count, patient_count, patient_ratio)

def META_0008(config, cdm_path, excel_path, sheetname):
    """
    sodium 데이터 수 확인
    """
    table_name = "measurement"
    column_name = "measurement_source_value"
    code_list = config["metadata"]["sodium"]
    id = inspect.currentframe().f_code.co_name
    cdm = pd.read_csv(os.path.join(cdm_path, table_name + ".csv"), dtype=str)
    feature = cdm[cdm[column_name].isin(code_list)]

    feature_count = len(feature)
    row_count = len(cdm)
    feature_patient_count = feature["person_id"].nunique()
    patient_count = cdm["person_id"].nunique()
    patient_ratio = feature_patient_count / patient_count

    find_excel_row_and_write_metadata_count(excel_path, sheetname, id, feature_count, row_count, feature_patient_count, patient_count, patient_ratio)

def META_0009(config, cdm_path, excel_path, sheetname):
    """
    potassium 데이터 수 확인
    """
    table_name = "measurement"
    column_name = "measurement_source_value"
    code_list = config["metadata"]["potassium"]
    id = inspect.currentframe().f_code.co_name
    cdm = pd.read_csv(os.path.join(cdm_path, table_name + ".csv"), dtype=str)
    feature = cdm[cdm[column_name].isin(code_list)]

    feature_count = len(feature)
    row_count = len(cdm)
    feature_patient_count = feature["person_id"].nunique()
    patient_count = cdm["person_id"].nunique()
    patient_ratio = feature_patient_count / patient_count

    find_excel_row_and_write_metadata_count(excel_path, sheetname, id, feature_count, row_count, feature_patient_count, patient_count, patient_ratio)

def META_0010(config, cdm_path, excel_path, sheetname):
    """
    ast 데이터 수 확인
    """
    table_name = "measurement"
    column_name = "measurement_source_value"
    code_list = config["metadata"]["ast"]
    id = inspect.currentframe().f_code.co_name
    cdm = pd.read_csv(os.path.join(cdm_path, table_name + ".csv"), dtype=str)
    feature = cdm[cdm[column_name].isin(code_list)]

    feature_count = len(feature)
    row_count = len(cdm)
    feature_patient_count = feature["person_id"].nunique()
    patient_count = cdm["person_id"].nunique()
    patient_ratio = feature_patient_count / patient_count

    find_excel_row_and_write_metadata_count(excel_path, sheetname, id, feature_count, row_count, feature_patient_count, patient_count, patient_ratio)

def META_0011(config, cdm_path, excel_path, sheetname):
    """
    alt 데이터 수 확인
    """
    table_name = "measurement"
    column_name = "measurement_source_value"
    code_list = config["metadata"]["alt"]
    id = inspect.currentframe().f_code.co_name
    cdm = pd.read_csv(os.path.join(cdm_path, table_name + ".csv"), dtype=str)
    feature = cdm[cdm[column_name].isin(code_list)]

    feature_count = len(feature)
    row_count = len(cdm)
    feature_patient_count = feature["person_id"].nunique()
    patient_count = cdm["person_id"].nunique()
    patient_ratio = feature_patient_count / patient_count

    find_excel_row_and_write_metadata_count(excel_path, sheetname, id, feature_count, row_count, feature_patient_count, patient_count, patient_ratio)

def META_0012(config, cdm_path, excel_path, sheetname):
    """
    total_bilirubin 데이터 수 확인
    """
    table_name = "measurement"
    column_name = "measurement_source_value"
    code_list = config["metadata"]["total_bilirubin"]
    id = inspect.currentframe().f_code.co_name
    cdm = pd.read_csv(os.path.join(cdm_path, table_name + ".csv"), dtype=str)
    feature = cdm[cdm[column_name].isin(code_list)]

    feature_count = len(feature)
    row_count = len(cdm)
    feature_patient_count = feature["person_id"].nunique()
    patient_count = cdm["person_id"].nunique()
    patient_ratio = feature_patient_count / patient_count

    find_excel_row_and_write_metadata_count(excel_path, sheetname, id, feature_count, row_count, feature_patient_count, patient_count, patient_ratio)

def META_0013(config, cdm_path, excel_path, sheetname):
    """
    total_protein 데이터 수 확인
    """
    table_name = "measurement"
    column_name = "measurement_source_value"
    code_list = config["metadata"]["total_protein"]
    id = inspect.currentframe().f_code.co_name
    cdm = pd.read_csv(os.path.join(cdm_path, table_name + ".csv"), dtype=str)
    feature = cdm[cdm[column_name].isin(code_list)]

    feature_count = len(feature)
    row_count = len(cdm)
    feature_patient_count = feature["person_id"].nunique()
    patient_count = cdm["person_id"].nunique()
    patient_ratio = feature_patient_count / patient_count

    find_excel_row_and_write_metadata_count(excel_path, sheetname, id, feature_count, row_count, feature_patient_count, patient_count, patient_ratio)

def META_0014(config, cdm_path, excel_path, sheetname):
    """
    albumin 데이터 수 확인
    """
    table_name = "measurement"
    column_name = "measurement_source_value"
    code_list = config["metadata"]["albumin"]
    id = inspect.currentframe().f_code.co_name
    cdm = pd.read_csv(os.path.join(cdm_path, table_name + ".csv"), dtype=str)
    feature = cdm[cdm[column_name].isin(code_list)]

    feature_count = len(feature)
    row_count = len(cdm)
    feature_patient_count = feature["person_id"].nunique()
    patient_count = cdm["person_id"].nunique()
    patient_ratio = feature_patient_count / patient_count

    find_excel_row_and_write_metadata_count(excel_path, sheetname, id, feature_count, row_count, feature_patient_count, patient_count, patient_ratio)

def META_0015(config, cdm_path, excel_path, sheetname):
    """
    bun 데이터 수 확인
    """
    table_name = "measurement"
    column_name = "measurement_source_value"
    code_list = config["metadata"]["bun"]
    id = inspect.currentframe().f_code.co_name
    cdm = pd.read_csv(os.path.join(cdm_path, table_name + ".csv"), dtype=str)
    feature = cdm[cdm[column_name].isin(code_list)]

    feature_count = len(feature)
    row_count = len(cdm)
    feature_patient_count = feature["person_id"].nunique()
    patient_count = cdm["person_id"].nunique()
    patient_ratio = feature_patient_count / patient_count

    find_excel_row_and_write_metadata_count(excel_path, sheetname, id, feature_count, row_count, feature_patient_count, patient_count, patient_ratio)

def META_0016(config, cdm_path, excel_path, sheetname):
    """
    creatinine 데이터 수 확인
    """
    table_name = "measurement"
    column_name = "measurement_source_value"
    code_list = config["metadata"]["creatinine"]
    id = inspect.currentframe().f_code.co_name
    cdm = pd.read_csv(os.path.join(cdm_path, table_name + ".csv"), dtype=str)
    feature = cdm[cdm[column_name].isin(code_list)]

    feature_count = len(feature)
    row_count = len(cdm)
    feature_patient_count = feature["person_id"].nunique()
    patient_count = cdm["person_id"].nunique()
    patient_ratio = feature_patient_count / patient_count

    find_excel_row_and_write_metadata_count(excel_path, sheetname, id, feature_count, row_count, feature_patient_count, patient_count, patient_ratio)

def META_0017(config, cdm_path, excel_path, sheetname):
    """
    egfr 데이터 수 확인
    """
    table_name = "measurement"
    column_name = "measurement_source_value"
    code_list = config["metadata"]["egfr"]
    id = inspect.currentframe().f_code.co_name
    cdm = pd.read_csv(os.path.join(cdm_path, table_name + ".csv"), dtype=str)
    feature = cdm[cdm[column_name].isin(code_list)]

    feature_count = len(feature)
    row_count = len(cdm)
    feature_patient_count = feature["person_id"].nunique()
    patient_count = cdm["person_id"].nunique()
    patient_ratio = feature_patient_count / patient_count

    find_excel_row_and_write_metadata_count(excel_path, sheetname, id, feature_count, row_count, feature_patient_count, patient_count, patient_ratio)

def META_0018(config, cdm_path, excel_path, sheetname):
    """
    crp 데이터 수 확인
    """
    table_name = "measurement"
    column_name = "measurement_source_value"
    code_list = config["metadata"]["crp"]
    id = inspect.currentframe().f_code.co_name
    cdm = pd.read_csv(os.path.join(cdm_path, table_name + ".csv"), dtype=str)
    feature = cdm[cdm[column_name].isin(code_list)]

    feature_count = len(feature)
    row_count = len(cdm)
    feature_patient_count = feature["person_id"].nunique()
    patient_count = cdm["person_id"].nunique()
    patient_ratio = feature_patient_count / patient_count

    find_excel_row_and_write_metadata_count(excel_path, sheetname, id, feature_count, row_count, feature_patient_count, patient_count, patient_ratio)

def META_0019(config, cdm_path, excel_path, sheetname):
    """
    troponin_i 데이터 수 확인
    """
    table_name = "measurement"
    column_name = "measurement_source_value"
    code_list = config["metadata"]["troponin_i"]
    id = inspect.currentframe().f_code.co_name
    cdm = pd.read_csv(os.path.join(cdm_path, table_name + ".csv"), dtype=str)
    feature = cdm[cdm[column_name].isin(code_list)]

    feature_count = len(feature)
    row_count = len(cdm)
    feature_patient_count = feature["person_id"].nunique()
    patient_count = cdm["person_id"].nunique()
    patient_ratio = feature_patient_count / patient_count

    find_excel_row_and_write_metadata_count(excel_path, sheetname, id, feature_count, row_count, feature_patient_count, patient_count, patient_ratio)

def META_0020(config, cdm_path, excel_path, sheetname):
    """
    ck_mb 데이터 수 확인
    """
    table_name = "measurement"
    column_name = "measurement_source_value"
    code_list = config["metadata"]["ck_mb"]
    id = inspect.currentframe().f_code.co_name
    cdm = pd.read_csv(os.path.join(cdm_path, table_name + ".csv"), dtype=str)
    feature = cdm[cdm[column_name].isin(code_list)]

    feature_count = len(feature)
    row_count = len(cdm)
    feature_patient_count = feature["person_id"].nunique()
    patient_count = cdm["person_id"].nunique()
    patient_ratio = feature_patient_count / patient_count

    find_excel_row_and_write_metadata_count(excel_path, sheetname, id, feature_count, row_count, feature_patient_count, patient_count, patient_ratio)

def META_0021(config, cdm_path, excel_path, sheetname):
    """
    ph 데이터 수 확인
    """
    table_name = "measurement"
    column_name = "measurement_source_value"
    code_list = config["metadata"]["ph"]
    id = inspect.currentframe().f_code.co_name
    cdm = pd.read_csv(os.path.join(cdm_path, table_name + ".csv"), dtype=str)
    feature = cdm[cdm[column_name].isin(code_list)]

    feature_count = len(feature)
    row_count = len(cdm)
    feature_patient_count = feature["person_id"].nunique()
    patient_count = cdm["person_id"].nunique()
    patient_ratio = feature_patient_count / patient_count

    find_excel_row_and_write_metadata_count(excel_path, sheetname, id, feature_count, row_count, feature_patient_count, patient_count, patient_ratio)

def META_0022(config, cdm_path, excel_path, sheetname):
    """
    paco2 데이터 수 확인
    """
    table_name = "measurement"
    column_name = "measurement_source_value"
    code_list = config["metadata"]["paco2"]
    id = inspect.currentframe().f_code.co_name
    cdm = pd.read_csv(os.path.join(cdm_path, table_name + ".csv"), dtype=str)
    feature = cdm[cdm[column_name].isin(code_list)]

    feature_count = len(feature)
    row_count = len(cdm)
    feature_patient_count = feature["person_id"].nunique()
    patient_count = cdm["person_id"].nunique()
    patient_ratio = feature_patient_count / patient_count

    find_excel_row_and_write_metadata_count(excel_path, sheetname, id, feature_count, row_count, feature_patient_count, patient_count, patient_ratio)

def META_0023(config, cdm_path, excel_path, sheetname):
    """
    pao2 데이터 수 확인
    """
    table_name = "measurement"
    column_name = "measurement_source_value"
    code_list = config["metadata"]["pao2"]
    id = inspect.currentframe().f_code.co_name
    cdm = pd.read_csv(os.path.join(cdm_path, table_name + ".csv"), dtype=str)
    feature = cdm[cdm[column_name].isin(code_list)]

    feature_count = len(feature)
    row_count = len(cdm)
    feature_patient_count = feature["person_id"].nunique()
    patient_count = cdm["person_id"].nunique()
    patient_ratio = feature_patient_count / patient_count

    find_excel_row_and_write_metadata_count(excel_path, sheetname, id, feature_count, row_count, feature_patient_count, patient_count, patient_ratio)

def META_0024(config, cdm_path, excel_path, sheetname):
    """
    arterial_ph 데이터 수 확인
    """
    table_name = "measurement"
    column_name = "measurement_source_value"
    code_list = config["metadata"]["arterial_ph"]
    id = inspect.currentframe().f_code.co_name
    cdm = pd.read_csv(os.path.join(cdm_path, table_name + ".csv"), dtype=str)
    feature = cdm[cdm[column_name].isin(code_list)]

    feature_count = len(feature)
    row_count = len(cdm)
    feature_patient_count = feature["person_id"].nunique()
    patient_count = cdm["person_id"].nunique()
    patient_ratio = feature_patient_count / patient_count

    find_excel_row_and_write_metadata_count(excel_path, sheetname, id, feature_count, row_count, feature_patient_count, patient_count, patient_ratio)

def META_0025(config, cdm_path, excel_path, sheetname):
    """
    체온 데이터 수 확인
    """
    table_name = "measurement"
    column_name = "measurement_concept_id" # vs는 concept_id로 조회
    code_list = config["metadata"]["temperature"]
    id = inspect.currentframe().f_code.co_name
    cdm = pd.read_csv(os.path.join(cdm_path, table_name + ".csv"))
    feature = cdm[cdm[column_name].isin(code_list)]

    feature_count = len(feature)
    row_count = len(cdm)
    feature_patient_count = feature["person_id"].nunique()
    patient_count = cdm["person_id"].nunique()
    patient_ratio = feature_patient_count / patient_count

    find_excel_row_and_write_metadata_count(excel_path, sheetname, id, feature_count, row_count, feature_patient_count, patient_count, patient_ratio)


def META_0026(config, cdm_path, excel_path, sheetname):
    """
    sbp 데이터 수 확인
    """
    table_name = "measurement"
    column_name = "measurement_concept_id" # vs는 concept_id로 조회
    code_list = config["metadata"]["sbp"]
    id = inspect.currentframe().f_code.co_name
    cdm = pd.read_csv(os.path.join(cdm_path, table_name + ".csv"))
    feature = cdm[cdm[column_name].isin(code_list)]

    feature_count = len(feature)
    row_count = len(cdm)
    feature_patient_count = feature["person_id"].nunique()
    patient_count = cdm["person_id"].nunique()
    patient_ratio = feature_patient_count / patient_count

    find_excel_row_and_write_metadata_count(excel_path, sheetname, id, feature_count, row_count, feature_patient_count, patient_count, patient_ratio)

def META_0027(config, cdm_path, excel_path, sheetname):
    """
    dbp 데이터 수 확인
    """
    table_name = "measurement"
    column_name = "measurement_concept_id" # vs는 concept_id로 조회
    code_list = config["metadata"]["dbp"]
    id = inspect.currentframe().f_code.co_name
    cdm = pd.read_csv(os.path.join(cdm_path, table_name + ".csv"))
    feature = cdm[cdm[column_name].isin(code_list)]

    feature_count = len(feature)
    row_count = len(cdm)
    feature_patient_count = feature["person_id"].nunique()
    patient_count = cdm["person_id"].nunique()
    patient_ratio = feature_patient_count / patient_count

    find_excel_row_and_write_metadata_count(excel_path, sheetname, id, feature_count, row_count, feature_patient_count, patient_count, patient_ratio)


def META_0028(config, cdm_path, excel_path, sheetname):
    """
    Heart rate 데이터 수 확인
    """
    table_name = "measurement"
    column_name = "measurement_concept_id" # vs는 concept_id로 조회
    code_list = config["metadata"]["heartrate"]
    id = inspect.currentframe().f_code.co_name
    cdm = pd.read_csv(os.path.join(cdm_path, table_name + ".csv"))
    feature = cdm[cdm[column_name].isin(code_list)]

    feature_count = len(feature)
    row_count = len(cdm)
    feature_patient_count = feature["person_id"].nunique()
    patient_count = cdm["person_id"].nunique()
    patient_ratio = feature_patient_count / patient_count

    find_excel_row_and_write_metadata_count(excel_path, sheetname, id, feature_count, row_count, feature_patient_count, patient_count, patient_ratio)

def META_0029(config, cdm_path, excel_path, sheetname):
    """
    respiratory_rate 데이터 수 확인
    """
    table_name = "measurement"
    column_name = "measurement_concept_id" # vs는 concept_id로 조회
    code_list = config["metadata"]["respiratory_rate"]
    id = inspect.currentframe().f_code.co_name
    cdm = pd.read_csv(os.path.join(cdm_path, table_name + ".csv"))
    feature = cdm[cdm[column_name].isin(code_list)]

    feature_count = len(feature)
    row_count = len(cdm)
    feature_patient_count = feature["person_id"].nunique()
    patient_count = cdm["person_id"].nunique()
    patient_ratio = feature_patient_count / patient_count

    find_excel_row_and_write_metadata_count(excel_path, sheetname, id, feature_count, row_count, feature_patient_count, patient_count, patient_ratio)

def META_0030(config, cdm_path, excel_path, sheetname):
    """
    bmi 데이터 수 확인
    """
    table_name = "measurement"
    column_name = "measurement_concept_id" # vs는 concept_id로 조회
    code_list = config["metadata"]["bmi"]
    id = inspect.currentframe().f_code.co_name
    cdm = pd.read_csv(os.path.join(cdm_path, table_name + ".csv"))
    feature = cdm[cdm[column_name].isin(code_list)]

    feature_count = len(feature)
    row_count = len(cdm)
    feature_patient_count = feature["person_id"].nunique()
    patient_count = cdm["person_id"].nunique()
    patient_ratio = feature_patient_count / patient_count

    find_excel_row_and_write_metadata_count(excel_path, sheetname, id, feature_count, row_count, feature_patient_count, patient_count, patient_ratio)

def META_0031(config, cdm_path, excel_path, sheetname):
    """
    height 데이터 수 확인
    """
    table_name = "measurement"
    column_name = "measurement_concept_id" # vs는 concept_id로 조회
    code_list = config["metadata"]["height"]
    id = inspect.currentframe().f_code.co_name
    cdm = pd.read_csv(os.path.join(cdm_path, table_name + ".csv"))
    feature = cdm[cdm[column_name].isin(code_list)]

    feature_count = len(feature)
    row_count = len(cdm)
    feature_patient_count = feature["person_id"].nunique()
    patient_count = cdm["person_id"].nunique()
    patient_ratio = feature_patient_count / patient_count

    find_excel_row_and_write_metadata_count(excel_path, sheetname, id, feature_count, row_count, feature_patient_count, patient_count, patient_ratio)

def META_0032(config, cdm_path, excel_path, sheetname):
    """
    weight 데이터 수 확인
    """
    table_name = "measurement"
    column_name = "measurement_concept_id" # vs는 concept_id로 조회
    code_list = config["metadata"]["weight"]
    id = inspect.currentframe().f_code.co_name
    cdm = pd.read_csv(os.path.join(cdm_path, table_name + ".csv"))
    feature = cdm[cdm[column_name].isin(code_list)]

    feature_count = len(feature)
    row_count = len(cdm)
    feature_patient_count = feature["person_id"].nunique()
    patient_count = cdm["person_id"].nunique()
    patient_ratio = feature_patient_count / patient_count

    find_excel_row_and_write_metadata_count(excel_path, sheetname, id, feature_count, row_count, feature_patient_count, patient_count, patient_ratio)
