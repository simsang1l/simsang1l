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
from QC.src.excel_util import find_excel_row_and_write_error_ratio

def DQ_0001(cdm_path, excel_path, sheetname):
    """
    person_source_value에 NULL이나 공백 있는지 진단
    """
    table_name = "person"
    column_name = "person_source_value"
    id = inspect.currentframe().f_code.co_name
    cdm = pd.read_csv(os.path.join(cdm_path, table_name + ".csv"), dtype=str)

    cdm[column_name] = cdm[column_name].str.strip()
    empty_count = sum(cdm[column_name] == '')

    null_count = cdm[column_name].isnull().sum()
    error_count = null_count + empty_count
    row_count = len(cdm)
    null_ratio = null_count / row_count

    find_excel_row_and_write_error_ratio(excel_path, sheetname, id, error_count, row_count, null_ratio)


def DQ_0002(cdm_path, excel_path, sheetname):
    """
    person_source_value는 중복값 없는지 확인
    """
    table_name = "person"
    column_name = "person_source_value"
    id = inspect.currentframe().f_code.co_name
    cdm = pd.read_csv(os.path.join(cdm_path, table_name + ".csv"), dtype=str)
 
    error_count = cdm[column_name].duplicated(keep=False).sum()
    row_count = len(cdm)
    null_ratio = error_count / row_count

    find_excel_row_and_write_error_ratio(excel_path, sheetname, id, error_count, row_count, null_ratio)

def DQ_0003(cdm_path, excel_path, sheetname):
    """
    year_of_birth가 올 해 이전 날짜로 채워져 있는지 확인
    """
    table_name = "person"
    column_name = "year_of_birth"
    id = inspect.currentframe().f_code.co_name
    cdm = pd.read_csv(os.path.join(cdm_path, table_name + ".csv"))

    error_count = len(cdm[cdm[column_name] > datetime.now().year])
    row_count = len(cdm)
    null_ratio = error_count / row_count

    find_excel_row_and_write_error_ratio(excel_path, sheetname, id, error_count, row_count, null_ratio)

def DQ_0004(cdm_path, excel_path, sheetname):
    """
    month_of_birth가 올 해 이전 날짜로 채워져 있는지 확인
    """
    table_name = "person"
    column_name = "month_of_birth"
    id = inspect.currentframe().f_code.co_name
    cdm = pd.read_csv(os.path.join(cdm_path, table_name + ".csv"))

    error_count = len(cdm[(cdm[column_name] > 12)| (cdm[column_name] < 1)])
    row_count = len(cdm)
    null_ratio = error_count / row_count

    find_excel_row_and_write_error_ratio(excel_path, sheetname, id, error_count, row_count, null_ratio)

def DQ_0005(cdm_path, excel_path, sheetname):
    """
    day_of_birth가 올 해 이전 날짜로 채워져 있는지 확인
    """
    table_name = "person"
    column_name = "day_of_birth"
    id = inspect.currentframe().f_code.co_name
    cdm = pd.read_csv(os.path.join(cdm_path, table_name + ".csv"))

    error_count = len(cdm[(cdm[column_name] > 31)| (cdm[column_name] < 1)])
    row_count = len(cdm)
    null_ratio = error_count / row_count

    find_excel_row_and_write_error_ratio(excel_path, sheetname, id, error_count, row_count, null_ratio)

def DQ_0006(cdm_path, excel_path, sheetname):
    """
    gender_source_value가 NULL이나 공백이 있는지 확인
    """
    table_name = "person"
    column_name = "gender_source_value"
    id = inspect.currentframe().f_code.co_name
    cdm = pd.read_csv(os.path.join(cdm_path, table_name + ".csv"))

    cdm[column_name] = cdm[column_name].str.strip()
    empty_count = sum(cdm[column_name] == '')
    null_count = cdm[column_name].isnull().sum()

    error_count = null_count + empty_count
    row_count = len(cdm)
    null_ratio = error_count / row_count

    find_excel_row_and_write_error_ratio(excel_path, sheetname, id, error_count, row_count, null_ratio)

def DQ_0007(cdm_path, excel_path, sheetname):
    """
    death_datetime이 birth_datetime보다 빠를 수 없음을 확인
    """
    table_name = "person"
    column_name = "death_datetime"
    id = inspect.currentframe().f_code.co_name
    cdm = pd.read_csv(os.path.join(cdm_path, table_name + ".csv"))

    cdm[column_name] = pd.to_datetime(cdm[column_name])
    cdm["birth_datetime"] = pd.to_datetime(cdm["birth_datetime"])

    error_count = len(cdm[cdm[column_name] < cdm["birth_datetime"]])
    row_count = len(cdm)
    null_ratio = error_count / row_count

    find_excel_row_and_write_error_ratio(excel_path, sheetname, id, error_count, row_count, null_ratio)

def DQ_0008(cdm_path, excel_path, sheetname):
    """
    visit_occurrence_id에 NULL이나 공백이 있는지 확인
    """
    table_name = "visit_occurrence"
    column_name = "visit_occurrence_id"
    id = inspect.currentframe().f_code.co_name
    cdm = pd.read_csv(os.path.join(cdm_path, table_name + ".csv"), dtype=str)

    cdm[column_name] = cdm[column_name].str.strip()
    empty_count = sum(cdm[column_name] == '')
    null_count = cdm[column_name].isnull().sum()

    error_count = null_count + empty_count
    row_count = len(cdm)
    null_ratio = error_count / row_count

    find_excel_row_and_write_error_ratio(excel_path, sheetname, id, error_count, row_count, null_ratio)

def DQ_0009(cdm_path, excel_path, sheetname):
    """
    visit_occurrence_id에 중복값이 있는지 확인
    """
    table_name = "visit_occurrence"
    column_name = "visit_occurrence_id"
    id = inspect.currentframe().f_code.co_name
    cdm = pd.read_csv(os.path.join(cdm_path, table_name + ".csv"))

    error_count = cdm[column_name].duplicated(keep=False).sum()
    row_count = len(cdm)
    null_ratio = error_count / row_count

    find_excel_row_and_write_error_ratio(excel_path, sheetname, id, error_count, row_count, null_ratio)

def DQ_0010(cdm_path, excel_path, sheetname):
    """
    visit_start_date가 visit_end_date보다 큰 데이터가 있는지 확인
    """
    table_name = "visit_occurrence"
    column_name = "visit_start_date"
    compare_column = "visit_end_date"
    id = inspect.currentframe().f_code.co_name
    cdm = pd.read_csv(os.path.join(cdm_path, table_name + ".csv"))

    error_count = len(cdm[cdm[column_name] > cdm[compare_column]])
    row_count = len(cdm)
    null_ratio = error_count / row_count

    find_excel_row_and_write_error_ratio(excel_path, sheetname, id, error_count, row_count, null_ratio)

def DQ_0011(cdm_path, excel_path, sheetname):
    """
    visit_source_value에 NULL이나 공백이 있는지 확인
    """
    table_name = "visit_occurrence"
    column_name = "visit_start_date"
    id = inspect.currentframe().f_code.co_name
    cdm = pd.read_csv(os.path.join(cdm_path, table_name + ".csv"), dtype=str)

    cdm[column_name] = cdm[column_name].str.strip()
    empty_count = sum(cdm[column_name] == '')
    null_count = cdm[column_name].isnull().sum()

    error_count = null_count + empty_count
    row_count = len(cdm)
    null_ratio = error_count / row_count

    find_excel_row_and_write_error_ratio(excel_path, sheetname, id, error_count, row_count, null_ratio)

def DQ_0012(cdm_path, excel_path, sheetname):
    """
    visit_start_datetime에 NULL이나 공백이 있는지 확인
    """
    table_name = "visit_occurrence"
    column_name = "visit_start_datetime"
    id = inspect.currentframe().f_code.co_name
    cdm = pd.read_csv(os.path.join(cdm_path, table_name + ".csv"), dtype=str)

    cdm[column_name] = cdm[column_name].str.strip()
    empty_count = sum(cdm[column_name] == '')
    null_count = cdm[column_name].isnull().sum()

    error_count = null_count + empty_count
    row_count = len(cdm)
    null_ratio = error_count / row_count

    find_excel_row_and_write_error_ratio(excel_path, sheetname, id, error_count, row_count, null_ratio)

def DQ_0013(cdm_path, excel_path, sheetname):
    """
    visit_occurrence의 person_id중 person테이블에 없는 환자인지 확인
    """
    table_name = "visit_occurrence"
    column_name = "person_id"
    id = inspect.currentframe().f_code.co_name
    cdm = pd.read_csv(os.path.join(cdm_path, table_name + ".csv"), dtype=str)
    person = pd.read_csv(os.path.join(cdm_path, "person" + ".csv"), dtype=str)

    cdm = pd.merge(cdm, person, on=column_name, how="outer") #, suffixes=('', '_p'))

    # NULL, 공백, person테이블과 관계 확인!
    cdm[column_name] = cdm[column_name].str.strip()
    empty_count = sum(cdm[column_name] == '')
    null_count = cdm[column_name].isnull().sum()

    error_count = null_count + empty_count
    row_count = len(cdm)
    null_ratio = error_count / row_count

    find_excel_row_and_write_error_ratio(excel_path, sheetname, id, error_count, row_count, null_ratio)

def DQ_0014(cdm_path, excel_path, sheetname):
    """
    visit_source_value에 NULL이나 공백이 있는지 확인
    """
    table_name = "visit_occurrence"
    column_name = "visit_source_value"
    id = inspect.currentframe().f_code.co_name
    cdm = pd.read_csv(os.path.join(cdm_path, table_name + ".csv"), dtype=str)

    # NULL, 공백, person테이블과 관계 확인!
    cdm[column_name] = cdm[column_name].str.strip()
    empty_count = sum(cdm[column_name] == '')
    null_count = cdm[column_name].isnull().sum()

    error_count = null_count + empty_count
    row_count = len(cdm)
    null_ratio = error_count / row_count

    find_excel_row_and_write_error_ratio(excel_path, sheetname, id, error_count, row_count, null_ratio)

def DQ_0015(cdm_path, excel_path, sheetname):
    """
    condition_occurrence테이블에 visit_occurrence_id컬럼의 NULL이나 공백이 있는지 확인
    """
    table_name = "condition_occurrence"
    column_name = "visit_occurrence_id"
    id = inspect.currentframe().f_code.co_name
    cdm = pd.read_csv(os.path.join(cdm_path, table_name + ".csv"), dtype=str)

    # NULL, 공백, person테이블과 관계 확인!
    cdm[column_name] = cdm[column_name].str.strip()
    empty_count = sum(cdm[column_name] == '')
    null_count = cdm[column_name].isnull().sum()

    error_count = null_count + empty_count
    row_count = len(cdm)
    null_ratio = error_count / row_count

    find_excel_row_and_write_error_ratio(excel_path, sheetname, id, error_count, row_count, null_ratio)

def DQ_0016(cdm_path, excel_path, sheetname):
    """
    drug_exposure테이블에 visit_occurrence_id컬럼의 NULL이나 공백이 있는지 확인
    """
    table_name = "drug_exposure"
    column_name = "visit_occurrence_id"
    id = inspect.currentframe().f_code.co_name
    cdm = pd.read_csv(os.path.join(cdm_path, table_name + ".csv"), dtype=str)

    # NULL, 공백, person테이블과 관계 확인!
    cdm[column_name] = cdm[column_name].str.strip()
    empty_count = sum(cdm[column_name] == '')
    null_count = cdm[column_name].isnull().sum()

    error_count = null_count + empty_count
    row_count = len(cdm)
    null_ratio = error_count / row_count

    find_excel_row_and_write_error_ratio(excel_path, sheetname, id, error_count, row_count, null_ratio)

def DQ_0017(cdm_path, excel_path, sheetname):
    """
    measurement테이블에 visit_occurrence_id컬럼의 NULL이나 공백이 있는지 확인
    """
    table_name = "measurement"
    column_name = "visit_occurrence_id"
    id = inspect.currentframe().f_code.co_name
    cdm = pd.read_csv(os.path.join(cdm_path, table_name + ".csv"), dtype=str)

    # NULL, 공백, person테이블과 관계 확인!
    cdm[column_name] = cdm[column_name].str.strip()
    empty_count = sum(cdm[column_name] == '')
    null_count = cdm[column_name].isnull().sum()

    error_count = null_count + empty_count
    row_count = len(cdm)
    null_ratio = error_count / row_count

    find_excel_row_and_write_error_ratio(excel_path, sheetname, id, error_count, row_count, null_ratio)

def DQ_0018(cdm_path, excel_path, sheetname):
    """
    procedure_occurrence테이블에 visit_occurrence_id컬럼의 NULL이나 공백이 있는지 확인
    """
    table_name = "procedure_occurrence"
    column_name = "visit_occurrence_id"
    id = inspect.currentframe().f_code.co_name
    cdm = pd.read_csv(os.path.join(cdm_path, table_name + ".csv"), dtype=str)

    # NULL, 공백, person테이블과 관계 확인!
    cdm[column_name] = cdm[column_name].str.strip()
    empty_count = sum(cdm[column_name] == '')
    null_count = cdm[column_name].isnull().sum()

    error_count = null_count + empty_count
    row_count = len(cdm)
    null_ratio = error_count / row_count

    find_excel_row_and_write_error_ratio(excel_path, sheetname, id, error_count, row_count, null_ratio)

def DQ_0019(cdm_path, excel_path, sheetname):
    """
    condition_occurrence테이블에 condition_occurrence_id컬럼의 NULL이나 공백이 있는지 확인
    """
    table_name = "condition_occurrence"
    column_name = "condition_occurrence_id"
    id = inspect.currentframe().f_code.co_name
    cdm = pd.read_csv(os.path.join(cdm_path, table_name + ".csv"), dtype=str)

    # NULL, 공백, person테이블과 관계 확인!
    cdm[column_name] = cdm[column_name].str.strip()
    empty_count = sum(cdm[column_name] == '')
    null_count = cdm[column_name].isnull().sum()

    error_count = null_count + empty_count
    row_count = len(cdm)
    null_ratio = error_count / row_count

    find_excel_row_and_write_error_ratio(excel_path, sheetname, id, error_count, row_count, null_ratio)

def DQ_0020(cdm_path, excel_path, sheetname):
    """
    condition_occurrence_id에 중복값이 있는지 확인
    """
    table_name = "condition_occurrence"
    column_name = "condition_occurrence_id"
    id = inspect.currentframe().f_code.co_name
    cdm = pd.read_csv(os.path.join(cdm_path, table_name + ".csv"))

    error_count = cdm[column_name].duplicated(keep=False).sum()
    row_count = len(cdm)
    null_ratio = error_count / row_count

    find_excel_row_and_write_error_ratio(excel_path, sheetname, id, error_count, row_count, null_ratio)

def DQ_0021(cdm_path, excel_path, sheetname):
    """
    condition_start_date가 condition_end_date보다 큰 데이터가 있는지 확인
    """
    table_name = "condition_occurrence"
    column_name = "condition_start_date"
    compare_column = "condition_end_date"
    id = inspect.currentframe().f_code.co_name
    cdm = pd.read_csv(os.path.join(cdm_path, table_name + ".csv"))

    error_count = len(cdm[cdm[column_name] > cdm[compare_column]])
    row_count = len(cdm)
    null_ratio = error_count / row_count

    find_excel_row_and_write_error_ratio(excel_path, sheetname, id, error_count, row_count, null_ratio)


def DQ_0022(cdm_path, excel_path, sheetname):
    """
    condition_start_date에 NULL이나 공백이 있는지 확인
    """
    table_name = "condition_occurrence"
    column_name = "condition_start_date"
    id = inspect.currentframe().f_code.co_name
    cdm = pd.read_csv(os.path.join(cdm_path, table_name + ".csv"), dtype=str)

    # NULL, 공백, person테이블과 관계 확인!
    cdm[column_name] = cdm[column_name].str.strip()
    empty_count = sum(cdm[column_name] == '')
    null_count = cdm[column_name].isnull().sum()

    error_count = null_count + empty_count
    row_count = len(cdm)
    null_ratio = error_count / row_count

    find_excel_row_and_write_error_ratio(excel_path, sheetname, id, error_count, row_count, null_ratio)

def DQ_0023(cdm_path, excel_path, sheetname):
    """
    condition_start_datetime에 NULL이나 공백이 있는지 확인
    """
    table_name = "condition_occurrence"
    column_name = "condition_start_datetime"
    id = inspect.currentframe().f_code.co_name
    cdm = pd.read_csv(os.path.join(cdm_path, table_name + ".csv"), dtype=str)

    # NULL, 공백, person테이블과 관계 확인!
    cdm[column_name] = cdm[column_name].str.strip()
    empty_count = sum(cdm[column_name] == '')
    null_count = cdm[column_name].isnull().sum()

    error_count = null_count + empty_count
    row_count = len(cdm)
    null_ratio = error_count / row_count

    find_excel_row_and_write_error_ratio(excel_path, sheetname, id, error_count, row_count, null_ratio)

def DQ_0024(cdm_path, excel_path, sheetname):
    """
    condition_source_value에 NULL이나 공백이 있는지 확인
    """
    table_name = "condition_occurrence"
    column_name = "condition_source_value"
    id = inspect.currentframe().f_code.co_name
    cdm = pd.read_csv(os.path.join(cdm_path, table_name + ".csv"))

    # NULL, 공백, person테이블과 관계 확인!
    cdm[column_name] = cdm[column_name].str.strip()
    empty_count = sum(cdm[column_name] == '')
    null_count = cdm[column_name].isnull().sum()

    error_count = null_count + empty_count
    row_count = len(cdm)
    null_ratio = error_count / row_count

    find_excel_row_and_write_error_ratio(excel_path, sheetname, id, error_count, row_count, null_ratio)

def DQ_0025(cdm_path, excel_path, sheetname):
    """
    condition_occurrence person_id중 person테이블에 없는 환자인지 확인
    """
    table_name = "condition_occurrence"
    column_name = "person_id"
    id = inspect.currentframe().f_code.co_name
    cdm = pd.read_csv(os.path.join(cdm_path, table_name + ".csv"), dtype=str)
    person = pd.read_csv(os.path.join(cdm_path, "person" + ".csv"), dtype=str)

    cdm = pd.merge(cdm, person, on=column_name, how="outer", suffixes=('', '_'))

    # NULL, 공백, person테이블과 관계 확인!
    cdm[column_name] = cdm[column_name].str.strip()
    empty_count = sum(cdm[column_name] == '')
    null_count = cdm[column_name].isnull().sum()

    error_count = null_count + empty_count
    row_count = len(cdm)
    null_ratio = error_count / row_count

    find_excel_row_and_write_error_ratio(excel_path, sheetname, id, error_count, row_count, null_ratio)

def DQ_0026(cdm_path, excel_path, sheetname):
    """
    drug_exposure_id에 NULL이나 공백이 있는지 확인
    """
    table_name = "drug_exposure"
    column_name = "drug_exposure_id"
    id = inspect.currentframe().f_code.co_name
    cdm = pd.read_csv(os.path.join(cdm_path, table_name + ".csv"), dtype=str)

    # NULL, 공백, person테이블과 관계 확인!
    cdm[column_name] = cdm[column_name].str.strip()
    empty_count = sum(cdm[column_name] == '')
    null_count = cdm[column_name].isnull().sum()

    error_count = null_count + empty_count
    row_count = len(cdm)
    null_ratio = error_count / row_count

    find_excel_row_and_write_error_ratio(excel_path, sheetname, id, error_count, row_count, null_ratio)

def DQ_0027(cdm_path, excel_path, sheetname):
    """
    drug_exposure_id에 중복값이 있는지 확인
    """
    table_name = "drug_exposure"
    column_name = "drug_exposure_id"
    id = inspect.currentframe().f_code.co_name
    cdm = pd.read_csv(os.path.join(cdm_path, table_name + ".csv"))

    error_count = cdm[column_name].duplicated(keep=False).sum()
    row_count = len(cdm)
    null_ratio = error_count / row_count

    find_excel_row_and_write_error_ratio(excel_path, sheetname, id, error_count, row_count, null_ratio)

def DQ_0028(cdm_path, excel_path, sheetname):
    """
    drug_exposure_start_date가 drug_exposure_end_date보다 큰 데이터가 있는지 확인
    """
    table_name = "drug_exposure"
    column_name = "drug_exposure_start_date"
    compare_column = "drug_exposure_end_date"
    id = inspect.currentframe().f_code.co_name
    cdm = pd.read_csv(os.path.join(cdm_path, table_name + ".csv"))

    error_count = len(cdm[cdm[column_name] > cdm[compare_column]])
    row_count = len(cdm)
    null_ratio = error_count / row_count

    find_excel_row_and_write_error_ratio(excel_path, sheetname, id, error_count, row_count, null_ratio)


def DQ_0029(cdm_path, excel_path, sheetname):
    """
    drug_exposure_start_date에 NULL이나 공백이 있는지 확인
    """
    table_name = "drug_exposure"
    column_name = "drug_exposure_start_date"
    id = inspect.currentframe().f_code.co_name
    cdm = pd.read_csv(os.path.join(cdm_path, table_name + ".csv"), dtype=str)

    # NULL, 공백, person테이블과 관계 확인!
    cdm[column_name] = cdm[column_name].str.strip()
    empty_count = sum(cdm[column_name] == '')
    null_count = cdm[column_name].isnull().sum()

    error_count = null_count + empty_count
    row_count = len(cdm)
    null_ratio = error_count / row_count

    find_excel_row_and_write_error_ratio(excel_path, sheetname, id, error_count, row_count, null_ratio)

def DQ_0030(cdm_path, excel_path, sheetname):
    """
    drug_exposure_start_datetime에 NULL이나 공백이 있는지 확인
    """
    table_name = "drug_exposure"
    column_name = "drug_exposure_start_datetime"
    id = inspect.currentframe().f_code.co_name
    cdm = pd.read_csv(os.path.join(cdm_path, table_name + ".csv"), dtype=str)

    # NULL, 공백, person테이블과 관계 확인!
    cdm[column_name] = cdm[column_name].str.strip()
    empty_count = sum(cdm[column_name] == '')
    null_count = cdm[column_name].isnull().sum()

    error_count = null_count + empty_count
    row_count = len(cdm)
    null_ratio = error_count / row_count

    find_excel_row_and_write_error_ratio(excel_path, sheetname, id, error_count, row_count, null_ratio)

def DQ_0031(cdm_path, excel_path, sheetname):
    """
    drug_exposure의 person_id중 person테이블에 없는 환자인지 확인
    """
    table_name = "drug_exposure"
    column_name = "person_id"
    id = inspect.currentframe().f_code.co_name
    cdm = pd.read_csv(os.path.join(cdm_path, table_name + ".csv"), dtype=str)
    person = pd.read_csv(os.path.join(cdm_path, "person" + ".csv"), dtype=str)

    cdm = pd.merge(cdm, person, on=column_name, how="outer", suffixes=('', '_'))

    # NULL, 공백, person테이블과 관계 확인!
    cdm[column_name] = cdm[column_name].str.strip()
    empty_count = sum(cdm[column_name] == '')
    null_count = cdm[column_name].isnull().sum()

    error_count = null_count + empty_count
    row_count = len(cdm)
    null_ratio = error_count / row_count

    find_excel_row_and_write_error_ratio(excel_path, sheetname, id, error_count, row_count, null_ratio)

def DQ_0032(cdm_path, excel_path, sheetname):
    """
    drug_source_value에 NULL이나 공백이 있는지 확인
    """
    table_name = "drug_exposure"
    column_name = "drug_source_value"
    id = inspect.currentframe().f_code.co_name
    cdm = pd.read_csv(os.path.join(cdm_path, table_name + ".csv"), dtype=str)

    # NULL, 공백, person테이블과 관계 확인!
    cdm[column_name] = cdm[column_name].str.strip()
    empty_count = sum(cdm[column_name] == '')
    null_count = cdm[column_name].isnull().sum()

    error_count = null_count + empty_count
    row_count = len(cdm)
    null_ratio = error_count / row_count

    find_excel_row_and_write_error_ratio(excel_path, sheetname, id, error_count, row_count, null_ratio)

def DQ_0033(cdm_path, excel_path, sheetname):
    """
    measurement_id에 NULL이나 공백이 있는지 확인
    """
    table_name = "measurement"
    column_name = "measurement_id"
    id = inspect.currentframe().f_code.co_name
    cdm = pd.read_csv(os.path.join(cdm_path, table_name + ".csv"), dtype=str)

    # NULL, 공백, person테이블과 관계 확인!
    cdm[column_name] = cdm[column_name].str.strip()
    empty_count = sum(cdm[column_name] == '')
    null_count = cdm[column_name].isnull().sum()

    error_count = null_count + empty_count
    row_count = len(cdm)
    null_ratio = error_count / row_count

    find_excel_row_and_write_error_ratio(excel_path, sheetname, id, error_count, row_count, null_ratio)

def DQ_0034(cdm_path, excel_path, sheetname):
    """
    measurement_id에 중복값이 있는지 확인
    """
    table_name = "measurement"
    column_name = "measurement_id"
    id = inspect.currentframe().f_code.co_name
    cdm = pd.read_csv(os.path.join(cdm_path, table_name + ".csv"))

    error_count = cdm[column_name].duplicated(keep=False).sum()
    row_count = len(cdm)
    null_ratio = error_count / row_count

    find_excel_row_and_write_error_ratio(excel_path, sheetname, id, error_count, row_count, null_ratio)

def DQ_0035(cdm_path, excel_path, sheetname):
    """
    measurement_date에 NULL이나 공백이 있는지 확인
    """
    table_name = "measurement"
    column_name = "measurement_date"
    id = inspect.currentframe().f_code.co_name
    cdm = pd.read_csv(os.path.join(cdm_path, table_name + ".csv"), dtype=str)

    # NULL, 공백, person테이블과 관계 확인!
    cdm[column_name] = cdm[column_name].str.strip()
    empty_count = sum(cdm[column_name] == '')
    null_count = cdm[column_name].isnull().sum()

    error_count = null_count + empty_count
    row_count = len(cdm)
    null_ratio = error_count / row_count

    find_excel_row_and_write_error_ratio(excel_path, sheetname, id, error_count, row_count, null_ratio)

def DQ_0036(cdm_path, excel_path, sheetname):
    """
    measurement_datetime에 NULL이나 공백이 있는지 확인
    """
    table_name = "measurement"
    column_name = "measurement_datetime"
    id = inspect.currentframe().f_code.co_name
    cdm = pd.read_csv(os.path.join(cdm_path, table_name + ".csv"), dtype=str)

    # NULL, 공백, person테이블과 관계 확인!
    cdm[column_name] = cdm[column_name].str.strip()
    empty_count = sum(cdm[column_name] == '')
    null_count = cdm[column_name].isnull().sum()

    error_count = null_count + empty_count
    row_count = len(cdm)
    null_ratio = error_count / row_count

    find_excel_row_and_write_error_ratio(excel_path, sheetname, id, error_count, row_count, null_ratio)

def DQ_0037(cdm_path, excel_path, sheetname):
    """
    measurement의 person_id중 person테이블에 없는 환자인지 확인
    """
    table_name = "measurement"
    column_name = "person_id"
    id = inspect.currentframe().f_code.co_name
    cdm = pd.read_csv(os.path.join(cdm_path, table_name + ".csv"), dtype=str)
    person = pd.read_csv(os.path.join(cdm_path, "person" + ".csv"), dtype=str)

    cdm = pd.merge(cdm, person, on=column_name, how="outer", suffixes=('', '_'))

    # NULL, 공백, person테이블과 관계 확인!
    cdm[column_name] = cdm[column_name].str.strip()
    empty_count = sum(cdm[column_name] == '')
    null_count = cdm[column_name].isnull().sum()

    error_count = null_count + empty_count
    row_count = len(cdm)
    null_ratio = error_count / row_count

    find_excel_row_and_write_error_ratio(excel_path, sheetname, id, error_count, row_count, null_ratio)

def DQ_0038(cdm_path, excel_path, sheetname):
    """
    measurement_source_value에 NULL이나 공백이 있는지 확인
    """
    table_name = "measurement"
    column_name = "measurement_source_value"
    id = inspect.currentframe().f_code.co_name
    cdm = pd.read_csv(os.path.join(cdm_path, table_name + ".csv"), dtype=str)

    # NULL, 공백, person테이블과 관계 확인!
    cdm[column_name] = cdm[column_name].str.strip()
    empty_count = sum(cdm[column_name] == '')
    null_count = cdm[column_name].isnull().sum()

    error_count = null_count + empty_count
    row_count = len(cdm)
    null_ratio = error_count / row_count

    find_excel_row_and_write_error_ratio(excel_path, sheetname, id, error_count, row_count, null_ratio)

def DQ_0039(cdm_path, excel_path, sheetname):
    """
    procedure_occurrence_id에 NULL이나 공백이 있는지 확인
    """
    table_name = "procedure_occurrence"
    column_name = "procedure_occurrence_id"
    id = inspect.currentframe().f_code.co_name
    cdm = pd.read_csv(os.path.join(cdm_path, table_name + ".csv"), dtype=str)

    # NULL, 공백, person테이블과 관계 확인!
    cdm[column_name] = cdm[column_name].str.strip()
    empty_count = sum(cdm[column_name] == '')
    null_count = cdm[column_name].isnull().sum()

    error_count = null_count + empty_count
    row_count = len(cdm)
    null_ratio = error_count / row_count

    find_excel_row_and_write_error_ratio(excel_path, sheetname, id, error_count, row_count, null_ratio)

def DQ_0040(cdm_path, excel_path, sheetname):
    """
    procedure_occurrence_id에 중복값이 있는지 확인
    """
    table_name = "procedure_occurrence"
    column_name = "procedure_occurrence_id"
    id = inspect.currentframe().f_code.co_name
    cdm = pd.read_csv(os.path.join(cdm_path, table_name + ".csv"))

    error_count = cdm[column_name].duplicated(keep=False).sum()
    row_count = len(cdm)
    null_ratio = error_count / row_count

    find_excel_row_and_write_error_ratio(excel_path, sheetname, id, error_count, row_count, null_ratio)

def DQ_0041(cdm_path, excel_path, sheetname):
    """
    procedure_date에 NULL이나 공백이 있는지 확인
    """
    table_name = "procedure_occurrence"
    column_name = "procedure_date"
    id = inspect.currentframe().f_code.co_name
    cdm = pd.read_csv(os.path.join(cdm_path, table_name + ".csv"), dtype=str)

    # NULL, 공백, person테이블과 관계 확인!
    cdm[column_name] = cdm[column_name].str.strip()
    empty_count = sum(cdm[column_name] == '')
    null_count = cdm[column_name].isnull().sum()

    error_count = null_count + empty_count
    row_count = len(cdm)
    null_ratio = error_count / row_count

    find_excel_row_and_write_error_ratio(excel_path, sheetname, id, error_count, row_count, null_ratio)

def DQ_0042(cdm_path, excel_path, sheetname):
    """
    procedure_datetime에 NULL이나 공백이 있는지 확인
    """
    table_name = "procedure_occurrence"
    column_name = "procedure_datetime"
    id = inspect.currentframe().f_code.co_name
    cdm = pd.read_csv(os.path.join(cdm_path, table_name + ".csv"), dtype=str)

    # NULL, 공백, person테이블과 관계 확인!
    cdm[column_name] = cdm[column_name].str.strip()
    empty_count = sum(cdm[column_name] == '')
    null_count = cdm[column_name].isnull().sum()

    error_count = null_count + empty_count
    row_count = len(cdm)
    null_ratio = error_count / row_count

    find_excel_row_and_write_error_ratio(excel_path, sheetname, id, error_count, row_count, null_ratio)

def DQ_0043(cdm_path, excel_path, sheetname):
    """
    procedure_occurrence의 person_id중 person테이블에 없는 환자인지 확인
    """
    table_name = "procedure_occurrence"
    column_name = "person_id"
    id = inspect.currentframe().f_code.co_name
    cdm = pd.read_csv(os.path.join(cdm_path, table_name + ".csv"), dtype=str)
    person = pd.read_csv(os.path.join(cdm_path, "person" + ".csv"), dtype=str)

    cdm = pd.merge(cdm, person, on=column_name, how="outer", suffixes=('', '_'))

    # NULL, 공백, person테이블과 관계 확인!
    cdm[column_name] = cdm[column_name].str.strip()
    empty_count = sum(cdm[column_name] == '')
    null_count = cdm[column_name].isnull().sum()

    error_count = null_count + empty_count
    row_count = len(cdm)
    null_ratio = error_count / row_count

    find_excel_row_and_write_error_ratio(excel_path, sheetname, id, error_count, row_count, null_ratio)

def DQ_0044(cdm_path, excel_path, sheetname):
    """
    procedure_source_value에 NULL이나 공백이 있는지 확인
    """
    table_name = "procedure_occurrence"
    column_name = "procedure_source_value"
    id = inspect.currentframe().f_code.co_name
    cdm = pd.read_csv(os.path.join(cdm_path, table_name + ".csv"))

    # NULL, 공백, person테이블과 관계 확인!
    cdm[column_name] = cdm[column_name].str.strip()
    empty_count = sum(cdm[column_name] == '')
    null_count = cdm[column_name].isnull().sum()

    error_count = null_count + empty_count
    row_count = len(cdm)
    null_ratio = error_count / row_count

    find_excel_row_and_write_error_ratio(excel_path, sheetname, id, error_count, row_count, null_ratio)

def DQ_0045(cdm_path, excel_path, sheetname):
    """
    observation_period_id에 NULL이나 공백이 있는지 확인
    """
    table_name = "observation_period"
    column_name = "observation_period_id"
    id = inspect.currentframe().f_code.co_name
    cdm = pd.read_csv(os.path.join(cdm_path, table_name + ".csv"), dtype=str)

    # NULL, 공백, person테이블과 관계 확인!
    cdm[column_name] = cdm[column_name].str.strip()
    empty_count = sum(cdm[column_name] == '')
    null_count = cdm[column_name].isnull().sum()

    error_count = null_count + empty_count
    row_count = len(cdm)
    null_ratio = error_count / row_count

    find_excel_row_and_write_error_ratio(excel_path, sheetname, id, error_count, row_count, null_ratio)

def DQ_0046(cdm_path, excel_path, sheetname):
    """
    observation_period_id에 중복값이 있는지 확인
    """
    table_name = "observation_period"
    column_name = "observation_period_id"
    id = inspect.currentframe().f_code.co_name
    cdm = pd.read_csv(os.path.join(cdm_path, table_name + ".csv"))

    error_count = cdm[column_name].duplicated(keep=False).sum()
    row_count = len(cdm)
    null_ratio = error_count / row_count

    find_excel_row_and_write_error_ratio(excel_path, sheetname, id, error_count, row_count, null_ratio)

def DQ_0047(cdm_path, excel_path, sheetname):
    """
    observation_period_start_date가 observation_period_end_date보다 큰 데이터가 있는지 확인
    """
    table_name = "observation_period"
    column_name = "observation_period_start_date"
    compare_column = "observation_period_end_date"
    id = inspect.currentframe().f_code.co_name
    cdm = pd.read_csv(os.path.join(cdm_path, table_name + ".csv"))

    error_count = len(cdm[cdm[column_name] > cdm[compare_column]])
    row_count = len(cdm)
    null_ratio = error_count / row_count

    find_excel_row_and_write_error_ratio(excel_path, sheetname, id, error_count, row_count, null_ratio)


def DQ_0048(cdm_path, excel_path, sheetname):
    """
    observation_period_start_date에 NULL이나 공백이 있는지 확인
    """
    table_name = "observation_period"
    column_name = "observation_period_start_date"
    id = inspect.currentframe().f_code.co_name
    cdm = pd.read_csv(os.path.join(cdm_path, table_name + ".csv"), dtype=str)

    # NULL, 공백, person테이블과 관계 확인!
    cdm[column_name] = cdm[column_name].str.strip()
    empty_count = sum(cdm[column_name] == '')
    null_count = cdm[column_name].isnull().sum()

    error_count = null_count + empty_count
    row_count = len(cdm)
    null_ratio = error_count / row_count

    find_excel_row_and_write_error_ratio(excel_path, sheetname, id, error_count, row_count, null_ratio)

def DQ_0049(cdm_path, excel_path, sheetname):
    """
    observation_period_end_date에 NULL이나 공백이 있는지 확인
    """
    table_name = "observation_period"
    column_name = "observation_period_end_date"
    id = inspect.currentframe().f_code.co_name
    cdm = pd.read_csv(os.path.join(cdm_path, table_name + ".csv"), dtype=str)

    # NULL, 공백, person테이블과 관계 확인!
    cdm[column_name] = cdm[column_name].str.strip()
    empty_count = sum(cdm[column_name] == '')
    null_count = cdm[column_name].isnull().sum()

    error_count = null_count + empty_count
    row_count = len(cdm)
    null_ratio = error_count / row_count

    find_excel_row_and_write_error_ratio(excel_path, sheetname, id, error_count, row_count, null_ratio)

def DQ_0050(cdm_path, excel_path, sheetname):
    """
    observation_period의 person_id중 person테이블에 없는 환자인지 확인
    """
    table_name = "observation_period"
    column_name = "person_id"
    id = inspect.currentframe().f_code.co_name
    cdm = pd.read_csv(os.path.join(cdm_path, table_name + ".csv"), dtype=str)
    person = pd.read_csv(os.path.join(cdm_path, "person" + ".csv"), dtype=str)

    cdm = pd.merge(cdm, person, on=column_name, how="outer", suffixes=('', '_'))

    # NULL, 공백, person테이블과 관계 확인!
    cdm[column_name] = cdm[column_name].str.strip()
    empty_count = sum(cdm[column_name] == '')
    null_count = cdm[column_name].isnull().sum()

    error_count = null_count + empty_count
    row_count = len(cdm)
    null_ratio = error_count / row_count

    find_excel_row_and_write_error_ratio(excel_path, sheetname, id, error_count, row_count, null_ratio)
