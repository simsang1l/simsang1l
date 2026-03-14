"""
procedure_occurrence테이블의 field별 기초 통계 정보 구하기
"""

import pandas as pd
import numpy as np
import os, sys
# 상위 디렉토리 (my_project 폴더)를 sys.path에 추가
current_dir = os.path.dirname(__file__)  # 현재 파일의 디렉토리
parent_dir = os.path.dirname(current_dir)  # 현재 디렉토리의 부모 디렉토리 (QC)
project_dir = os.path.dirname(parent_dir)  # QC의 부모 디렉토리 (my_project)
sys.path.insert(0, project_dir)
from QC.src.excel_util import write_df_to_excel

def procedure_occurrence_field_summary(cdm_path, excel_path, sheetname):
    table_name = "procedure_occurrence"
    id = 9
    cdm = pd.read_csv(os.path.join(cdm_path, table_name + ".csv"))

    cdm["procedure_date"] = pd.to_datetime(cdm["procedure_date"])
    cdm["procedure_datetime"] = pd.to_datetime(cdm["procedure_datetime"])
    cdm["처방일"] = pd.to_datetime(cdm["처방일"])
    cdm["수술일"] = pd.to_datetime(cdm["수술일"])
    
    # dataframe.describe에는 numeric데이터에 대해서는 고유값(unique)수가 제공되지 않아 직접 계산
    # dataframe.describe에는 datetime데이터에 대해서는 고유값(unique)수와 최빈값(top), 최빈값수(freq)가 제공되지 않아 직접 계산
    manual_summary_list = []
    for col in cdm.columns:
        value_counts = cdm[col].value_counts(dropna=True)  # NaN 값은 무시
        if not value_counts.empty:
            mode_value = value_counts.index[0]  # 최빈값
            mode_count = value_counts.iloc[0]  # 최빈값의 등장 횟수
        else:
            mode_value = np.nan
            mode_count = 0

        mode_value = cdm[col].mode()[0] if not cdm[col].mode().empty else np.nan
        manual_summary_list.append([col, cdm[col].nunique(), mode_value, mode_count])

    manual_summary = pd.DataFrame(manual_summary_list, columns = ("column_name", "manual_unique", "manual_top", "manual_freq"))

    summary = cdm.describe(include="all", datetime_is_numeric=True).T
    summary.index.name = "column_name"

    null_count = cdm.isnull().sum()
    null_count = null_count.rename("null_count")
    null_count.index.name = "column_name"
    null_count = null_count.reset_index()

    null_ratio = cdm.isnull().mean()
    null_ratio = null_ratio.rename("null_ratio")
    null_ratio.index.name = "column_name"
    null_ratio = null_ratio.reset_index()

    result = pd.merge(summary, null_count, on="column_name", how="outer")
    result = pd.merge(result, null_ratio, on="column_name", how="outer")
    result = pd.merge(result, manual_summary, on="column_name", how="outer")

    result["id"] = f"field_{str(id).zfill(4)}"
    result["table_name"] = table_name
    result["unique"] = np.select([result["unique"].isnull()], [result["manual_unique"]], default=result["unique"])
    result["top"] = np.select([result["top"].isnull()], [result["manual_top"]], default=result["top"])
    result["freq"] = np.select([result["freq"].isnull()], [result["manual_freq"]], default=result["freq"])

    result = result[["id", "table_name", "column_name", "count", "unique",
                "top", "freq", "mean", "std", "min", "25%", "50%", "75%", "max", "null_count", "null_ratio"]]
    
    df = pd.DataFrame(result)    
    write_df_to_excel(excel_path, sheetname, df)



