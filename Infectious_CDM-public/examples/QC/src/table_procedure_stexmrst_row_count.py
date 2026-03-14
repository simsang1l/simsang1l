"""
사용한 원본데이터와 변환된 CDM procedure_stexmrst테이블의 건수 및 환자수 확인
"""

import pandas as pd
import os, sys
# 상위 디렉토리 (my_project 폴더)를 sys.path에 추가
current_dir = os.path.dirname(__file__)  # 현재 파일의 디렉토리
parent_dir = os.path.dirname(current_dir)  # 현재 디렉토리의 부모 디렉토리 (QC)
project_dir = os.path.dirname(parent_dir)  # QC의 부모 디렉토리 (my_project)
sys.path.insert(0, project_dir)
from QC.src.excel_util import write_df_to_excel
from DataTransformer import ProcedureStexmrstTransformer

def procedure_stexmrst_row_count(config, cdm_path, source_path, excel_path, config_path, sheetname):
    table_name = "procedure_stexmrst"
    source_table = config[table_name]["data"]["source_data"]
    patno = config["person_source_value"]
    orddate = config[table_name]["columns"]["orddate"]
    cdm_date = "처방일"

    source = ProcedureStexmrstTransformer(config_path).process_source()
    cdm = pd.read_csv(os.path.join(cdm_path, table_name + ".csv"))

    source['year_month'] = source[orddate].dt.to_period('M')
    cdm[cdm_date] = pd.to_datetime(cdm[cdm_date])
    cdm['year_month'] = cdm[cdm_date].dt.to_period('M')

    summary_source = source.groupby("year_month").agg(
        total_count=(patno, "count"),
        unique_patients=(patno, "nunique")
    ).reset_index()
    summary_cdm = cdm.groupby("year_month").agg(
        total_count_cdm=("person_id", "count"),
        unique_patients_cdm=("person_id", "nunique")
    ).reset_index()

    summary = pd.merge(summary_source, summary_cdm, on="year_month", how="outer")
    summary = summary.sort_values(by="year_month", ascending=False)
 
    summary["id"] = 10
    summary["source_table"] = f"{source_table}"
    summary["cdm_table"] = table_name
    summary["row_rate"] = summary["total_count_cdm"] / summary["total_count"]
    summary["patients_rate"] = summary["unique_patients_cdm"] / summary["unique_patients"]
    summary = summary[["id", "source_table", "year_month", "total_count", "unique_patients",
                "cdm_table", "total_count_cdm", "unique_patients_cdm", "row_rate", "patients_rate"]]
    
    df = pd.DataFrame(summary)    

    write_df_to_excel(excel_path, sheetname, df)



