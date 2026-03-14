"""
사용한 원본데이터와 변환된 CDM provider테이블의 건수 및 환자수 확인
"""

import pandas as pd
import os

def provider_row_count(config, cdm_path, source_path, excel_path):
    table_name = "provider"
    source_table = config[table_name]["data"]["source_data"]
    provider = config[table_name]["columns"]["provider_source_value"]
    source = pd.read_csv(os.path.join(source_path, source_table + ".csv"))
    cdm = pd.read_csv(os.path.join(cdm_path, table_name + ".csv"))

    source_length = len(source)
    cdm_length = len(cdm)
    source_person_count = source[provider].nunique()
    cdm_person_count = cdm["provider_id"].nunique()

    data = [[2, source_table, None, source_length, source_person_count, table_name, cdm_length, cdm_person_count, round(source_length / cdm_length, 3), round(cdm_person_count / source_person_count, 3)]]
    df = pd.DataFrame(data)

    with pd.ExcelWriter(excel_path, engine="openpyxl", mode = "a", if_sheet_exists='overlay') as writer:
        sheetname = "원본비교결과"
        df.to_excel(writer, sheet_name=sheetname, startrow=writer.sheets[sheetname].max_row , index=False, header=None)



