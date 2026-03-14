import pandas as pd
import yaml
import os

def load_config(config_path):
    """
    YAML 설정 파일을 로드합니다.
    """
    with open(config_path, 'r', encoding="utf-8") as file:
        return yaml.safe_load(file)
    
unit_concept_synonym = [
	[9438, "/10^5"],
	[8549, "/10^6"],
	[8786, "/HPF"],
	[8647, "/ul"],
	[8647, "/㎕"],
	[8695, "EU/mL"],
	[8829, "EU/mL"],
	[9157, "GPL-U/ml"],
	[9157, "GPL-U/mL"],
	[8923, "IU/L"],
	[8985, "IU/mL"],
	[8985, "IU/ml"],
	[8985, "IU/㎖"],
    [8985, "_IU/mL"],
	[8529, "Index"],
	[9158, "MPL-U/mL"],
	[9158, "MPL-U/ml"],
	[9388, "PRU"],
	[8645, "U/L"],
	[8645, "U/l"],
	[8763, "U/mL"],
	[8763, "U/ml"],
	[8859, "UG/ML"],
	[8859, "ug/ml"],
    [8859, "μg/mL"],
    [8859, "㎍/mL"],
	[720869, "mL/min/1.73 m^2"],
	[8799, "copies/mL"],
	[8799, "copies/ml"],
	[8713, "g/dl"],
    [8636, "g/l"],
	[8583, "fl"],
	[8576, "mg/㎗"],
	[8840, "mg/dl"],
	[8876, "mmHg"],
	[8751, "mg/l"],
	[8587, "ml"],
    [9570, "ml/dl"],
    [8752, "mm/hr"],
    [9572, "mm²"],
    [8817, "ng/dl"],
    [8817, "ng/㎗"],
    [8842, "ng/ml"],
    [8842, "ng/㎖"],
    [9020, "ng/ml/hr"],
    [8763, "u/mL"], 
    [8763, "u/ml"],
    [8845, "pg/㎖"],
    [8523, "ratio"],
    [44777566, "score"],
    [8555, "sec"],
    [8749, "uMol/L"],
    [8749, "μmol/ℓ"],
    [8837, "ug/dl"],
    [9014, "ug/g creat"],
    [8748, "μg/L"],
    [8748, "ug/ℓ"],
    [9448, "세"],
    [8848, "x10³/㎕"],
    [9550, "mIU/mL"],
    [9550, "mIU/ml"],
    [9093, "uIU/mL"],
    [8862, "mOsm/Kg"],
    [8860, "uU/mL"],
    [8860, "uU/ml"],
    [9093, "uIU/ml"],
    [8510, "U"],
    [8510, "Units"],
    [8510, "Unit"],
    [8909, "mg/day"],
    [8784, "cells/㎕"],
    [8910, "mmol/day"],
    [720842, "x10^6/Kg"],
    [8906, "ug/day"],
    [9384, "점"],
    [8875, "uEq/L"],
    [9557, "mEq/L"]
]

config = load_config("./config.yaml")

# unit_concept_id 불러오기
df = pd.DataFrame(unit_concept_synonym, columns = ["concept_id", "concept_synonym_name"])
unit_concept = pd.read_csv(os.path.join(config["CDM_path"], "concept_unit.csv"))
unit_concept = unit_concept[unit_concept["vocabulary_id"]=="UCUM"]

### concept_synonym 만들기 ##
concept_synonym = pd.merge(df, unit_concept, left_on="concept_id", right_on="concept_id", how="inner")
concept_synonym = concept_synonym[["concept_id", "concept_name", "concept_synonym_name"]]
concept_synonym.to_csv(os.path.join(config["CDM_path"],"unit_concept_synonym.csv"), index = False)

"""
### 전북대병원 unit 매핑 리스트 구하기 ###
measurement = pd.read_csv(os.path.join(config["CDM_path"], "measurement.csv"))
concept_synonym = pd.read_csv(os.path.join(config["CDM_path"], "unit_concept_synonym.csv"))
# 원본 unit과 concept_synonym과 매핑
measurement = measurement[["unit_concept_id", "unit_source_value"]].drop_duplicates()
df = pd.merge(measurement, concept_synonym, left_on="unit_source_value", right_on="concept_synonym_name", how = "left")
# 중복되는 데이터 제거
df = df[["unit_concept_id", "unit_source_value", "concept_id", "concept_name"]].drop_duplicates()
# unit_concept_id먼저 매핑 후 concept_id 매핑하여 리스트 만들기
df["unit_concept_id"] = df["unit_concept_id"].combine_first(df["concept_id"])
df = df[["unit_concept_id", "concept_name", "unit_source_value"]]
# csv 파일 만들기 
df.to_csv("./unit_concept_id_list.csv", index = False)


### 매핑률 변화 구하기 ###
measurement_map_rate = len(measurement[measurement["unit_concept_id"].notnull()]) / len(measurement)
unit_map_rate =  len(df[df["unit_concept_id"].notnull()]) / len(df)

map_rate = {
    "concept_synonym 적용 전 매핑된 코드 수": [len(measurement[measurement["unit_concept_id"].notnull()])],
    "concept_synonym 적용 후 매핑된 코드 수": [len(df[df["unit_concept_id"].notnull()])],
    "전체 코드 수": [len(df)],
    "concept_synonym 적용 전 매핑률": [measurement_map_rate],
    "concept_synonym 적용 후 매핑률": [unit_map_rate]
            }
pd.DataFrame(map_rate).to_csv("unit_map_rate.csv", index=False)
"""