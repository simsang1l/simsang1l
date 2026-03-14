import pandas as pd
from DataTransformer import DataTransformer 
import os

config = DataTransformer("config.yaml").load_config("config.yaml")

# 의약품 코드매핑 정보
drug = pd.read_csv(os.path.join(config["CDM_path"],"BarCodeData_20240401.txt"), delimiter='|', encoding='cp949', dtype=str)
atc = pd.read_csv(os.path.join(config["CDM_path"], "atcindex.csv"), dtype=str)
atc = atc[["ATC 코드","ATC 코드명"]]

edi_atc = pd.merge(drug, atc, left_on="ATC코드", right_on="ATC 코드", how = "left")
# concept_edi_atc["concept_name"] = concept_edi_atc["concept_name"].str.split(';')[0]

edi_atc.to_csv(os.path.join(config["CDM_path"], "edi_atc.csv"), index=False)