import pandas as pd
import yaml, os

def load_config(config_path):
    """
    YAML 설정 파일을 로드합니다.
    """
    with open(config_path, 'r', encoding="utf-8") as file:
        return yaml.safe_load(file)
        
def Add_UnitConceptId(yaml_path, hospital, unit_path):
    config = load_config(yaml_path)
    
    # measurement.csv 파일 불러오기
    measurement = pd.read_csv(os.path.join(config['CDM_path'], "measurement.csv"))

    unit_list = []
    start_concept_id = 10**9
    
    # unit_concept_id가 없는 항목 가져오기
    measurement = measurement[measurement["unit_concept_id"].isnull() & measurement["unit_source_value"].notnull()]
    measurement_unit = measurement["unit_source_value"].drop_duplicates()
    measurement_unit = measurement_unit.values

    for i in range(len(measurement_unit)):
        # concept_id, concept_name, domain_id, vocabulary_id, concept_class_id, standard_concept, concept_code, valid_start_date, valid_end_date, invalid_reason
        unit_list.append([start_concept_id + i, measurement_unit[i], "Unit", hospital, None, None, measurement_unit[i], "1970-01-01", "2099-12-31", None])

    df_unit = pd.DataFrame(unit_list)
    df_unit.to_csv(unit_path, mode='a', encoding="utf-8", header=False, index = False)
    print(f"추가된 unit {len(measurement_unit)}개 \n 목록: {measurement_unit}")

if __name__ == "__main__":
    hospital = ["CNUH", "DSMC", "JBUH", "KNUH"]
    unit_path = "../concept_unit.csv"
    Add_UnitConceptId("config.yaml", hospital[2], unit_path)