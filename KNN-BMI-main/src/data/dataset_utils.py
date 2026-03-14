# src/data/dataset_utils.py

import os
import pandas as pd
import numpy as np
import logging
from datetime import timedelta
import yaml
from typing import Tuple, Optional, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from src.utils.utils import load_config

# =======================================================
# 데이터 전처리 유틸
# =======================================================


def convert_to_float64(data):
    """BMI_REFERENCE 내부 데이터를 np.float64로 변환"""
    if isinstance(data, dict):
        return {key: convert_to_float64(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [np.float64(x) for x in data]  # 리스트 값 변환 (percentiles)
    else:
        return np.float64(data)  # 개별 값 변환


def bmi_zscore(weight, length, ga, gender):
    __doc__ = """
        BMI Curves for Preterm Infants 논문을 바탕으로
        LMS방법을 활용하여 BMI-Zscore를 계산
        BMI = (g/cm^2)*10
    """
    # 성별 : {재태 연령 : {L, M, S, Percentiles[3rd, 10th, 25th, 50th, 75th, 90th, 97th]}}
    BMI_REFERENCE = {
        # male
        1: {
            24: {"L": 0.162649, "M": 6.925844, "S": 0.104742, "percentiles": [5.60, 6.01, 6.46, 6.93, 7.42, 7.95, 8.51]},
            25: {"L": 0.163671, "M": 7.20182, "S": 0.106296, "percentiles": [5.80, 6.24, 6.71, 7.20, 7.73, 8.29, 8.88]},
            26: {"L": 0.164706, "M": 7.510527, "S": 0.107421, "percentiles": [6.03, 6.50, 6.99, 7.51, 8.06, 8.65, 9.28]},
            27: {"L": 0.165763, "M": 7.827689, "S": 0.109134, "percentiles": [6.27, 6.76, 7.28, 7.83, 8.41, 9.04, 9.70]},
            28: {"L": 0.166843, "M": 8.131427, "S": 0.109847, "percentiles": [6.50, 7.01, 7.55, 8.13, 8.75, 9.40, 10.09]},
            29: {"L": 0.167922, "M": 8.489882, "S": 0.111159, "percentiles": [6.77, 7.31, 7.88, 8.49, 9.14, 9.83, 10.56]},
            30: {"L": 0.168978, "M": 8.904723, "S": 0.113935, "percentiles": [7.06, 7.63, 8.25, 8.90, 9.60, 10.35, 11.14]},
            31: {"L": 0.169983, "M": 9.375006, "S": 0.112906, "percentiles": [7.45, 8.05, 8.69, 9.38, 10.10, 10.88, 11.70]},
            32: {"L": 0.170912, "M": 9.857181, "S": 0.110895, "percentiles": [7.86, 8.49, 9.15, 9.86, 10.61, 11.41, 12.25]},
            33: {"L": 0.171754, "M": 10.37792, "S": 0.110858, "percentiles": [8.28, 8.93, 9.63, 10.38, 11.17, 12.01, 12.90]},
            34: {"L": 0.172493, "M": 10.90406, "S": 0.112346, "percentiles": [8.67, 9.37, 10.11, 10.90, 11.75, 12.64, 13.59]},
            35: {"L": 0.173106, "M": 11.44493, "S": 0.117447, "percentiles": [9.00, 9.76, 10.58, 11.44, 12.37, 13.36, 14.41]},
            36: {"L": 0.173621, "M": 11.99975, "S": 0.120805, "percentiles": [9.38, 10.19, 11.06, 12.00, 13.00, 14.07, 15.20]},
            37: {"L": 0.174061, "M": 12.51268, "S": 0.118914, "percentiles": [9.81, 10.65, 11.55, 12.51, 13.54, 14.63, 15.80]},
            38: {"L": 0.174445, "M": 13.03349, "S": 0.114496, "percentiles": [10.32, 11.16, 12.07, 13.03, 14.06, 15.15, 16.31]},
            39: {"L": 0.174802, "M": 13.28914, "S": 0.108389, "percentiles": [10.65, 11.48, 12.36, 13.29, 14.28, 15.33, 16.44]},
            40: {"L": 0.175141, "M": 13.39009, "S": 0.104435, "percentiles": [10.82, 11.63, 12.48, 13.39, 14.35, 15.37, 16.44]},
            41: {"L": 0.175463, "M": 13.47175, "S": 0.103808, "percentiles": [10.90, 11.71, 12.57, 13.47, 14.43, 15.45, 16.52]}
        },
        # female
        2: {
            24: {"L": 0.20436, "M": 6.734847, "S": 0.109993, "percentiles": [5.38, 5.80, 6.26, 6.73, 7.24, 7.78, 8.35]},
            25: {"L": 0.240113, "M": 7.005395, "S": 0.111203, "percentiles": [5.57, 6.02, 6.50, 7.01, 7.54, 8.10, 8.70]},
            26: {"L": 0.244235, "M": 7.276484, "S": 0.111862, "percentiles": [5.78, 6.25, 6.75, 7.28, 7.83, 8.42, 9.05]},
            27: {"L": 0.207293, "M": 7.559349, "S": 0.112801, "percentiles": [6.00, 6.49, 7.01, 7.56, 8.15, 8.77, 9.42]},
            28: {"L": 0.152314, "M": 7.887805, "S": 0.114883, "percentiles": [6.24, 6.76, 7.30, 7.89, 8.51, 9.18, 9.89]},
            29: {"L": 0.157302, "M": 8.263847, "S": 0.113994, "percentiles": [6.55, 7.09, 7.66, 8.26, 8.91, 9.60, 10.34]},
            30: {"L": 0.196138, "M": 8.690219, "S": 0.112902, "percentiles": [6.90, 7.46, 8.06, 8.69, 9.36, 10.08, 10.84]},
            31: {"L": 0.220911, "M": 9.159153, "S": 0.11425, "percentiles": [7.24, 7.84, 8.48, 9.16, 9.88, 10.64, 11.45]},
            32: {"L": 0.234645, "M": 9.651788, "S": 0.115124, "percentiles": [7.62, 8.25, 8.93, 9.65, 10.41, 11.22, 12.08]},
            33: {"L": 0.22088, "M": 10.18235, "S": 0.1142, "percentiles": [8.05, 8.72, 9.43, 10.18, 10.98, 11.83, 12.72]},
            34: {"L": 0.156873, "M": 10.73633, "S": 0.114924, "percentiles": [8.50, 9.19, 9.94, 10.74, 11.59, 12.49, 13.46]},
            35: {"L": 0.13732, "M": 11.28477, "S": 0.121735, "percentiles": [8.81, 9.58, 10.40, 11.28, 12.23, 13.25, 14.34]},
            36: {"L": 0.214643, "M": 11.84423, "S": 0.128108, "percentiles": [9.10, 9.95, 10.87, 11.84, 12.89, 14.01, 15.20]},
            37: {"L": 0.247743, "M": 12.39839, "S": 0.126778, "percentiles": [9.54, 10.43, 11.38, 12.40, 13.48, 14.63, 15.86]},
            38: {"L": 0.196413, "M": 12.86351, "S": 0.117996, "percentiles": [10.10, 10.96, 11.88, 12.86, 13.91, 15.02, 16.20]},
            39: {"L": 0.141546, "M": 13.18563, "S": 0.110283, "percentiles": [10.54, 11.36, 12.25, 13.19, 14.19, 15.25, 16.38]},
            40: {"L": 0.211803, "M": 13.36399, "S": 0.10492, "percentiles": [10.78, 11.59, 12.45, 13.36, 14.32, 15.34, 16.41]},
            41: {"L": 0.388779, "M": 13.44849, "S": 0.104309, "percentiles": [10.82, 11.66, 12.53, 13.45, 14.40, 15.40, 16.44]}
        }
    }

    if gender not in BMI_REFERENCE:
        return None, None  # 성별이 잘못된 경우 NaN 반환

    # 📌 모든 데이터를 np.float64로 변환
    BMI_REFERENCE = convert_to_float64(BMI_REFERENCE)

    lms_data = BMI_REFERENCE[gender].get(ga, None)
    if lms_data is None:
        return None, None  # GA가 없는 경우 NaN 반환

    lms_data = BMI_REFERENCE[gender][ga]
    l, m, s = lms_data["L"], lms_data["M"], lms_data["S"]
    percentiles = lms_data["percentiles"]

    weight = np.float64(weight)
    length = np.float64(length)

    # BMI 계산 ((g/cm^2) * 10)
    bmi = (weight / (length ** 2)) * 10

    # Z-score계산
    if l == 0:
        z_score = (bmi / m - 1) / s
    else:
        z_score = ((bmi / m) ** l - 1) / (l * s)

    # Percentile 비교
    percentile_labels = ["3rd", "10th", "25th", "50th", "75th", "90th", "97th"]
    percentile_category = "Below 3rd"

    for i, p in enumerate(percentiles):
        if bmi < p:
            percentile_category = percentile_labels[i]
            break
    else:
        percentile_category = "Above 97th"

    return bmi, z_score  # , percentile_category


male_table = pd.read_excel("./ref/tab_bmi_boys_p_0_2.xlsx")
female_table = pd.read_excel("./ref/tab_bmi_girls_p_0_2.xlsx")


def WHO_bmi_zscore(weight, length, age_months, gender):
    __doc__ = """
        WHO의 BMI 기준을 바탕으로 LMS방법을 활용하여 BMI-Zscore를 계산 (https://www.who.int/toolkits/child-growth-standards/standards/body-mass-index-for-age-bmi-for-age)
        BMI = kg/m²
        :param weight: 체중(kg)
        :param length: 신장(cm)
        :param age_months: 개월 수
        :param gender: 1 (남성), 2 (여성)
        :return: BMI z-score
    """
    # 1) 입력값 검증
    if (weight is None or length is None or age_months is None or gender is None):
        return None

    # 체중, 신장, 개월 수가 음수이거나 0이면 계산 불가
    if weight <= 0 or length <= 0 or age_months < 0:
        return None

    # 성별이 1, 2가 아니면 None 반환
    if gender not in [1, 2]:
        return None

    height_m = length / 100
    weight_kg = weight * 0.001
    bmi = weight_kg / (height_m ** 2)

    # 성별에 따라 테이블 선택
    if gender == 1:  # 'male':
        lms_table = male_table
    elif gender == 2:  # 'female':
        lms_table = female_table
    # else:
    #     raise ValueError("성별은 'male' 또는 'female'이어야 합니다.")

    # 정확한 month 값으로만 LMS 값 선택
    row = lms_table[lms_table['Month'] == age_months]

    # if row.empty:
    #     raise ValueError(f"해당 month({age_months})에 대한 LMS 값이 없습니다.")

    # LMS 값 추출
    L = row.iloc[0]['L'].item()
    M = row.iloc[0]['M'].item()
    S = row.iloc[0]['S'].item()

    # LMS 방법으로 z-score 계산
    if L != 0:
        z_score = ((bmi / M) ** L - 1) / (L * S)
    else:
        # L이 0인 경우 로그 변환 사용
        z_score = np.log(bmi / M) / S

    return z_score


bmi_male_table_days = pd.read_excel(
    "./ref/bfa-boys-zscore-expanded-tables.xlsx")
bmi_female_table_days = pd.read_excel(
    "./ref/bfa-girls-zscore-expanded-tables.xlsx")


def WHO_bmi_zscore_for_days(weight, length, age_days, gender):
    """
        WHO의 BMI 기준을 바탕으로 LMS방법을 활용하여 BMI-Zscore를 계산 (https://www.who.int/toolkits/child-growth-standards/standards/body-mass-index-for-age-bmi-for-age)
        BMI = kg/m²
        :param weight: 체중(kg)
        :param length: 신장(cm)
        :param age_days: 일 수
        :param gender: 1 (남성), 2 (여성)
        :return: BMI z-score
    """
    # 1) 입력값 검증
    if (weight is None or length is None or age_days is None or gender is None):
        return None

    # 체중, 신장, 개월 수가 음수이거나 0이면 계산 불가
    if weight <= 0 or length <= 0 or age_days < 0 or np.isnan(age_days):
        return None

    # 성별이 1, 2가 아니면 None 반환
    if gender not in [1, 2]:
        return None

    age_days = int(age_days)

    height_m = length / 100
    weight_kg = weight  # * 0.001
    bmi = weight_kg / (height_m ** 2)

    # 성별에 따라 테이블 선택
    if gender == 1:  # 'male':
        lms_table = bmi_male_table_days
    elif gender == 2:  # 'female':
        lms_table = bmi_female_table_days
    else:
        return None

    # 정확한 day 값으로만 LMS 값 선택
    row = lms_table[lms_table['Day'] == age_days]

    # 해당 age_days에 대한 데이터가 없는 경우 None 반환
    if row.empty:
        return None

    try:
        # LMS 값 추출
        L = row.iloc[0]['L'].item()
        M = row.iloc[0]['M'].item()
        S = row.iloc[0]['S'].item()

        # LMS 방법으로 z-score 계산
        if L != 0:
            z_score = ((bmi / M) ** L - 1) / (L * S)
        else:
            # L이 0인 경우 로그 변환 사용
            z_score = np.log(bmi / M) / S

        return z_score
    except (IndexError, KeyError, ValueError) as e:
        # 데이터 추출 중 오류 발생 시 None 반환
        return None


weight_male_table_days = pd.read_excel(
    "./ref/wfa-boys-zscore-expanded-tables.xlsx")
weight_female_table_days = pd.read_excel(
    "./ref/wfa-girls-zscore-expanded-tables.xlsx")


def WHO_weight_zscore_for_days(weight, age_days, gender):
    """
        WHO의 체중 기준을 바탕으로 LMS방법을 활용하여 Weight-Zscore를 계산 (https://www.who.int/tools/child-growth-standards/standards/weight-for-age)
        :param weight: 체중(kg)
        :param age_days: 일 수
        :param gender: 1 (남성), 2 (여성)
        :return: Weight z-score
    """
    # 1) 입력값 검증
    if (weight is None or age_days is None or gender is None):
        return None

    # 체중, 일 수가 음수이거나 0이면 계산 불가
    if weight <= 0 or age_days < 0 or np.isnan(age_days):
        return None

    # 성별이 1, 2가 아니면 None 반환
    if gender not in [1, 2]:
        return None

    age_days = int(age_days)

    weight_kg = weight  # * 0.001

    # 성별에 따라 테이블 선택
    if gender == 1:  # 'male':
        lms_table = weight_male_table_days
    elif gender == 2:  # 'female':
        lms_table = weight_female_table_days
    else:
        return None

    # 정확한 day 값으로만 LMS 값 선택
    row = lms_table[lms_table['Day'] == age_days]

    # 해당 age_days에 대한 데이터가 없는 경우 None 반환
    if row.empty:
        return None

    try:
        # LMS 값 추출
        L = row.iloc[0]['L'].item()
        M = row.iloc[0]['M'].item()
        S = row.iloc[0]['S'].item()

        # LMS 방법으로 z-score 계산
        if L != 0:
            z_score = ((weight_kg / M) ** L - 1) / (L * S)
        else:
            # L이 0인 경우 로그 변환 사용
            z_score = np.log(weight_kg / M) / S

        return z_score
    except (IndexError, KeyError, ValueError) as e:
        # 데이터 추출 중 오류 발생 시 None 반환
        return None


height_male_table_days = pd.read_excel(
    "./ref/lhfa-boys-zscore-expanded-tables.xlsx")
height_female_table_days = pd.read_excel(
    "./ref/lhfa-girls-zscore-expanded-tables.xlsx")


def WHO_height_zscore_for_days(height, age_days, gender):
    """
        WHO의 신장 기준을 바탕으로 LMS방법을 활용하여 Height-Zscore를 계산 (https://www.who.int/tools/child-growth-standards/standards/length-height-for-age)
        :param height: 신장 (cm)
        :param age_days: 일 수
        :param gender: 1 (남성), 2 (여성)
        :return: Height z-score
    """
    # 1) 입력값 검증
    if (height is None or age_days is None or gender is None):
        return None

    # 체중, 신장, 개월 수가 음수이거나 0이면 계산 불가
    if height <= 0 or age_days < 0 or np.isnan(age_days):
        return None

    # 성별이 1, 2가 아니면 None 반환
    if gender not in [1, 2]:
        return None

    age_days = int(age_days)

    # 성별에 따라 테이블 선택
    if gender == 1:  # 'male':
        lms_table = height_male_table_days
    elif gender == 2:  # 'female':
        lms_table = height_female_table_days
    else:
        return None

    # 정확한 day 값으로만 LMS 값 선택
    row = lms_table[lms_table['Day'] == age_days]

    # 해당 age_days에 대한 데이터가 없는 경우 None 반환
    if row.empty:
        return None

    try:
        # LMS 값 추출
        L = row.iloc[0]['L'].item()
        M = row.iloc[0]['M'].item()
        S = row.iloc[0]['S'].item()

        # LMS 방법으로 z-score 계산
        if L != 0:
            z_score = ((height / M) ** L - 1) / (L * S)
        else:
            # L이 0인 경우 로그 변환 사용
            z_score = np.log(height / M) / S

        return z_score
    except (IndexError, KeyError, ValueError) as e:
        # 데이터 추출 중 오류 발생 시 None 반환
        return None


def preprocess(raw_data: pd.DataFrame, config_path: str) -> pd.DataFrame:
    """
    원시 데이터 전처리
    새로운 컬럼 추가, label컬럼 추가, 원본 값 변경

    Args:
        raw_data: 원시 데이터
        config_path: 설정 파일 경로

    Returns:
        전처리된 데이터
    """
    config = load_config(config_path)

    print('초기 features shape :', raw_data.shape)

    # 입원시 BMI 변수 추가
    raw_data[["birth_bmi", "birth_bmi_zscore"]] = raw_data.apply(
        lambda row: pd.Series(
            bmi_zscore(row["bwei"], row["bhei"],
                       row["gagew"], row["sex_sys_val"])
        ),
        axis=1
    )
    raw_data["birth_bmi"] = (raw_data["bwei"] / 1000) / \
        (raw_data["bhei"] / 100) ** 2

    # 퇴원 BMI 관련 변수 추가
    # missing value 2613
    # features[["dcd_bmi", "dcd_bmi_zscore"]] = features.apply(
    #     lambda row: pd.Series(
    #         bmi_zscore(row["dcdwt"], row["dcdht"], row["gagew"], row["sex_sys_val"])
    #     ),
    #     axis=1
    # )

    columns = ["birthdt", "admd", "fdcdt", "ntetdt", "ntetdty", "iperrdt", "phudstdt", "date36",
               "pdaddt", "acldt", "pmiodt", "avegftrdt", "sftfudt", "efydt", "dt1", "dt2"]
    for i in columns:
        raw_data[i] = pd.to_datetime(raw_data[i])

    raw_data["dcd_days"] = (raw_data["admd"] - raw_data["birthdt"] +
                            pd.to_timedelta(raw_data["stday"], unit='D')).dt.days  # 출생부터 퇴원시까지 일수
    raw_data["corrected_age"] = (
        raw_data["gagew"] * 7 + raw_data["gaged"] + raw_data["dcd_days"])  # 교정나이
    raw_data["corrected_agew"] = (
        raw_data["gagew"] * 7 + raw_data["gaged"] + raw_data["dcd_days"]) // 7  # 교정주수
    raw_data["corrected_aged"] = (
        raw_data["gagew"] * 7 + raw_data["gaged"] + raw_data["dcd_days"]) % 7  # 교정주수의 일
    # features["dcd_aged"] = (features["fdcdt"]- features["birthdt"]).dt.days # 퇴원시 나이 (일수)
    # features["dcd_agem"] = features["dcd_aged"] // 30.4375 # 퇴원시 나이(개월)
    raw_data["age_months"] = np.where(
        raw_data["corrected_age"] < config["experiment"]["age_cutoff"] * 7, 0, (raw_data["corrected_age"] - config["experiment"]["age_cutoff"] * 7) // 30.4375)
    # features["age_months"] = np.where(features["corrected_age"] < 40 * 7, 0, (features["corrected_age"] - 40 * 7 ) // 30.4375)
    raw_data["stday36"] = (raw_data["date36"] - raw_data["admd"]).dt.days
    raw_data["stday36"] = raw_data["stday36"].fillna(0)  # 교정 36주 전 퇴원한 경우?
    raw_data["stday28"] = (
        raw_data["birthdt"] + pd.to_timedelta(28, unit='D') - raw_data["admd"]).dt.days
    raw_data["stday7"] = (
        raw_data["birthdt"] + pd.to_timedelta(7, unit='D') - raw_data["admd"]).dt.days
    raw_data.loc[raw_data["stday7"] < 0, "stday7"] = 0  # 출생 후 입원한 경우 0으로 처리
    raw_data.loc[raw_data["stday28"] < 0, "stday28"] = 0  # 출생 후 입원한 경우 0으로 처리

    # 출생 28일 시점 침습적, 비침습적 인공호흡기 평균 사용 기간
    raw_data["iarvppd7"] = round((raw_data["iarvppd"] /
                                  raw_data["stday"]) * raw_data["stday7"], 1)
    raw_data["niarvrpd7"] = round((raw_data["niarvrpd"] /
                                   raw_data["stday"]) * raw_data["stday7"], 1)
    raw_data["invfpod7"] = round((raw_data["invfpod"] /
                                  raw_data["stday"]) * raw_data["stday7"], 1)
    raw_data["aoxyuppd7"] = round((raw_data["aoxyuppd"] /
                                   raw_data["stday"]) * raw_data["stday7"], 1)
    raw_data["niarvhfnc7"] = round((raw_data["niarvhfnc"] /
                                    raw_data["stday"]) * raw_data["stday7"], 1)
    raw_data["iarvppd7"] = raw_data["iarvppd7"].fillna(0)
    raw_data["niarvrpd7"] = raw_data["niarvrpd7"].fillna(0)
    raw_data["invfpod7"] = raw_data["invfpod7"].fillna(0)
    raw_data["aoxyuppd7"] = raw_data["aoxyuppd7"].fillna(0)
    raw_data["niarvhfnc7"] = raw_data["niarvhfnc7"].fillna(0)

    # 출생 28일 시점 침습적, 비침습적 인공호흡기 평균 사용 기간
    raw_data["iarvppd28"] = round(
        (raw_data["iarvppd"] / raw_data["stday"]) * raw_data["stday28"], 1)
    raw_data["niarvrpd28"] = round(
        (raw_data["niarvrpd"] / raw_data["stday"]) * raw_data["stday28"], 1)
    raw_data["invfpod28"] = round(
        (raw_data["invfpod"] / raw_data["stday"]) * raw_data["stday28"], 1)
    raw_data["niarvhfnc28"] = round(
        (raw_data["niarvhfnc"] / raw_data["stday"]) * raw_data["stday28"], 1)
    raw_data["aoxyuppd28"] = round(
        (raw_data["aoxyuppd"] / raw_data["stday"]) * raw_data["stday28"], 1)
    raw_data["iarvppd28"] = raw_data["iarvppd28"].fillna(0)
    raw_data["niarvrpd28"] = raw_data["niarvrpd28"].fillna(0)
    raw_data["invfpod28"] = raw_data["invfpod28"].fillna(0)
    raw_data["aoxyuppd28"] = raw_data["aoxyuppd28"].fillna(0)
    raw_data["niarvhfnc28"] = raw_data["niarvhfnc28"].fillna(0)

    # 교정 36주 시점 침습적, 비침습적 인공호흡기 평균 사용 기간
    raw_data["iarvppd36"] = round((raw_data["iarvppd"] /
                                   raw_data["stday"]) * raw_data["stday36"], 1)
    raw_data["niarvrpd36"] = round((
        raw_data["niarvrpd"] / raw_data["stday"]) * raw_data["stday36"], 1)
    raw_data["invfpod36"] = round((raw_data["invfpod"] /
                                   raw_data["stday"]) * raw_data["stday36"], 1)
    raw_data["aoxyuppd36"] = round((raw_data["aoxyuppd"] /
                                    raw_data["stday"]) * raw_data["stday36"], 1)
    raw_data["niarvhfnc36"] = round((raw_data["niarvhfnc"] /
                                     raw_data["stday"]) * raw_data["stday36"], 1)
    raw_data["iarvppd36"] = raw_data["iarvppd36"].fillna(0)
    raw_data["niarvrpd36"] = raw_data["niarvrpd36"].fillna(0)
    raw_data["invfpod36"] = raw_data["invfpod36"].fillna(0)
    raw_data["aoxyuppd36"] = raw_data["aoxyuppd36"].fillna(0)
    raw_data["niarvhfnc36"] = raw_data["niarvhfnc36"].fillna(0)

    #########################
    ### FollowUp BMI 계산  ###
    #########################

    # F/U 나이 계산
    # FU1은 교정 48개월
    # FU2는 만 36개월
    cut_age = 43 * 7
    raw_data["days1"] = (raw_data["dt1"] - raw_data["fdcdt"]).dt.days
    raw_data["birth_age1"] = (raw_data["dt1"] - raw_data["birthdt"]).dt.days
    # + raw_data["age_months"]
    raw_data["corrected_agem1"] = raw_data["days1"] // 30.4375
    raw_data["birth_agem1"] = (
        raw_data["dt1"] - raw_data["birthdt"]).dt.days // 30.4375
    raw_data["days2"] = (raw_data["dt2"] - raw_data["fdcdt"]).dt.days
    raw_data["birth_age2"] = (raw_data["dt2"] - raw_data["birthdt"]).dt.days
    # + raw_data["age_months"]
    raw_data["corrected_agem2"] = raw_data["days2"] // 30.4375
    raw_data["birth_agem2"] = (
        raw_data["dt2"] - raw_data["birthdt"]).dt.days // 30.4375

    raw_data["dcd_bmi"] = (raw_data["dcdwt"] / 1000) / \
        (raw_data["dcdht"] / 100) ** 2
    raw_data["bmi1"] = raw_data["wt1"] / (raw_data["ht1"] / 100) ** 2
    raw_data["bmi2"] = raw_data["wt2"] / (raw_data["ht2"] / 100) ** 2

    raw_data["wt1_zscore"] = raw_data.apply(lambda row: WHO_weight_zscore_for_days(
        row["wt1"], row["days1"], row["sex_sys_val"]), axis=1)
    raw_data["wt2_zscore"] = raw_data.apply(lambda row: WHO_weight_zscore_for_days(
        row["wt2"], row["birth_age2"], row["sex_sys_val"]), axis=1)
    raw_data["ht1_zscore"] = raw_data.apply(lambda row: WHO_height_zscore_for_days(
        row["ht1"], row["days1"], row["sex_sys_val"]), axis=1)
    raw_data["ht2_zscore"] = raw_data.apply(lambda row: WHO_height_zscore_for_days(
        row["ht2"], row["birth_age2"], row["sex_sys_val"]), axis=1)
    raw_data["bmi1_zscore"] = raw_data.apply(lambda row: WHO_bmi_zscore_for_days(
        row["wt1"], row["ht1"], row["days1"], row["sex_sys_val"]), axis=1)
    raw_data["bmi2_zscore"] = raw_data.apply(lambda row: WHO_bmi_zscore_for_days(
        row["wt2"], row["ht2"], row["birth_age2"], row["sex_sys_val"]), axis=1)

    raw_data["dcd_bmi_zscore"] = raw_data.apply(
        lambda row: WHO_bmi_zscore(
            row["dcdwt"], row["dcdht"], row["age_months"], row["sex_sys_val"]),
        axis=1
    )

    # features.groupby(["gagew", 'sex_sys_val'])[['dcd_bmi', 'dcd_bmi_zscore']].describe().reset_index().sort_values(['sex_sys_val','gagew']).to_csv(os.path.join(result_path, 'bmi_by_ga.csv'))
    mapping = {
        "심화평가": 1,
        "심화평가권고": 1,
        "추적검사요망": 2,
        "또래수준": 3,
        "빠른수준": 4,
    }

    # 매핑 적용하여 새로운 컬럼에 저장
    raw_data["dgmtr1"] = raw_data["dgmbp1"].map(mapping)
    raw_data["dgmtr2"] = raw_data["dgmbp2"].map(mapping)
    raw_data["dfmtr1"] = raw_data["dfmbp1"].map(mapping)
    raw_data["dfmtr2"] = raw_data["dfmbp2"].map(mapping)
    raw_data["rctr1"] = raw_data["rcbp1"].map(mapping)
    raw_data["rctr2"] = raw_data["rcbp2"].map(mapping)
    raw_data["lgtr1"] = raw_data["lgbp1"].map(mapping)
    raw_data["lgtr2"] = raw_data["lgbp2"].map(mapping)
    raw_data["sctr1"] = raw_data["scbp1"].map(mapping)
    raw_data["sctr2"] = raw_data["scbp2"].map(mapping)
    raw_data["shtr1"] = raw_data["shbp1"].map(mapping)
    raw_data["shtr2"] = raw_data["shbp2"].map(mapping)

    ##########################
    ###     모름 값 처리    ###
    ##########################
    raw_data["chor"] = raw_data["chor"].replace(3, 1)
    raw_data["chor"] = raw_data["chor"].replace(np.nan, 1)

    raw_data["prom"] = raw_data["prom"].replace(3, 1)
    raw_data["prom"] = raw_data["prom"].replace(np.nan, 1)

    raw_data["ster"] = raw_data["ster"].replace(3, 1)
    raw_data["ster"] = raw_data["ster"].replace(np.nan, 1)

    raw_data["atbyn"] = raw_data["atbyn"].replace(3, np.nan)  # 3=모름
    raw_data["atbyn"] = raw_data["atbyn"].replace(np.nan, 1)  # 3=모름

    raw_data["bpia"] = raw_data["bpia"].replace(3, 2)
    raw_data["bpia"] = raw_data["bpia"].replace(np.nan, 2)

    raw_data["phh"] = raw_data["phh"].fillna(1)
    raw_data["phh"] = raw_data["phh"].replace(3, 1)

    # 초기 소생술 빈값을 없다고 봐도 될려나?
    raw_data["resu"] = raw_data["resu"].replace(3, 1)
    raw_data["resuo"] = raw_data["resuo"].replace(np.nan, 1)  # 1=없음
    raw_data["resup"] = raw_data["resup"].replace(np.nan, 1)  # 1=없음
    raw_data["resui"] = raw_data["resui"].replace(np.nan, 1)  # 1=없음
    raw_data["resuh"] = raw_data["resuh"].replace(np.nan, 1)  # 1=없음
    raw_data["resue"] = raw_data["resue"].replace(np.nan, 1)  # 1=없음
    raw_data["resuc"] = raw_data["resuc"].replace(np.nan, 1)  # 1=없음

    # 저혈압 약재
    # lbpd1: Inotropics, lbpd2: Hydrocortisone, lbpd3: Others
    raw_data['lbpd1'] = raw_data['lbpd1'].replace(np.nan, 0)  # 0=없음
    raw_data['lbpd1'] = raw_data['lbpd1'].replace(1, 1)  # 1=사용

    raw_data['lbpd2'] = raw_data['lbpd2'].replace(np.nan, 0)  # 0=없음
    raw_data['lbpd2'] = raw_data['lbpd2'].replace(2, 1)  # 2->1=사용

    raw_data['lbpd3'] = raw_data['lbpd3'].replace(np.nan, 0)  # 0=없음
    raw_data['lbpd3'] = raw_data['lbpd3'].replace(3, 1)  # 3->1=사용

    # Hypotension_Tx: 저혈압 치료 유무 (lbpd1, lbpd2, lbpd3 중 하나라도 사용한 경우)
    raw_data['Hypotension_Tx'] = 0
    raw_data.loc[(raw_data['lbpd1'] == 1) | (raw_data['lbpd2'] == 1)
                 | (raw_data['lbpd3'] == 1), 'Hypotension_Tx'] = 1

    raw_data['phud1'] = raw_data['phud1'].replace(np.nan, 0)  # 0=없음
    raw_data.loc[raw_data['phud'].str.contains('1', na=False), 'phud1'] = 1
    raw_data['phud2'] = raw_data['phud2'].replace(np.nan, 0)  # 0=없음
    raw_data.loc[raw_data['phud'].str.contains('2', na=False), 'phud2'] = 1
    # raw_data.loc[raw_data['phudstdt'].notna(), 'phud2'] = 1
    raw_data['phud2'] = raw_data['phud2'].replace(2, 1)
    raw_data['phud3'] = raw_data['phud3'].replace(np.nan, 0)  # 0=없음
    raw_data.loc[raw_data['phud'].str.contains('3', na=False), 'phud3'] = 1
    # raw_data.loc[raw_data['phudstdt'].notna(), 'phud3'] = 1
    raw_data['phud3'] = raw_data['phud3'].replace(3, 1)
    raw_data['phud4'] = raw_data['phud4'].replace(np.nan, 0)  # 0=없음
    raw_data.loc[raw_data['phud'].str.contains('4', na=False), 'phud4'] = 1
    # raw_data.loc[raw_data['phudstdt'].notna(), 'phud4'] = 1
    raw_data['phud4'] = raw_data['phud4'].replace(4, 1)
    raw_data['phud5'] = raw_data['phud5'].replace(np.nan, 0)  # 0=없음
    raw_data.loc[raw_data['phud'].str.contains('5', na=False), 'phud5'] = 1
    # raw_data.loc[raw_data['phudstdt'].notna(), 'phud5'] = 1
    raw_data['phud5'] = raw_data['phud5'].replace(5, 1)
    raw_data['phud6'] = raw_data['phud6'].replace(np.nan, 0)  # 0=없음
    raw_data.loc[raw_data['phud'].str.contains('6', na=False), 'phud6'] = 1
    # raw_data.loc[raw_data['phudstdt'].notna(), 'phud6'] = 1
    raw_data['phud6'] = raw_data['phud6'].replace(6, 1)

    raw_data["phud_yn"] = 0
    raw_data.loc[(raw_data["phud2"] == 1) | (raw_data["phud3"] == 1) | (raw_data["phud4"] == 1) | (
        raw_data["phud5"] == 1) | (raw_data["phud6"] == 1) | (raw_data["phudstdt"].notna()), 'phud_yn'] = 1

    raw_data['inhg'] = raw_data['inhg'].replace(6, 1)  # 6=뇌 영상검사 미시행
    raw_data['phh'] = raw_data['phh'].replace(np.nan, 1)  # 1=없음
    raw_data['pvl'] = raw_data['pvl'].replace(np.nan, 1)  # 1=없음
    raw_data['pvl'] = raw_data['pvl'].replace(3, 1)  # 1=없음
    raw_data['seps'] = raw_data['seps'].replace(np.nan, 1)  # 0=없음
    raw_data['seps'] = raw_data['seps'].replace(3, 1)  # 0=없음

    # 산전스테로이드 약제 관련 변수
    # Dexamethasone
    raw_data.loc[raw_data['sterd'].str.contains('1', na=False), 'sterd1'] = 1
    raw_data['sterd1'] = raw_data['sterd1'].replace(np.nan, 0)  # 0=사용안함
    # Betamethasone
    raw_data.loc[raw_data['sterd'].str.contains('2', na=False), 'sterd2'] = 2
    raw_data['sterd2'] = raw_data['sterd2'].replace(2, 1)  # 1=사용
    raw_data['sterd2'] = raw_data['sterd2'].replace(np.nan, 0)  # 0=사용안함

    # ## steroid 사용 방법
    # # 전신
    # features.loc[features['strdut'].str.contains('1', na=False), 'strdut1'] = 1
    # features['strdut1'] = features['strdut1'].replace(np.nan, 0)
    # # 흡입
    # features.loc[features['strdut'].str.contains('2', na=False), 'strdut2'] = 2
    # features['strdut2'] = features['strdut2'].replace(2, 1) # 1=사용
    # features['strdut2'] = features['strdut2'].replace(np.nan, 0)

    # ## 전신 steroid약제 종류
    # # Dexamethasone
    # features.loc[features['strdud'].str.contains('1', na=False), 'strdud1'] = 1
    # features['strdud1'] = features['strdud1'].replace(np.nan, 0) # 0=사용안함
    # # Hydrocortison
    # features.loc[features['strdud'].str.contains('2', na=False), 'strdud2'] = 2
    # features['strdud2'] = features['strdud2'].replace(2, 1)
    # features['strdud2'] = features['strdud2'].replace(np.nan, 0) # 0=사용안함
    # # Prednisolone
    # features.loc[features['strdud'].str.contains('3', na=False), 'strdud3'] = 3
    # features['strdud3'] = features['strdud3'].replace(3, 1)
    # features['strdud3'] = features['strdud3'].replace(np.nan, 0) # 0=사용안함
    # # etc
    # features.loc[features['strdud'].str.contains('4', na=False), 'strdud4'] = 4
    # features['strdud4'] = features['strdud4'].replace(4, 1)
    # features['strdud4'] = features['strdud4'].replace(np.nan, 0) # 0=사용안함

    raw_data['efyn'] = raw_data['efyn'].replace(np.nan, 1)  # 1=아니오
    raw_data['efyn'] = raw_data['efyn'].replace(3, 1)  # 1=아니오
    raw_data['pdad'] = raw_data['pdad'].replace(np.nan, 1)  # 1=아니오
    raw_data['ntet'] = raw_data['ntet'].replace(3, 1)  # 3=모름|1=아니오

    raw_data.loc[raw_data["ntety"] >= 2, "ntety"] = 2
    raw_data['ntety'] = raw_data['ntety'].fillna(1)

    raw_data.loc[raw_data["iperr"] >= 2, "iperr"] = 2
    raw_data['iperr'] = raw_data['iperr'].fillna(1)

    raw_data['pmio'] = raw_data['pmio'].replace(np.nan, 1)  # 1=아니오
    raw_data['pmio'] = raw_data['pmio'].replace(3, np.nan)  # 1=아니오
    raw_data['avegftr'] = raw_data['avegftr'].replace(np.nan, 1)  # 1=아니오
    raw_data["eye_Tx"] = 0
    raw_data.loc[(raw_data["pmio"] == 1) | (
        raw_data["avegftr"] == 1), "eye_Tx"] = 1

    ##########################
    ##  출생 후 7일, 28일이내 내역  ##
    ##########################
    birth7_28_cols = {
        "ntet": "ntetdt",
        "ntety": "ntetdty",
        "iperr": "iperrdt",
        "phud": "phudstdt",
        "pdad": "pdaddt",
        "acl": "acldt",
        "pmio": "pmiodt",
        "avegftr": "avegftrdt",
        "sftfu": "sftfudt",
        "efy": "efydt",
    }
    for col, dt_col in birth7_28_cols.items():
        raw_data[f"{col}_diff_birth_days"] = (
            raw_data[dt_col] - raw_data["birthdt"]).dt.days

        raw_data[f"{col}_7"] = np.select(
            [raw_data[f"{col}_diff_birth_days"] <= 7], [1], default=0)
        raw_data[f"{col}_28"] = np.select(
            [(raw_data[f"{col}_diff_birth_days"] <= 28)], [1], default=0)

        raw_data[f"{col}_36"] = np.select(
            [(raw_data[dt_col] < raw_data["date36"])], [1], default=0)

    ##########################
    ##     변수 전처리       ##
    ##########################
    raw_data['nrp_grade'] = 1
    raw_data.loc[raw_data['resuo'] == 2, 'nrp_grade'] = 2
    raw_data.loc[raw_data['resuc'] == 2, 'nrp_grade'] = 2
    raw_data.loc[raw_data['resup'] == 2, 'nrp_grade'] = 3
    raw_data.loc[raw_data['resui'] == 2, 'nrp_grade'] = 4
    raw_data.loc[raw_data['resuh'] == 2, 'nrp_grade'] = 5
    raw_data.loc[raw_data['resue'] == 2, 'nrp_grade'] = 6
    raw_data.loc[raw_data['resu'] == 3, 'nrp_grade'] = 1  # 미시행

    raw_data["birthdt"] = pd.to_datetime(
        raw_data["birthdt"].astype(str), format='%Y-%m-%d')
    raw_data['prom_duration'] = (pd.to_datetime(raw_data['birthdt'].astype(str), format='%Y-%m-%d') -
                                 pd.to_datetime(raw_data['promd'].astype(str), format='%Y-%m-%d') + timedelta(days=1)).dt.days
    # PROM: 조기양막파수 여부 (1 없음 | 2 있음 | 3 모름)
    raw_data.loc[raw_data['prom'] == 1, 'prom_duration'] = 0
    raw_data.loc[raw_data['prom'] == 3,
                 'prom_duration'] = 0  # PROM 을 모르는 경우 없음
    # raw_data.loc[raw_data['prom_duration'] > 30,
    #              'prom_duration'] = 30  # PROM 30일 이상은 30으로 변경
    raw_data.loc[raw_data['prom_duration'] < 0,
                 'prom_duration'] = np.nan  # PROM 1미만은 제거

    # 3. 산전 스테로이드 투여
    # ANS: 산전 스테로이드 투여 유무 (1=없음|2=있음|3=모름)
    # 모름 데이터는 None으로 변경
    raw_data.loc[raw_data['ster'] == 3, 'ster'] = 1

    ############################
    ## 약물 관련 데이터 전처리 ##
    ############################

    # PH(폐동맥고혈압) 약물 수정
    # features['phud'] = 1 # 폐동맥 고혈압은 있지만 빈값인 경우, 사용 약물은 없는 경우일까??
    # features.loc[features['phud1'] == 1, 'phud'] = 2 # 1=일산화질소
    # features.loc[features['phud2'] == 1, 'phud'] = 3 # 2=Sildenafil
    # features.loc[features['phud3'] == 1, 'phud'] = 4 # 3=Iloprost
    # features.loc[features['phud4'] == 1, 'phud'] = 5 # 4=Bosentan
    # features.loc[features['phud5'] == 1, 'phud'] = 6 # 5=Milrinone
    # features.loc[features['phud6'] == 1, 'phud'] = 7 # 6=기타

    # BPD 약물 변수 생성 (전신 steroid약제?)
    # features['strdud_sub'] = 1
    # features.loc[features['strdud1'] == 1, 'strdud_sub'] = 2 # Dexamethasone
    # features.loc[features['strdud2'] == 2, 'strdud_sub'] = 3 # Hydrocortison
    # features.loc[features['strdud3'] == 3, 'strdud_sub'] = 4 # Prednisolone
    # features.loc[features['strdud4'] == 4, 'strdud_sub'] = 5 # Etc

    # BE 기호 수정 (출생 1시간 이내 혈액 가스의 base excess)
    # features.loc[features['sign'] == 1, 'sign'] = 1 # 1=+
    # features.loc[features['sign'] == 2, 'sign'] = -1 # 2=-

    # # 동맥관 개존(PDA) 약물 투여 경로 변수(paddr) 수정
    # features['paddr'] = 1
    # features.loc[features['paddr1'] == 1, 'paddr'] = 2 # 정맥
    # features.loc[features['paddr2'] == 2, 'paddr'] = 3 # 경구

    # # 세균성 패혈증 횟수(bsf) 수정
    # features['bsf'] = 0 # 치료없음
    # features.loc[features['bsf1'] == 1, 'bsf'] = 1 # 1회
    # features.loc[features['bsf2'] == 1, 'bsf'] = 2 # 2회
    # features.loc[features['bsf3'] == 1, 'bsf'] = 3 # 3회이상

    # ROP_Stage(pmirnsg) 수정
    # features.loc[features['pmirnsgnd'] == 1, 'pmirnsg'] = 6 # 병기시행안함 추가
    # print(preprocessing['pmirnsg'].value_counts())

    #############################
    # feature 추출..?
    # [1] 인적사항

    # 출생일 타입 변경
    # features["birthdt"] = pd.to_datetime(features["birthdt"].astype(str), format = '%Y-%m-%d')

    # 출생 연도 추출
    raw_data["birth_year"] = raw_data["birthdt"].dt.year

    # 성별
    raw_data["sex"] = raw_data["sex_sys_val"]

    # [2] 산모 및 부성 정보

    # 산모 나이(mage), 컬럼명 변경없이 사용
    # 산모의 임신력(gran), 컬럼명 변경없이 사용
    # 산모의 출산력(parn), 컬럼명 변경없이 사용

    # 임신중 양수량
    # raw_data["amni"] = raw_data["amni"].replace(4, np.nan)
    # AF: 임신중 양수량 (1=정상|2=과소증|3=과다증|4=모름)
    # AF_amount: 임신중 양수량(양을 기준으로 재배치) (1=과소증|2=정상|3=과다증|4=모름)
    raw_data.loc[raw_data['amni'] == 2, 'AF_amount'] = 1
    raw_data.loc[raw_data['amni'] == 1, 'AF_amount'] = 2
    raw_data.loc[raw_data['amni'] == 3, 'AF_amount'] = 3
    raw_data.loc[raw_data['amni'] == 4, 'AF_amount'] = 4
    # AF_oloigo: 임신중 양수과소증 (1=정상 또는 과다증|2=과소증|3=모름)
    raw_data.loc[raw_data['amni'] == 2, 'AF_oligo'] = 2
    raw_data.loc[raw_data['amni'] == 1, 'AF_oligo'] = 1
    raw_data.loc[raw_data['amni'] == 3, 'AF_oligo'] = 1
    raw_data.loc[raw_data['amni'] == 4, 'AF_oligo'] = 3
    # AF_poly: 임신중 양수과다증 (1=정상 또는 과소증|2=과다증|3=모름)
    raw_data.loc[raw_data['amni'] == 2, 'AF_poly'] = 1
    raw_data.loc[raw_data['amni'] == 1, 'AF_poly'] = 1
    raw_data.loc[raw_data['amni'] == 3, 'AF_poly'] = 2
    raw_data.loc[raw_data['amni'] == 4, 'AF_poly'] = 3

    # 산모 교육 정도(medu), 컬럼명 변경없이 사용, (1=초졸|2=중졸|3=고졸|4=대졸 이상|5=모름)
    # 산모 출신 국가(mcou), 컬럼명 변경없이 사용, (1=대한민국|2=베트남|3=중국|4=필리핀|5=일본|6=캄보디아|7=미국|8=태국|9=몽골|10=기타)
    # 부성 교육 정도(fedu), 컬럼명 변경없이 사용, (1=초졸|2=중졸|3=고졸|4=대졸 이상|5=모름)
    # 부성 출신 국가(fcou), 컬럼명 변경없이 사용, (1=대한민국|2=베트남|3=중국|4=필리핀|5=일본|6=캄보디아|7=미국|8=태국|9=몽골|10=기타)
    # 결혼 상태(merr), 컬럼명 변경없이 사용, (1=결혼|2=이혼|3=미혼|4=미혼 동거)

    # 현 임신 정보
    # 다태아 수(mulg), 컬럼명 변경없이 사용, (1=Singleton|2=Twin|3=Triplet|4=Quadruplet 이상)
    raw_data['mulg_yn'] = 1
    raw_data.loc[raw_data["mulg"] >= 2, 'mulg_yn'] = 2
    raw_data['gran_yn'] = 0
    raw_data.loc[raw_data["gran"] >= 2, 'gran_yn'] = 1
    raw_data['parn_yn'] = 0
    raw_data.loc[raw_data["parn"] >= 1, 'parn_yn'] = 1

    # 임신 과정(prep), 컬럼명 변경없이 사용 (1=자연임신|2:IVF)
    # 당뇨(dm), 컬럼명 변경없이 사용 (1=없음|2=GDM|3=Overt DM)

    # 산모 당뇨 유무
    values1 = []
    for value in raw_data['dm']:
        if value == 1:
            values1.append(1)
        elif value == 2:
            values1.append(2)
        else:
            values1.append(2)
    # dm2: 산모 당뇨(임신성 당뇨 + 본태성 당뇨) 유무 (1=없음|2=GDM or Overt DM)
    raw_data['Mat_dm'] = values1

    # 5. 고혈압 병력
    # Mat_HTN: 산모 고혈압 유무 (1=없음|2=PIH|3=Chronic HTN)
    # htn2 : 산모 고혈압 유무, 산모 만성 고혈압 유무
    values1 = []
    for value in raw_data['htn']:
        if value == 1:
            values1.append(1)
        elif value == 2:
            values1.append(2)
        else:
            values1.append(2)
    # Mat_HTN2: 산모 고혈압(임신성 고혈압 + 본태성 고혈압) 유무 (1=없음|2=PIH or Chronic HTN)
    raw_data['Mat_HTN'] = values1

    # IVH_Severe 변수 설정 (A or B)
    # A: 1주일 이내 뇌초음파검사에서 grade 3이상인 환자
    inhg_to_severe = {
        1: 1,  # grade 0
        2: 1,  # grade 1
        3: 2,  # grade 2
        4: 2,  # grade 3
        5: 2,  # grade 4
        6: 1   # 미시행
    }

    raw_data['IVH_Severe'] = raw_data['inhg'].map(inhg_to_severe)

    # 감영질환
    # 패혈증 혈액 배양 미시행한 경우 None으로 변경, chi square검정을 하기에 너무 적은 수
    raw_data.loc[raw_data['seps'] == 3, 'seps'] = 1

    # 출생 28일 이내 sepsis 여부
    cols = ["birthdt", "bsfdt1", "bsfdt2",
            "bsfdt3", "fsfdt1", "fsfdt2", "ntetdt"]
    for col in cols:
        raw_data[col] = pd.to_datetime(raw_data[col])

    # EOS, LOS 여부
    eos_condition = [
        (raw_data["bsfdt1"].notna()) & (
            (raw_data["bsfdt1"] - raw_data["birthdt"]).dt.days <= 3),
        (raw_data["bsfdt2"].notna()) & (
            (raw_data["bsfdt2"] - raw_data["birthdt"]).dt.days <= 3),
        (raw_data["bsfdt3"].notna()) & (
            (raw_data["bsfdt3"] - raw_data["birthdt"]).dt.days <= 3),
        (raw_data["fsfdt1"].notna()) & (
            (raw_data["fsfdt1"] - raw_data["birthdt"]).dt.days <= 3),
        (raw_data["fsfdt2"].notna()) & (
            (raw_data["fsfdt2"] - raw_data["birthdt"]).dt.days <= 3),
    ]
    eos_value = [1, 1, 1, 1, 1]
    raw_data["eos_yn"] = np.select(eos_condition, eos_value, default=0)

    los_condition = [
        (raw_data["bsfdt1"].notna()) & (
            (raw_data["bsfdt1"] - raw_data["birthdt"]).dt.days > 3),
        (raw_data["bsfdt2"].notna()) & (
            (raw_data["bsfdt2"] - raw_data["birthdt"]).dt.days > 3),
        (raw_data["bsfdt3"].notna()) & (
            (raw_data["bsfdt3"] - raw_data["birthdt"]).dt.days > 3),
        (raw_data["fsfdt1"].notna()) & (
            (raw_data["fsfdt1"] - raw_data["birthdt"]).dt.days > 3),
        (raw_data["fsfdt2"].notna()) & (
            (raw_data["fsfdt2"] - raw_data["birthdt"]).dt.days > 3),
    ]
    los_value = [1, 1, 1, 1, 1]
    raw_data["los_yn"] = np.select(los_condition, los_value, default=0)

    # los 7 days
    los_condition = [
        (raw_data["bsfdt1"].notna()) & (
            (3 <= (raw_data["bsfdt1"] - raw_data["birthdt"]).dt.days) & ((raw_data["bsfdt1"] - raw_data["birthdt"]).dt.days <= 7)),
        (raw_data["bsfdt2"].notna()) & (
            (3 <= (raw_data["bsfdt2"] - raw_data["birthdt"]).dt.days) & ((raw_data["bsfdt2"] - raw_data["birthdt"]).dt.days <= 7)),
        (raw_data["bsfdt3"].notna()) & (
            (3 <= (raw_data["bsfdt3"] - raw_data["birthdt"]).dt.days) & ((raw_data["bsfdt3"] - raw_data["birthdt"]).dt.days <= 7)),
        (raw_data["fsfdt1"].notna()) & (
            (3 <= (raw_data["fsfdt1"] - raw_data["birthdt"]).dt.days) & ((raw_data["fsfdt1"] - raw_data["birthdt"]).dt.days <= 7)),
        (raw_data["fsfdt2"].notna()) & (
            (3 <= (raw_data["fsfdt2"] - raw_data["birthdt"]).dt.days) & ((raw_data["fsfdt2"] - raw_data["birthdt"]).dt.days <= 7)),
    ]
    los_value = [1, 1, 1, 1, 1]
    raw_data["los7"] = np.select(los_condition, los_value, default=0)

    # los 28 days
    los_condition = [
        (raw_data["bsfdt1"].notna()) & (
            (3 <= (raw_data["bsfdt1"] - raw_data["birthdt"]).dt.days) & ((raw_data["bsfdt1"] - raw_data["birthdt"]).dt.days <= 28)),
        (raw_data["bsfdt2"].notna()) & (
            (3 <= (raw_data["bsfdt2"] - raw_data["birthdt"]).dt.days) & ((raw_data["bsfdt2"] - raw_data["birthdt"]).dt.days <= 28)),
        (raw_data["bsfdt3"].notna()) & (
            (3 <= (raw_data["bsfdt3"] - raw_data["birthdt"]).dt.days) & ((raw_data["bsfdt3"] - raw_data["birthdt"]).dt.days <= 28)),
        (raw_data["fsfdt1"].notna()) & (
            (3 <= (raw_data["fsfdt1"] - raw_data["birthdt"]).dt.days) & ((raw_data["fsfdt1"] - raw_data["birthdt"]).dt.days <= 28)),
        (raw_data["fsfdt2"].notna()) & (
            (3 <= (raw_data["fsfdt2"] - raw_data["birthdt"]).dt.days) & ((raw_data["fsfdt2"] - raw_data["birthdt"]).dt.days <= 28)),
    ]
    los_value = [1, 1, 1, 1, 1]
    raw_data["los28"] = np.select(los_condition, los_value, default=0)

    # los CA 36 Weeks
    los_condition = [
        (raw_data["bsfdt1"].notna()) & (
            (3 <= (raw_data["bsfdt1"] - raw_data["birthdt"]).dt.days) & (raw_data["bsfdt1"] < raw_data["date36"])),
        (raw_data["bsfdt2"].notna()) & (
            (3 <= (raw_data["bsfdt2"] - raw_data["birthdt"]).dt.days) & (raw_data["bsfdt2"] < raw_data["date36"])),
        (raw_data["bsfdt3"].notna()) & (
            (3 <= (raw_data["bsfdt3"] - raw_data["birthdt"]).dt.days) & (raw_data["bsfdt3"] < raw_data["date36"])),
        (raw_data["fsfdt1"].notna()) & (
            (3 <= (raw_data["fsfdt1"] - raw_data["birthdt"]).dt.days) & (raw_data["fsfdt1"] < raw_data["date36"])),
        (raw_data["fsfdt2"].notna()) & (
            (3 <= (raw_data["fsfdt2"] - raw_data["birthdt"]).dt.days) & (raw_data["fsfdt2"] < raw_data["date36"])),
    ]
    los_value = [1, 1, 1, 1, 1]
    raw_data["los36"] = np.select(los_condition, los_value, default=0)

    # 출생 28일 이내 NEC grade 2 여부(NEC 진단 날짜가 나와 있는지 확인 필요)
    raw_data["ntet_grade2"] = np.select([(raw_data["ntet"] == 2) & (
        (raw_data["ntetdt"] - raw_data["birthdt"]).dt.days <= 28)], [2], default=1)

    # raw_data["bdp"] = raw_data["bdp"].replace(5, np.nan)
    # BPD여부 non vs mild moderate severe
    raw_data["bdp_yn"] = 0
    raw_data.loc[raw_data["bdp"].isin([2, 3, 4]), "bdp_yn"] = 1
    raw_data.loc[raw_data["bdp"] == 5, "bdp_yn"] = 0
    raw_data["bdp_yn"].value_counts()

    raw_data["oxyt28"] = raw_data["oxyt28"].replace(4, 1)
    # ventilation구분 0: None, 1: 비침습적, 2: 침습적
    raw_data.loc[raw_data["arre28"] == 7, 'arre28'] = 1  # 1=비침습적
    raw_data.loc[raw_data["arre28"] == 6, 'arre28'] = 2  # 2=침습적
    raw_data["arre28"] = raw_data["arre28"].fillna(0)

    raw_data["arre36"] = raw_data["arre36"].replace(np.nan, 0)

    raw_data["oxyt36"] = raw_data["oxyt36"].fillna(0)
    raw_data.loc[raw_data["oxyt36"] == 1, 'oxyt36'] = 0
    raw_data.loc[raw_data["oxyt36"] == 2, 'oxyt36'] = 1  # 30%미만 산소
    raw_data.loc[raw_data["oxyt36"] == 3, 'oxyt36'] = 2  # 30%이상 산소

    raw_data.loc[raw_data["arre36"] == 7, 'arre36'] = 1  # 비침습적 환기
    raw_data.loc[raw_data["arre36"] == 6, 'arre36'] = 2  # 침습적 환기

    raw_data["vent_severe_28"] = 0  # 미시행
    raw_data.loc[(raw_data["arre28"] == 1) & (raw_data["oxyt28"]
                                              == 2), "vent_severe_28"] = 1  # 비침습적 + 산소 30% 미만
    raw_data.loc[(raw_data["arre28"] == 1) & (
        raw_data["oxyt28"].isin([3])), "vent_severe_28"] = 2  # 비침습적 + 산소 30% 이상
    raw_data.loc[(raw_data["arre28"] == 2) & (
        raw_data["oxyt28"].isin([2])), "vent_severe_28"] = 3  # 침습적 + 산소 30% 미만
    raw_data.loc[(raw_data["arre28"] == 2) & (
        raw_data["oxyt28"].isin([3])), "vent_severe_28"] = 4  # 침습적 + 산소 30% 이상
    raw_data["vent_severe_36"] = 0  # 미시행
    raw_data.loc[(raw_data["arre36"] == 1) & (raw_data["oxyt36"]
                                              == 1), "vent_severe_36"] = 1  # 비침습적 + 산소 30% 미만
    raw_data.loc[(raw_data["arre36"] == 1) & (
        raw_data["oxyt36"].isin([2])), "vent_severe_36"] = 2  # 비침습적 + 산소 30% 이상
    raw_data.loc[(raw_data["arre36"] == 2) & (
        raw_data["oxyt36"].isin([1])), "vent_severe_36"] = 3  # 침습적 + 산소 30% 이상
    raw_data.loc[(raw_data["arre36"] == 2) & (
        raw_data["oxyt36"].isin([2])), "vent_severe_36"] = 4  # 침습적 + 산소 30% 이상

    raw_data["severe_bpd"] = 0
    raw_data.loc[raw_data["oxyt36"] == 2, "severe_bpd"] = 1
    raw_data.loc[raw_data["arre36"].isin([1, 2]), "severe_bpd"] = 1

    raw_data["pmirnsg"] = raw_data["pmirnsg"].fillna(0)
    raw_data["avegftr"] = raw_data["avegftr"].fillna(0)
    raw_data["niarvhfnc"] = raw_data["niarvhfnc"].fillna(0)
    raw_data["aoxyuppd"] = raw_data["aoxyuppd"].fillna(0)

    raw_data["rop_yn"] = 0
    raw_data.loc[raw_data["pmirnsg"] >= 1, 'rop_yn'] = 1

    raw_data["rop3_yn"] = 0
    raw_data.loc[raw_data["pmirnsg"] >= 3, 'rop3_yn'] = 1
    raw_data["ga_28"] = 0
    raw_data.loc[raw_data["gagew"] < 28, 'ga_28'] = 1

    #
    raw_data["severe_morbidity_score"] = (
        (raw_data["bdp"] >= 3).astype(int)
        + (raw_data["inhg"] >= 3).astype(int)
        + (raw_data["ntet"] == 2).astype(int)
        + (raw_data["pmirnsg"] >= 3).astype(int)
    )

    #########################
    ##      label정의      ##
    #########################
    raw_data.loc[(config["experiment"]["zscore_cutoff"] * -1 <= raw_data['dcd_bmi_zscore'])
                 & (raw_data['dcd_bmi_zscore'] <= config["experiment"]["zscore_cutoff"]), 'label'] = 0
    raw_data.loc[raw_data['dcd_bmi_zscore'] < -1 *
                 config["experiment"]["zscore_cutoff"], 'label'] = 1
    raw_data.loc[raw_data['dcd_bmi_zscore'] >
                 config["experiment"]["zscore_cutoff"], 'label'] = 2

    raw_data["bmi_diff"] = raw_data["dcd_bmi"] - raw_data["birth_bmi"]
    raw_data["bmi_zscore_diff"] = raw_data["dcd_bmi_zscore"] - \
        raw_data["birth_bmi_zscore"]
    logging.info(f"전처리 후 데이터 수: {raw_data.shape}")

    return raw_data


def filter_data(preprocessed_data: pd.DataFrame, config_path: str, features_path: str) -> pd.DataFrame:
    """
    전처리된 데이터 필터링

    제외조건:
      GA32주 미만
      사망한 환자
      성별을 모르는 환자
      퇴원시 신장 혹은 체중이 없는 환자
    최종 데이터:
      퇴원시 BMI Z-score가 High 혹은 Normal인 환자

    Args:
        preprocessed_data: 전처리된 데이터
        config_path: 설정 파일 경로
        features_path: 특성 정의 파일 경로

    Returns:
        필터링된 데이터
    """
    config = load_config(config_path)
    # GA조건 걸기
    logging.info("===== 재태주수가 32주 미만인 환자들만 포함 =====")
    ga_data = preprocessed_data[preprocessed_data["gagew"]
                                < config["experiment"]["ga_cutoff"]]
    logging.info(
        f"GA32주 미만 조건 적용 후 shape: {ga_data.shape}, {preprocessed_data.shape[0] - ga_data.shape[0]}건 제외")
    ga_wt_data = ga_data[ga_data["bwei"] < 1500]
    logging.info(
        f'출생 시 몸무게 1500g이상 환자 제거 :: {ga_wt_data.shape}, {ga_data.shape[0] - ga_wt_data.shape[0]}건 제외 ')

    # 제외조건 적용
    logging.info('========== features_dataset 전처리 ==========')
    ga_wt_death_data = ga_wt_data[(
        # ga_wt_data['death'].isna()) & (ga_wt_data["bdp"] != 5) & (ga_wt_data["death36"] != 1)]
        ga_wt_data['death'].isna()) & (ga_wt_data["bdp"] != 5)]
    logging.info(
        f'사망한 환자 제거 :: {ga_wt_death_data.shape}, {ga_wt_data.shape[0] - ga_wt_death_data.shape[0]}건 제외')
    ga_wt_death_sex_data = ga_wt_death_data[ga_wt_death_data['sex_sys_val'] != 3]
    logging.info(
        f'성별을 모르는 환자 제거 :: {ga_wt_death_sex_data.shape}, {ga_wt_death_data.shape[0] - ga_wt_death_sex_data.shape[0]}건 제외')

    # 퇴원시 교정주수가 34주 미만 44주 6일 초과인 환자 제거
    ga_wt_death_ca_data = ga_wt_death_sex_data[
        (ga_wt_death_sex_data["corrected_agew"] >= 34) &
        (ga_wt_death_sex_data["corrected_agew"] <= 44)
    ]
    logging.info(
        f'퇴원시 교정주수가 34주 미만 44주 6일 초과인 환자 제거 :: {ga_wt_death_ca_data.shape}, {ga_wt_death_sex_data.shape[0] - ga_wt_death_ca_data.shape[0]}건 제외')

    # 퇴원시 신장 혹은 체중이 없는 환자 제거
    ga_wt_death_sex_wtht_data = ga_wt_death_ca_data.dropna(
        subset=["dcdht", "dcdwt"])
    logging.info(
        f'퇴원 시 신장 혹은 체중이 없는 환자 제거 :: {ga_wt_death_sex_wtht_data.shape}, {ga_wt_death_ca_data.shape[0] - ga_wt_death_sex_wtht_data.shape[0]}건 제외')

    # 극단적인 값 없애기
    ga_wt_death_sex_wtht_outlier_data = ga_wt_death_sex_wtht_data[
        ga_wt_death_sex_wtht_data["birth_bmi"] < 30]
    inclusion_condition = (
        ((ga_wt_death_sex_wtht_outlier_data["bwei"] < 1000) & (ga_wt_death_sex_wtht_outlier_data["gagew"] == 22)) |
        ((ga_wt_death_sex_wtht_outlier_data["bwei"] < 1400) & (ga_wt_death_sex_wtht_outlier_data["gagew"] == 25)) |
        (ga_wt_death_sex_wtht_outlier_data["dcd_bmi"] < 40)
    )

    ga_wt_death_sex_wtht_outlier_data = ga_wt_death_sex_wtht_outlier_data[inclusion_condition]
    logging.info(
        f'극단적인 값 제거 :: {ga_wt_death_sex_wtht_outlier_data.shape}, {ga_wt_death_sex_wtht_data.shape[0] - ga_wt_death_sex_wtht_outlier_data.shape[0]}건 제외')

    return ga_wt_death_sex_wtht_outlier_data


def dropna_subset(df):
    logging.info("============= dropna subset ==================")
    logging.info(f"input shape: {df.shape}")
    result = df.dropna(subset=["apgs1", "apgs5"])

    logging.info(
        f"after drop apgar score subset shape: {result.shape}, {df.shape[0] - result.shape[0]}건 제외")
    return result


def split_data(filtered_data: pd.DataFrame, config_path: str, features_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    config = load_config(config_path)
    features_config = load_config(features_path)
    target_features = features_config["derived_columns"]["label"][0]

    ######################################
    # external validation dataset 나누기 #
    ######################################
    filtered_data["birthdt"] = pd.to_datetime(filtered_data["birthdt"])
    features_external = filtered_data[filtered_data['birthdt'].dt.year >
                                      config["experiment"]["split_year"]]
    features_dataset = filtered_data[filtered_data['birthdt'].dt.year <=
                                     config["experiment"]["split_year"]]

    train_data, test_data = train_test_split(
        features_dataset, test_size=config["training"]["split_ratio"], random_state=config["training"]["seed"], stratify=features_dataset[target_features])
    logging.info(
        f"overall: {features_dataset.shape},  train_data shape: {train_data.shape}, test_data shape: {test_data.shape}")
    logging.info(
        f"train_data value_count: {train_data[target_features].value_counts()}, test_data shape: {test_data[target_features].value_counts()}")
    logging.info(f"external_data: {features_external.shape}")
    logging.info(
        f"external_data value_count: {train_data[target_features].value_counts()}")
    features_external_final = features_external.dropna(
        subset=["birth_bmi_zscore"])
    logging.info(
        f"validation set에서 birth_BMI_zscore 결측 값 제거, {features_external_final.shape}, {features_external.shape[0] - features_external_final.shape[0]} 건 제외")

    return features_external_final, train_data, test_data


def postprocess(train_data: pd.DataFrame, test_data: pd.DataFrame, config_path: str, features_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # 결측 데이터 처리
    train_dataset = train_data.copy()
    test_dataset = test_data.copy()
    train_dataset = train_dataset.dropna(
        subset=["apgs1", "apgs5", "bwei", "bhei"])
    test_dataset = test_dataset.dropna(
        subset=["apgs1", "apgs5", "bwei", "bhei"])

    train_dataset[["birth_bmi", "birth_bmi_zscore"]] = train_dataset.apply(
        lambda row: pd.Series(
            bmi_zscore(row["bwei"], row["bhei"],
                       row["gagew"], row["sex_sys_val"])
        ),
        axis=1)
    train_dataset["birth_bmi"] = (train_dataset["bwei"] / 1000) / \
        (train_dataset["bhei"] / 100) ** 2

    test_dataset[["birth_bmi", "birth_bmi_zscore"]] = test_dataset.apply(
        lambda row: pd.Series(
            bmi_zscore(row["bwei"], row["bhei"],
                       row["gagew"], row["sex_sys_val"])
        ),
        axis=1)
    test_dataset["birth_bmi"] = (test_dataset["bwei"] / 1000) / \
        (test_dataset["bhei"] / 100) ** 2

    train_dataset = train_dataset.dropna(subset=["birth_bmi_zscore"])
    test_dataset = test_dataset.dropna(subset=["birth_bmi_zscore"])

    # 데이터 보간
    train_ml_data = train_data.copy()
    test_ml_data = test_data.copy()
    train_ml_data["apgs1"] = train_ml_data["apgs1"].fillna(
        train_ml_data["apgs1"].mode()[0])
    train_ml_data["apgs5"] = train_ml_data["apgs5"].fillna(
        train_ml_data["apgs5"].mode()[0])
    test_ml_data["apgs1"] = test_ml_data["apgs1"].fillna(
        train_ml_data["apgs1"].mode()[0])
    test_ml_data["apgs5"] = test_ml_data["apgs5"].fillna(
        train_ml_data["apgs5"].mode()[0])
    train_ml_data["bhei"] = train_ml_data["bhei"].fillna(
        train_ml_data["bhei"].mean())
    train_ml_data["bwei"] = train_ml_data["bwei"].fillna(
        train_ml_data["bwei"].mean())
    train_ml_data[["birth_bmi", "birth_bmi_zscore"]] = train_ml_data.apply(
        lambda row: pd.Series(
            bmi_zscore(row["bwei"], row["bhei"],
                       row["gagew"], row["sex_sys_val"])
        ),
        axis=1)
    train_ml_data["birth_bmi"] = (
        train_ml_data["bwei"] / 1000) / (train_ml_data["bhei"] / 100) ** 2

    test_ml_data["bhei"] = test_ml_data["bhei"].fillna(
        train_ml_data["bhei"].mean())
    test_ml_data["bwei"] = test_ml_data["bwei"].fillna(
        train_ml_data["bwei"].mean())
    test_ml_data[["birth_bmi", "birth_bmi_zscore"]] = test_ml_data.apply(
        lambda row: pd.Series(
            bmi_zscore(row["bwei"], row["bhei"],
                       row["gagew"], row["sex_sys_val"])
        ),
        axis=1)
    test_ml_data["birth_bmi"] = (test_ml_data["bwei"] / 1000) / \
        (test_ml_data["bhei"] / 100) ** 2

    train_data_final = train_ml_data.dropna(subset=["birth_bmi_zscore"])
    test_data_final = test_ml_data.dropna(subset=["birth_bmi_zscore"])

    logging.info(
        f"Train data, Birth BMI Z-score에서 NULL값 제외 후:: {train_data_final.shape}, {train_ml_data.shape[0] - train_data_final.shape[0]}건 제외")
    logging.info(
        f"Test data, Birth BMI Z-score에서 NULL값 제외 후:: {test_data_final.shape}, {test_ml_data.shape[0] - test_data_final.shape[0]}건 제외")
    logging.info(
        f"train, test 제외조건 적용 후 데이터 수: Train::{train_dataset.shape}, {train_data.shape[0] - train_dataset.shape[0]}건 제외, Test::{test_dataset.shape}, {test_data.shape[0] - test_dataset.shape[0]}건 제외")

    derivation_data = pd.concat([train_dataset, test_dataset])
    derivation_data_ml = pd.concat([train_data_final, test_data_final])
    logging.info(f"Derivation data:: {derivation_data.shape}")
    logging.info(f"Derivation data ml:: {derivation_data_ml.shape}")

    return train_dataset, test_dataset, train_data_final, test_data_final, derivation_data, derivation_data_ml
