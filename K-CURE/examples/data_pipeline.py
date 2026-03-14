"""
K-CURE 폐암 생존 분석 — 데이터 통합 파이프라인

20GB+ SAS 원시 데이터 8개 테이블을 환자 1명 = 1행 코호트로 통합한다.
대회 VM의 메모리 제한으로 인해 chunk 기반 스트리밍과 merge_asof를 활용했다.

본인 담당: 데이터 통합 파이프라인 · 전처리 · EDA
"""

import pandas as pd
import numpy as np
from pathlib import Path


# ──────────────────────────────────────────────
# 1. 유틸리티
# ──────────────────────────────────────────────

def fill_date(x):
    """K-CURE 데이터의 날짜 형식 통일 (4자리/6자리/8자리 → YYYYMMDD)"""
    if pd.isna(x):
        return None
    x = str(x)
    if len(x) == 4:
        return x + "0101"
    elif len(x) == 6:
        return x + "01"
    elif len(x) == 8:
        return x
    return None


def read_sas_chunked(filepath, filter_fn=None, chunksize=100_000, encoding="cp1252"):
    """
    메모리에 올릴 수 없는 SAS 파일을 chunk 단위로 읽으면서
    조건에 맞는 행만 필터링하여 반환한다.

    T200(진료)·T300(처방) 테이블은 각각 수천만 행, 총 20GB 이상이어서
    pd.read_sas()로 전체 로딩하면 OOM이 발생한다.
    """
    chunks = pd.read_sas(filepath, chunksize=chunksize, encoding=encoding)
    filtered_list = []

    for chunk in chunks:
        if filter_fn is not None:
            chunk = filter_fn(chunk)
        if not chunk.empty:
            filtered_list.append(chunk)

    return pd.concat(filtered_list, ignore_index=True) if filtered_list else pd.DataFrame()


# ──────────────────────────────────────────────
# 2. 테이블별 로딩 및 전처리
# ──────────────────────────────────────────────

KEY = "SN_KEY"


def load_registry_death(rgst_path, death_path):
    """암등록 + 사망 테이블 결합, 5년 생존 여부 계산"""
    rgst = pd.read_sas(rgst_path, encoding="cp1252")
    death = pd.read_sas(death_path, encoding="cp1252")

    rgst["FDX_tmp"] = pd.to_datetime(rgst["FDX"].apply(fill_date), format="%Y%m%d")
    death["DREGDATE"] = pd.to_datetime(death["DREGDATE"])

    merged = rgst.merge(death, on=KEY, how="left")

    # 5년 이내 사망 여부
    merged["survive"] = np.where(
        (merged["DREGDATE"] <= merged["FDX_tmp"] + pd.DateOffset(years=5))
        & merged["DREGDATE"].notnull(),
        "1",
        "0",
    )

    # 치료 이력 컬럼 분리 (수술/화학/방사선/면역/호르몬)
    merged["TX"] = merged["TX"].fillna("00000")
    merged[["op", "chemo", "rt", "immune", "hormon"]] = (
        merged["TX"]
        .astype(str)
        .str.pad(5, fillchar="0")
        .apply(lambda x: pd.Series(list(x)))
        .astype(int)
    )

    return merged


def unify_questionnaire(g1q_0708_path, g1q_0917_path, g1q_1823_path):
    """
    3개 시기(07-08, 09-17, 18-23) 문진 데이터의 컬럼명·코딩 체계를 통일한 뒤 concat.

    시기마다 흡연·음주·과거력 관련 컬럼명과 코딩 체계가 달라서
    단순 concat이 불가능하다. 예: Q_DRK_FRQ_V0108 → Q_DRK_FRQ_V09N
    """
    g1q_0708 = pd.read_sas(g1q_0708_path, encoding="cp1252")
    g1q_0917 = pd.read_sas(g1q_0917_path, encoding="cp1252")
    g1q_1823 = pd.read_sas(g1q_1823_path, encoding="cp1252")

    # 07-08: 과거력 코드를 09-23 형식의 개별 컬럼으로 변환
    disease_map = {
        "6": "Q_PHX_DX_STK",    # 뇌졸중
        "4": "Q_PHX_DX_HTDZ",   # 고혈압
        "7": "Q_PHX_DX_HTN",    # 당뇨병
    }
    for phx_col in ["Q_PHX1_DZ", "Q_PHX2_DZ", "Q_PHX3_DZ"]:
        for code, target_col in disease_map.items():
            g1q_0708.loc[g1q_0708[phx_col] == code, target_col] = "1"
        # 암 포함 기타
        g1q_0708.loc[g1q_0708[phx_col].isin(["8", "9"]), "Q_PHX_DX_ETC"] = "1"

    # 18-23: 흡연 컬럼 통일
    g1q_1823["Q_PA_VD"] = g1q_1823["Q_PA_VD_FRQ"]
    g1q_1823.loc[g1q_1823["Q_SMK_NOW_YN"] == "0", "Q_SMK_PRE_DRT"] = g1q_1823["Q_SMK_DRT"]
    g1q_1823.loc[g1q_1823["Q_SMK_NOW_YN"] == "0", "Q_SMK_PRE_AMT"] = g1q_1823["Q_SMK_AMT"]
    g1q_1823.loc[g1q_1823["Q_SMK_NOW_YN"] == "1", "Q_SMK_NOW_DRT"] = g1q_1823["Q_SMK_DRT"]
    g1q_1823.loc[g1q_1823["Q_SMK_NOW_YN"] == "1", "Q_SMK_NOW_AMT"] = g1q_1823["Q_SMK_AMT"]

    return pd.concat([g1q_0708, g1q_0917, g1q_1823], ignore_index=True)


def extract_surgery(t200_path):
    """T200(진료) 테이블에서 수술 내역만 chunk 단위로 추출"""
    return read_sas_chunked(
        t200_path,
        filter_fn=lambda chunk: chunk[chunk["SOPR_YN"] == "9"],
    )


def extract_anticancer_drugs(t300_path):
    """
    T300(처방) 테이블에서 항암제 처방만 chunk 단위로 추출.
    항암제 주성분코드 30종은 약물명 기반으로 직접 매핑 테이블을 구성했다.
    """
    drug_codes = {
        "pemetrexed": [4812], "gemcitabine": [1649],
        "paclitaxel": [2078, 5037], "docetaxel": [1485],
        "etoposide": [1571], "cisplatin": [1348], "carboplatin": [1238],
        "gefitinib": [4530], "erlotinib": [4774], "bevacizumab": [5546],
        "nivolumab": [6390], "pembrolizumab": [6400],
        "atezolizumab": [6577], "durvalumab": [6769],
        "afatinib": [6261], "dacomitinib": [6916], "osimertinib": [6525],
        "lazertinib": [6951], "crizotinib": [6175], "ceritinib": [6344],
        "alectinib": [6562], "brigatinib": [6757], "lorlatinib": [6997],
        "entrectinib": [6886], "dabrafenib": [6631], "trametinib": [6454],
        "tepotinib": [7400], "trastuzumab": [2428, 6260, 7015, 7016, 7136],
        "larotrectinib": [6889],
    }
    code_list = [str(c) for codes in drug_codes.values() for c in codes]

    return read_sas_chunked(
        t300_path,
        filter_fn=lambda chunk: chunk[chunk["GNL_CD"].isin(code_list)],
    )


# ──────────────────────────────────────────────
# 3. 테이블 통합 — merge_asof 기반 시간축 정렬
# ──────────────────────────────────────────────

def merge_temporal(base_df, right_df, base_date_col, right_date_col, direction="backward"):
    """
    시간축이 다른 두 테이블을 merge_asof로 결합.

    환자의 진단일과 건강검진일은 일치하지 않는다.
    단순 merge(on=KEY)를 쓰면 매칭이 안 되거나,
    진단 이후 검진 결과가 붙는 오류가 생긴다.
    임상적으로 진단 직전 검진 결과만 의미가 있으므로
    merge_asof(direction="backward")을 사용한다.
    """
    base_sorted = base_df.sort_values(base_date_col)
    right_sorted = right_df.sort_values(right_date_col)

    return pd.merge_asof(
        left=base_sorted,
        right=right_sorted,
        by=KEY,
        left_on=base_date_col,
        right_on=right_date_col,
        direction=direction,
    )


def build_cohort(data_dir):
    """
    8개 테이블을 순차적으로 통합하여 환자 1명 = 1행 코호트 생성.

    결합 순서:
    1. 암등록 + 사망 → 기준 테이블 (left join)
    2. T200 → 수술 이력 추출 (chunk → merge_asof forward)
    3. T300 → 항암제 처방 추출 (chunk → merge)
    4. 보험료 → merge_asof nearest
    5. 검진 → merge_asof nearest
    6. 문진 → 컬럼 통일 → concat → merge_asof nearest
    """
    data_dir = Path(data_dir)

    # 1) 암등록 + 사망
    cohort = load_registry_death(
        data_dir / "smpl_3rd_lc_rgst.sas7bdat",
        data_dir / "smpl_3rd_lc_death.sas7bdat",
    )

    # NSCLC + SEER 병기 + C34 원발부위 필터링
    cohort = cohort[cohort["MCODE"].isin(["80463", "80703", "81403", "80123"])]
    cohort = cohort[cohort["SEER_GRP"].isin(["1", "2", "3", "4", "5"])]
    cohort = cohort[cohort["TCODE"].str.contains("C34")]

    # 2) T200에서 수술 내역 결합
    t200_op = extract_surgery(data_dir / "smpl_3rd_lc_t200.sas7bdat")
    t200_op["RECU_FR_DD"] = pd.to_datetime(t200_op["RECU_FR_DD"])
    t200_op["RECU_TO_DD"] = pd.to_datetime(t200_op["RECU_TO_DD"])
    t200_op = t200_op[t200_op["M_SICK_CD"].str.contains("C34")]
    t200_op = t200_op.sort_values("RECU_FR_DD").drop_duplicates(subset=[KEY], keep="first")

    cohort = merge_temporal(cohort, t200_op, "FDX_tmp", "RECU_FR_DD", direction="forward")

    # 3) T300에서 항암제 처방 결합
    t300_drug = extract_anticancer_drugs(data_dir / "smpl_3rd_lc_t300.sas7bdat")
    t300_drug["RECU_FR_DD"] = pd.to_datetime(t300_drug["RECU_FR_DD"])
    t300_drug = t300_drug.sort_values("RECU_FR_DD").drop_duplicates(subset=[KEY], keep="first")
    cohort = merge_temporal(cohort, t300_drug, "FDX_tmp", "RECU_FR_DD", direction="forward")

    # 4) 보험료
    bfc = pd.read_sas(data_dir / "smpl_3rd_lc_bfc.sas7bdat", encoding="cp1252")
    bfc["STD_YYYY_tmp"] = pd.to_datetime(bfc["STD_YYYY"].astype(str) + "-12-31")
    cohort = merge_temporal(cohort, bfc, "RECU_TO_DD", "STD_YYYY_tmp", direction="nearest")

    # 4) 건강검진
    g1e = pd.read_sas(data_dir / "smpl_3rd_lc_g1e.sas7bdat", encoding="cp1252")
    g1e = g1e.rename(columns={"EXMD_BZ_YYYY": "EXMD_BZ_YYYY_g1e"})
    g1e["EXMD_BZ_YYYY_g1e_tmp"] = pd.to_datetime(g1e["EXMD_BZ_YYYY_g1e"].astype(str) + "-12-31")
    cohort = merge_temporal(cohort, g1e, "RECU_TO_DD", "EXMD_BZ_YYYY_g1e_tmp", direction="nearest")

    # 5) 문진 (3개 시기 통일 후 결합)
    g1q = unify_questionnaire(
        data_dir / "smpl_3rd_lc_g1q_0708.sas7bdat",
        data_dir / "smpl_3rd_lc_g1q_0917.sas7bdat",
        data_dir / "smpl_3rd_lc_g1q_1823.sas7bdat",
    )
    g1q = g1q.rename(columns={"EXMD_BZ_YYYY": "EXMD_BZ_YYYY_g1q"})
    g1q["EXMD_BZ_YYYY_g1q_tmp"] = pd.to_datetime(g1q["EXMD_BZ_YYYY_g1q"].astype(str) + "-12-31")
    cohort = merge_temporal(cohort, g1q, "RECU_TO_DD", "EXMD_BZ_YYYY_g1q_tmp", direction="nearest")

    return cohort


# ──────────────────────────────────────────────
# 4. 전처리 — 임상적 의미 기반 결측 처치
# ──────────────────────────────────────────────

def impute_biomarkers(df, biomarker_cols, group_cols=("SEX", "AGE_GRP", "BMI_GRP")):
    """
    바이오마커 결측치를 성별·연령·BMI 그룹별 중앙값으로 대치.
    그룹 내 중앙값이 없으면 전체 중앙값으로 fallback한다.

    혈액 수치는 성별·연령대에 따라 정상 범위가 다르므로
    전체 중앙값은 왜곡 위험이 있다.
    """
    for col in biomarker_cols:
        group_median = df.groupby(list(group_cols))[col].transform("median")
        df[col] = df[col].fillna(group_median)
        df[col] = df[col].fillna(df[col].median())
    return df
