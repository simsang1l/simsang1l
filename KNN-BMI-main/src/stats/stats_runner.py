from src.utils.utils import *
from src.stats.stats_utils import *
import pandas as pd
import logging
import warnings

# matplotlib 디버그 메시지 숨기기
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

# seaborn 경고 메시지 숨기기
warnings.filterwarnings('ignore', category=UserWarning, module='seaborn')

execution_time = get_korea_time()


def run_screening(config_path, features_path):
    data = load_data("split", "train", config_path)
    results = variable_screening(data, config_path, features_path)
    save_result_csv(results, path_type="screening", result_type="screening",
                    execution_time=execution_time, config_path=config_path, index=True)

# def run_screening_fu(config_path, features_path):
#     data = load_data("split", "train", config_path)
#     results = variable_screening_fu(data, config_path, features_path)
#     save_result_csv(results, path_type="screening", result_type="screening_fu", execution_time=execution_time, config_path=config_path, index = True)


def run_bsid(config_path, features_path):
    data = load_data("split", "train", config_path)
    bsid2, bsid3 = make_tableone_neurologic(data, features_path)
    save_result_csv(bsid2, path_type="tableone", result_type="bsid2",
                    execution_time=execution_time, config_path=config_path, index=True)
    save_result_csv(bsid3, path_type="tableone", result_type="bsid3",
                    execution_time=execution_time, config_path=config_path, index=True)


def run_posthoc(config_path, features_path):
    data = load_data("split", "train", config_path)
    # result = posthoc(data, features_path)
    result = posthoc_comparison(data, features_path)
    save_result_csv(result, path_type="screening", result_type="posthoc",
                    execution_time=execution_time, config_path=config_path)


def run_compare_train_test(config_path, features_path):
    train_data = load_data("split", "train", config_path)
    test_data = load_data("split", "test", config_path)
    results = compare_train_test(train_data, test_data)
    save_result_csv(results, path_type="tableone", result_type="compare_train_test",
                    execution_time=execution_time, config_path=config_path, index=True)


def run_elasticnet(config_path, features_path):
    data = load_data("split", "train", config_path)
    results = make_elasticnet(
        data, config_path, features_path, 'corr_sig_20251104')
    save_result_csv(results, path_type="logistic_regression", result_type="elasticnet",
                    execution_time=execution_time, config_path=config_path, index=True)


def run_backward(config_path, features_path):
    data = load_data("split", "train", config_path)
    testset = load_data("split", "test", config_path)
    train_model = backward_mnlogit(data, features_path)

    validation_backward(testset, features_path, train_model)


def run_tableone(config_path, features_path):
    data = load_data("split", "train", config_path)
    results = create_tableone(data, config_path, features_path)
    save_result_csv(results, path_type="tableone", result_type="tableone_full",
                    execution_time=execution_time, config_path=config_path, index=True)
    results = chi_sqaure(data, features_path)
    save_result_csv(results, path_type="tableone", result_type="chi_square",
                    execution_time=execution_time, config_path=config_path, index=True)


def run_derivation_tableone(config_path, features_path):
    data = load_data("split", "derivation", config_path)
    results = create_derivation_tableone(data, config_path, features_path)
    save_result_csv(results, path_type="tableone", result_type="tableone_full_derivation",
                    execution_time=execution_time, config_path=config_path, index=True)


def run_followup_tableone(config_path, features_path):
    data = load_data("split", "train", config_path)
    fu_data, results = create_followup_tableone(
        data, config_path, features_path)
    save_result_csv(results, path_type="tableone", result_type="tableone_followup",
                    execution_time=execution_time, config_path=config_path, index=True)
    results = variable_screening_fu(fu_data, config_path, features_path)
    save_result_csv(results, path_type="screening", result_type="screening_fu",
                    execution_time=execution_time, config_path=config_path, index=True)


def run_followup_outcomes(config_path, features_path):
    """
    2세와 3세 시점의 follow-up outcomes에 대한 단변량 분석
    BSID, KDST, Weight/Height, BMI 변수들의 분포와 p-value 계산
    """
    data = load_data("split", "train", config_path)
    features_config = load_config(features_path)

    # 변수 그룹 정의
    bsid2 = ["cognit1", "lang1", "motor1"]
    bsid3 = ["cognit2", "lang2", "motor2"]

    kdst2 = features_config["initial_columns"]["followup"]["KDST"]["KDST2"]
    kdst3 = features_config["initial_columns"]["followup"]["KDST"]["KDST3"]
    wtht2 = features_config["initial_columns"]["followup"]["wtht"]["wtht2"]
    wtht3 = features_config["initial_columns"]["followup"]["wtht"]["wtht3"]
    bmi2 = features_config["derived_columns"]["followup"]["bmi"]["bmi2"]
    bmi3 = features_config["derived_columns"]["followup"]["bmi"]["bmi3"]
    target = features_config["derived_columns"]["label"][0]

    data = data[(data["corrected_agem1"] >= 18) & (data["corrected_agem1"] <= 30)
                & (data["birth_agem2"] >= 30) & (data["birth_agem2"] <= 42)]

    # 각 변수 그룹별로 데이터 준비
    # 1. Weight/Height/BMI 데이터
    wtht_bmi_data2 = data[wtht2 + bmi2 + [target]].dropna().copy()
    wtht_bmi_data3 = data[wtht3 + bmi3 + [target]].dropna().copy()
    wtht_bmi_data2['bmi_group'] = wtht_bmi_data2[target].map(
        {0: 'Normal', 1: 'Low', 2: 'High'})
    wtht_bmi_data3['bmi_group'] = wtht_bmi_data3[target].map(
        {0: 'Normal', 1: 'Low', 2: 'High'})

    # 2. KDST 데이터
    kdst_data2 = data[kdst2 + [target]].dropna().copy()
    kdst_data3 = data[kdst3 + [target]].dropna().copy()
    kdst_data2['bmi_group'] = kdst_data2[target].map(
        {0: 'Normal', 1: 'Low', 2: 'High'})
    kdst_data3['bmi_group'] = kdst_data3[target].map(
        {0: 'Normal', 1: 'Low', 2: 'High'})

    # KDST 분류 적용
    for col in kdst2:
        kdst_data2[col] = kdst_data2[col].apply(classify_kdst)
    for col in kdst3:
        kdst_data3[col] = kdst_data3[col].apply(classify_kdst)

    # 3. BSID 데이터
    bsid_data2 = data[bsid2 + [target]].dropna().copy()
    bsid_data3 = data[bsid3 + [target]].dropna().copy()
    bsid_data2['bmi_group'] = bsid_data2[target].map(
        {0: 'Normal', 1: 'Low', 2: 'High'})
    bsid_data3['bmi_group'] = bsid_data3[target].map(
        {0: 'Normal', 1: 'Low', 2: 'High'})

    # BSID 분류 적용
    for col in bsid2:
        bsid_data2[col] = bsid_data2[col].apply(classify_bsid3)
    for col in bsid3:
        bsid_data3[col] = bsid_data3[col].apply(classify_bsid3)

    logging.info(
        f"Weight/Height/BMI - 2yr: {wtht_bmi_data2.shape}, 3yr: {wtht_bmi_data3.shape}")
    logging.info(f"KDST - 2yr: {kdst_data2.shape}, 3yr: {kdst_data3.shape}")
    logging.info(f"BSID - 2yr: {bsid_data2.shape}, 3yr: {bsid_data3.shape}")

    # 변수 타입 분류 함수
    def classify_variables(df, exclude_cols=['bmi_group', target]):
        """변수를 연속형과 범주형으로 분류"""
        continuous_vars = []
        categorical_vars = []

        for col in df.columns:
            if col in exclude_cols:
                continue

            if col in df.columns:
                # 숫자형이고 고유값이 10개 이상이면 연속형으로 간주
                if df[col].dtype in ['int64', 'float64'] and df[col].nunique() >= 10:
                    continuous_vars.append(col)
                else:
                    categorical_vars.append(col)

        return continuous_vars, categorical_vars

    # === Weight/Height/BMI 분석 ===
    logging.info("=== Weight/Height/BMI 분석 시작 ===")

    # 2세 Weight/Height/BMI 분석
    wtht_continuous_vars2, wtht_categorical_vars2 = classify_variables(
        wtht_bmi_data2)
    logging.info(f"Weight/Height/BMI 2세 연속형: {wtht_continuous_vars2}")
    logging.info(f"Weight/Height/BMI 2세 범주형: {wtht_categorical_vars2}")

    wtht_continuous_results2 = analyze_univariate_continuous(
        wtht_bmi_data2, 'bmi_group', wtht_continuous_vars2) if wtht_continuous_vars2 else pd.DataFrame()
    wtht_categorical_results2 = analyze_univariate_categorical(
        wtht_bmi_data2, 'bmi_group', wtht_categorical_vars2) if wtht_categorical_vars2 else pd.DataFrame()

    # 3세 Weight/Height/BMI 분석
    wtht_continuous_vars3, wtht_categorical_vars3 = classify_variables(
        wtht_bmi_data3)
    logging.info(f"Weight/Height/BMI 3세 연속형: {wtht_continuous_vars3}")
    logging.info(f"Weight/Height/BMI 3세 범주형: {wtht_categorical_vars3}")

    wtht_continuous_results3 = analyze_univariate_continuous(
        wtht_bmi_data3, 'bmi_group', wtht_continuous_vars3) if wtht_continuous_vars3 else pd.DataFrame()
    wtht_categorical_results3 = analyze_univariate_categorical(
        wtht_bmi_data3, 'bmi_group', wtht_categorical_vars3) if wtht_categorical_vars3 else pd.DataFrame()

    # Weight/Height/BMI 결과 합치기
    wtht_results2 = pd.DataFrame()
    for result_df in [wtht_continuous_results2, wtht_categorical_results2]:
        if not result_df.empty:
            wtht_results2 = pd.concat(
                [wtht_results2, result_df], ignore_index=True)

    wtht_results3 = pd.DataFrame()
    for result_df in [wtht_continuous_results3, wtht_categorical_results3]:
        if not result_df.empty:
            wtht_results3 = pd.concat(
                [wtht_results3, result_df], ignore_index=True)

    # === KDST 분석 ===
    logging.info("=== KDST 분석 시작 ===")

    # 2세 KDST 분석
    kdst_continuous_vars2, kdst_categorical_vars2 = classify_variables(
        kdst_data2)
    logging.info(f"KDST 2세 연속형: {kdst_continuous_vars2}")
    logging.info(f"KDST 2세 범주형: {kdst_categorical_vars2}")

    kdst_continuous_results2 = analyze_univariate_continuous(
        kdst_data2, 'bmi_group', kdst_continuous_vars2) if kdst_continuous_vars2 else pd.DataFrame()
    kdst_categorical_results2 = analyze_univariate_categorical(
        kdst_data2, 'bmi_group', kdst_categorical_vars2) if kdst_categorical_vars2 else pd.DataFrame()

    # 3세 KDST 분석
    kdst_continuous_vars3, kdst_categorical_vars3 = classify_variables(
        kdst_data3)
    logging.info(f"KDST 3세 연속형: {kdst_continuous_vars3}")
    logging.info(f"KDST 3세 범주형: {kdst_categorical_vars3}")

    kdst_continuous_results3 = analyze_univariate_continuous(
        kdst_data3, 'bmi_group', kdst_continuous_vars3) if kdst_continuous_vars3 else pd.DataFrame()
    kdst_categorical_results3 = analyze_univariate_categorical(
        kdst_data3, 'bmi_group', kdst_categorical_vars3) if kdst_categorical_vars3 else pd.DataFrame()

    # KDST 결과 합치기
    kdst_results2 = pd.DataFrame()
    for result_df in [kdst_continuous_results2, kdst_categorical_results2]:
        if not result_df.empty:
            kdst_results2 = pd.concat(
                [kdst_results2, result_df], ignore_index=True)

    kdst_results3 = pd.DataFrame()
    for result_df in [kdst_continuous_results3, kdst_categorical_results3]:
        if not result_df.empty:
            kdst_results3 = pd.concat(
                [kdst_results3, result_df], ignore_index=True)

    # === BSID 분석 ===
    logging.info("=== BSID 분석 시작 ===")

    # 2세 BSID 분석
    bsid_continuous_vars2, bsid_categorical_vars2 = classify_variables(
        bsid_data2)
    logging.info(f"BSID 2세 연속형: {bsid_continuous_vars2}")
    logging.info(f"BSID 2세 범주형: {bsid_categorical_vars2}")

    bsid_continuous_results2 = analyze_univariate_continuous(
        bsid_data2, 'bmi_group', bsid_continuous_vars2) if bsid_continuous_vars2 else pd.DataFrame()
    bsid_categorical_results2 = analyze_univariate_categorical(
        bsid_data2, 'bmi_group', bsid_categorical_vars2) if bsid_categorical_vars2 else pd.DataFrame()

    # 3세 BSID 분석
    bsid_continuous_vars3, bsid_categorical_vars3 = classify_variables(
        bsid_data3)
    logging.info(f"BSID 3세 연속형: {bsid_continuous_vars3}")
    logging.info(f"BSID 3세 범주형: {bsid_categorical_vars3}")

    bsid_continuous_results3 = analyze_univariate_continuous(
        bsid_data3, 'bmi_group', bsid_continuous_vars3) if bsid_continuous_vars3 else pd.DataFrame()
    bsid_categorical_results3 = analyze_univariate_categorical(
        bsid_data3, 'bmi_group', bsid_categorical_vars3) if bsid_categorical_vars3 else pd.DataFrame()

    # BSID 결과 합치기
    bsid_results2 = pd.DataFrame()
    for result_df in [bsid_continuous_results2, bsid_categorical_results2]:
        if not result_df.empty:
            bsid_results2 = pd.concat(
                [bsid_results2, result_df], ignore_index=True)

    bsid_results3 = pd.DataFrame()
    for result_df in [bsid_continuous_results3, bsid_categorical_results3]:
        if not result_df.empty:
            bsid_results3 = pd.concat(
                [bsid_results3, result_df], ignore_index=True)

    # === 각 그룹별 결과 저장 ===

    # # Weight/Height/BMI 결과 저장
    # if not wtht_results2.empty:
    #     save_result_csv(wtht_results2, path_type="followup_outcomes", result_type="wtht_bmi_2year",
    #                     execution_time=execution_time, config_path=config_path, index=False)
    # if not wtht_results3.empty:
    #     save_result_csv(wtht_results3, path_type="followup_outcomes", result_type="wtht_bmi_3year",
    #                     execution_time=execution_time, config_path=config_path, index=False)

    # # KDST 결과 저장
    # if not kdst_results2.empty:
    #     save_result_csv(kdst_results2, path_type="followup_outcomes", result_type="kdst_2year",
    #                     execution_time=execution_time, config_path=config_path, index=False)
    # if not kdst_results3.empty:
    #     save_result_csv(kdst_results3, path_type="followup_outcomes", result_type="kdst_3year",
    #                     execution_time=execution_time, config_path=config_path, index=False)

    # # BSID 결과 저장
    # if not bsid_results2.empty:
    #     save_result_csv(bsid_results2, path_type="followup_outcomes", result_type="bsid_2year",
    #                     execution_time=execution_time, config_path=config_path, index=False)
    # if not bsid_results3.empty:
    #     save_result_csv(bsid_results3, path_type="followup_outcomes", result_type="bsid_3year",
    #                     execution_time=execution_time, config_path=config_path, index=False)

    # === 각 그룹별 결과를 전체 2세/3세 결과로 통합 ===

    # 2세 전체 결과 합치기 (Weight/Height/BMI + KDST + BSID)
    combined_results2 = pd.DataFrame()
    for result_df in [wtht_results2, kdst_results2, bsid_results2]:
        if not result_df.empty:
            combined_results2 = pd.concat(
                [combined_results2, result_df], ignore_index=True)

    # 3세 전체 결과 합치기 (Weight/Height/BMI + KDST + BSID)
    combined_results3 = pd.DataFrame()
    for result_df in [wtht_results3, kdst_results3, bsid_results3]:
        if not result_df.empty:
            combined_results3 = pd.concat(
                [combined_results3, result_df], ignore_index=True)

    # 2세/3세 개별 저장
    if not combined_results2.empty:
        save_result_csv(combined_results2, path_type="followup_outcomes", result_type="followup_outcomes_2year",
                        execution_time=execution_time, config_path=config_path, index=False)
    if not combined_results3.empty:
        save_result_csv(combined_results3, path_type="followup_outcomes", result_type="followup_outcomes_3year",
                        execution_time=execution_time, config_path=config_path, index=False)

    # 2세/3세 통합 결과 생성
    combined_results = create_combined_followup_results(
        combined_results2, combined_results3)

    # 통합 결과 저장
    if not combined_results.empty:
        save_result_csv(combined_results, path_type="followup_outcomes",
                        result_type="followup_outcomes",
                        execution_time=execution_time, config_path=config_path, index=False)

    # 유의한 변수들 추출 (p < 0.05) - 통합 결과에서 추출
    significant_vars = []

    # 통합 결과에서 유의한 변수 추출
    if not combined_results.empty:
        for _, row in combined_results.iterrows():
            if row['p_value'] < 0.05:
                significant_vars.append({
                    'variable': row['variable'],
                    'group': row['group'],
                    'var_type': row.get('var_type', 'unknown'),
                    'p_value': row['p_value'],
                    'test_name': row.get('test_name', 'unknown')
                })

    # 유의한 변수들 저장
    if significant_vars:
        sig_vars_df = pd.DataFrame(significant_vars)
        save_result_csv(sig_vars_df, path_type="followup_outcomes",
                        result_type="significant_vars",
                        execution_time=execution_time, config_path=config_path, index=False)

    # 그래프 생성
    logging.info("=== 그래프 생성 시작 ===")
    combined_results = combined_results[combined_results["group"] != "Overall"]
    print(combined_results.shape)
    # 통합된 데이터를 사용한 기본 그래프 생성
    basic_fig = create_followup_basic_plots(
        combined_results, 'group', kdst2, kdst3)
    save_plot(basic_fig, path_type="followup_outcomes", result_type="basic_plots",
              execution_time=execution_time, config_path=config_path)


def run_followup_bmi_tableone(config_path, features_path):
    data = load_data("split", "train", config_path)
    fu_data, results = create_followup_bmi_tableone(
        data, config_path, features_path)
    data, bsid2, bsid3, kdst = make_tableone_neurologic(fu_data, features_path)
    stats = variable_stats(data, features_path)
    chisq = chisq_posthoc(data, features_path)
    # save_result_csv(fu_data, path_type="tableone", result_type="fu_data", execution_time=execution_time, config_path=config_path, index = True)
    save_result_csv(results, path_type="tableone", result_type="tableone_followup",
                    execution_time=execution_time, config_path=config_path, index=True)
    save_result_csv(bsid2, path_type="tableone", result_type="bsid2",
                    execution_time=execution_time, config_path=config_path, index=True)
    save_result_csv(bsid3, path_type="tableone", result_type="bsid3",
                    execution_time=execution_time, config_path=config_path, index=True)
    save_result_csv(kdst, path_type="tableone", result_type="kdst",
                    execution_time=execution_time, config_path=config_path, index=True)
    save_result_csv(stats, path_type="tableone", result_type="fu_stats",
                    execution_time=execution_time, config_path=config_path, index=True)
    save_result_csv(chisq, path_type="tableone", result_type="chisq_posthcoc",
                    execution_time=execution_time, config_path=config_path, index=True)
    # save_result_csv(results, path_type="tableone", result_type="chi_square_neurologic", execution_time=execution_time, config_path=config_path, index = True)


def run_chisq_posthoc(config_path, features_path):
    data = load_data("split", "train", config_path)
    result = chisq_posthoc_comparison(data, features_path)
    save_result_csv(result, path_type="tableone", result_type="chisq_posthoc",
                    execution_time=execution_time, config_path=config_path, index=True)


def run_demographics(config_path, features_path):
    data = load_data("split", "train", config_path)
    results = create_demographics(data, config_path, features_path)
    save_result_csv(results, path_type="tableone", result_type="demographics",
                    execution_time=execution_time, config_path=config_path, index=True)


def run_all_demographics(config_path, features_path):
    data = load_data("split", "derivation", config_path)
    results = create_demographics(data, config_path, features_path)
    save_result_csv(results, path_type="tableone", result_type="derivation_demographics",
                    execution_time=execution_time, config_path=config_path, index=True)


def run_corr(config_path, features_path):
    data = load_data("split", "train", config_path)
    results = create_corr(data, config_path, features_path)
    save_result_csv(results, path_type="corr", result_type="corr",
                    execution_time=execution_time, config_path=config_path, index=True)
    save_heatmap(results, path_type="corr", result_type="corr",
                 execution_time=execution_time, config_path=config_path)


def run_corr_significant(config_path, features_path):
    data = load_data("split", "train", config_path)
    results = create_corr_significant(
        data, config_path, features_path, vars='corr_sig_20251104')
    save_result_csv(results, path_type="corr", result_type="corr_significant",
                    execution_time=execution_time, config_path=config_path, index=True)
    save_heatmap(results, path_type="corr", result_type="corr_significant",
                 execution_time=execution_time, config_path=config_path)


def run_vif(config_path, features_path):
    data = load_data("split", "train", config_path)
    results = create_vif(data, config_path, features_path,
                         vars='corr_sig_20251104')
    save_result_csv(results, path_type="vif", result_type="vif",
                    execution_time=execution_time, config_path=config_path, index=True)


def run_multi_lr(config_path, features_path, vars):
    data = load_data("split", "train", config_path)
    # results = Multivariate_MNLogit(data, config_path, features_path)
    results = Multivariate_MNLogit(
        data, config_path, features_path, vars)
    # save_result_csv(results.summary(), path_type="logistic_regression", result_type="multivariate_logistic_regression", execution_time=execution_time, config_path=config_path, index = True)
    odds_ratio = create_odds_ratio_df(results, features_path)
    save_result_csv(odds_ratio, path_type="logistic_regression", result_type="multivariate_odds_ratio",
                    execution_time=execution_time, config_path=config_path, index=True)

    fig_high_bmi = plot_forest(
        odds_ratio,
        title="High BMI vs Normal BMI: OR with 95% CI",
        comparison_col="High vs Normal OR",
        ci_lower_col="High vs Normal 95% CI Lower",
        ci_upper_col="High vs Normal 95% CI Upper"
    )

    fig_low_bmi = plot_forest(
        odds_ratios=odds_ratio,
        title="Low BMI vs Normal BMI: OR with 95% CI",
        comparison_col="Low vs Normal OR",
        ci_lower_col="Low vs Normal 95% CI Lower",
        ci_upper_col="Low vs Normal 95% CI Upper"
    )

    save_plot(fig_high_bmi, path_type="logistic_regression", result_type="multivariate_odds_ratio_high_bmi",
              execution_time=execution_time, config_path=config_path)
    save_plot(fig_low_bmi, path_type="logistic_regression", result_type="multivariate_odds_ratio_low_bmi",
              execution_time=execution_time, config_path=config_path)

    # =======================================
    # 유의한 변수만 처리
    # significant_odds_ratio = odds_ratio[odds_ratio['p_value'] < 0.05].copy()
    sginificant_odds_ratio_low = odds_ratio[odds_ratio["p-value (Low vs Normal)"]
                                            < 0.05]
    sginificant_odds_ratio_high = odds_ratio[odds_ratio["p-value (High vs Normal)"]
                                             < 0.05]
    save_result_csv(sginificant_odds_ratio_low, path_type="logistic_regression",
                    result_type="multivariate_odds_ratio_low_bmi_significant", execution_time=execution_time, config_path=config_path, index=True)
    save_result_csv(sginificant_odds_ratio_high, path_type="logistic_regression",
                    result_type="multivariate_odds_ratio_high_bmi_significant", execution_time=execution_time, config_path=config_path, index=True)

    fig_high_bmi = plot_forest(
        sginificant_odds_ratio_high,
        title="High BMI vs Normal BMI: OR with 95% CI",
        comparison_col="High vs Normal OR",
        ci_lower_col="High vs Normal 95% CI Lower",
        ci_upper_col="High vs Normal 95% CI Upper"
    )

    fig_low_bmi = plot_forest(
        odds_ratios=sginificant_odds_ratio_low,
        title="Low BMI vs Normal BMI: OR with 95% CI",
        comparison_col="Low vs Normal OR",
        ci_lower_col="Low vs Normal 95% CI Lower",
        ci_upper_col="Low vs Normal 95% CI Upper"
    )

    save_plot(fig_high_bmi, path_type="logistic_regression", result_type="multivariate_odds_ratio_high_bmi_significant",
              execution_time=execution_time, config_path=config_path)
    save_plot(fig_low_bmi, path_type="logistic_regression", result_type="multivariate_odds_ratio_low_bmi_significant",
              execution_time=execution_time, config_path=config_path)


def run_uni_lr(config_path, features_path, vars):
    data = load_data("split", "train", config_path)
    # results = Univariate_Logistic_Regression(data, config_path, features_path)
    results = Univariate_MNLogit(
        data, config_path, features_path, vars)
    # summary_df = create_univariate_logit_publication_table(results)
    # print('summary_df:::', summary_df)
    # importance_df = get_feature_importance(results)

    # OR과 95% CI를 둘째자리까지 반올림하여 "OR (lower - upper)" 형태로 표시하는 컬럼 추가
    results["OR (95% CI)"] = (
        results["CI_2.5%"].round(2).astype(str) + " - " +
        results["CI_97.5%"].round(2).astype(str)
    )

    save_result_csv(results, path_type="logistic_regression", result_type="univariate_odds_ratio",
                    execution_time=execution_time, config_path=config_path, index=True)

    results["display_name"] = results["feature"] + \
        " (" + results["level"] + ")"
    results_low = results[results["comparison"] == "Normal vs Low"].copy()
    results_high = results[results["comparison"] == "Normal vs High"].copy()
    fig_low_bmi = plot_forest(
        odds_ratios=results_low,
        title="Normal vs Low: Odds Ratio with 95% CI",
        comparison_col="OR",
        ci_lower_col="CI_2.5%",
        ci_upper_col="CI_97.5%"
    )

    fig_high_bmi = plot_forest(
        odds_ratios=results_high,
        title="Normal vs high: Odds Ratio with 95% CI",
        comparison_col="OR",
        ci_lower_col="CI_2.5%",
        ci_upper_col="CI_97.5%"
    )

    save_plot(fig_high_bmi, path_type="logistic_regression", result_type="univariate_odds_ratio_high_bmi",
              execution_time=execution_time, config_path=config_path)
    save_plot(fig_low_bmi, path_type="logistic_regression", result_type="univariate_odds_ratio_low_bmi",
              execution_time=execution_time, config_path=config_path)


def run_lr(config_path, features_path):
    run_uni_lr(config_path, features_path, vars='corr_sig_20251104')
    # run_binary_lr(config_path, features_path)
    run_multi_lr(config_path, features_path, vars='corr_sig_20251104')


def run_binary_lr(config_path, features_path):
    data = load_data("split", "train", config_path)
    low = data[data["label"].isin([0, 1])]
    high = data[data["label"].isin([0, 2])]

    # run_uni_lr(config_path, features_path)

    low_results = Binary_Logit(
        low, config_path, features_path, pos_label=1, neg_label=0)
    high_results = Binary_Logit(
        high, config_path, features_path, pos_label=2, neg_label=0)

    low_result = logit_to_df(low_results, prefix="Low")
    high_result = logit_to_df(high_results, prefix="High")

    # p < 0.05 변수만 플롯 그리기
    # low_columns = ["lbp_2", "atbyn_2.0", "resui_2.0", "resuo_2.0", "strdu_2",
    #                "eythtran_2", "vent_severe_1", "bdp_yn_2", "birth_bmi", "invfpod", "stday"]
    # high_columns = ["atbyn_2.0", "vent_severe_36_3", "vent_severe_36_1",
    #                 "bdp_yn_2", "birth_bmi", "invfpod", "gagew", "stday"]
    low_result_sig = low_result[low_result["Low_p"] < 0.05]
    high_result_sig = high_result[high_result["High_p"] < 0.05]

    # low_result_sig = low_result_sig[low_result_sig["display_name"].isin(
    #     low_columns)].copy().reset_index(drop=True)
    # high_result_sig = high_result_sig[high_result_sig["display_name"].isin(
    #     high_columns)].copy().reset_index(drop=True)

    save_result_csv(low_result, path_type="logistic_regression", result_type="binary_odds_ratio_low_bmi",
                    execution_time=execution_time, config_path=config_path, index=True)
    save_result_csv(high_result, path_type="logistic_regression", result_type="binary_odds_ratio_high_bmi",
                    execution_time=execution_time, config_path=config_path, index=True)

    # Low vs Normal 플롯
    fig_log = save_forest_plot_clean(low_result, features_path,
                                     or_col="Low_OR",
                                     ci_lo_col="Low_CI_low",
                                     ci_hi_col="Low_CI_high",
                                     title="Low BMI vs Normal BMI: OR with 95% CI")  # — Adjusted OR (95% CI)")

    # High vs Normal 플롯
    fig_high = save_forest_plot_clean(high_result, features_path,
                                      or_col="High_OR",
                                      ci_lo_col="High_CI_low",
                                      ci_hi_col="High_CI_high",
                                      title="High BMI vs Normal BMI: OR with 95% CI")  # — Adjusted OR (95% CI)")

    save_plot(fig_high, path_type="logistic_regression", result_type="binary_odds_ratio_high_bmi",
              execution_time=execution_time, config_path=config_path)
    save_plot(fig_log, path_type="logistic_regression", result_type="binary_odds_ratio_low_bmi",
              execution_time=execution_time, config_path=config_path)

    # Low vs Normal 플롯
    fig_log = save_forest_plot_clean(low_result_sig, features_path,
                                     or_col="Low_OR",
                                     ci_lo_col="Low_CI_low",
                                     ci_hi_col="Low_CI_high",
                                     title="Low BMI vs Normal BMI: Adjusted OR with 95% CI")

    # High vs Normal 플롯
    fig_high = save_forest_plot_clean(high_result_sig, features_path,
                                      or_col="High_OR",
                                      ci_lo_col="High_CI_low",
                                      ci_hi_col="High_CI_high",
                                      title="High BMI vs Normal BMI: Adjusted OR with 95% CI")

    save_plot(fig_high, path_type="logistic_regression", result_type="binary_odds_ratio_high_bmi_significant",
              execution_time=execution_time, config_path=config_path)
    save_plot(fig_log, path_type="logistic_regression", result_type="binary_odds_ratio_low_bmi_significant",
              execution_time=execution_time, config_path=config_path)


def run_abnormal_lr(config_path, features_path):
    data = load_data("split", "train", config_path)
    data["label"] = data["label"].replace(2, 1)
    logging.info(
        f'========== run abnormal lr:: {data["label"].value_counts()} ==========')

    results = Binary_Logit(
        data, config_path, features_path, pos_label=1, neg_label=0)
    df_result = logit_to_df(results, prefix="Abnormal")

    save_result_csv(df_result, path_type="logistic_regression", result_type="binary_odds_ratio_abnormal_bmi",
                    execution_time=execution_time, config_path=config_path, index=True)
    # Abnormal vs Normal 플롯
    fig_high = save_forest_plot_clean(df_result, features_path,
                                      or_col="Abnormal_OR",
                                      ci_lo_col="Abnormal_CI_low",
                                      ci_hi_col="Abnormal_CI_high",
                                      title="Abnormal BMI vs Normal — Adjusted OR (95% CI)"
                                      )

    save_plot(fig_high, path_type="logistic_regression", result_type="binary_odds_ratio_abnormal_bmi",
              execution_time=execution_time, config_path=config_path)


def run_abnormal_lr_univariate(config_path, features_path):
    """Abnormal vs Normal BMI에 대한 Univariate Logistic Regression 실행"""
    data = load_data("split", "train", config_path)
    data["label"] = data["label"].replace(2, 1)
    logging.info(
        f'========== run abnormal lr univariate:: {data["label"].value_counts()} ==========')

    # Univariate logistic regression 실행
    results = Univariate_Logit(
        data, config_path, features_path, pos_label=1, neg_label=0)
    df_result = logit_to_df_univariate(results, prefix="Abnormal")

    # 결과 저장
    save_result_csv(df_result, path_type="logistic_regression", result_type="univariate_odds_ratio_abnormal_bmi",
                    execution_time=execution_time, config_path=config_path, index=True)

    # Forest plot 생성
    if not df_result.empty:
        fig_univariate = save_forest_plot_clean(df_result, features_path,
                                                or_col="Abnormal_OR",
                                                ci_lo_col="Abnormal_CI_low",
                                                ci_hi_col="Abnormal_CI_high",
                                                title="Abnormal BMI vs Normal — Univariate OR (95% CI)"
                                                )

        save_plot(fig_univariate, path_type="logistic_regression", result_type="univariate_odds_ratio_abnormal_bmi",
                  execution_time=execution_time, config_path=config_path)
    else:
        logging.warning("Univariate 결과가 비어있어 forest plot을 생성하지 않습니다.")


def run_abnormal_lr_univariate_multivariate(config_path, features_path):
    """Abnormal vs Normal BMI에 대한 Univariate + Multivariate Logistic Regression 실행"""
    data = load_data("split", "train", config_path)
    data["label"] = data["label"].replace(2, 1)
    logging.info(
        f'========== run abnormal lr univariate + multivariate:: {data["label"].value_counts()} ==========')

    # 1. Univariate logistic regression 실행
    logging.info("========== Univariate Logistic Regression 시작 ==========")
    univariate_results = Univariate_Logit(
        data, config_path, features_path, pos_label=1, neg_label=0)
    df_univariate = logit_to_df_univariate(
        univariate_results, prefix="Abnormal")

    # Univariate 결과 저장
    save_result_csv(df_univariate, path_type="logistic_regression", result_type="univariate_odds_ratio_abnormal_bmi",
                    execution_time=execution_time, config_path=config_path, index=True)

    # Univariate Forest plot 생성
    if not df_univariate.empty:
        fig_univariate = save_forest_plot_clean(df_univariate, features_path,
                                                or_col="Abnormal_OR",
                                                ci_lo_col="Abnormal_CI_low",
                                                ci_hi_col="Abnormal_CI_high",
                                                title="Abnormal BMI vs Normal — Univariate OR (95% CI)"
                                                )

        save_plot(fig_univariate, path_type="logistic_regression", result_type="univariate_odds_ratio_abnormal_bmi",
                  execution_time=execution_time, config_path=config_path)
    else:
        logging.warning("Univariate 결과가 비어있어 forest plot을 생성하지 않습니다.")

    # 2. Multivariate logistic regression 실행
    logging.info("========== Multivariate Logistic Regression 시작 ==========")
    multivariate_results = Binary_Logit(
        data, config_path, features_path, pos_label=1, neg_label=0)
    df_multivariate = logit_to_df(multivariate_results, prefix="Abnormal")

    # Multivariate 결과 저장
    save_result_csv(df_multivariate, path_type="logistic_regression", result_type="multivariate_odds_ratio_abnormal_bmi",
                    execution_time=execution_time, config_path=config_path, index=True)

    # Multivariate Forest plot 생성
    if not df_multivariate.empty:
        fig_multivariate = save_forest_plot_clean(df_multivariate, features_path,
                                                  or_col="Abnormal_OR",
                                                  ci_lo_col="Abnormal_CI_low",
                                                  ci_hi_col="Abnormal_CI_high",
                                                  title="Abnormal BMI vs Normal — Multivariate OR (95% CI)"
                                                  )

        save_plot(fig_multivariate, path_type="logistic_regression",
                  result_type="multivariate_odds_ratio_abnormal_bmi", execution_time=execution_time, config_path=config_path)
    else:
        logging.warning("Multivariate 결과가 비어있어 forest plot을 생성하지 않습니다.")

    logging.info(
        "========== Univariate + Multivariate Logistic Regression 완료 ==========")


def run_sensitivity_analysis(config_path, features_path):
    data = load_data("split", "train", config_path)
    results = Sensitivityanalysis(data, features_path)
    save_result_csv(results, path_type="logistic_regression", result_type="sensitivity_analysis",
                    execution_time=execution_time, config_path=config_path, index=True)


def run_adjusted_logit(config_path, features_path):
    data = load_data("split", "train", config_path)
    results = adjusted_logit(data, features_path)
    save_result_csv(results, path_type="logistic_regression", result_type="adjusted_logit",
                    execution_time=execution_time, config_path=config_path, index=True)


def run_feature_selection(config_path, features_path, method="elasticnet"):
    """
    다양한 변수 선택 방법을 통합적으로 제공
    method: 'rfe', 'rf', 'elasticnet'
    """
    # 데이터 로딩
    data = load_data("split", "train", config_path)
    features_config = load_config(features_path)
    features = 'corr_sig_20251104'
    features_category = features_config[features]["category"]
    features_continuous = features_config[features]["continuous"]

    label = features_config["derived_columns"]["label"][0]
    X = data.drop(columns=[label])
    X = X[features_category + features_continuous]
    X_encoded = pd.get_dummies(X, columns=features_category, drop_first=True)
    # 2. 스케일링 (연속형 변수만)
    scaler = StandardScaler()
    X_encoded[features_continuous] = scaler.fit_transform(
        X_encoded[features_continuous])
    y = data[label]
    # 변수 선택
    if method == "rfe":
        selected = feature_selection(
            X_encoded, y, method=method, n_features=20)
    elif method == "rf":
        selected = feature_selection(X_encoded, y, method=method, top_n=10)
    elif method == "elasticnet":
        selected = feature_selection(X_encoded, y, method=method, n_repeats=10)
    else:
        raise ValueError(f"Unknown method: {method}")

    print(f"선택된 변수({method}):", selected)
    # 결과 저장 등 추가 가능


def run_backward(config_path, features_path):
    data = load_data("split", "train", config_path)
    results = backward_selection(data, features_path)
    # save_result_csv(results, path_type="logistic_regression", result_type="backward_selection", execution_time=execution_time, config_path=config_path, index = True)
