import itertools
from tableone import TableOne
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.oneway import anova_oneway
import pandas as pd
import numpy as np
import os
from src.utils.utils import load_config
import logging
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
import pingouin as pg
import seaborn as sns
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import statsmodels.formula.api as smf
from tqdm import tqdm
import re
import warnings

# matplotlib 디버그 메시지 숨기기
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

# seaborn 경고 메시지 숨기기
warnings.filterwarnings('ignore', category=UserWarning, module='seaborn')


def variable_screening(
        df: pd.DataFrame,
        config_path: str,
        features_path: str,
        alpha: float = 0.05,
        adjust_method: str = "holm"
) -> pd.DataFrame:
    """
    1차 변수 스크리닝 + Holm 보정

    Parameters
    ----------
    df : DataFrame 전체 데이터
    alpha : float, 유의수준
    adjust_method : str, p-value 보정 방법 ('holm', 'fdr_bh', ...)

    Returns
    -------
    result_df : DataFrame
        variable, var_type, test_used, stat, p_raw, p_adj, keep(True/False)
    """
    logging.info("=========== variable screening Start =============")
    # config 파일 불러오기
    config = load_config(config_path)
    features_config = load_config(features_path)
    initial_columns = features_config["initial_columns"]
    derived_columns = features_config["derived_columns"]

    # 범주형, 연속형 변수 구분
    category_features = initial_columns["category"] + \
        derived_columns["category"]
    continuous_features = initial_columns["numeric"] + \
        derived_columns["numeric"]
    target_features = initial_columns["target"] + derived_columns["target"]
    target = derived_columns["label"][0]

    results = []
    y = df[target]

    # ── 1) 범주형: χ² 검정 ──────────────────────────────────
    for v in category_features:
        tbl = pd.crosstab(df[v], y)
        chi2, p, dof, _ = stats.chi2_contingency(tbl)
        results.append({
            "variable": v,
            "var_type": "categorical",
            "test_used": "chi2",
            "stat": chi2,
            "p_raw": p
        })

    # # ── 2) 연속형: Levene → ANOVA or Kruskal ──────────────
    # for v in continuous_features:
    #     groups = [df.loc[y == g, v].dropna() for g in y.unique()]

    #     # 등분산성
    #     stat_levene, p_levene = stats.levene(*groups, center="median")

    #     if p_levene >= alpha:                      # 등분산 OK → ANOVA
    #         stat, p = stats.f_oneway(*groups)
    #         test_name = "anova"
    #     else:                                      # 등분산 X → Kruskal-Wallis
    #         stat, p = stats.kruskal(*groups)
    #         test_name = "kruskal"

        # results.append({
        #     "variable": v,
        #     "var_type": "continuous",
        #     "test_used": test_name,
        #     "stat": stat,
        #     "p_raw": p,
        #     "p_levene": p_levene
        # })

     # ── 2) 연속형: Welch ANOVA (등분산성은 참고용) ─────────────
    for v in continuous_features:
        groups = [df.loc[y == g, v].dropna() for g in y.unique()]
        # 등분산성 p 값(보고용)
        _, p_levene = stats.levene(*groups, center="median")

        # Welch ANOVA
        res = anova_oneway(df[v], groups=y, use_var="unequal")
        stat, p = res.statistic, res.pvalue

        results.append({
            "variable": v,
            "var_type": "continuous",
            "test_used": "welch",
            "stat": stat,
            "p_raw": p,
            "p_levene": p_levene
        })

    # ── 3) Holm 보정 (p-value 다중 비교 보정)────────────────────
    res_df = pd.DataFrame(results)
    reject, p_adj, _, _ = multipletests(res_df["p_raw"], alpha=alpha,
                                        method=adjust_method)
    res_df["p_adj"] = p_adj
    res_df["keep"] = reject        # True → 유의

    return res_df.sort_values("p_adj").reset_index(drop=True)


def variable_screening_fu(
        df: pd.DataFrame,
        config_path: str,
        features_path: str,
        alpha: float = 0.05,
        adjust_method: str = "holm"
) -> pd.DataFrame:
    logging.info("=========== variable screening fu Start =============")
    # config 파일 불러오기
    features_config = load_config(features_path)
    derived_columns = features_config["derived_columns"]
    target = derived_columns["label"]

    df = df.reset_index()
    logging.info(f'df shape:: \n {df.head()}')
    logging.info(f'df shape:: {df.shape}')
    df = df.dropna(axis=0)
    logging.info(f'df shape:: {df.shape}')

    # 데이터 wide format -> long format으로 변환
    df_long = pd.wide_to_long(df,
                              stubnames=["ht", "wt", "bmi", "dgmtr",
                                         "dfmtr", "rctr", "sctr", "lgtr", "shtr"],
                              i="index",
                              j='time',
                              suffix='\d+'
                              ).reset_index()
    df_long["time"] = df_long["time"].astype("category")
    logging.info(f'df shape:: {df_long.shape}')
    logging.info(f'df shape:: \n {df_long.head()}')

    variables = ['wt', 'ht', 'bmi', "dgmtr",
                 "dfmtr", "rctr", "sctr", "lgtr", "shtr"]

    # 빈 DataFrame 준비 (논문용 Table 구조)
    all_mixed_anova = pd.DataFrame()

    # 루프 돌면서 저장
    for var in variables:
        print(f"\n=== Mixed ANOVA for {var} ===")

        anova_result = pg.mixed_anova(dv=var,
                                      within='time',
                                      between='label',
                                      subject='index',
                                      data=df_long)

        print(anova_result)

        # 논문용 Table에 필요한 column만 추출하고 rename
        anova_table = anova_result[['Source', 'F',
                                    'DF1', 'DF2', 'p-unc', 'np2']].copy()
        anova_table['Variable'] = var

        # 컬럼 이름 논문 스타일로 변경
        anova_table = anova_table.rename(columns={
            'Source': 'Source',
            'F': 'F',
            'DF1': 'DF1',
            'DF2': 'DF2',
            'p-unc': 'p-value',
            'np2': 'η² (Effect Size)',
            'Variable': 'Variable'
        })

        # 누적 저장
        all_mixed_anova = pd.concat(
            [all_mixed_anova, anova_table], ignore_index=True)

    plot_df = df_long.groupby(['label', 'time']).agg(
        mean_wt=('wt', 'mean'),
        sd_wt=('wt', 'std'),
        mean_ht=('ht', 'mean'),
        sd_ht=('ht', 'std'),
        mean_fu_bmi=('bmi', 'mean'),
        sd_fu_bmi=('bmi', 'std')
    ).reset_index()

    # 예시: wt 그리기
    plt.figure(figsize=(6, 4))
    sns.lineplot(data=plot_df, x='time', y='mean_fu_bmi',
                 hue='label', marker='o')
    plt.title('Group x Time Interaction for Weight')
    plt.xlabel('Time')
    plt.ylabel('Weight (kg)')
    plt.legend(title='Group')
    plt.tight_layout()
    plt.show()

    return all_mixed_anova

    # pairwise_group_t1 = pg.pairwise_ttests(dv='bmi',
    #                                    between='label',
    #                                    within='time',
    #                                    subject='index',
    #                                    data=df_long,
    #                                    padjust='holm',  # 다중비교 보정
    #                                    effsize='hedges')  # 효과 크기도 같이 계산
    # print("\n=== Pairwise comparisons for fu_bmi ===")
    # print(pairwise_group_t1)


def posthoc(df, features_path):
    """
    DataFrame을 입력받아 **연속형 변수**에 대해 그룹별 Games-Howell 사후 분석을 수행하며,
    각 비교쌍에 대해 A, B 그룹의 샘플 수(n_A, n_B)를 포함하여 반환합니다.

    Parameters:
    - df : pd.DataFrame
    - features_path : str, 변수 정보가 담긴 yaml 경로

    Returns:
    - pd.DataFrame: Games-Howell 결과 + n_A, n_B 포함
    """
    logging.info("======== posthoc 시작 ============")
    results = []
    features_config = load_config(features_path)
    group_col = features_config["derived_columns"]["label"][0]
    columns = (
        features_config["initial_columns"]["followup"]["wtht"]["wtht2"] +
        features_config["initial_columns"]["followup"]["wtht"]["wtht3"] +
        features_config["derived_columns"]["followup"]["bmi"]["bmi2"] +
        features_config["derived_columns"]["followup"]["bmi"]["bmi3"]
    )
    df = df[(df["corrected_agem1"] >= 18) & (df["corrected_agem1"] <= 30)
            & (df["birth_agem2"] >= 30) & (df["birth_agem2"] <= 42)]
    df = df.dropna(subset=["wt1", "wt2", "ht1", "ht2",
                   "bmi1", "bmi2", "bmi1_zscore", "bmi2_zscore"])

    for col in columns:
        if col == group_col:
            continue

        temp = df[[group_col, col]].dropna()
        if temp[group_col].nunique() < 2 or temp[col].nunique() < 2:
            continue

        try:
            res = pg.pairwise_gameshowell(dv=col, between=group_col, data=temp)
            res["variable"] = col

            # 각 그룹의 n 계산
            n_count = temp.groupby(group_col).size().reset_index()
            n_count.columns = [group_col, 'n']

            # A 그룹 n 추가
            res = res.merge(n_count, how='left', left_on=[
                            'A'], right_on=[group_col])
            res = res.rename(columns={'n': 'n_A'}).drop(columns=[group_col])

            # B 그룹 n 추가
            res = res.merge(n_count, how='left', left_on=[
                            'B'], right_on=[group_col])
            res = res.rename(columns={'n': 'n_B'}).drop(columns=[group_col])

            results.append(res)

        except Exception as e:
            print(f"[!] 변수 {col} 처리 중 오류 발생: {e}")

    if results:
        return pd.concat(results, axis=0, ignore_index=True)
    else:
        return pd.DataFrame(columns=[
            "A", "B", "mean(A)", "mean(B)", "diff", "se", "T", "dof",
            "pval", "CI95%", "hedges", "variable", "n_A", "n_B"
        ])


def make_elasticnet(data: pd.DataFrame, config_path: str, features_path: str, vars: str):
    logging.info("============= make elasticnet ===================")
    config = load_config(config_path)
    features_config = load_config(features_path)

    label = features_config["derived_columns"]["label"]

    # 모든 컬럼 합치기
    cat_cols = features_config[vars]["category"]
    num_cols = features_config[vars]["continuous"]
    columns = cat_cols + num_cols + label

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols),
            ("num", StandardScaler(), num_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    pipe = Pipeline(steps=[
        ("pre", pre),  # ColumnTransformer
        ("clf", LogisticRegression(
            penalty="elasticnet", solver="saga",  # penalty = 'elasticnet'
            # penalty="l1", solver="saga",
            class_weight='balanced', multi_class="multinomial",
            max_iter=5000, random_state=42  # (n_jobs는 saga에서 무시되어도 존재는 함)
        )),
    ])
    param_grid = {
        "clf__C":        np.logspace(-2, 1, 20),
        "clf__l1_ratio": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    }
    data = data[columns]
    data = data.dropna(axis=0)
    logging.info(f"{data[label].value_counts()}")
    y_train = data[label]
    X_train = data[columns]
    X_train = X_train.drop(columns="label")

    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    grid = GridSearchCV(pipe, param_grid, cv=cv,
                        scoring="roc_auc_ovr_weighted",
                        n_jobs=-1, refit=True)
    grid.fit(X_train, y_train)

    best_pipe = grid.best_estimator_
    pre_name = "pre" if "pre" in best_pipe.named_steps else "columntransformer"
    clf_name = "clf" if "clf" in best_pipe.named_steps else "logisticregression"

    pre = best_pipe.named_steps[pre_name]
    best_lr = best_pipe.named_steps[clf_name]

    feat_names = pre.get_feature_names_out()            # 인코딩 후 피처명
    # [n_classes, n_features_after_encoding]
    coef = best_lr.coef_
    mask = (coef != 0).any(axis=0)

    # 길이 안전 체크
    if len(mask) != len(feat_names):
        raise RuntimeError(
            f"Shape mismatch: coef {coef.shape}, feats {len(feat_names)}")

    sel_features = [f for f, m in zip(feat_names, mask) if m]  # 선택된(인코딩된) 피처

    # 원 변수 단위로 묶기 (보고용)
    def base_var(name: str) -> str:
        # 마지막 언더스코어(_) 뒤의 부분만 제거, 없으면 그대로 유지
        return name.rsplit("_", 1)[0] if "_" in name else name

    sel_vars = sorted(set(base_var(f) for f in sel_features))

    # ── 로깅 ───────────────────────────────────────────────────────────
    # l1_ratio는 파라미터이므로 best_params_ 또는 get_params()로 접근해야 함
    best_l1_ratio = grid.best_params_['clf__l1_ratio']
    best_C = grid.best_params_['clf__C']
    flavour = ("Ridge-like" if best_l1_ratio <= 0.25 else
               "LASSO-like" if best_l1_ratio >= 0.75 else
               "Balanced elastic-net")
    logging.info(f"Best l1_ratio : {best_l1_ratio:.3f} -> {flavour}")
    logging.info(f"Best C        : {best_C:.6g}")
    logging.info(
        f"Selected features (encoded): {len(sel_features)} → {sel_features}")
    logging.info(f"Selected variables (grouped): {len(sel_vars)} → {sel_vars}")

    # ── 반환(원하는 형태 택1) ─────────────────────────────────────────
    # 1) 원 변수 단위(보고/해석 용이)
    return pd.DataFrame({"variable": sel_vars})


def backward_mnlogit(df, features_path, alpha=0.1):
    features_config = load_config(features_path)

    label = features_config["derived_columns"]["label"]
    vars_ = features_config["elastic_column_20250610"]["category"] + \
        features_config["elastic_column_20250610"]["continuous"]

    y = df[label]
    X = sm.add_constant(df[vars_])
    keep = list(X.columns)

    while True:
        mdl = sm.MNLogit(y, X[keep]).fit(method='newton', disp=0)
        pmax = mdl.pvalues.drop('const').max(axis=1)  # 변수별 최대 p
        worst = pmax.idxmax()
        if pmax[worst] > alpha:
            keep.remove(worst)
        else:
            break
    final_vars, train_model = keep[1:], sm.MNLogit(
        y, X[keep]).fit(method='newton', disp=0)
    logging.info(f'Backward → {len(final_vars)} vars: {final_vars}')

    return train_model


def validation_backward(df, features_path, train_model):
    features_config = load_config(features_path)

    label = features_config["derived_columns"]["label"][0]
    vars_ = features_config["backward_column_20250610"]["category"] + \
        features_config["backward_column_20250610"]["continuous"]
    # vars_ = features_config["elastic_column_20250610"]["category"] + features_config["elastic_column_20250610"]["continuous"]

    y_test = df[label]
    X_test = sm.add_constant(df[vars_])
    # 3. 테스트 세트 성능 평가 ──────────────────────────────────────────
    X_test_sel = sm.add_constant(X_test[vars_])
    y_prob = train_model.predict(X_test_sel)          # (n, K) 확률
    macro_auc = roc_auc_score(
        y_test, y_prob, multi_class='ovr', average='macro')
    logging.info(f'Macro-AUC on TEST:{round(macro_auc, 3)}')

    # 예측 클래스 및 리포트
    y_pred = y_prob.idxmax(axis=1).astype(y_test.dtype)
    logging.info(f'\n {classification_report(y_test, y_pred)}')

    # 혼동행렬(참고)
    logging.info(f'Confusion matrix\n, {confusion_matrix(y_test, y_pred)}')


def compare_train_test(train_data, test_data):
    """
    train 데이터와 test 데이터를 비교하는 함수

    Args:
        train_data (pd.DataFrame): 학습 데이터
        test_data (pd.DataFrame): 테스트 데이터

    Returns:
        pd.DataFrame: train과 test 데이터의 비교 결과
    """
    # 필요한 컬럼만 불러오기
    features = ['gagew', 'bwei', 'birth_bmi',
                'dcdwt', 'dcdht', 'dcd_bmi', 'dcd_bmi_zscore', 'corrected_agew', 'label']
    train_data = train_data[features]
    test_data = test_data[features]

    # 데이터셋 구분을 위한 컬럼 추가
    train_data['dataset'] = 'train'
    test_data['dataset'] = 'test'

    # 데이터 합치기
    combined_data = pd.concat(
        [train_data, test_data], axis=0, ignore_index=True)
    logging.info(f"{combined_data.shape}")
    logging.info(combined_data.head())

    # TableOne 생성
    table = TableOne(
        combined_data,
        continuous=features[:8],
        categorical=['label'],
        groupby='dataset',
        pval=True,
        smd=True,
        htest_name=True
    )

    # 결과를 DataFrame으로 변환
    results = table.tableone

    return results


def _contingency_from_groups(groups):
    """TableOne이 넘겨주는 groups(list[np.ndarray])로 r×c 분할표 생성."""
    arrays = [np.asarray(g) for g in groups]
    # NaN 제거
    arrays = [a[~pd.isna(a)] for a in arrays]
    # 레벨(범주) 일관화
    levels = np.unique(np.concatenate(arrays)) if arrays and len(
        arrays[0]) else np.array([])
    # 빈 데이터 방어
    if levels.size == 0 or len(arrays) == 0:
        return np.array([[0]]), levels
    # r×c 테이블 (행=범주, 열=그룹)
    table = np.array([[np.sum(a == lvl) for a in arrays] for lvl in levels])
    return table, levels


def htest_wrapper(groups, *args, **kwargs):
    """
    TableOne용 공통 htest:
      - 2×2  : Fisher's exact
      - 2×k  : Fisher–Freeman–Halton (fisher 패키지 사용)
      - r×c  : Chi-squared (correction=False)
    반환: float p-value
    """
    table, levels = _contingency_from_groups(groups)
    r, c = table.shape

    # 데이터가 너무 빈약하면 p=1.0 반환
    if table.sum() == 0 or r == 0 or c == 0:
        htest_wrapper.__name__ = "N/A"
        return 1.0

    if r == 2 and c == 2:
        # 2×2 Fisher 정확검정
        _, p = fisher_exact(table)
        htest_wrapper.__name__ = "Fisher's exact (2×2)"
        return float(p)

    if r == 2 and c >= 2:
        # 2×k Fisher–Freeman–Halton
        if _HAS_FFH:
            p = pvalue_nway(table).two_tail
            htest_wrapper.__name__ = "Fisher–Freeman–Halton (2×k)"
            return float(p)
        else:
            # fallback: chi-square
            _, p, _, _ = chi2_contingency(table, correction=False)
            htest_wrapper.__name__ = "Chi-squared (fallback; install `fisher`)"
            return float(p)

    # 일반 r×c: chi-square
    _, p, _, _ = chi2_contingency(table, correction=False)
    htest_wrapper.__name__ = "Chi-squared (r×c)"
    return float(p)


def create_tableone(data: pd.DataFrame, config_path: str, features_path: str) -> None:
    """TableOne 통계 테이블을 생성하고 저장합니다.

    Args:
        data (pd.DataFrame): 분석할 데이터프레임
        config_path (str): 설정 파일 경로
        features_path (str): 특성 정의 파일 경로
    """
    config = load_config(config_path)
    features_config = load_config(features_path)
    result_path = config["paths"]["tableone"]
    file_name = config["results"]["tableone_full"]
    full_path = os.path.join(result_path, file_name)

    initial_columns = features_config["initial_columns"]
    derived_columns = features_config["derived_columns"]

    category_features = initial_columns["category"] + \
        derived_columns["category"]
    numeric_features = initial_columns["numeric"] + derived_columns["numeric"]
    target_features = initial_columns["target"] + derived_columns["target"]
    label = derived_columns["label"]

    # 모든 컬럼 합치기
    columns = category_features + numeric_features + target_features + label

    # 데이터 전처리
    data = data[columns].copy()  # 원본 데이터 보존

    # # 결측치 처리
    # data = data.dropna(axis=0)

    # BMI 그룹 생성
    data['bmi_group'] = data['label'].map({0: 'Normal', 1: 'Low', 2: 'High'})
    data = data.drop(columns="label")
    logging.info(f"{data.isnull().sum().to_dict()}")
    # 카테고리형 변수 처리
    # for col, mapping in features_config["value_map"].items():
    #     data[col] = data[col].map(mapping)

    # data["gagew"] = data["gagew"] / 7
    # data["bwei"] = data["bwei"] / 1000
    # data["iarvppd"] = data["iarvppd"] / 7
    # data["niarvrpd"] = data["niarvrpd"] / 7
    # data["aoxyuppd"] = data["aoxyuppd"] / 7
    # data["niarvhfnc"] = data["niarvhfnc"] / 7
    # data["invfpod"] = data["invfpod"] / 7
    # data["stday"] = data["stday"] / 7
    try:
        table1 = TableOne(data,
                          #   columns=columns,
                          categorical=category_features,
                          continuous=numeric_features + target_features,
                          groupby='bmi_group',
                          pval=True,
                          smd=True,
                          htest={'phud_pma': htest_wrapper},
                          htest_name=True,
                          )
        logging.info(f"TableOne 결과가 저장되었습니다: {full_path}")
        logging.info(f"TableOne 결과:\n{table1.tabulate(tablefmt='latex')}")
        return table1

    except Exception as e:
        logging.error(f"TableOne 생성 중 오류 발생: {str(e)}")
        # logging.error(f"데이터 샘플:\n{data[columns].head()}")
        raise


def create_derivation_tableone(data: pd.DataFrame, config_path: str, features_path: str) -> None:
    """TableOne 통계 테이블을 생성하고 저장합니다.

    Args:
        data (pd.DataFrame): 분석할 데이터프레임
        config_path (str): 설정 파일 경로
        features_path (str): 특성 정의 파일 경로
    """
    config = load_config(config_path)
    features_config = load_config(features_path)
    result_path = config["paths"]["tableone"]
    file_name = config["results"]["tableone_full"]
    full_path = os.path.join(result_path, file_name)

    initial_columns = features_config["initial_columns"]
    derived_columns = features_config["derived_columns"]

    category_features = initial_columns["category"] + \
        derived_columns["category"]
    numeric_features = initial_columns["numeric"] + derived_columns["numeric"]
    target_features = initial_columns["target"] + derived_columns["target"]
    label = derived_columns["label"]

    # 모든 컬럼 합치기
    columns = category_features + numeric_features + target_features + label

    # 데이터 전처리
    data = data[columns].copy()  # 원본 데이터 보존

    # # 결측치 처리
    # data = data.dropna(axis=0)

    # BMI 그룹 생성
    data['bmi_group'] = data['label'].map({0: 'Normal', 1: 'Low', 2: 'High'})
    data = data.drop(columns="label")
    logging.info(f"{data.isnull().sum().to_dict()}")
    # 카테고리형 변수 처리
    # for col, mapping in features_config["value_map"].items():
    #     data[col] = data[col].map(mapping)

    # data["gagew"] = data["gagew"] / 7
    # data["bwei"] = data["bwei"] / 1000
    # data["iarvppd"] = data["iarvppd"] / 7
    # data["niarvrpd"] = data["niarvrpd"] / 7
    # data["aoxyuppd"] = data["aoxyuppd"] / 7
    # data["niarvhfnc"] = data["niarvhfnc"] / 7
    # data["invfpod"] = data["invfpod"] / 7
    # data["stday"] = data["stday"] / 7
    try:
        table1 = TableOne(data,
                          #   columns=columns,
                          categorical=category_features,
                          continuous=numeric_features + target_features,
                          groupby='bmi_group',
                          pval=True,
                          smd=True,
                          htest_name=True,
                          )
        logging.info(f"TableOne 결과가 저장되었습니다: {full_path}")
        logging.info(f"TableOne 결과:\n{table1.tabulate(tablefmt='latex')}")
        return table1

    except Exception as e:
        logging.error(f"TableOne 생성 중 오류 발생: {str(e)}")
        # logging.error(f"데이터 샘플:\n{data[columns].head()}")
        raise


def create_followup_tableone(data: pd.DataFrame, config_path: str, features_path: str) -> None:
    """TableOne 통계 테이블을 생성하고 저장합니다.

    Args:
        data (pd.DataFrame): 분석할 데이터프레임
        config_path (str): 설정 파일 경로
        features_path (str): 특성 정의 파일 경로
    """
    config = load_config(config_path)
    features_config = load_config(features_path)
    result_path = config["paths"]["tableone"]
    file_name = config["results"]["tableone_followup"]
    full_path = os.path.join(result_path, file_name)

    initial_columns = features_config["initial_columns"]
    derived_columns = features_config["derived_columns"]
    label = derived_columns["label"]

    # 모든 컬럼 합치기
    columns = initial_columns["followup"]["base"][:1] + initial_columns["followup"]["numeric"] + initial_columns["followup"]["category"] +\
        derived_columns["followup"]["base"] + \
        derived_columns["followup"]["numeric"] + label

    # 데이터 전처리
    data = data[columns].copy()  # 원본 데이터 보존
    logging.info(f"followup tableone dropna 이전: {data.shape}")

    data = data[(data["corrected_agem1"] >= 18) & (data["corrected_agem1"] <= 30) & (
        data["birth_agem2"] >= 30) & (data["birth_agem2"] <= 42)]
    logging.info(f"followup tableone 개월 조건 적용:: {data.shape}")

    # 결측치 처리
    logging.info(f"followup tableone dropna 이전: {data.columns}")
    # data = data.drop(columns=["hc1", "hc2"])
    data = data.dropna(axis=0)
    logging.info(f"followup tableone dropna: {data.shape}")
    logging.info(f"followup tableone dropna 이전: {data.columns}")

    # BMI 그룹 생성
    data['bmi_group'] = data['label'].map({0: 'Normal', 1: 'Low', 2: 'High'})
    # data = data.drop(columns=label)
    logging.info(f"{data.head()}")

    # 카테고리형 변수 처리
    # for col, mapping in features_config["value_map"].items():
    #     data[col] = data[col].map(mapping)

    try:
        table1 = TableOne(data,
                          continuous=initial_columns["followup"]["numeric"] +
                          derived_columns["followup"]["numeric"],
                          categorical=initial_columns["followup"]["category"],
                          groupby='bmi_group',
                          pval=True,
                          smd=True,
                          htest_name=True,
                          )
        logging.info(f"TableOne 결과가 저장되었습니다: {full_path}")
        logging.info(f"TableOne 결과:\n{table1.tabulate(tablefmt='latex')}")
        return data, table1

    except Exception as e:
        logging.error(f"TableOne 생성 중 오류 발생: {str(e)}")
        # logging.error(f"데이터 샘플:\n{data[columns].head()}")
        raise


def create_followup_bmi_tableone(data: pd.DataFrame, config_path: str, features_path: str) -> None:
    """TableOne 통계 테이블을 생성하고 저장합니다.

    Args:
        data (pd.DataFrame): 분석할 데이터프레임
        config_path (str): 설정 파일 경로
        features_path (str): 특성 정의 파일 경로
    """
    config = load_config(config_path)
    features_config = load_config(features_path)
    result_path = config["paths"]["tableone"]
    file_name = config["results"]["tableone_followup"]
    full_path = os.path.join(result_path, file_name)

    initial_columns = features_config["initial_columns"]
    derived_columns = features_config["derived_columns"]
    label = derived_columns["label"]

    bsid2 = features_config["initial_columns"]["followup"]["BSID"]["BSID2"]
    bsid3 = features_config["initial_columns"]["followup"]["BSID"]["BSID3"]
    kdst = features_config["initial_columns"]["followup"]["KDST"]

    # 모든 컬럼 합치기
    columns = initial_columns["followup"]["base"][:1] + initial_columns["followup"]["numeric"] + \
        derived_columns["followup"]["base"] + \
        derived_columns["followup"]["numeric"]

    # 데이터 전처리
    data = data[columns+label + bsid2+bsid3+kdst].copy()  # 원본 데이터 보존
    logging.info(f"followup tableone dropna 이전: {data.shape}")

    data = data[(data["corrected_agem1"] >= 18) & (data["corrected_agem1"] <= 30) & (
        data["birth_agem2"] >= 30) & (data["birth_agem2"] <= 42)]
    logging.info(f"followup tableone 개월 조건 적용:: {data.shape}")

    # 결측치 처리
    logging.info(f"followup tableone dropna 이전: {data.columns}")
    # data = data.drop(columns=["hc1", "hc2"])
    data_2year = data.dropna(subset=["wt1", "ht1",
                                     "bmi1",  "bmi1_zscore"], axis=0)
    data_3year = data.dropna(subset=["wt1", "wt2", "ht1", "ht2",
                                     "bmi1", "bmi2", "bmi1_zscore", "bmi2_zscore"], axis=0)
    logging.info(f"2year followup tableone dropna: {data_2year.shape}")
    logging.info(f"2year followup tableone dropna 이전: {data_2year.columns}")
    logging.info(f"3year followup tableone dropna: {data_3year.shape}")
    logging.info(f"3year followup tableone dropna 이전: {data_3year.columns}")

    # BMI 그룹 생성
    data_2year['bmi_group'] = data_2year['label'].map(
        {0: 'Normal', 1: 'Low', 2: 'High'})
    data_3year['bmi_group'] = data_3year['label'].map(
        {0: 'Normal', 1: 'Low', 2: 'High'})
    # data = data.drop(columns=label)
    logging.info(f"{data_2year.head()}")
    logging.info(f"{data_3year.head()}")

    tmp = ['corrected_agem1', 'birth_agem2', 'birth_agem1']
    for col in tmp:
        data_2year[col] = pd.to_numeric(data_2year[col], errors='coerce')
        data_3year[col] = pd.to_numeric(data_3year[col], errors='coerce')

    try:
        table1 = TableOne(data_2year[columns+["bmi_group"]],
                          continuous=initial_columns["followup"]["numeric"] +
                          derived_columns["followup"]["numeric"],
                          categorical=[],  # initial_columns["followup"]["category"],
                          groupby='bmi_group',
                          pval=True,
                          smd=True,
                          htest_name=True,
                          )
        logging.info(f"2year TableOne 결과가 저장되었습니다: {full_path}")
        logging.info(
            f"2year TableOne 결과:\n{table1.tabulate(tablefmt='latex')}")

        table2 = TableOne(data_3year[columns+["bmi_group"]],
                          continuous=initial_columns["followup"]["numeric"] +
                          derived_columns["followup"]["numeric"],
                          categorical=[],  # initial_columns["followup"]["category"],
                          groupby='bmi_group',
                          pval=True,
                          smd=True,
                          htest_name=True,
                          )
        logging.info(f"3year TableOne 결과가 저장되었습니다: {full_path}")
        logging.info(
            f"3year TableOne 결과:\n{table2.tabulate(tablefmt='latex')}")
        return data_2year, data_3year, table1, table2

    except Exception as e:
        logging.error(f"TableOne 생성 중 오류 발생: {str(e)}")
        # logging.error(f"데이터 샘플:\n{data[columns].head()}")
        raise


def chi_sqaure(data, features_path):
    features_path = load_config(features_path)
    category_features = features_path["initial_columns"]["category"]
    # category_features = features_path["initial_columns"]["followup"]["category"]
    results = []
    y = data[features_path["derived_columns"]["label"][0]]

    for v in category_features:
        tbl = pd.crosstab(data[v], y)
        chi2, p, dof, _ = stats.chi2_contingency(tbl)
        results.append({
            "variable": v,
            "var_type": "categorical",
            "test_used": "chi2",
            "stat": chi2,
            "p_raw": p
        })
    return pd.DataFrame(results)

# ── 유틸 ────────────────────────────────────────────────────────────


def cramers_v(chi2, n, k):
    return np.sqrt(chi2 / (n * (k - 1)))


def _pairwise_chisq_one(df_sub, group_col, var_col,
                        alpha=0.05, correction=False, adjust="holm"):
    """
    group_col vs var_col(범주형)에 대해
    전체 χ² + pairwise χ²/Fisher + 다중비교 보정.
    + 변수별 사용 행 수(n_used) 기록
    """
    n_used = len(df_sub)              # ★ 추가: 이 변수에 실제로 쓰인 행 수
    full_ct = pd.crosstab(df_sub[group_col], df_sub[var_col])
    chi2, p_all, dof, _ = chi2_contingency(full_ct, correction=False)

    records = []
    for g1, g2 in itertools.combinations(full_ct.index, 2):
        ct_pair = full_ct.loc[[g1, g2]]
        exp_pair = chi2_contingency(ct_pair, correction=False)[3]
        need_fisher = (exp_pair < 5).sum() > 0.2 * exp_pair.size

        if need_fisher and ct_pair.shape[1] == 2:
            p_raw = fisher_exact(ct_pair)[1]
            chi_pair = np.nan
        else:
            chi_pair, p_raw, _, _ = chi2_contingency(ct_pair,
                                                     correction=correction)

        n_pair = ct_pair.values.sum()
        V = (np.sqrt(chi_pair / (n_pair * (ct_pair.shape[1]-1)))
             if not np.isnan(chi_pair) else np.nan)

        records.append({
            "variable":  var_col,
            "group1":    g1,
            "group2":    g2,
            "p_raw":     p_raw,
            "chi2":      chi_pair,
            "cramers_v": V,
            "n_used":    n_used          # ★ 추가: 변수 전체 분석에 사용된 행 수
        })

    # 다중비교 보정
    p_adj = multipletests([r["p_raw"] for r in records], method=adjust)[1]
    for rec, p in zip(records, p_adj):
        rec["p_adj"] = p
    return pd.DataFrame(records)


# ── 메인 함수 ───────────────────────────────────────────────────────
def chisq_posthoc(df: pd.DataFrame,
                  features_path: str,
                  group_col: str = "bmi_group",
                  alpha: float = 0.05,
                  correction: bool = False,
                  adjust: str = "holm") -> pd.DataFrame:

    cfg = load_config(features_path)
    kdst = cfg["initial_columns"]["followup"]["KDST"]
    bsid2 = cfg["initial_columns"]["followup"]["BSID"]["BSID2"]
    bsid3 = cfg["initial_columns"]["followup"]["BSID"]["BSID3"]

    # BSID 점수 → 등급
    for col in bsid2:
        df[f"{col}_grp"] = df[col].apply(classify_bsid2)
    for col in bsid3:
        df[f"{col}_grp"] = df[col].apply(classify_bsid3)

    var_cols = kdst + [f"{c}_grp" for c in bsid2] + [f"{c}_grp" for c in bsid3]
    n_total = len(df)                # ★ 추가: 전체 데이터 행 수

    all_out = []
    for v in var_cols:
        sub = df[[group_col, v]].dropna()
        if sub[v].nunique() < 2:
            continue
        res = _pairwise_chisq_one(sub, group_col, v,
                                  alpha=alpha,
                                  correction=correction,
                                  adjust=adjust)
        if res is not None:
            res["n_total"] = n_total   # ★ 추가: 분석 전 전체 행 수
            all_out.append(res)

    return (pd.concat(all_out, ignore_index=True)
            if all_out else pd.DataFrame())


def create_demographics(data: pd.DataFrame, config_path: str, features_path: str) -> None:
    """TableOne 통계 테이블을 생성하고 저장합니다.

    Args:
        data (pd.DataFrame): 분석할 데이터프레임
        config_path (str): 설정 파일 경로
        features_path (str): 특성 정의 파일 경로
    """
    config = load_config(config_path)
    features_config = load_config(features_path)
    result_path = config["paths"]["tableone"]
    file_name = config["results"]["demographics"]
    full_path = os.path.join(result_path, file_name)

    initial_columns = features_config["demographics"]
    derived_columns = features_config["derived_columns"]

    category_features = initial_columns["category"]
    numeric_features = initial_columns["numeric"]
    label = derived_columns["label"]

    # 모든 컬럼 합치기
    columns = category_features + numeric_features + label

    # 데이터 전처리
    data = data[columns].copy()  # 원본 데이터 보존

    # # 결측치 처리
    # data = data.dropna(subset=["birth_bmi_zscore", "apgs1", "apgs5"], axis=0)

    # BMI 그룹 생성
    data['bmi_group'] = data['label'].map({0: 'Normal', 1: 'Low', 2: 'High'})
    data = data.drop(columns="label")
    logging.info(f"{data.head()}")
    # 카테고리형 변수 처리
    # for col, mapping in features_config["value_map"].items():
    #     data[col] = data[col].map(mapping)

    try:
        table1 = TableOne(data,
                          #   columns=columns,
                          categorical=category_features,
                          continuous=numeric_features,
                          groupby='bmi_group',
                          pval=True,
                          smd=True,
                          htest_name=True,
                          )
        logging.info(f"TableOne 결과가 저장되었습니다: {full_path}")
        logging.info(f"TableOne 결과:\n{table1.tabulate(tablefmt='latex')}")
        return table1

    except Exception as e:
        logging.error(f"TableOne 생성 중 오류 발생: {str(e)}")
        # logging.error(f"데이터 샘플:\n{data[columns].head()}")
        raise


def classify_bsid2(score):
    if score >= 85:
        return 'normal'
    elif score >= 70:
        return 'at risk'
    elif score < 70:
        return 'delay'


def classify_bsid3(score):
    if score >= 115:
        return 'above average'
    elif score >= 85:
        return 'average'
    elif score >= 70:
        return 'Mild delay'
    elif score < 70:
        return 'delay'


def classify_kdst(score):
    if score == 1:
        return 'In-depth assessment advised'
    elif score == 2:
        return 'Follow-up evaluation needed'
    elif score == 3:
        return 'peer level'
    elif score == 4:
        return 'Advanced for age'


def make_tableone_neurologic(data: pd.DataFrame, features_path: str) -> None:
    """단발성으로 TableOne 결과를 뽑아내기 위함

    Args:
        data (pd.DataFrame): 분석할 데이터프레임
        config_path (str): 설정 파일 경로
        features_path (str): 특성 정의 파일 경로
    """

    features_config = load_config(features_path)
    bsid2 = features_config["initial_columns"]["followup"]["BSID"]["BSID2"]
    bsid3 = features_config["initial_columns"]["followup"]["BSID"]["BSID3"]
    kdst = features_config["initial_columns"]["followup"]["KDST"]
    target = features_config["derived_columns"]["label"][0]

    # 데이터 전처리
    data = data[bsid2 + bsid3 + kdst + [target]].copy()  # 원본 데이터 보존
    y = data[target]

    # BMI 그룹 생성
    data['bmi_group'] = data[target].map({0: 'Normal', 1: 'Low', 2: 'High'})
    data = data.drop(columns=target)

    # # 코드 실행을 위한 na값 처리
    # for col in bsid2 + bsid3 + kdst:
    #     data[col] = data[col].fillna(0)

    for col in bsid2:
        if col == "bmi_group":
            continue
        data[f"{col}_grp"] = data[col].apply(classify_bsid2)

    for col in bsid3:
        if col == "bmi_group":
            continue
        data[f"{col}_grp"] = data[col].apply(classify_bsid3)

    data_bsid2 = data[bsid2 + ["bmi_group"]]
    for col in data_bsid2.columns:
        if col == "bmi_group":
            continue
        data_bsid2[f"{col}_grp"] = data_bsid2[col].apply(classify_bsid2)
    data_bsid2 = data_bsid2.dropna()

    data_bsid3 = data[bsid3 + ["bmi_group"]]
    for col in data_bsid3.columns:
        if col == "bmi_group":
            continue
        data_bsid3[f"{col}_grp"] = data_bsid3[col].apply(classify_bsid3)
    data_bsid3 = data_bsid3.dropna()

    data_kdst = data[kdst + ["bmi_group"]]
    data_kdst = data_kdst.dropna()

    try:
        table1 = TableOne(data_bsid2,
                          columns=data_bsid2.columns.tolist(),
                          groupby='bmi_group',
                          pval=True,
                          smd=True,
                          htest_name=True,
                          )

        table2 = TableOne(data_bsid3,
                          columns=data_bsid3.columns.tolist(),
                          groupby='bmi_group',
                          pval=True,
                          smd=True,
                          htest_name=True,
                          )
        table3 = TableOne(data_kdst,
                          columns=data_kdst.columns.tolist(),
                          groupby='bmi_group',
                          pval=True,
                          smd=True,
                          htest_name=True,
                          )
        return data, table1, table2, table3

    except Exception as e:
        logging.error(f"TableOne 생성 중 오류 발생: {str(e)}")
        raise


def create_corr(data: pd.DataFrame, config_path: str, features_path: str):
    """Corr 통계 테이블을 생성하고 저장합니다.

    Args:
        data (pd.DataFrame): 분석할 데이터프레임
        config_path (str): 설정 파일 경로
        features_path (str): 특성 정의 파일 경로
    """
    try:
        config = load_config(config_path)
        features_config = load_config(features_path)
        result_path = config["paths"]["corr"]
        file_name = config["results"]["corr"]
        full_path = os.path.join(result_path, file_name)

        initial_columns = features_config["initial_columns"]
        derived_columns = features_config["derived_columns"]

        category_features = initial_columns["category"] + \
            derived_columns["category"]
        numeric_features = initial_columns["numeric"] + \
            derived_columns["numeric"]
        target_features = initial_columns["target"] + derived_columns["target"]
        label = derived_columns["label"]

        # 모든 컬럼 합치기
        columns = category_features + numeric_features + label + target_features

        # 데이터 전처리
        data = data[columns].copy()  # 원본 데이터 보존

        results = data.corr(method='pearson')

        # # 변수명 매핑 적용
        # column_map = features_config.get("column_map", {})
        # if column_map:
        #     # 인덱스와 컬럼 모두 매핑 적용
        #     results = results.rename(index=column_map, columns=column_map)
        #     logging.info(f"변수명 매핑이 적용되었습니다. 매핑된 변수 수: {len(column_map)}")

        logging.info(f"Correlation 결과가 저장되었습니다: {full_path}")

        return results

    except Exception as e:
        logging.error(f"Correlation 생성 중 오류 발생: {str(e)}")
        raise


def create_corr_significant(data: pd.DataFrame, config_path: str, features_path: str, vars: str):
    """Corr 통계 테이블을 생성하고 저장합니다.

    Args:
        data (pd.DataFrame): 분석할 데이터프레임
        config_path (str): 설정 파일 경로
        features_path (str): 특성 정의 파일 경로
    """
    try:
        features_config = load_config(features_path)
        derived_columns = features_config["derived_columns"]

        label = derived_columns["target"][-1]
        category_features = features_config[vars]["category"]
        numeric_features = features_config[vars]["continuous"]

        # 모든 컬럼 합치기
        columns = category_features + numeric_features + [label]

        # 데이터 전처리
        data = data[columns].copy()  # 원본 데이터 보존

        results = data.corr(method='pearson')

        # 변수명 매핑 적용
        column_map = features_config.get("column_map", {})
        if column_map:
            # 인덱스와 컬럼 모두 매핑 적용
            results = results.rename(index=column_map, columns=column_map)
        logging.info(f"변수 수: {len(category_features + numeric_features)}")

        # logging.info(f"Correlation 결과가 저장되었습니다: {full_path}")

        return results

    except Exception as e:
        logging.error(f"Correlation 생성 중 오류 발생: {str(e)}")
        raise


def create_vif(data: pd.DataFrame, config_path: str, features_path: str, vars: str):
    """VIF 통계 테이블을 생성
    다중 공선성(Variance Inflation Factor, VIF)

    Args:
        data (pd.DataFrame): 분석할 데이터프레임
        config_path (str): 설정 파일 경로
        features_path (str): 특성 정의 파일 경로
    """
    try:
        config = load_config(config_path)
        features_config = load_config(features_path)
        result_path = config["paths"]["vif"]
        file_name = config["results"]["vif"]
        full_path = os.path.join(result_path, file_name)

        # target_features = features_config["derived_columns"]["target"]
        target = features_config["derived_columns"]["label"]
        corr_features = features_config[vars]["category"] + \
            features_config[vars]["continuous"] + target

        data_stat = data[corr_features].copy()
        data_stat = data_stat.dropna(axis=0)

        X = data_stat.drop(target, axis=1)
        y = data_stat[target]

        X = sm.add_constant(X)  # 독립변수에만 절편 추가

        vif_data = pd.DataFrame()
        vif_data["Variable"] = X.columns

        vif_data["VIF"] = [variance_inflation_factor(
            X.values, i) for i in range(X.shape[1])]

        logging.info(f"VIF 결과가 저장되었습니다: {full_path}")
        return vif_data

    except Exception as e:
        logging.error(f"VIF 생성 중 오류 발생: {str(e)}")
        raise


def Binary_Logit(
        data: pd.DataFrame,
        config_path: str,
        features_path: str,
        pos_label,        # e.g. 'Low'  (Positive class: 1)
        neg_label,        # e.g. 'Normal' (Reference class: 0)
        maxiter: int = 1000):
    """
    Binomial Logistic Regression (statsmodels.Logit)
    ------------------------------------------------
    Parameters
    ----------
    data          : DataFrame  (원 데이터; 범주형·연속형·라벨 포함)
    config_path   : str        (load_config 에서 읽을 YAML 경로)
    features_path : str        (feature 목록 YAML 경로)
    pos_label     : str        양성 클래스 이름  (1)
    neg_label     : str        음성 클래스 이름  (0, reference)
    maxiter       : int        최대 반복 (default 300)

    Returns
    -------
    result : statsmodels.discrete.discrete_model.BinaryResults
    """

    # 1. 설정 로드
    cfg = load_config(config_path)
    feat_cfg = load_config(features_path)

    cats = feat_cfg["elastic_column_20250831"]["category"]
    nums = feat_cfg["elastic_column_20250831"]["continuous"]
    target = feat_cfg["derived_columns"]["label"]   # ex) ['label']

    # # 2. 분석 대상 두 그룹만 필터
    # subset = data[data[target[0]].isin([pos_label, neg_label])].copy()

    # 3. 결측 제거 및 단위 변환(필요 시)
    subset = data[cats + nums + target].dropna()

    scale_vars = ["iarvppd", "niarvrpd",
                  "aoxyuppd", "niarvhfnc", "invfpod", "stday"]
    for v in scale_vars:
        if v in subset.columns:
            subset[v] = subset[v] / 7
    if "bwei" in subset.columns:
        subset["bwei"] = subset["bwei"] / 1000

    # 4. y 인코딩: pos=1, neg=0
    subset["y_bin"] = (subset[target[0]] == pos_label).astype(int)
    y = subset["y_bin"]

    # 5. X: 범주형 더미 + 연속형 + 상수
    X = subset[cats + nums]
    X = pd.get_dummies(X, columns=cats, drop_first=True).astype(float)
    X = sm.add_constant(X, has_constant="add")      # intercept

    # 6. 모델 적합
    model = sm.Logit(y, X)
    result = model.fit(maxiter=maxiter, disp=False)

    # 7. 로그
    logging.info(
        f"[Binary Logit] {pos_label} vs {neg_label} rows: {subset.shape}")
    logging.info(
        f"[Binary Logit] converged: {result.mle_retvals['converged']}")
    logging.info(result.summary())

    return result


def Multivariate_MNLogit(data: pd.DataFrame,
                         config_path: str,
                         features_path: str,
                         cols,
                         maxiter: int = 500):
    """
    Multinomial Logistic Regression (MNLogit)
    – y: label Series
    – X: 범주형→더미(k-1) + 연속형 그대로 + intercept
    – sample_weight: 샘플별 가중치 (None이면 가중치 없이)
    """

    # 1) 설정 로드
    cfg = load_config(config_path)
    feat_cfg = load_config(features_path)

    # 2) 컬럼 리스트
    cats = feat_cfg[cols]["category"]
    nums = feat_cfg[cols]["continuous"]
    target = feat_cfg["derived_columns"]["label"]   # ['label']

    # 3) NA 제거 & y, X 분리
    df = data[cats + nums + target].dropna()
    y = df[target].squeeze()                       # Series
    class_counts = np.bincount(y)
    class_weights = {i: 1.0/count for i, count in enumerate(class_counts)}
    sample_weight = np.array([class_weights[cls] for cls in y])

    vars = ["iarvppd", "niarvrpd", "aoxyuppd", "niarvhfnc", "invfpod", "stday"]
    for var in vars:
        if var in df.columns:
            df[var] = df[var] / 7
    if "bwei" in df.columns:
        df["bwei"] = df["bwei"] / 1000

    # 4) X 만들기: 범주형만 dummies
    X = df[cats + nums]
    X = pd.get_dummies(X,
                       columns=cats,
                       drop_first=True).astype(float)
    X = sm.add_constant(X, has_constant='add')

    # 5) 모델 학습
    model = sm.MNLogit(y, X)
    result = model.fit(maxiter=maxiter, disp=False)
    # if sample_weight is not None:
    #     result = model.fit(maxiter=maxiter, disp=False, weights=sample_weight)
    # else:
    #     result = model.fit(maxiter=maxiter, disp=False)

    # 6) 로깅
    logging.info(f"Drop NA in Multivariate LR: {df.shape}")
    logging.info(
        f"Multivariate MNLogit converged: {result.mle_retvals['converged']}")
    logging.info(result.summary())

    return result


def Multivariate_Logit(data: pd.DataFrame,
                       config_path: str,
                       features_path: str,
                       maxiter: int = 500):
    """
    Multinomial Logistic Regression (MNLogit)
    – y: label Series
    – X: 범주형→더미(k-1) + 연속형 그대로 + intercept
    – sample_weight: 샘플별 가중치 (None이면 가중치 없이)
    """

    # 1) 설정 로드
    cfg = load_config(config_path)
    feat_cfg = load_config(features_path)

    # 2) 컬럼 리스트
    cats = feat_cfg["elastic_column_20250831"]["category"]
    nums = feat_cfg["elastic_column_20250831"]["continuous"]
    target = feat_cfg["derived_columns"]["label"]   # ['label']

    # 3) NA 제거 & y, X 분리
    df = data[cats + nums + target].dropna()
    y = df[target].squeeze()                       # Series
    class_counts = np.bincount(y)
    class_weights = {i: 1.0/count for i, count in enumerate(class_counts)}
    sample_weight = np.array([class_weights[cls] for cls in y])

    vars = ["iarvppd", "niarvrpd", "aoxyuppd", "niarvhfnc", "invfpod", "stday"]
    for var in vars:
        if var in df.columns:
            df[var] = df[var] / 7
    if "bwei" in df.columns:
        df["bwei"] = df["bwei"] / 1000

    # 4) X 만들기: 범주형만 dummies
    X = df[cats + nums]
    X = pd.get_dummies(X,
                       columns=cats,
                       drop_first=True).astype(float)
    X = sm.add_constant(X, has_constant='add')

    # 5) 모델 학습
    model = sm.Logit(y, X)
    result = model.fit(maxiter=maxiter, disp=False)
    # if sample_weight is not None:
    #     result = model.fit(maxiter=maxiter, disp=False, weights=sample_weight)
    # else:
    #     result = model.fit(maxiter=maxiter, disp=False)

    # 6) 로깅
    logging.info(f"Drop NA in Multivariate LR: {df.shape}")
    logging.info(
        f"Multivariate MNLogit converged: {result.mle_retvals['converged']}")
    logging.info(result.summary())

    return result


def Univariate_MNLogit(data: pd.DataFrame,
                       config_path: str,
                       features_path: str,
                       cols,
                       maxiter: int = 500) -> pd.DataFrame:
    """
    Univariate Multinomial Logistic Regression
    – MNLogit을 써서 각 feature별로 univariate 회귀 실행
    – baseline은 'Normal'로 지정 → Normal vs Low, Normal vs High 비교
    – OR, 95% CI, p-value, level, comparison을 DataFrame으로 반환 후 CSV 저장
    – sample_weight: 샘플별 가중치 (None이면 가중치 없이)
    """

    # 1) 설정 로드 & 결과 경로 준비
    cfg = load_config(config_path)
    feat_cfg = load_config(features_path)
    out_dir = cfg["paths"]["logistic_regression"]
    out_file = cfg["results"]["univariate_logistic_regression"]

    # 2) 컬럼 명세 & 결측치 제거
    cats = feat_cfg[cols]["category"]
    nums = feat_cfg[cols]["continuous"]
    target = feat_cfg["derived_columns"]["label"]    # e.g. ['label']
    df = data[cats + nums + target].dropna()

    vars = ["iarvppd", "niarvrpd", "aoxyuppd", "niarvhfnc", "invfpod", "stday"]
    for var in vars:
        if var in df.columns:
            df[var] = df[var] / 7
    if "bwei" in df.columns:
        df["bwei"] = df["bwei"] / 1000

    # 3) y를 재인코딩: baseline='Normal'
    orig_order = cfg["experiment"]["label_order"]   # ["Normal","Low","High"]
    baseline_name = "Normal"
    new_order = [lab for lab in orig_order if lab !=
                 baseline_name] + [baseline_name]
    # new_order == ["Low","High","Normal"]

    # 실제 컬럼명 꺼내기
    target_col = target[0]  # 'label'

    # 문자열 라벨 생성
    label_map_inv = {0: "Normal", 1: "Low", 2: "High"}
    df["label_str"] = df[target_col].map(label_map_inv)

    # Categorical로 변환
    y_cat = pd.Categorical(df["label_str"],
                           categories=new_order,
                           ordered=True)
    y = y_cat.codes

    class_counts = np.bincount(y)
    class_weights = {i: 1.0/count for i, count in enumerate(class_counts)}
    sample_weight = np.array([class_weights[cls] for cls in y])

    # 4) 레이블 맵 (code→이름)
    label_map = {i: lab for i, lab in enumerate(new_order)}

    records = []
    # 5) feature별 univariate loop
    for feat in cats + nums:
        logging.info(f"[Univariate] '{feat}' 시작")

        # 5.1) X 매트릭스 구성
        if feat in cats:
            # 범주형 → k-1개 더미
            Xf = pd.get_dummies(df[feat],
                                prefix=feat,
                                drop_first=True).astype(float)
            # 각 더미 컬럼에서 레벨만 추출하기 위한 함수
            def strip_level(col): return col.replace(f"{feat}_", "")
            levels = [strip_level(c) for c in Xf.columns]
        else:
            # 연속형 → 그대로
            Xf = df[[feat]].astype(float)
            def strip_level(col): return col
            levels = [feat]

        X = sm.add_constant(Xf, has_constant='add').astype(float)

        try:
            # 5.2) MNLogit 적합
            model = sm.MNLogit(y, X)
            if sample_weight is not None:
                res = model.fit(maxiter=maxiter, disp=False,
                                weights=sample_weight)
            else:
                res = model.fit(maxiter=maxiter, disp=False)

            logging.info(f"Univariate MNLogit summary: {res.summary()}")
            # 5.3) 결과 추출: params, bse, pvalues 사용
            params = res.params    # DataFrame: index=var, columns=[0,1] (코드값)
            bse = res.bse       # 동일한 shape
            pvals = res.pvalues  # 동일한 shape

            # columns 0,1 은 각각 Low vs Normal, High vs Normal
            for class_idx in params.columns:
                comp_str = f"{baseline_name} vs {label_map[class_idx]}"
                for var in params.index:
                    if var == "const":
                        continue

                    # raw coefficient & 표준오차
                    coef = params.loc[var, class_idx]
                    se = bse.loc[var, class_idx]

                    # 95% CI on log-odds
                    lower_log = coef - 1.96*se
                    upper_log = coef + 1.96*se

                    # exp → OR, CI
                    or_val = np.exp(coef)
                    ci_low = np.exp(lower_log)
                    ci_high = np.exp(upper_log)
                    pval = pvals.loc[var, class_idx]

                    # level 이름: dummy면 접두사 제거, 아니면 변수명
                    level = strip_level(var)

                    records.append({
                        "feature": feat,
                        "level": level,
                        "class": class_idx,      # 0 또는 1
                        "comparison": comp_str,        # "Normal vs Low" or "Normal vs High"
                        "OR": or_val,
                        "CI_2.5%": ci_low,
                        "CI_97.5%": ci_high,
                        "pvalue": pval
                    })

            logging.info(
                f"[Univariate] '{feat}' 완료 (llf={res.llf:.1f}, pseudoR2={res.prsquared:.3f})\n")

        except Exception as e:
            logging.error(f"[Univariate] '{feat}' 에러: {e}")

    # 6) DataFrame 생성 & 정렬
    df_res = pd.DataFrame.from_records(records,
                                       columns=["feature", "level", "class", "comparison", "OR", "CI_2.5%", "CI_97.5%", "pvalue"])

    # 원하는 순서: Normal vs Low, 그다음 Normal vs High
    order = [f"{baseline_name} vs {lab}" for lab in new_order[:-1]]
    df_res["comparison"] = pd.Categorical(df_res["comparison"],
                                          categories=order,
                                          ordered=True)
    df_res = df_res.sort_values(["comparison", "feature", "level"])

    # # 7) 저장 & 반환
    # os.makedirs(os.path.dirname(full_path), exist_ok=True)
    # df_res.to_csv(full_path, index=False)
    # logging.info(f"Univariate MNLogit 결과 저장: {full_path}")

    return df_res


def create_odds_ratio_df(result, features_path):
    feature_config = load_config(features_path)

    # 변환된 y 값의 개수 확인
    unique, counts = np.unique(result.model.endog, return_counts=True)
    endog_counts = dict(zip(unique, counts))
    print('endog_counts:', endog_counts)

    # Odds Ratio(OR) 및 신뢰구간 계산
    or_df = np.exp(result.params).reset_index()
    ci_df = np.exp(result.conf_int()).reset_index()
    p_values_df = result.pvalues.reset_index()

    print(ci_df.shape)
    print('ci_df head:\n', ci_df.head())
    print('or_df head:\n', or_df.head())

    # ------------------------------------------------------
    # label = 2 (Low vs Normal), label = 1 (High vs Normal) 로 분리
    # (※ 모델 구조에 따라 label 숫자는 달라질 수 있으니 실제 결과에 맞춰 확인)
    # ------------------------------------------------------
    ci_low_vs_normal = ci_df[ci_df["label"] == '1'].copy()
    ci_high_vs_normal = ci_df[ci_df["label"] == '2'].copy()

    # 컬럼명 변경
    ci_low_vs_normal.columns = [
        "label", "variable", "Low vs Normal 95% CI Lower", "Low vs Normal 95% CI Upper"]
    ci_high_vs_normal.columns = [
        "label", "variable", "High vs Normal 95% CI Lower", "High vs Normal 95% CI Upper"]

    # 불필요한 label 컬럼 제거
    ci_low_vs_normal.drop(columns=["label"], inplace=True)
    ci_high_vs_normal.drop(columns=["label"], inplace=True)

    # ------------------------------------------------------
    # OR, 신뢰구간, p-value 변환 및 컬럼명 변경
    # ------------------------------------------------------
    or_df = np.exp(result.params).reset_index()
    or_df.columns = ["variable", "Low vs Normal OR", "High vs Normal OR"]

    p_values_df = result.pvalues.reset_index()
    p_values_df.columns = [
        "variable", "p-value (Low vs Normal)", "p-value (High vs Normal)"]

    print('or_df:\n', or_df.head())
    print('ci_low_vs_normal:\n', ci_low_vs_normal.head())
    print('p_values_df:\n', p_values_df.head())

    # ------------------------------------------------------
    # 4) 병합 수행 (inner join)
    #    순서: or_df → ci_low_vs_normal → ci_high_vs_normal → p_values_df
    # ------------------------------------------------------
    odds_ratios = or_df.merge(ci_low_vs_normal, on="variable", how="inner")
    odds_ratios = odds_ratios.merge(
        ci_high_vs_normal, on="variable", how="inner")
    odds_ratios = odds_ratios.merge(p_values_df, on="variable", how="inner")

    # const 행 제외
    odds_ratios = odds_ratios[odds_ratios["variable"] != "const"]

    # ------------------------------------------------------
    # 5) 원하는 컬럼 순서로 재배치
    # ------------------------------------------------------
    final_columns = [
        "variable",
        "High vs Normal OR",
        "High vs Normal 95% CI Lower",
        "High vs Normal 95% CI Upper",
        "p-value (High vs Normal)",
        "Low vs Normal OR",
        "Low vs Normal 95% CI Lower",
        "Low vs Normal 95% CI Upper",
        "p-value (Low vs Normal)",
    ]
    odds_ratios = odds_ratios[final_columns]

    column_map = feature_config['column_map']
    odds_ratios["display_name"] = odds_ratios["variable"].map(
        column_map).fillna(odds_ratios["variable"])

    # ------------------------------------------------------
    # 6) OR과 95% CI를 둘째자리까지 반올림하여 "lower - upper" 형태로 표시하는 컬럼 추가
    # ------------------------------------------------------
    # High vs Normal OR (95% CI) 컬럼 생성
    odds_ratios["High vs Normal OR (95% CI)"] = (
        odds_ratios["High vs Normal 95% CI Lower"].round(2).astype(str) + " - " +
        odds_ratios["High vs Normal 95% CI Upper"].round(2).astype(str)
    )

    # Low vs Normal OR (95% CI) 컬럼 생성
    odds_ratios["Low vs Normal OR (95% CI)"] = (
        odds_ratios["Low vs Normal 95% CI Lower"].round(2).astype(str) + " - " +
        odds_ratios["Low vs Normal 95% CI Upper"].round(2).astype(str)
    )

    # ------------------------------------------------------
    # 7) 최종 결과 확인 및 저장
    # ------------------------------------------------------
    logging.info(f"최종 병합된 odds_ratios 데이터 개수: {odds_ratios.shape}")
    logging.info(odds_ratios.head())

    # # CSV로 저장 (result_path와 파일명은 필요에 맞게 조정)
    # odds_ratios.to_csv(os.path.join(result_path, '2_odds_ratios.csv'), index=True)

    return odds_ratios


def plot_forest(odds_ratios, title, comparison_col, ci_lower_col, ci_upper_col, max_display=11, min_display=0.05, save_path=None):
    """
    포레스트 플롯을 그리고 선택적으로 저장한다.
    폰트 크기에 맞춰 그래프 요소들도 조정

    Parameters:
    -----------
    save_path : str, optional
        저장할 파일 경로. None이면 저장하지 않음.
    """
    variables = odds_ratios["display_name"]
    OR = odds_ratios[comparison_col]
    lower_CI = odds_ratios[ci_lower_col]
    upper_CI = odds_ratios[ci_upper_col]

    OR_orig = OR.copy()
    lower_orig = lower_CI.copy()
    upper_orig = upper_CI.copy()

    OR_plot = np.clip(OR, min_display, max_display)
    lower_plot = np.clip(lower_CI, min_display, max_display)
    upper_plot = np.clip(upper_CI, min_display, max_display)

    xerr_lower = OR_plot - lower_plot
    xerr_upper = upper_plot - OR_plot

    fig, ax = plt.subplots(figsize=(16, len(variables) * 0.9))  # 더 크게 조정
    plt.subplots_adjust(left=0.3, right=0.95, top=0.9, bottom=0.1)  # 여백 최적화

    y_pos = np.arange(len(variables))

    # 에러바 스타일을 폰트에 맞춰 조정
    ax.errorbar(OR_plot, y_pos,
                xerr=[xerr_lower, xerr_upper],
                fmt='o',
                color='black',
                markersize=8,      # 점 크기 증가
                capsize=5,         # 캐치 크기 증가
                capthick=2.5,      # 캐치 굵기 증가
                elinewidth=2.5)    # 에러바 선 굵기 증가

    # 기준선도 굵게 조정
    ax.axvline(1, color='gray', linestyle='--', lw=2)

    ax.set_xscale("log")
    ax.set_xlim(min_display, max_display)

    ticks = [min_display, 0.1, 0.5, 0.7, 1, 2, 5, 10]
    ax.set_xticks(ticks)
    ax.set_xticklabels([min_display, "", "", "", 1, "", "",
                       10], fontsize=16)  # x축 라벨 폰트 크기 추가

    # 축 눈금 크기 조정
    ax.tick_params(axis='both', which='major', labelsize=16, width=2, length=6)
    ax.tick_params(axis='both', which='minor', width=1.5, length=4)

    # 텍스트 위치를 더 체계적으로 계산
    for i in range(len(variables)):
        annotation = f"{OR_orig.iloc[i]:.2f} ({lower_orig.iloc[i]:.2f}, {upper_orig.iloc[i]:.2f})"

        # 각 변수별로 에러바의 최대 범위를 계산
        error_end = upper_plot.iloc[i]

        # 텍스트 위치를 에러바 끝에서 충분히 떨어뜨리되, 그래프 범위 내에 배치
        if error_end < max_display * 0.3:  # 에러바가 짧은 경우
            annot_x = max_display * 0.4  # 고정 위치
        elif error_end < max_display * 0.6:  # 에러바가 중간인 경우
            annot_x = max_display * 0.65
        else:  # 에러바가 긴 경우
            annot_x = max_display * 0.8

        # 텍스트 박스 스타일 개선 (폰트 크기 증가)
        ax.text(annot_x, y_pos[i], annotation, va='center', ha='center', fontsize=18,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='white', alpha=0.9))

    ax.set_yticks(y_pos)
    ax.set_yticklabels(variables, fontsize=22)  # y축 라벨 폰트 크기 증가
    ax.tick_params(axis='y', pad=8)  # y축 패딩 증가
    ax.set_title(title, fontsize=24, pad=20)  # 제목 폰트 크기 증가

    plt.tight_layout()

    # 저장 기능
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[✓] Forest plot saved ➜  {save_path}")

    return fig


def logit_to_df(result, prefix="Low"):
    """statsmodels Logit 결과 → DataFrame 한 줄"""
    coefs = result.params
    conf = result.conf_int()
    pvals = result.pvalues
    rows = []
    for var in coefs.index.drop("const"):
        rows.append({
            "display_name": var,
            f"{prefix}_OR":     np.exp(coefs[var]),
            f"{prefix}_CI_low": np.exp(conf.loc[var, 0]),
            f"{prefix}_CI_high": np.exp(conf.loc[var, 1]),
            f"{prefix}_p":      pvals[var]
        })
    return pd.DataFrame(rows)


def save_forest_plot(df: pd.DataFrame,
                     features_path,
                     or_col: str,
                     ci_lo_col: str,
                     ci_hi_col: str,
                     label_col: str = "display_name",
                     title: str = "",
                     xlim: tuple = (0.05, 20),
                     xticks: list = [0.05, 0.1, 0.5, 0.7, 1, 2, 5, 10]):
    """
    포레스트 플롯을 저장(PNG·PDF 등)한다.
    """
    # df = df.sort_values(or_col)                 # 보기 좋게 정렬(선택)
    # df = df.sort_values("display_name")                 # 보기 좋게 정렬(선택)
    feature_config = load_config(features_path)
    column_map = feature_config['column_map']
    df["display_name"] = df["display_name"].map(
        column_map).fillna(df["display_name"])

    y_pos = np.arange(len(df))

    OR = df[or_col].values.astype(float)
    lo = df[ci_lo_col].values.astype(float)
    hi = df[ci_hi_col].values.astype(float)

    fig, ax = plt.subplots(figsize=(10, len(df) * 0.7))
    ax.errorbar(OR, y_pos,
                xerr=[OR - lo, hi - OR],
                fmt='o', color='black', capsize=3)
    ax.axvline(1, color='gray', ls='--', lw=1)

    ax.set_xscale('log')
    ax.set_xlim(*xlim)
    ax.set_xticks(xticks)
    ax.set_xticklabels([0.05, "", "", "", 1, "", "", 10])
    # ax.set_xlabel("Odds Ratio ()")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df[label_col], fontsize=16)
    ax.set_title(title)

    # 오른쪽에 숫자 주석
    for i, txt in enumerate(df[label_col]):
        annot = f"{OR[i]:.2f} ({lo[i]:.2f}, {hi[i]:.2f})"
        ax.text(xlim[1] * 1.05, y_pos[i], annot,
                va='center', ha='left', fontsize=16)

    plt.tight_layout()
    # fig.savefig(path, dpi=300, bbox_inches="tight")
    # plt.close(fig)
    # print(f"[✓] Forest plot saved ➜  {os.path.abspath(path)}")
    return fig


def save_forest_plot_clean(df: pd.DataFrame,
                           features_path,
                           or_col: str,
                           ci_lo_col: str,
                           ci_hi_col: str,
                           label_col: str = "display_name",
                           title: str = "",
                           xlim: tuple = (0.05, 20),
                           xticks: list = [0.05, 0.1, 0.5, 0.7, 1, 2, 5, 10]):
    """
    포레스트 플롯을 저장 - 깔끔한 버전 (배경 박스 없음)
    폰트 크기에 맞춰 그래프 요소들도 조정
    """
    feature_config = load_config(features_path)
    column_map = feature_config['column_map']
    df["display_name"] = df["display_name"].map(
        column_map).fillna(df["display_name"])

    y_pos = np.arange(len(df))

    OR = df[or_col].values.astype(float)
    lo = df[ci_lo_col].values.astype(float)
    hi = df[ci_hi_col].values.astype(float)

    # figure 크기를 더 크게 조정
    fig, ax = plt.subplots(figsize=(14, len(df) * 0.8))

    # 에러바 스타일을 폰트에 맞춰 조정
    ax.errorbar(OR, y_pos,
                xerr=[OR - lo, hi - OR],
                fmt='o',
                color='black',
                markersize=8,      # 점 크기 증가 (기본값 6 -> 8)
                capsize=5,         # 캐치 크기 증가 (기본값 3 -> 5)
                capthick=2.5,      # 캐치 굵기 증가 (기본값 1 -> 2.5)
                elinewidth=2.5)    # 에러바 선 굵기 증가 (기본값 1 -> 2.5)

    # 기준선도 굵게 조정
    ax.axvline(1, color='gray', ls='--', lw=2)  # 선 굵기 증가 (1 -> 2)

    ax.set_xscale('log')
    ax.set_xlim(*xlim)
    ax.set_xticks(xticks)
    ax.set_xticklabels([0.05, "", "", "", 1, "", "", 10],
                       fontsize=16)  # x축 라벨 폰트 크기 추가
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df[label_col], fontsize=22)
    ax.set_title(title, fontsize=24, pad=20)  # 제목 폰트 크기와 패딩 조정

    # 축 눈금 크기 조정
    ax.tick_params(axis='both', which='major', labelsize=16, width=2, length=6)
    ax.tick_params(axis='both', which='minor', width=1.5, length=4)

    # 그래프 내부 상단에 텍스트 배치 (각 변수별로, 겹치지 않도록)
    for i, txt in enumerate(df[label_col]):
        annot = f"{OR[i]:.2f} ({lo[i]:.2f}, {hi[i]:.2f})"
        # 각 점의 에러바와 겹치지 않도록 위치 계산
        max_error_x = max(hi[i], OR[i] * 1.2)  # 에러바 끝점 또는 OR*1.2 중 큰 값
        text_x = min(max_error_x * 1.3, xlim[1] * 0.85)  # 에러바에서 30% 떨어진 위치
        ax.text(text_x, y_pos[i], annot,
                va='center', ha='left', fontsize=18,
                weight='normal')  # 폰트 굵기 조정 옵션

    # 여백 조정
    plt.subplots_adjust(left=0.3, right=0.95, top=0.9, bottom=0.1)
    plt.tight_layout()
    return fig


def variable_stats(data: pd.DataFrame, features_path: str):
    import numpy as np
    cfg = load_config(features_path)
    results = []

    i_columns = cfg["initial_columns"]["followup"]["numeric"]
    d_columns = cfg["derived_columns"]["followup"]["numeric"]
    bsid2 = cfg["initial_columns"]["followup"]["BSID"]["BSID2"]
    bsid3 = cfg["initial_columns"]["followup"]["BSID"]["BSID3"]
    kdst = cfg["initial_columns"]["followup"]["KDST"]
    group_col = "bmi_group"  # cfg["derived_columns"]["label"][0]

    features = data.columns.tolist()  # i_columns + d_columns + bsid2 + bsid3 + kdst
    logging.info(f"dtype:: {data[features].dtypes}")
    group_values = [g for g in data[group_col].dropna().unique()]

    for col in features:
        if col not in data.columns or col == group_col:
            continue
        col_data = data[[col, group_col]]
        null_count = col_data[col].isnull().sum()
        count = len(col_data)
        null_ratio = null_count / count if count > 0 else np.nan

        # 범주형 변수
        if data[col].dtype == 'object' or len(data[col].unique()) < 10:
            # 통계검정: chi2
            try:
                contingency = pd.crosstab(col_data[col], col_data[group_col])
                chi2, p, _, _ = stats.chi2_contingency(contingency)
                test_used = 'chi2'
            except Exception:
                p = np.nan
                test_used = 'chi2 (fail)'
            # 각 값별로 빈도/비율
            value_list = col_data[col].dropna().unique()
            for v in value_list:
                row = {
                    'variable': col,
                    'var_type': 'categorical',
                    'test_used': test_used,
                    'p_value': p,
                    'value': v,
                    'count': count,
                    'null_count': null_count,
                    'null_ratio': null_ratio
                }
                for g in group_values:
                    group_df = col_data[col_data[group_col] == g][col]
                    group_total = group_df.shape[0]
                    value_count = (group_df == v).sum()
                    value_percent = value_count / group_total if group_total > 0 else np.nan
                    row[f'group_{g}_count'] = value_count
                    row[f'group_{g}_percent'] = value_percent
                results.append(row)
        else:
            # 연속형 변수
            # 그룹별 통계
            group_stats = {}
            for g in group_values:
                group_df = col_data[col_data[group_col] == g][col].dropna()
                group_stats[f'group_{g}_n'] = group_df.count()
                group_stats[f'group_{g}_mean'] = group_df.mean()
                group_stats[f'group_{g}_std'] = group_df.std()
                group_stats[f'group_{g}_median'] = group_df.median()
                group_stats[f'group_{g}_min'] = group_df.min()
                group_stats[f'group_{g}_max'] = group_df.max()
                group_stats[f'group_{g}_25%'] = group_df.quantile(0.25)
                group_stats[f'group_{g}_75%'] = group_df.quantile(0.75)
            # 통계검정
            groups = [col_data[col_data[group_col] == g]
                      [col].dropna().values for g in group_values]
            try:
                if len(groups) == 2:
                    _, p = stats.ttest_ind(
                        groups[0], groups[1], nan_policy='omit')
                    test_used = 't-test'
                else:
                    _, p = stats.f_oneway(*groups)
                    test_used = 'ANOVA'
            except Exception:
                p = np.nan
                test_used = 't-test/ANOVA (fail)'
            row = {
                'variable': col,
                'var_type': 'continuous',
                'test_used': test_used,
                'p_value': p,
                'count': count,
                'null_count': null_count,
                'null_ratio': null_ratio
            }
            row.update(group_stats)
            results.append(row)
    return pd.DataFrame(results)


def Univariate_Logit(
        data: pd.DataFrame,
        config_path: str,
        features_path: str,
        pos_label,        # e.g. 1 (Abnormal)
        neg_label,        # e.g. 0 (Normal)
        maxiter: int = 1000):
    """
    Univariate Binomial Logistic Regression (statsmodels.Logit)
    각 변수별로 개별적인 이항 로지스틱 회귀를 실행
    ------------------------------------------------
    Parameters
    ----------
    data          : DataFrame  (원 데이터; 범주형·연속형·라벨 포함)
    config_path   : str        (load_config 에서 읽을 YAML 경로)
    features_path : str        (feature 목록 YAML 경로)
    pos_label     : int        양성 클래스 값  (1)
    neg_label     : int        음성 클래스 값  (0, reference)
    maxiter       : int        최대 반복 (default 1000)

    Returns
    -------
    results : dict  각 변수별 statsmodels 결과를 담은 딕셔너리
    """

    # 1. 설정 로드
    cfg = load_config(config_path)
    feat_cfg = load_config(features_path)

    cats = feat_cfg["elastic_column_20250831"]["category"]
    nums = feat_cfg["elastic_column_20250831"]["continuous"]
    target = feat_cfg["derived_columns"]["label"]   # ex) ['label']

    # 2. 결측 제거 및 단위 변환(필요 시)
    subset = data[cats + nums + target].dropna()

    scale_vars = ["iarvppd", "niarvrpd",
                  "aoxyuppd", "niarvhfnc", "invfpod", "stday"]
    for v in scale_vars:
        if v in subset.columns:
            subset[v] = subset[v] / 7
    if "bwei" in subset.columns:
        subset["bwei"] = subset["bwei"] / 1000

    # 3. y 인코딩: pos=1, neg=0
    subset["y_bin"] = (subset[target[0]] == pos_label).astype(int)
    y = subset["y_bin"]

    # 4. 각 변수별로 univariate logistic regression 실행
    results = {}
    all_vars = cats + nums

    for var in all_vars:
        try:
            logging.info(f"[Univariate Logit] '{var}' 시작")

            # 해당 변수만 선택
            if var in cats:
                # 범주형 변수: 더미 변수 생성
                X_var = pd.get_dummies(subset[[var]], columns=[
                                       var], drop_first=True).astype(float)
            else:
                # 연속형 변수: 그대로 사용
                X_var = subset[[var]].astype(float)

            # 상수항 추가
            X = sm.add_constant(X_var, has_constant="add")

            # 모델 적합
            model = sm.Logit(y, X)
            result = model.fit(maxiter=maxiter, disp=False)

            results[var] = result

            logging.info(
                f"[Univariate Logit] '{var}' 완료 (converged: {result.mle_retvals['converged']})")

        except Exception as e:
            logging.error(f"[Univariate Logit] '{var}' 에러: {e}")
            results[var] = None

    logging.info(
        f"[Univariate Logit] 총 {len([r for r in results.values() if r is not None])}/{len(all_vars)} 변수 성공")

    return results


def logit_to_df_univariate(results, prefix="Abnormal"):
    """Univariate statsmodels Logit 결과들 → DataFrame"""
    rows = []

    for var_name, result in results.items():
        if result is None:
            continue

        try:
            coefs = result.params
            conf = result.conf_int()
            pvals = result.pvalues

            # const는 제외하고 실제 변수들만 처리
            for var in coefs.index:
                if var == "const":
                    continue

                rows.append({
                    "Variable": var_name,
                    "display_name": var,
                    f"{prefix}_OR": np.exp(coefs[var]),
                    f"{prefix}_CI_low": np.exp(conf.loc[var, 0]),
                    f"{prefix}_CI_high": np.exp(conf.loc[var, 1]),
                    f"{prefix}_p": pvals[var]
                })

        except Exception as e:
            logging.error(f"[logit_to_df_univariate] '{var_name}' 변환 에러: {e}")

    return pd.DataFrame(rows)


SAFE_NAME = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')


def qwrap(var: str) -> str:
    """formula 에 넣기에 안전하지 않으면 Q("…")로 감싼다."""
    return var if SAFE_NAME.fullmatch(var) else f'Q("{var}")'


def Sensitivityanalysis(data: pd.DataFrame, features_path: str) -> pd.DataFrame:
    cfg = load_config(features_path)

    # (1) 전체 변수
    cat_cols = cfg["elastic_column_20250610"]["category"]
    cont_cols = cfg["elastic_column_20250610"]["continuous"]
    all_columns = cat_cols + cont_cols

    # (2) 관심 변수
    int_cat = cfg["confounders_20250610"]["category"]
    int_con = cfg["confounders_20250610"]["continuous"]
    interest_vars = int_cat + int_con

    data['resuo'] = data['resuo'].astype('Int64')      # 1, 2 → Int
    data['resui'] = data['resui'].astype('Int64')

    # (3) 혼란변수 목록 (관심 변수 제외)
    confounders = [c for c in all_columns if c not in interest_vars]
    results = []

    for conf in tqdm(confounders):
        model_vars = interest_vars + [conf]

        # ---------- 더미 인코딩 ----------
        df_enc = data.copy()
        sel_cols = []
        for v in model_vars:
            if v in cat_cols:
                dmy = pd.get_dummies(df_enc[v], prefix=v, drop_first=True)
                df_enc = pd.concat([df_enc, dmy], axis=1)
                sel_cols.extend(dmy.columns)
            else:
                sel_cols.append(v)

        # ---------- formula ----------
        safe_cols = [qwrap(c) for c in sel_cols]          # 안전성 검사 → Q() 감싸기
        formula = 'label ~ ' + ' + '.join(safe_cols)

        try:
            mdl = smf.mnlogit(formula, data=df_enc).fit(disp=0)

            # MNLogit → params: (행 = 변수, 열 = (k-1)개의 outcome)
            for iv in interest_vars:
                # 열 이름이 iv 자체이거나 'iv_level' 더미일 수 있음
                rows = [r for r in mdl.params.index if r.startswith(iv)]
                for r in rows:
                    OR = np.exp(mdl.params.loc[r])
                    p = mdl.pvalues.loc[r]
                    results.append(
                        {'confounder': conf,
                         'param':      r,
                         'OR':         OR.to_dict(),      # outcome 별 OR
                         'p_value':    p.to_dict()}
                    )
            result_df = pd.DataFrame(results)
            rows = []
            for _, row in result_df.iterrows():
                or_dict = row['OR']       # {0: OR0, 1: OR1, …}
                p_dict = row['p_value']  # {0: p0,  1: p1, …}

                for out in or_dict.keys():  # outcome code (0, 1, …)
                    rows.append({
                        'confounder': row['confounder'],
                        'param': row['param'],
                        'outcome': out,           # 모델 내부 outcome 코드
                        'OR': or_dict[out],
                        'p_value': p_dict[out]
                    })

            long_df = pd.DataFrame(rows)
        except Exception as e:
            print(f"모델 실패: {conf}, 에러: {e}")
            continue

    return long_df


def adjusted_logit(data: pd.DataFrame, features_path: str) -> pd.DataFrame:
    cfg = load_config(features_path)

    # ① 전체 컬럼 목록 ─────────────────────────────────────────
    cat_cols_cfg = cfg["elastic_column_20250610"]["category"]
    cont_cols = cfg["elastic_column_20250610"]["continuous"]
    label = cfg["derived_columns"]["label"][0]               # ex) 'label'
    all_columns = cat_cols_cfg + cont_cols                      # ★ 전체
    data = data[all_columns + [label]].copy()            # 깊은 복사

    # ② 혼란변수 목록 ─────────────────────────────────────────
    conf_cat = cfg["confounders_20250610"]["category"]
    conf_con = cfg["confounders_20250610"]["continuous"]
    confounders = conf_cat + conf_con                            # ★ 혼란 변수

    # ③ 관심(주효과) 변수 = 전체 − 혼란 ───────────────────────
    exposure_vars = [c for c in all_columns if c not in confounders]

    # ④ 정수형 범주를 category 로 변환
    force_cat = ['resuo', 'resui', 'atbyn', 'phud1']
    for v in force_cat:
        if v in data.columns:
            data[v] = data[v].astype('Int64').astype('category')
            if v not in cat_cols_cfg:
                cat_cols_cfg.append(v)

    # ⑤ 더미 생성
    df_enc = pd.get_dummies(data, columns=cat_cols_cfg, drop_first=True)

    # ⑥ formula 작성
    X_cols = []
    for v in exposure_vars + confounders:        # ← ★ 두 리스트 모두 X 로 사용
        if v in cat_cols_cfg:
            X_cols += [c for c in df_enc.columns if c.startswith(f'{v}_')]
        else:
            X_cols.append(v)
    formula = f'{label} ~ ' + ' + '.join(sorted(set(X_cols)))

    # ⑦ 모델 적합
    mdl = smf.mnlogit(formula, data=df_enc).fit(method='newton',
                                                maxiter=500, disp=0)

    # ❹ Adjusted OR, 95 % CI, p-value 추출 ──────────────────────
    rows, z = [], 1.96
    outcome_map = {0: "Low_vs_Normal", 1: "High_vs_Normal"}   # MNLogit 의 두 컬럼

    # ★ exposure + confounder **전체**를 순회
    for var in exposure_vars + confounders:

        # ── ① 모델 계수 이름(더미 포함) 수집
        if var in cat_cols_cfg:                                     # 범주형
            terms = [r for r in mdl.params.index if r.startswith(f'{var}_')]
        else:                                                       # 연속형
            terms = [var] if var in mdl.params.index else []

        # 추정 안 된 변수(변이 0·공선성)
        if not terms:
            logging.info(f"  ※ {var} 계수 없음 (드롭)")
            continue

        # ── ② outcome (Low·High) 별 OR·CI·p 계산
        for term in terms:
            for out in mdl.params.columns:                          # 0,1
                coef = mdl.params.loc[term, out]
                se = mdl.bse.loc[term, out]
                OR = np.exp(coef)
                CI_lo, CI_hi = np.exp(coef - z*se), np.exp(coef + z*se)
                p = mdl.pvalues.loc[term, out]

                rows.append({
                    'variable': var,
                    'parameter': term,
                    'comparison': outcome_map.get(out, f'Outcome_{out}'),
                    'OR_adj': OR,
                    'CI_95_low': CI_lo,
                    'CI_95_high': CI_hi,
                    'CI_95%': f'{CI_lo:.3f} – {CI_hi:.3f}',
                    'p_adj': p,
                    'significant': p < 0.05
                })

    adj_df = (pd.DataFrame(rows)
                .sort_values(['variable', 'comparison', 'parameter'])
                .reset_index(drop=True))

    # ───────────────────────────────────── ⑨ 요약 로그
    if not adj_df.empty:
        logging.info(f"총 {len(adj_df)} 행(계수) 추출, "
                     f"유의(p < 0.05) {adj_df['significant'].sum()} 행")

        sig = adj_df[adj_df['significant']].head(10)
        if not sig.empty:
            logging.info("유의 상위 10개:")
            for _, r in sig.iterrows():
                logging.info(f"  {r['variable']} - {r['comparison']}: "
                             f"OR={r['OR_adj']:.2f}  p={r['p_adj']:.4g}")

    return adj_df


###################################################
################ 변수선택 방법들 모음 ################
###################################################

def select_features_rfe(
    X: pd.DataFrame, y: pd.Series, n_features: int = 10, random_state: int = 42
) -> list:
    """
    RFE(Recursive Feature Elimination)로 변수 선택
    Args:
        X: feature DataFrame
        y: target Series
        n_features: 최종 선택할 변수 개수
    Returns:
        선택된 변수명 리스트
    """
    model = LogisticRegression(max_iter=1000, random_state=random_state)
    selector = RFE(model, n_features_to_select=n_features)
    selector.fit(X, y)
    selected = X.columns[selector.support_].tolist()
    return selected


def select_features_rf_importance(
    X: pd.DataFrame, y: pd.Series, top_n: int = 10, random_state: int = 42
) -> list:
    """
    랜덤포레스트 변수 중요도 기반 상위 변수 선택
    Args:
        X: feature DataFrame
        y: target Series
        top_n: 상위 n개 변수 선택
    Returns:
        선택된 변수명 리스트
    """
    model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    model.fit(X, y)
    importances = pd.Series(model.feature_importances_, index=X.columns)
    selected = importances.sort_values(
        ascending=False).head(top_n).index.tolist()
    return selected


def select_features_elasticnet_stability(
    X: pd.DataFrame, y: pd.Series, n_repeats: int = 10, random_state: int = 42
) -> list:
    """
    ElasticNet을 여러 번 반복하여 자주 선택되는 변수만 남김
    Args:
        X: feature DataFrame
        y: target Series
        n_repeats: 반복 횟수
    Returns:
        선택된 변수명 리스트
    """
    np.random.seed(random_state)
    selected_counts = pd.Series(0, index=X.columns)
    for i in range(n_repeats):
        model = LogisticRegressionCV(
            Cs=10, cv=5, penalty='elasticnet', solver='saga', l1_ratios=[0.5],
            max_iter=1000, random_state=random_state + i, n_jobs=-1
        )
        model.fit(X, y)
        coef = np.abs(model.coef_).mean(axis=0)
        selected_counts += (coef > 1e-5)
    # 50% 이상 반복에서 선택된 변수만 남김
    selected = selected_counts[selected_counts >=
                               n_repeats // 2].index.tolist()
    return selected


def feature_selection(
    X: pd.DataFrame, y: pd.Series, method: str = "rfe", **kwargs
) -> list:
    """
    다양한 변수 선택 방법을 통합적으로 제공
    method: 'rfe', 'rf', 'elasticnet'
    """
    if method == "rfe":
        return select_features_rfe(X, y, **kwargs)
    elif method == "rf":
        return select_features_rf_importance(X, y, **kwargs)
    elif method == "elasticnet":
        return select_features_elasticnet_stability(X, y, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")


def backward_selection(df, features_path, model_type='auto', criterion='bic', alpha=0.1):
    """
    features.yaml의 구조를 참고하여 backward selection을 수행하는 함수
    - model_type: 'logit'(이항), 'mnlogit'(다항), 'ols'(연속형)
    - criterion: 'aic' 또는 'bic'
    """
    import statsmodels.api as sm
    from src.utils.utils import load_config
    import numpy as np
    import logging

    features_config = load_config(features_path)
    label = features_config["derived_columns"]["label"][0]
    vars_ = features_config["elastic_column_20250822"]["category"] + \
        features_config["elastic_column_20250822"]["continuous"]

    X = df[vars_].copy()
    y = df[label]
    X = sm.add_constant(X)
    cols = list(X.columns)

    # 클래스 개수에 따라 자동 선택
    if model_type == 'auto':
        if y.nunique() > 2:
            model_type = 'mnlogit'
        else:
            model_type = 'logit'

    def get_model(X, y):
        if model_type == 'logit':
            return sm.Logit(y, X).fit(disp=0)
        elif model_type == 'ols':
            return sm.OLS(y, X).fit()
        elif model_type == 'mnlogit':
            return sm.MNLogit(y, X).fit(method='newton', disp=0)
        else:
            raise ValueError('model_type은 logit, mnlogit, ols만 지원')

    def get_criterion(model):
        if criterion == 'aic':
            return model.aic
        elif criterion == 'bic':
            return model.bic
        else:
            raise ValueError('criterion은 aic 또는 bic만 지원')

    best_crit = np.inf
    best_cols = cols.copy()
    improved = True

    while improved and len(cols) > 1:
        improved = False
        crits = []
        for c in cols:
            if c == 'const':
                continue
            test_cols = [col for col in cols if col != c]
            try:
                model = get_model(X[test_cols], y)
                crit = get_criterion(model)
                crits.append((crit, c, model))
            except Exception as e:
                continue
        if not crits:
            break
        crits.sort()
        min_crit, remove_col, best_model = crits[0]
        if min_crit < best_crit:
            best_crit = min_crit
            cols.remove(remove_col)
            best_cols = cols.copy()
            improved = True
            logging.info(
                f"변수 '{remove_col}' 제거, {criterion.upper()}={min_crit:.3f}")
        else:
            break

    # 최종 모델 적합
    final_model = get_model(X[best_cols], y)
    logging.info(f"최종 선택 변수: {best_cols}")
    logging.info(f"최종 {criterion.upper()}: {get_criterion(final_model):.3f}")
    return best_cols, final_model


# ==============================================
# 단변량 분석 모듈
# ==============================================

def analyze_univariate_continuous(df, group_col, variables):
    """
    연속형 변수에 대한 단변량 분석

    Parameters:
    -----------
    df : pd.DataFrame
    group_col : str
    variables : list

    Returns:
    --------
    pd.DataFrame : 분석 결과
    """
    results = []
    groups = df[group_col].unique()

    for var in variables:
        if var not in df.columns:
            continue

        # 전체 통계
        overall_stats = {
            'variable': var,
            'group': 'Overall',
            'n': df[var].count(),
            'mean': df[var].mean().round(2),
            'std': df[var].std().round(2),
            'median': df[var].median(),
            'min': df[var].min(),
            'max': df[var].max(),
            'q25': df[var].quantile(0.25),
            'q75': df[var].quantile(0.75)
        }

        # 그룹별 통계
        group_stats = []
        for group in groups:
            group_data = df[df[group_col] == group][var].dropna()
            if len(group_data) > 0:
                group_stats.append({
                    'variable': var,
                    'group': str(group),
                    'n': len(group_data),
                    'mean': group_data.mean().round(2),
                    'std': group_data.std().round(2),
                    'median': group_data.median(),
                    'min': group_data.min(),
                    'max': group_data.max(),
                    'q25': group_data.quantile(0.25),
                    'q75': group_data.quantile(0.75)
                })

        # 통계 검정
        group_data_list = [df[df[group_col] == group][var].dropna()
                           for group in groups]
        group_data_list = [data for data in group_data_list if len(data) > 0]

        if len(group_data_list) >= 2:
            try:
                # 등분산성 검정
                _, levene_p = stats.levene(*group_data_list)

                if levene_p >= 0.05:  # 등분산 가정 만족
                    _, p_value = stats.f_oneway(*group_data_list)
                    test_name = 'ANOVA'
                else:  # 등분산 가정 불만족
                    _, p_value = stats.kruskal(*group_data_list)
                    test_name = 'Kruskal-Wallis'

            except:
                p_value = np.nan
                test_name = 'Failed'
                levene_p = np.nan
        else:
            p_value = np.nan
            test_name = 'Insufficient groups'
            levene_p = np.nan

        # p-value를 모든 행에 추가
        overall_stats.update({
            'p_value': p_value.round(2),
            'test_name': test_name,
            'levene_p': levene_p
        })

        for stat in group_stats:
            stat.update({
                'p_value': p_value.round(2),
                'test_name': test_name,
                'levene_p': levene_p
            })

        results.extend([overall_stats] + group_stats)

    return pd.DataFrame(results)


def analyze_univariate_categorical(df, group_col, variables):
    """
    범주형 변수에 대한 단변량 분석

    Parameters:
    -----------
    df : pd.DataFrame
    group_col : str
    variables : list

    Returns:
    --------
    pd.DataFrame : 분석 결과
    """
    results = []
    groups = df[group_col].unique()

    for var in variables:
        if var not in df.columns:
            continue

        # 전체 통계
        overall_counts = df[var].value_counts()
        total_n = len(df[var].dropna())

        for category, count in overall_counts.items():
            overall_stats = {
                'variable': var,
                'group': 'Overall',
                'category': str(category),
                'n': count,
                'percent': round(count / total_n * 100, 2)
            }
            results.append(overall_stats)

        # 그룹별 통계
        for group in groups:
            group_data = df[df[group_col] == group][var].dropna()
            if len(group_data) > 0:
                group_counts = group_data.value_counts()
                group_total = len(group_data)

                for category, count in group_counts.items():
                    group_stats = {
                        'variable': var,
                        'group': str(group),
                        'category': str(category),
                        'n': count,
                        'percent': round(count / group_total * 100, 2)
                    }
                    results.append(group_stats)

        # 통계 검정 (Chi-square)
        try:
            contingency = pd.crosstab(df[group_col], df[var])
            if contingency.shape[0] >= 2 and contingency.shape[1] >= 2:
                chi2, p_value, dof, expected = stats.chi2_contingency(
                    contingency)
                test_name = 'Chi-square'
            else:
                p_value = np.nan
                test_name = 'Insufficient categories'
        except:
            p_value = np.nan
            test_name = 'Failed'

        # p-value를 해당 변수의 모든 행에 추가
        for result in results:
            if result['variable'] == var:
                result.update({
                    'p_value': p_value.round(2),
                    'test_name': test_name
                })

    return pd.DataFrame(results)


def create_tableone_style_continuous(df, group_col, variables):
    """
    연속형 변수에 대한 TableOne 스타일 분석

    Parameters:
    -----------
    df : pd.DataFrame
    group_col : str
    variables : list

    Returns:
    --------
    pd.DataFrame : TableOne 스타일 결과
    """
    results = []
    groups = sorted(df[group_col].unique())

    for var in variables:
        if var not in df.columns:
            continue

        # 전체 통계
        overall_data = df[var].dropna()
        overall_n = len(overall_data)
        overall_mean = overall_data.mean()
        overall_std = overall_data.std()

        # 그룹별 통계
        group_stats = {}
        for group in groups:
            group_data = df[df[group_col] == group][var].dropna()
            if len(group_data) > 0:
                group_stats[group] = {
                    'n': len(group_data),
                    'mean': group_data.mean(),
                    'std': group_data.std()
                }
            else:
                group_stats[group] = {
                    'n': 0,
                    'mean': np.nan,
                    'std': np.nan
                }

        # 통계 검정
        group_data_list = [df[df[group_col] == group][var].dropna()
                           for group in groups]
        group_data_list = [data for data in group_data_list if len(data) > 0]

        if len(group_data_list) >= 2:
            try:
                # 등분산성 검정
                _, levene_p = stats.levene(*group_data_list)

                if levene_p >= 0.05:  # 등분산 가정 만족
                    _, p_value = stats.f_oneway(*group_data_list)
                    test_name = 'ANOVA'
                else:  # 등분산 가정 불만족
                    _, p_value = stats.kruskal(*group_data_list)
                    test_name = 'Kruskal-Wallis'

            except:
                p_value = np.nan
                test_name = 'Failed'
        else:
            p_value = np.nan
            test_name = 'Insufficient groups'

        # 결과 행 생성
        result_row = {
            'Variable': var,
            'Overall': f"{overall_mean:.1f} ({overall_std:.1f})",
            'Overall_n': overall_n
        }

        # 그룹별 결과 추가
        for group in groups:
            stats = group_stats[group]
            if stats['n'] > 0:
                result_row[f'{group}'] = f"{stats['mean']:.1f} ({stats['std']:.1f})"
                result_row[f'{group}_n'] = stats['n']
            else:
                result_row[f'{group}'] = "NA"
                result_row[f'{group}_n'] = 0

        # p-value 추가
        if p_value < 0.001:
            result_row['p_value'] = '<0.001'
        elif p_value < 0.05:
            result_row['p_value'] = f"{p_value:.3f}"
        else:
            result_row['p_value'] = f"{p_value:.3f}"

        result_row['test_name'] = test_name
        results.append(result_row)

    return pd.DataFrame(results)


def create_tableone_style_categorical(df, group_col, variables):
    """
    범주형 변수에 대한 TableOne 스타일 분석

    Parameters:
    -----------
    df : pd.DataFrame
    group_col : str
    variables : list

    Returns:
    --------
    pd.DataFrame : TableOne 스타일 결과
    """
    results = []
    groups = sorted(df[group_col].unique())

    for var in variables:
        if var not in df.columns:
            continue

        # 전체 통계
        overall_counts = df[var].value_counts()
        total_n = len(df[var].dropna())

        # 그룹별 통계
        group_counts = {}
        for group in groups:
            group_data = df[df[group_col] == group][var].dropna()
            if len(group_data) > 0:
                group_counts[group] = group_data.value_counts()
            else:
                group_counts[group] = pd.Series()

        # 통계 검정 (Chi-square)
        try:
            contingency = pd.crosstab(df[group_col], df[var])
            if contingency.shape[0] >= 2 and contingency.shape[1] >= 2:
                chi2, p_value, dof, expected = stats.chi2_contingency(
                    contingency)
                test_name = 'Chi-square'
            else:
                p_value = np.nan
                test_name = 'Insufficient categories'
        except:
            p_value = np.nan
            test_name = 'Failed'

        # 각 카테고리별로 결과 행 생성
        for category in overall_counts.index:
            result_row = {
                'Variable': f"{var} ({category})",
                'Overall': f"{overall_counts[category]} ({overall_counts[category]/total_n*100:.1f}%)",
                'Overall_n': total_n
            }

            # 그룹별 결과 추가
            for group in groups:
                if category in group_counts[group]:
                    count = group_counts[group][category]
                    group_total = len(df[df[group_col] == group][var].dropna())
                    if group_total > 0:
                        result_row[f'{group}'] = f"{count} ({count/group_total*100:.1f}%)"
                        result_row[f'{group}_n'] = group_total
                    else:
                        result_row[f'{group}'] = "0 (0.0%)"
                        result_row[f'{group}_n'] = 0
                else:
                    result_row[f'{group}'] = "0 (0.0%)"
                    result_row[f'{group}_n'] = len(
                        df[df[group_col] == group][var].dropna())

            # p-value 추가
            if p_value < 0.001:
                result_row['p_value'] = '<0.001'
            elif p_value < 0.05:
                result_row['p_value'] = f"{p_value:.3f}"
            else:
                result_row['p_value'] = f"{p_value:.3f}"

            result_row['test_name'] = test_name
            results.append(result_row)

    return pd.DataFrame(results)


def create_tableone_style_summary(df, group_col, continuous_vars=None, categorical_vars=None):
    """
    TableOne 스타일의 통합 요약 테이블 생성

    Parameters:
    -----------
    df : pd.DataFrame
    group_col : str
    continuous_vars : list, optional
    categorical_vars : list, optional

    Returns:
    --------
    pd.DataFrame : 통합된 TableOne 스타일 결과
    """
    results = []

    # 연속형 변수 분석
    if continuous_vars:
        continuous_results = create_tableone_style_continuous(
            df, group_col, continuous_vars)
        continuous_results['var_type'] = 'continuous'
        results.append(continuous_results)

    # 범주형 변수 분석
    if categorical_vars:
        categorical_results = create_tableone_style_categorical(
            df, group_col, categorical_vars)
        categorical_results['var_type'] = 'categorical'
        results.append(categorical_results)

    if results:
        combined_results = pd.concat(results, ignore_index=True)
        return combined_results
    else:
        return pd.DataFrame()


def create_followup_basic_plots(combined_results, group_col='group', kdst_2year_vars=None, kdst_3year_vars=None):
    """
    통합된 follow-up 데이터를 사용해서 기본적인 그래프 생성

    Parameters:
    -----------
    combined_results : pd.DataFrame - 통합된 follow-up 결과 데이터 (age 컬럼 포함)
    group_col : str - 그룹 컬럼명

    Returns:
    --------
    matplotlib.figure.Figure : 기본 그래프
    """
    fig, axes = plt.subplots(5, 6, figsize=(24, 20))
    fig.suptitle('Follow-up Outcomes by BMI Group',
                 fontsize=14, fontweight='bold', y=0.98)

    colors = {'Normal': '#2E8B57', 'Low': '#FF6B6B', 'High': '#4ECDC4'}

    # 1행: BMI 그룹별 분포 (2세, 3세) + 추가 변수들
    # 2세 그룹별 분포 (wt1 기준)
    ax1 = axes[0, 0]
    data_2year_wt1 = combined_results[(combined_results['age'] == '2year') & (
        combined_results['variable'] == 'wt1')]
    group_counts_2 = data_2year_wt1.set_index('group')[
        'n']
    total_2year = group_counts_2.sum()
    group_percentages_2 = (group_counts_2 / total_2year * 100).round(1)

    wedges1, texts1, autotexts1 = ax1.pie(group_counts_2.values, labels=group_counts_2.index,
                                          autopct='%1.1f%%', colors=[colors.get(g, '#808080') for g in group_counts_2.index])
    ax1.set_title('BMI Group Distribution (2-year)',
                  fontsize=12, fontweight='bold')

    # 3세 그룹별 분포 (wt2 기준)
    ax2 = axes[0, 1]
    data_3year_wt2 = combined_results[(combined_results['age'] == '3year') & (
        combined_results['variable'] == 'wt2')]
    group_counts_3 = data_3year_wt2.set_index('group')[
        'n']
    total_3year = group_counts_3.sum()
    group_percentages_3 = (group_counts_3 / total_3year * 100).round(1)

    wedges2, texts2, autotexts2 = ax2.pie(group_counts_3.values, labels=group_counts_3.index,
                                          autopct='%1.1f%%', colors=[colors.get(g, '#808080') for g in group_counts_3.index])
    ax2.set_title('BMI Group Distribution (3-year)',
                  fontsize=12, fontweight='bold')

    # 나머지 첫 번째 행 (3-6번째 위치)는 완전히 숨김
    for i in range(2, 6):
        ax = axes[0, i]
        ax.axis('off')  # 축 완전히 숨기기
        ax.set_visible(False)  # subplot 자체를 안보이게 하기

    # 2행: Weight/Height/BMI 관련 변수들 (6개 모두)
    second_row_vars = ['wt1', 'wt1_zscore', 'ht1',
                       'ht1_zscore', 'bmi1', 'bmi1_zscore']
    second_row_titles = [
        'Weight (kg)', 'Weight Z-score', 'Height (cm)', 'Height Z-score', 'BMI (kg/m²)', 'BMI Z-score']

    for i, (var, title) in enumerate(zip(second_row_vars, second_row_titles)):
        ax = axes[1, i]

        # 2year와 3year 데이터를 각각 추출
        var_2year = var
        var_3year = var.replace(
            '1', '2') if '1' in var else var.replace('2', '1')

        data_2year = combined_results[(combined_results['age'] == '2year') &
                                      (combined_results['variable'] == var_2year)]
        data_3year = combined_results[(combined_results['age'] == '3year') &
                                      (combined_results['variable'] == var_3year)]

        if not data_2year.empty and not data_3year.empty:
            x_pos = [0, 1]  # 2-year, 3-year (더 가운데로 이동)

            # 각 그룹별로 error bar 그리기
            for group in ['Normal', 'Low', 'High']:
                group_data_2year = data_2year[data_2year[group_col] == group]
                group_data_3year = data_3year[data_3year[group_col] == group]

                means = []
                sems = []

                # 2year 데이터
                if not group_data_2year.empty:
                    means.append(group_data_2year['mean'].iloc[0])
                    sems.append(group_data_2year['std'].iloc[0])
                else:
                    means.append(np.nan)
                    sems.append(np.nan)

                # 3year 데이터
                if not group_data_3year.empty:
                    means.append(group_data_3year['mean'].iloc[0])
                    sems.append(group_data_3year['std'].iloc[0])
                else:
                    means.append(np.nan)
                    sems.append(np.nan)

                # Error bar 그리기
                if not any(np.isnan(means)):
                    ax.errorbar(x_pos, means, yerr=sems, marker='o',
                                color=colors.get(group, '#808080'), label=group, capsize=5,
                                linewidth=2, markersize=8)

            ax.set_xlabel('Age', fontsize=10)
            ax.set_ylabel(title.split('(')[0].strip(), fontsize=10)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(['2-year', '3-year'])
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlim(-0.5, 1.5)  # x축 범위 조정으로 더 가운데 집중
        else:
            ax.text(0.5, 0.5, f'No data found for {var}', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12)
            ax.set_title(title, fontsize=12, fontweight='bold')

    # 3행: KDST 변수들 (2-year)
    # 4행: KDST 변수들 (3-year)
    if kdst_2year_vars and kdst_3year_vars:

        # KDST 변수명과 제목 매핑
        kdst_var_names = {
            'dgmtr1': 'Gross Motor', 'dfmtr1': 'Fine Motor', 'rctr1': 'Cognitive',
            'lgtr1': 'Language', 'sctr1': 'Social', 'shtr1': 'Self-help'
        }

        # kdst1은 제외하고 나머지 변수들만 처리
        filtered_kdst_vars = [
            var for var in kdst_2year_vars if 'kdst1' not in var.lower()]

        # 3행: 2-year KDST 변수들
        for i, var in enumerate(filtered_kdst_vars):
            if i >= 6:
                break

            ax = axes[2, i]
            var_2year = var
            data_2year = combined_results[(combined_results['age'] == '2year') &
                                          (combined_results['variable'] == var_2year)]

            if not data_2year.empty:
                # 범주형 변수의 카테고리들 추출
                categories_2year = data_2year['category'].unique()
                all_categories = sorted(list(categories_2year))

                if len(all_categories) > 0:
                    # Simple Stacked Bar plot (BMI 그룹별)
                    x = np.arange(len(all_categories))
                    width = 0.7
                    bottom = np.zeros(len(all_categories))

                    # 전체 데이터에서 각 카테고리-그룹 조합의 비율 계산
                    total_data_count = data_2year['n'].sum(
                    ) if 'n' in data_2year.columns else len(data_2year)
                    group_percentages = {'Normal': [], 'Low': [], 'High': []}

                    for cat in all_categories:
                        cat_data = data_2year[data_2year['category'] == cat]

                        # 각 그룹별 데이터 수집
                        for group in ['Normal', 'Low', 'High']:
                            cat_group_data = cat_data[cat_data[group_col] == group]
                            if not cat_group_data.empty:
                                count = cat_group_data['n'].iloc[0] if 'n' in cat_group_data.columns else 0
                                # 전체 데이터 대비 비율 계산
                                percentage = (
                                    count / total_data_count * 100) if total_data_count > 0 else 0
                                group_percentages[group].append(percentage)
                            else:
                                group_percentages[group].append(0)

                    # Stacked bar 그리기
                    for group in ['Normal', 'Low', 'High']:
                        values = group_percentages[group]
                        bars = ax.bar(x, values, width, bottom=bottom,
                                      label=group, color=colors.get(
                                          group, '#808080'),
                                      alpha=0.8, edgecolor='white', linewidth=1)

                        # 각 스택 위에 퍼센트 표시 (15% 이상인 경우만)
                        for j, (bar, value) in enumerate(zip(bars, values)):
                            if value >= 15:
                                height = bottom[j] + value/2
                                ax.text(bar.get_x() + bar.get_width()/2., height,
                                        f'{value:.0f}%', ha='center', va='center',
                                        fontsize=9, fontweight='bold', color='white')

                        bottom += values

                    ax.set_xlabel('Categories', fontsize=10)
                    ax.set_ylabel('Percentage (%)', fontsize=10)
                    ax.set_title(f'{kdst_var_names.get(var, var)} (2-year)',
                                 fontsize=12, fontweight='bold')
                    ax.set_xticks(x)
                    ax.set_xticklabels(all_categories, rotation=45, ha='right')
                    ax.legend(fontsize=8)
                    ax.grid(True, alpha=0.3, axis='y')
                else:
                    ax.text(0.5, 0.5, f'No categories found for {var}', ha='center', va='center',
                            transform=ax.transAxes, fontsize=12)
                    ax.set_title(f'{kdst_var_names.get(var, var)} (2-year)',
                                 fontsize=12, fontweight='bold')
            else:
                ax.text(0.5, 0.5, f'No data found for {var}', ha='center', va='center',
                        transform=ax.transAxes, fontsize=12)
                ax.set_title(f'{kdst_var_names.get(var, var)} (2-year)',
                             fontsize=12, fontweight='bold')

        # 4행: 3-year KDST 변수들
        for i, var in enumerate(filtered_kdst_vars):
            if i >= 6:
                break

            ax = axes[3, i]
            var_3year = var.replace(
                '1', '2') if '1' in var else var.replace('2', '1')
            data_3year = combined_results[(combined_results['age'] == '3year') &
                                          (combined_results['variable'] == var_3year)]

            if not data_3year.empty:
                # 범주형 변수의 카테고리들 추출
                categories_3year = data_3year['category'].unique()
                all_categories = sorted(list(categories_3year))

                if len(all_categories) > 0:
                    # Simple Stacked Bar plot (BMI 그룹별)
                    x = np.arange(len(all_categories))
                    width = 0.7
                    bottom = np.zeros(len(all_categories))

                    # 전체 데이터에서 각 카테고리-그룹 조합의 비율 계산
                    total_data_count = data_3year['n'].sum(
                    ) if 'n' in data_3year.columns else len(data_3year)
                    group_percentages = {'Normal': [], 'Low': [], 'High': []}

                    for cat in all_categories:
                        cat_data = data_3year[data_3year['category'] == cat]

                        # 각 그룹별 데이터 수집
                        for group in ['Normal', 'Low', 'High']:
                            cat_group_data = cat_data[cat_data[group_col] == group]
                            if not cat_group_data.empty:
                                count = cat_group_data['n'].iloc[0] if 'n' in cat_group_data.columns else 0
                                # 전체 데이터 대비 비율 계산
                                percentage = (
                                    count / total_data_count * 100) if total_data_count > 0 else 0
                                group_percentages[group].append(percentage)
                            else:
                                group_percentages[group].append(0)

                    # Stacked bar 그리기
                    for group in ['Normal', 'Low', 'High']:
                        values = group_percentages[group]
                        bars = ax.bar(x, values, width, bottom=bottom,
                                      label=group, color=colors.get(
                                          group, '#808080'),
                                      alpha=0.8, edgecolor='white', linewidth=1)

                        # 각 스택 위에 퍼센트 표시 (15% 이상인 경우만)
                        for j, (bar, value) in enumerate(zip(bars, values)):
                            if value >= 15:
                                height = bottom[j] + value/2
                                ax.text(bar.get_x() + bar.get_width()/2., height,
                                        f'{value:.0f}%', ha='center', va='center',
                                        fontsize=9, fontweight='bold', color='white')

                        bottom += values

                    ax.set_xlabel('Categories', fontsize=10)
                    ax.set_ylabel('Percentage (%)', fontsize=10)
                    ax.set_title(f'{kdst_var_names.get(var, var)} (3-year)',
                                 fontsize=12, fontweight='bold')
                    ax.set_xticks(x)
                    ax.set_xticklabels(all_categories, rotation=45, ha='right')
                    ax.legend(fontsize=8)
                    ax.grid(True, alpha=0.3, axis='y')
                else:
                    ax.text(0.5, 0.5, f'No categories found for {var}', ha='center', va='center',
                            transform=ax.transAxes, fontsize=12)
                    ax.set_title(f'{kdst_var_names.get(var, var)} (3-year)',
                                 fontsize=12, fontweight='bold')
            else:
                ax.text(0.5, 0.5, f'No data found for {var}', ha='center', va='center',
                        transform=ax.transAxes, fontsize=12)
                ax.set_title(f'{kdst_var_names.get(var, var)} (3-year)',
                             fontsize=12, fontweight='bold')

    # 5행: BSID 변수들 (3개 변수 × 2년령)
    # BSID는 3개 변수 × 2년령으로 배치
    bsid_base_vars = ['cognit', 'lang', 'motor']
    bsid_var_names = {'cognit': 'Cognitive',
                      'lang': 'Language', 'motor': 'Motor'}

    # 디버깅: BSID 변수들이 combined_results에 있는지 확인
    available_vars = combined_results['variable'].unique()
    print(f"Available variables in combined_results: {sorted(available_vars)}")

    # 3개 변수 × 2년령 = 6개 위치에 배치
    for i, base_var in enumerate(bsid_base_vars):
        # 2-year 데이터 (위치: i*2)
        if i*2 < 6:
            ax = axes[4, i*2]
            var_2year = f"{base_var}1"

            print(f"Looking for BSID 2-year variable: {var_2year}")

            data_2year = combined_results[(combined_results['age'] == '2year') &
                                          (combined_results['variable'] == var_2year)]

            print(f"Data found - 2year: {len(data_2year)}")

            if not data_2year.empty:
                # Simple Stacked Bar plot (BMI 그룹별) - 2year 데이터
                categories_2year = data_2year['category'].unique()
                all_categories = sorted(list(categories_2year))

                if len(all_categories) > 0:
                    x = np.arange(len(all_categories))
                    width = 0.7
                    bottom = np.zeros(len(all_categories))

                    # 전체 데이터에서 각 카테고리-그룹 조합의 비율 계산
                    total_data_count = data_2year['n'].sum(
                    ) if 'n' in data_2year.columns else len(data_2year)
                    group_percentages = {'Normal': [], 'Low': [], 'High': []}

                    for cat in all_categories:
                        cat_data = data_2year[data_2year['category'] == cat]

                        # 각 그룹별 데이터 수집
                        for group in ['Normal', 'Low', 'High']:
                            cat_group_data = cat_data[cat_data[group_col] == group]
                            if not cat_group_data.empty:
                                count = cat_group_data['n'].iloc[0] if 'n' in cat_group_data.columns else 0
                                # 전체 데이터 대비 비율 계산
                                percentage = (
                                    count / total_data_count * 100) if total_data_count > 0 else 0
                                group_percentages[group].append(percentage)
                            else:
                                group_percentages[group].append(0)

                    # Stacked bar 그리기
                    for group in ['Normal', 'Low', 'High']:
                        values = group_percentages[group]
                        bars = ax.bar(x, values, width, bottom=bottom,
                                      label=group, color=colors.get(
                                          group, '#808080'),
                                      alpha=0.8, edgecolor='white', linewidth=1)

                        # 각 스택 위에 퍼센트 표시 (15% 이상인 경우만)
                        for j, (bar, value) in enumerate(zip(bars, values)):
                            if value >= 15:
                                height = bottom[j] + value/2
                                ax.text(bar.get_x() + bar.get_width()/2., height,
                                        f'{value:.0f}%', ha='center', va='center',
                                        fontsize=9, fontweight='bold', color='white')

                        bottom += values

                    ax.set_xlabel('Categories', fontsize=10)
                    ax.set_ylabel('Percentage (%)', fontsize=10)
                    ax.set_title(f'{bsid_var_names[base_var]} (2-year)',
                                 fontsize=12, fontweight='bold')
                    ax.set_xticks(x)
                    ax.set_xticklabels(all_categories, rotation=45, ha='right')
                    ax.legend(fontsize=8)
                    ax.grid(True, alpha=0.3, axis='y')
                else:
                    ax.text(0.5, 0.5, f'No categories found for {var_2year}', ha='center', va='center',
                            transform=ax.transAxes, fontsize=12)
                    ax.set_title(f'{bsid_var_names[base_var]} (2-year)',
                                 fontsize=12, fontweight='bold')
            else:
                ax.text(0.5, 0.5, f'No data found for {var_2year}', ha='center', va='center',
                        transform=ax.transAxes, fontsize=12)
                ax.set_title(f'{bsid_var_names[base_var]} (2-year)',
                             fontsize=12, fontweight='bold')

        # 3-year 데이터 (위치: i*2+1)
        if i*2+1 < 6:
            ax = axes[4, i*2+1]
            var_3year = f"{base_var}2"

            print(f"Looking for BSID 3-year variable: {var_3year}")

            data_3year = combined_results[(combined_results['age'] == '3year') &
                                          (combined_results['variable'] == var_3year)]

            print(f"Data found - 3year: {len(data_3year)}")

            if not data_3year.empty:
                # Simple Stacked Bar plot (BMI 그룹별) - 3year 데이터
                categories_3year = data_3year['category'].unique()
                all_categories = sorted(list(categories_3year))

                if len(all_categories) > 0:
                    x = np.arange(len(all_categories))
                    width = 0.7
                    bottom = np.zeros(len(all_categories))

                    # 전체 데이터에서 각 카테고리-그룹 조합의 비율 계산
                    total_data_count = data_3year['n'].sum(
                    ) if 'n' in data_3year.columns else len(data_3year)
                    group_percentages = {'Normal': [], 'Low': [], 'High': []}

                    for cat in all_categories:
                        cat_data = data_3year[data_3year['category'] == cat]

                        # 각 그룹별 데이터 수집
                        for group in ['Normal', 'Low', 'High']:
                            cat_group_data = cat_data[cat_data[group_col] == group]
                            if not cat_group_data.empty:
                                count = cat_group_data['n'].iloc[0] if 'n' in cat_group_data.columns else 0
                                # 전체 데이터 대비 비율 계산
                                percentage = (
                                    count / total_data_count * 100) if total_data_count > 0 else 0
                                group_percentages[group].append(percentage)
                            else:
                                group_percentages[group].append(0)

                    # Stacked bar 그리기
                    for group in ['Normal', 'Low', 'High']:
                        values = group_percentages[group]
                        bars = ax.bar(x, values, width, bottom=bottom,
                                      label=group, color=colors.get(
                                          group, '#808080'),
                                      alpha=0.8, edgecolor='white', linewidth=1)

                        # 각 스택 위에 퍼센트 표시 (15% 이상인 경우만)
                        for j, (bar, value) in enumerate(zip(bars, values)):
                            if value >= 15:
                                height = bottom[j] + value/2
                                ax.text(bar.get_x() + bar.get_width()/2., height,
                                        f'{value:.0f}%', ha='center', va='center',
                                        fontsize=9, fontweight='bold', color='white')

                        bottom += values

                    ax.set_xlabel('Categories', fontsize=10)
                    ax.set_ylabel('Percentage (%)', fontsize=10)
                    ax.set_title(f'{bsid_var_names[base_var]} (3-year)',
                                 fontsize=12, fontweight='bold')
                    ax.set_xticks(x)
                    ax.set_xticklabels(all_categories, rotation=45, ha='right')
                    ax.legend(fontsize=8)
                    ax.grid(True, alpha=0.3, axis='y')
                else:
                    ax.text(0.5, 0.5, f'No categories found for {var_3year}', ha='center', va='center',
                            transform=ax.transAxes, fontsize=12)
                    ax.set_title(f'{bsid_var_names[base_var]} (3-year)',
                                 fontsize=12, fontweight='bold')
            else:
                ax.text(0.5, 0.5, f'No data found for {var_3year}', ha='center', va='center',
                        transform=ax.transAxes, fontsize=12)
                ax.set_title(f'{bsid_var_names[base_var]} (3-year)',
                             fontsize=12, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def create_combined_followup_results(combined_results2, combined_results3):
    """
    2-year와 3-year 결과를 통합된 Long Format으로 변환

    Parameters:
    -----------
    combined_results2 : pd.DataFrame - 2-year 결과
    combined_results3 : pd.DataFrame - 3-year 결과

    Returns:
    --------
    pd.DataFrame : 통합된 결과 (age 컬럼 포함)
    """
    # 2-year 데이터에 age 컬럼 추가
    combined_results2['age'] = '2year'

    # 3-year 데이터에 age 컬럼 추가
    combined_results3['age'] = '3year'

    # 두 데이터프레임 합치기
    combined_results = pd.concat(
        [combined_results2, combined_results3], ignore_index=True)

    # 컬럼 순서 재정렬
    column_order = ['variable', 'group', 'age', 'n', 'mean', 'std', 'median', 'min', 'max',
                    'q25', 'q75', 'p_value', 'test_name', 'levene_p', 'category', 'percent']

    # 존재하는 컬럼만 선택
    existing_columns = [
        col for col in column_order if col in combined_results.columns]
    combined_results = combined_results[existing_columns]

    return combined_results


def posthoc_2year(df, features_path):
    """
    2년 추적관찰 데이터에 대해 그룹별 Games-Howell 사후 분석을 수행합니다.
    2년 데이터에 맞는 dropna 조건을 적용합니다.

    Parameters:
    - df : pd.DataFrame
    - features_path : str, 변수 정보가 담긴 yaml 경로

    Returns:
    - pd.DataFrame: Games-Howell 결과 + n_A, n_B 포함 (2년 데이터)
    """
    logging.info("======== posthoc 2년 데이터 분석 시작 ============")
    results = []
    features_config = load_config(features_path)
    group_col = features_config["derived_columns"]["label"][0]

    # 2년 데이터 컬럼들
    columns = (
        features_config["initial_columns"]["followup"]["wtht"]["wtht2"] +
        features_config["derived_columns"]["followup"]["bmi"]["bmi2"]
    )

    # 기본 필터링
    df = df[(df["corrected_agem1"] >= 18) & (df["corrected_agem1"] <= 30)
            & (df["birth_agem2"] >= 30) & (df["birth_agem2"] <= 42)]

    # 2년 데이터에 맞는 dropna 적용 (wt1, ht1, bmi1 관련)
    df = df.dropna(subset=["wt1", "ht1", "bmi1", "bmi1_zscore"])

    logging.info(f"2년 데이터 필터링 후 샘플 수: {len(df)}")

    for col in columns:
        if col == group_col:
            continue

        temp = df[[group_col, col]].dropna()
        if temp[group_col].nunique() < 2 or temp[col].nunique() < 2:
            continue

        try:
            res = pg.pairwise_gameshowell(dv=col, between=group_col, data=temp)
            res["variable"] = col
            res["data_type"] = "2year"  # 데이터 유형 표시

            # 각 그룹의 n 계산
            n_count = temp.groupby(group_col).size().reset_index()
            n_count.columns = [group_col, 'n']

            # A 그룹 n 추가
            res = res.merge(n_count, how='left', left_on=[
                            'A'], right_on=[group_col])
            res = res.rename(columns={'n': 'n_A'}).drop(columns=[group_col])

            # B 그룹 n 추가
            res = res.merge(n_count, how='left', left_on=[
                            'B'], right_on=[group_col])
            res = res.rename(columns={'n': 'n_B'}).drop(columns=[group_col])

            results.append(res)

        except Exception as e:
            print(f"[!] 2년 데이터 변수 {col} 처리 중 오류 발생: {e}")

    if results:
        result_df = pd.concat(results, axis=0, ignore_index=True)
        logging.info(f"2년 데이터 분석 완료: {len(result_df)}개 결과")
        return result_df
    else:
        return pd.DataFrame(columns=[
            "A", "B", "mean(A)", "mean(B)", "diff", "se", "T", "dof",
            "pval", "CI95%", "hedges", "variable", "data_type", "n_A", "n_B"
        ])


def posthoc_3year(df, features_path):
    """
    3년 추적관찰 데이터에 대해 그룹별 Games-Howell 사후 분석을 수행합니다.
    3년 데이터에 맞는 dropna 조건을 적용합니다.

    Parameters:
    - df : pd.DataFrame
    - features_path : str, 변수 정보가 담긴 yaml 경로

    Returns:
    - pd.DataFrame: Games-Howell 결과 + n_A, n_B 포함 (3년 데이터)
    """
    logging.info("======== posthoc 3년 데이터 분석 시작 ============")
    results = []
    features_config = load_config(features_path)
    group_col = features_config["derived_columns"]["label"][0]

    # 3년 데이터 컬럼들
    columns = (
        features_config["initial_columns"]["followup"]["wtht"]["wtht3"] +
        features_config["derived_columns"]["followup"]["bmi"]["bmi3"]
    )

    # 기본 필터링
    df = df[(df["corrected_agem1"] >= 18) & (df["corrected_agem1"] <= 30)
            & (df["birth_agem2"] >= 30) & (df["birth_agem2"] <= 42)]

    # 3년 데이터에 맞는 dropna 적용 (wt2, ht2, bmi2 관련)
    df = df.dropna(subset=["wt2", "ht2", "bmi2", "bmi2_zscore"])

    logging.info(f"3년 데이터 필터링 후 샘플 수: {len(df)}")

    for col in columns:
        if col == group_col:
            continue

        temp = df[[group_col, col]].dropna()
        if temp[group_col].nunique() < 2 or temp[col].nunique() < 2:
            continue

        try:
            res = pg.pairwise_gameshowell(dv=col, between=group_col, data=temp)
            res["variable"] = col
            res["data_type"] = "3year"  # 데이터 유형 표시

            # 각 그룹의 n 계산
            n_count = temp.groupby(group_col).size().reset_index()
            n_count.columns = [group_col, 'n']

            # A 그룹 n 추가
            res = res.merge(n_count, how='left', left_on=[
                            'A'], right_on=[group_col])
            res = res.rename(columns={'n': 'n_A'}).drop(columns=[group_col])

            # B 그룹 n 추가
            res = res.merge(n_count, how='left', left_on=[
                            'B'], right_on=[group_col])
            res = res.rename(columns={'n': 'n_B'}).drop(columns=[group_col])

            results.append(res)

        except Exception as e:
            print(f"[!] 3년 데이터 변수 {col} 처리 중 오류 발생: {e}")

    if results:
        result_df = pd.concat(results, axis=0, ignore_index=True)
        logging.info(f"3년 데이터 분석 완료: {len(result_df)}개 결과")
        return result_df
    else:
        return pd.DataFrame(columns=[
            "A", "B", "mean(A)", "mean(B)", "diff", "se", "T", "dof",
            "pval", "CI95%", "hedges", "variable", "data_type", "n_A", "n_B"
        ])


def posthoc_comparison(df, features_path):
    """
    2년과 3년 추적관찰 데이터를 각각 분석하여 비교 결과를 반환합니다.
    각 데이터에 맞는 dropna 조건이 별도로 적용됩니다.

    Parameters:
    - df : pd.DataFrame
    - features_path : str, 변수 정보가 담긴 yaml 경로

    Returns:
    - dict: {"2year": DataFrame, "3year": DataFrame, "combined": DataFrame}
    """
    logging.info("======== 2년/3년 데이터 별도 분석 및 비교 시작 ============")

    # 각각 별도 dropna 조건으로 분석
    result_2year = posthoc_2year(df, features_path)
    result_3year = posthoc_3year(df, features_path)

    # 결과 합치기
    combined_results = pd.concat(
        [result_2year, result_3year], axis=0, ignore_index=True)

    logging.info("======== 분석 결과 요약 ============")
    logging.info(f"2년 데이터 분석 결과: {len(result_2year)}개")
    logging.info(f"3년 데이터 분석 결과: {len(result_3year)}개")
    logging.info(f"전체 결합 결과: {len(combined_results)}개")

    if not result_2year.empty:
        logging.info(
            f"2년 데이터 분석 변수: {result_2year['variable'].unique().tolist()}")
    if not result_3year.empty:
        logging.info(
            f"3년 데이터 분석 변수: {result_3year['variable'].unique().tolist()}")

    return combined_results


def chisq_posthoc_2year(df: pd.DataFrame,
                        features_path: str,
                        group_col: str = "bmi_group",
                        alpha: float = 0.05,
                        correction: bool = False,
                        adjust: str = "holm") -> pd.DataFrame:
    """
    2년 추적관찰 데이터에 대해 KDST와 BSID 카이제곱 사후 분석을 수행합니다.
    2년 데이터에 맞는 dropna 조건을 적용합니다.

    Parameters:
    -----------
    df : pd.DataFrame
    features_path : str
    group_col : str, default "bmi_group"
    alpha : float, default 0.05
    correction : bool, default False
    adjust : str, default "holm"

    Returns:
    --------
    pd.DataFrame: 카이제곱 사후 분석 결과 (2년 데이터)
    """
    logging.info("======== chisq_posthoc 2년 데이터 분석 시작 ============")

    cfg = load_config(features_path)
    kdst2 = cfg["initial_columns"]["followup"]["KDST"]["KDST2"]
    # bsid2 = cfg["initial_columns"]["followup"]["BSID"]["BSID3"]
    bsid2 = ["cognit1", "lang1", "motor1"]

    # 라벨 컬럼 이름 가져오기
    label_col = cfg["derived_columns"]["label"][0]  # "label"

    # 2년 데이터 복사본 생성
    df_2year = df.copy()
    df_2year['bmi_group'] = df_2year[label_col].map(
        {0: 'Normal', 1: 'Low', 2: 'High'})

    # 기본 필터링
    df_2year = df_2year[(df_2year["corrected_agem1"] >= 18) & (df_2year["corrected_agem1"] <= 30)
                        & (df_2year["birth_agem2"] >= 30) & (df_2year["birth_agem2"] <= 42)]

    # 2년 KDST 데이터에 맞는 dropna 적용
    kdst_vars = kdst2 + [group_col]
    df_kdst_2year = df_2year[kdst_vars].dropna()
    logging.info(f"2년 KDST 데이터 필터링 후 샘플 수: {len(df_kdst_2year)}")

    # 2년 BSID 데이터에 맞는 dropna 적용
    bsid_vars = bsid2 + [group_col]
    df_bsid_2year = df_2year[bsid_vars].dropna()
    logging.info(f"2년 BSID 데이터 필터링 후 샘플 수: {len(df_bsid_2year)}")

    # BSID 점수 → 등급 변환
    for col in bsid2:
        df_bsid_2year[f"{col}_grp"] = df_bsid_2year[col].apply(classify_bsid2)

    # 분석할 변수 목록
    var_cols = kdst2 + [f"{c}_grp" for c in bsid2]

    # 각 데이터셋 결합 (KDST와 BSID 모두 포함)
    all_data_2year = pd.concat([
        df_kdst_2year[kdst2 + [group_col]],
        df_bsid_2year[[f"{c}_grp" for c in bsid2] + [group_col]]
    ], axis=1)

    # 중복된 group_col 제거
    all_data_2year = all_data_2year.loc[:,
                                        ~all_data_2year.columns.duplicated()]

    n_total = len(df_2year)  # 필터링 전 전체 데이터 행 수

    all_out = []
    for v in var_cols:
        if v not in all_data_2year.columns:
            continue

        sub = all_data_2year[[group_col, v]].dropna()
        if sub[v].nunique() < 2:
            continue

        res = _pairwise_chisq_one(sub, group_col, v,
                                  alpha=alpha,
                                  correction=correction,
                                  adjust=adjust)
        if res is not None:
            res["n_total"] = n_total
            res["data_type"] = "2year"  # 데이터 유형 표시
            all_out.append(res)

    if all_out:
        result_df = pd.concat(all_out, ignore_index=True)
        logging.info(f"2년 카이제곱 분석 완료: {len(result_df)}개 결과")
        return result_df
    else:
        return pd.DataFrame()


def chisq_posthoc_3year(df: pd.DataFrame,
                        features_path: str,
                        group_col: str = "bmi_group",
                        alpha: float = 0.05,
                        correction: bool = False,
                        adjust: str = "holm") -> pd.DataFrame:
    """
    3년 추적관찰 데이터에 대해 KDST와 BSID 카이제곱 사후 분석을 수행합니다.
    3년 데이터에 맞는 dropna 조건을 적용합니다.

    Parameters:
    -----------
    df : pd.DataFrame
    features_path : str
    group_col : str, default "bmi_group"
    alpha : float, default 0.05
    correction : bool, default False
    adjust : str, default "holm"

    Returns:
    --------
    pd.DataFrame: 카이제곱 사후 분석 결과 (3년 데이터)
    """
    logging.info("======== chisq_posthoc 3년 데이터 분석 시작 ============")

    cfg = load_config(features_path)
    kdst3 = cfg["initial_columns"]["followup"]["KDST"]["KDST3"]
    # bsid3 = cfg["initial_columns"]["followup"]["BSID"]["BSID3"]
    bsid3 = ["cognit2", "lang2", "motor2"]

    # 라벨 컬럼 이름 가져오기
    label_col = cfg["derived_columns"]["label"][0]  # "label"

    # 3년 데이터 복사본 생성
    df_3year = df.copy()
    df_3year['bmi_group'] = df_3year[label_col].map(
        {0: 'Normal', 1: 'Low', 2: 'High'})

    # 기본 필터링
    df_3year = df_3year[(df_3year["corrected_agem1"] >= 18) & (df_3year["corrected_agem1"] <= 30)
                        & (df_3year["birth_agem2"] >= 30) & (df_3year["birth_agem2"] <= 42)]

    # 3년 KDST 데이터에 맞는 dropna 적용
    kdst_vars = kdst3 + [group_col]
    df_kdst_3year = df_3year[kdst_vars].dropna()
    logging.info(f"3년 KDST 데이터 필터링 후 샘플 수: {len(df_kdst_3year)}")

    # 3년 BSID 데이터에 맞는 dropna 적용
    bsid_vars = bsid3 + [group_col]
    df_bsid_3year = df_3year[bsid_vars].dropna()
    logging.info(f"3년 BSID 데이터 필터링 후 샘플 수: {len(df_bsid_3year)}")

    # BSID 점수 → 등급 변환
    for col in bsid3:
        df_bsid_3year[f"{col}_grp"] = df_bsid_3year[col].apply(classify_bsid3)

    # 분석할 변수 목록
    var_cols = kdst3 + [f"{c}_grp" for c in bsid3]

    # 각 데이터셋 결합 (KDST와 BSID 모두 포함)
    all_data_3year = pd.concat([
        df_kdst_3year[kdst3 + [group_col]],
        df_bsid_3year[[f"{c}_grp" for c in bsid3] + [group_col]]
    ], axis=1)

    # 중복된 group_col 제거
    all_data_3year = all_data_3year.loc[:,
                                        ~all_data_3year.columns.duplicated()]

    n_total = len(df_3year)  # 필터링 전 전체 데이터 행 수

    all_out = []
    for v in var_cols:
        if v not in all_data_3year.columns:
            continue

        sub = all_data_3year[[group_col, v]].dropna()
        if sub[v].nunique() < 2:
            continue

        res = _pairwise_chisq_one(sub, group_col, v,
                                  alpha=alpha,
                                  correction=correction,
                                  adjust=adjust)
        if res is not None:
            res["n_total"] = n_total
            res["data_type"] = "3year"  # 데이터 유형 표시
            all_out.append(res)

    if all_out:
        result_df = pd.concat(all_out, ignore_index=True)
        logging.info(f"3년 카이제곱 분석 완료: {len(result_df)}개 결과")
        return result_df
    else:
        return pd.DataFrame()


def chisq_posthoc_comparison(df: pd.DataFrame,
                             features_path: str,
                             group_col: str = "bmi_group",
                             alpha: float = 0.05,
                             correction: bool = False,
                             adjust: str = "holm") -> pd.DataFrame:
    """
    2년과 3년 추적관찰 데이터를 각각 분석하여 KDST와 BSID 카이제곱 사후 분석 결과를 반환합니다.
    각 데이터에 맞는 dropna 조건이 별도로 적용됩니다.

    Parameters:
    -----------
    df : pd.DataFrame
    features_path : str
    group_col : str, default "bmi_group"
    alpha : float, default 0.05
    correction : bool, default False
    adjust : str, default "holm"

    Returns:
    --------
    pd.DataFrame: 2년과 3년 결과가 결합된 DataFrame
    """
    logging.info("======== 2년/3년 카이제곱 사후 분석 비교 시작 ============")

    # 각각 별도 dropna 조건으로 분석
    result_2year = chisq_posthoc_2year(
        df, features_path, group_col, alpha, correction, adjust)
    result_3year = chisq_posthoc_3year(
        df, features_path, group_col, alpha, correction, adjust)

    # 결과 합치기
    combined_results = pd.concat(
        [result_2year, result_3year], axis=0, ignore_index=True)

    logging.info("======== 카이제곱 분석 결과 요약 ============")
    logging.info(f"2년 데이터 분석 결과: {len(result_2year)}개")
    logging.info(f"3년 데이터 분석 결과: {len(result_3year)}개")
    logging.info(f"전체 결합 결과: {len(combined_results)}개")

    if not result_2year.empty:
        logging.info(
            f"2년 데이터 분석 변수: {result_2year['variable'].unique().tolist()}")
    if not result_3year.empty:
        logging.info(
            f"3년 데이터 분석 변수: {result_3year['variable'].unique().tolist()}")

    return combined_results
