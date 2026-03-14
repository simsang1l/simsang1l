import statsmodels.api as sm
import pandas as pd

from src.utils.utils import load_config, load_data

config = "conf/config.yaml"
feature_config = load_config("conf/data/features.yaml")

data = load_data("split", "train", config)

categorical_features = feature_config["initial_columns"]["category"] + feature_config["derived_columns"]["category"]
numerical_features = feature_config["initial_columns"]["numeric"] + feature_config["derived_columns"]["numeric"]
dcd_bmi = feature_config["derived_columns"]["target"][2]
dcd_bmi_zscore = feature_config["derived_columns"]["target"][3]

data = data[categorical_features + numerical_features+[dcd_bmi, dcd_bmi_zscore]].dropna(axis = 0)
X = data[categorical_features + numerical_features]
y_bmi = data[dcd_bmi]
y_bmiz = data[dcd_bmi_zscore]

X = sm.add_constant(X)

model_bmi = sm.OLS(y_bmi, X)
result_bmi = model_bmi.fit()
# print(result_bmi.summary())

model_bmiz = sm.OLS(y_bmiz, X)
result_bmiz = model_bmiz.fit()
# print(result_bmiz.summary())

# 4. 결과 정리 함수
def make_result_table(model, prefix):
    coef = model.params
    conf = model.conf_int()
    pval = model.pvalues

    return pd.DataFrame({
        f'{prefix} Coef': coef.round(3),
        f'{prefix} 95% CI': conf.apply(lambda row: f"({row[0]:.3f}, {row[1]:.3f})", axis=1),
        f'{prefix} p-value': pval.round(4)
    })

# 5. 결과 테이블 병합
result_bmi = make_result_table(result_bmi, 'BMI')
result_bmiz = make_result_table(result_bmiz, 'BMI-Z')

final_result = pd.concat([result_bmi, result_bmiz], axis=1)

# 6. CSV 저장
final_result.to_csv("bmi_regression_results.csv")

# # 시각적으로 확인
# import ace_tools as tools; tools.display_dataframe_to_user(name="BMI 회귀 결과", dataframe=final_result)