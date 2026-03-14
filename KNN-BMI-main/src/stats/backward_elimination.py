import pandas as pd
import statsmodels.api as sm
import pandas as pd
from src.utils.utils import load_data, load_config

config = load_config("conf/config.yaml")
data = load_data("split", "train", config)

feature_config = load_config("conf/data/features.yaml")
categorical_features = feature_config["initial_columns"]["category"] + feature_config["derived_columns"]["category"]
numerical_features = feature_config["initial_columns"]["numeric"] + feature_config["derived_columns"]["numeric"]
dcd_bmi = feature_config["derived_columns"]["target"][2]
dcd_bmi_zscore = feature_config["derived_columns"]["target"][3]
target = feature_config["derived_columns"]["target"][-1]

def backward_elimination(X, y, significance_level=0.05):
    """
    X: 독립 변수 (DataFrame)
    y: 종속 변수 (Series)
    significance_level: 제거 기준 p-value
    """
    X = sm.add_constant(X)  # 절편 추가
    cols = list(X.columns)

    while len(cols) > 0:
        X_selected = X[cols]
        model = sm.Logit(y, X_selected).fit(disp=False)
        p_values = model.pvalues
        max_pval = p_values.max()
        if max_pval > significance_level:
            excluded_var = p_values.idxmax()
            print(f"Removing '{excluded_var}' (p={max_pval:.4f})")
            cols.remove(excluded_var)
        else:
            break

    print("\nFinal selected variables:")
    print(cols)
    return X[cols]

backward_elimination(data[categorical_features + numerical_features], data[target])