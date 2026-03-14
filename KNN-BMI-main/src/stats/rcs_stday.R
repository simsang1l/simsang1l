setwd("D:/programing/AI/KNN-BMI")
library(tidyverse)
library(rms)

# 사용할 변수 정의
vars_used <- c("dcd_bmi_zscore", "gagew", "sex_sys_val",
               "stday")


# 데이터 로드 & factor 변환
train_df <- read_csv("./data/split/train_dataset.csv",
                     locale     = locale(encoding = "UTF-8"),
                     col_select = all_of(vars_used)) |>
  mutate(
    sex_sys_val = factor(sex_sys_val)         # 필요 시 factor 변환 (범주형 변수인 경우)
  )


# datadist 설정
dd <- datadist(train_df)
options(datadist = "dd")


# ols: 연속형 종속변수일 때 사용
# rcs: knot개수 k 지정 (rcs(,,,, k))
fit_ols <- ols(
  stday ~ rcs(dcd_bmi_zscore, 4) + gagew + sex_sys_val,
  data = train_df,
  x = TRUE,   # 함수 내부에서 예측·부트스트랩 등에 필요
  y = TRUE    # "
)



# 1) 시나리오(공변량 고정값) 정의
ref_gagew <- median(train_df$gagew, na.rm = TRUE)
ref_sex   <- levels(train_df$sex_sys_val)[1]

# 2) Predict() → RCS 곡선 + CI
p_rcs <- Predict(fit_ols,
                 dcd_bmi_zscore,                # x축
                 gagew       = ref_gagew,
                 sex_sys_val = ref_sex, 
                 conf.int    = 0.95)            # 기본 0.95

plot(p_rcs,                     # 기본 rms 플롯
     xlab = "BMI z-score at discharge",
     ylab = "Hospital Stay",
     xlim         = c(-6, 6),
     # ylim         = c(0, 1),
     lwd  = 2)

# ---------------------------------------------
x <- train_df$dcd_bmi_zscore  # 독립 변수 (BMI z-score)
y <- train_df$stday   # 종속 변수 (입원일수)

# RCS 모델 (5개 절편)
fit_rcs <- ols(y ~ rcs(x, 5), data=train_df)

# anova()로 비선형성 검정
anova(fit_rcs)
