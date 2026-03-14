setwd("D:/programing/AI/KNN-BMI")
library(tidyverse)
library(rms)

# ─────────────────────────────────────────────
# 0. 변수 목록 공통 선언
vars_used <- c("dcd_bmi_zscore", "gagew", "sex_sys_val",
               "bdp_yn", "stday", "label", "rop_yn", "vent_severe")

# ─────────────────────────────────────────────
# 1. 학습용 데이터 읽기
train_df <- read_csv(
  "./data/split/train_dataset.csv",
  locale     = locale(encoding = "UTF-8"),
  col_select = all_of(vars_used)
) %>%
  mutate(
    sex_sys_val  = factor(sex_sys_val),
    bdp_yn       = factor(bdp_yn),
    label        = factor(label),
    rop_yn       = factor(rop_yn),
    vent_severe  = factor(vent_severe)
  )

# 2. 모델 적합 (rms::lrm)
dd <- datadist(train_df); options(datadist = "dd")
fit_bdp_grp <- lrm(
  bdp_yn ~ rcs(dcd_bmi_zscore, 3) + gagew + sex_sys_val,
  data = train_df
)
anova(fit_bdp_grp)

# ─────────────────────────────────────────────
# 3. 검증용 데이터 읽기  ← 여기를 바꿔서 test를 사용
test_df <- read_csv(
  "./data/split/test_dataset.csv",
  locale     = locale(encoding = "UTF-8"),
  col_select = all_of(vars_used)
) %>%
  mutate(
    sex_sys_val  = factor(sex_sys_val, levels = levels(train_df$sex_sys_val)),
    bdp_yn       = factor(bdp_yn,      levels = levels(train_df$bdp_yn)),
    label        = factor(label,       levels = levels(train_df$label)),
    rop_yn       = factor(rop_yn,      levels = levels(train_df$rop_yn)),
    vent_severe  = factor(vent_severe, levels = levels(train_df$vent_severe))
  )

# ─────────────────────────────────────────────
# 4. test set에 대한 RCS 예측곡선 생성
# test 데이터의 dcd_bmi_zscore 범위 기준으로 예측점 생성
pred_test <- Predict(
  fit_bdp_grp,
  dcd_bmi_zscore = seq(min(test_df$dcd_bmi_zscore, na.rm = TRUE), 
                       max(test_df$dcd_bmi_zscore, na.rm = TRUE), 
                       length.out = 200),
  gagew = median(test_df$gagew, na.rm = TRUE),        # test 데이터 기준 중앙값
  sex_sys_val = "1",  # 또는 test 데이터에서 가장 빈번한 값
  fun = plogis
)


# ─────────────────────────────────────────────
# 5. 시각화 (여백·축 고정)
tight.theme <- list(
  layout.heights = list(top.padding = 0, bottom.padding = 0),
  layout.widths  = list(left.padding= 0, right.padding  = 0)
)

plot(pred_test,
     xlab         = "BMI z-score at discharge",
     ylab         = "Predicted probability of BPD",
     lwd          = 2,
     xlim         = c(-6, 6),
     ylim         = c(0, 1),
     par.settings = tight.theme,
     auto.key     = FALSE    # 범례 없으면 완전 타이트하게
)


