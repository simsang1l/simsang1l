setwd("D:/programing/AI/KNN-BMI")
library(fastDummies)
library(tidyverse)
library(rms)

# ─────────────────────────────────────────────
# 0. 변수 목록 공통 선언
vars_used <- c("dcd_bmi_zscore", "gagew", "sex_sys_val",
               "bdp_yn", "stday", "label", "rop_yn", "vent_severe_36")

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
    vent_severe_36  = factor(vent_severe_36)
  )

#################
# 더미 변수화
#################
train_df <- dummy_cols(train_df, select_columns = "vent_severe_36")
colnames(train_df)

# ─────────────────────────────────────────────
# 검증용 데이터 읽기
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
    vent_severe_36  = factor(vent_severe_36, levels = levels(train_df$vent_severe_36))
  )

# ─────────────────────────────────────────────
# 2. 모델 적합 (rms::lrm)
dd <- datadist(train_df); options(datadist = "dd")
fit_multinomial <- lrm(
  vent_severe_36 ~ rcs(dcd_bmi_zscore, 3) + gagew + sex_sys_val,
  data = train_df
)

# ─────────────────────────────────────────────
# 3. 각 레벨별 예측 곡선 생성
bmi_range <- seq(min(train_df$dcd_bmi_zscore, na.rm = TRUE), 
                 max(train_df$dcd_bmi_zscore, na.rm = TRUE), 
                 length.out = 200)

# 모든 vent_severe_36 레벨에 대한 예측
pred_all_levels <- Predict(
  fit_multinomial,
  dcd_bmi_zscore = bmi_range,
  gagew = median(train_df$gagew, na.rm = TRUE),
  sex_sys_val = "1",
  fun = plogis
)



# ─────────────────────────────────────────────
# 방법 2: 더미변수를 사용한 개별 모델들의 통합 시각화
train_df_dummy <- dummy_cols(train_df, select_columns = "vent_severe_36")
dummy_cols <- grep("vent_severe_36_", names(train_df_dummy), value = TRUE)

# 각 더미변수별 모델 적합
models <- list()
predictions <- list()

for (col in dummy_cols) {
  # 각 더미변수에 대한 개별 모델
  formula_str <- paste(col, "~ rcs(dcd_bmi_zscore, 3) + gagew + sex_sys_val")
  models[[col]] <- lrm(as.formula(formula_str), data = train_df_dummy)
  
  # 예측
  predictions[[col]] <- Predict(
    models[[col]],
    dcd_bmi_zscore = bmi_range,
    gagew = median(test_df$gagew, na.rm = TRUE),
    sex_sys_val = "1",
    fun = plogis
  )
}

# 예측 결과 통합
combined_pred <- do.call(rbind, lapply(names(predictions), function(name) {
  pred_df <- data.frame(predictions[[name]])
  pred_df$model <- name
  return(pred_df)
}))

# 시각화
p2 <- ggplot(combined_pred, aes(x = dcd_bmi_zscore, y = yhat, color = model)) +
  geom_line(size = 1.5) +
  geom_ribbon(aes(ymin = lower, ymax = upper, fill = model), alpha = 0.2) +
  scale_color_viridis_d(name = "Dummy Variable") +
  scale_fill_viridis_d(name = "Dummy Variable") +
  labs(
    x = "BMI z-score at discharge",
    y = "Predicted probability",
    title = "RCS Curves for Dummy Variables"
  ) +
  xlim(-6, 6) +
  ylim(0, 1) +
  theme_minimal() +
  theme(
    legend.position = "bottom",
    plot.title = element_text(hjust = 0.5)
  )

print(p2)




