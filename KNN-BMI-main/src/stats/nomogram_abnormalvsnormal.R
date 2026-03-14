# 공통 설정 ------------------------------------------------------------------
setwd("D:/programing/AI/KNN-BMI")
library(rms)
library(yaml)

# YAML 설정 파일 --------------------------------------------------------------
config           <- yaml.load_file("./conf/data/features.yaml")
continuous_cols  <- c("aoxyuppd", "invfpod", "niarvrpd", "stday", "birth_bmi_zscore", "gagew") 
continuous_cols
# c("invfpod", "birth_bmi", "stday")
categorical_vars <- c("vent_severe", "nrp_grade", "lbp", "atbyn") 
categorical_vars
# c("atbyn", "vent_severe", "vent_severe_36", "bdp_yn", "rop_yn", "strdu", "resuo")
target           <- config$derived_columns$label
features         <- c(target, continuous_cols, categorical_vars)

# 데이터 로딩 및 전처리 --------------------------------------------------------
data <- read.csv("./data/split/train_dataset.csv", stringsAsFactors = FALSE)
data[[target]]          <- as.integer(data[[target]])
data[continuous_cols]   <- lapply(data[continuous_cols], as.numeric)
data[categorical_vars]  <- lapply(data[categorical_vars], factor)
data                    <- na.omit(data[, features])

# ➡️ rcs 적용: continuous 변수에 rcs() 적용한 식으로 변환
rcs_terms     <- paste0("rcs(", continuous_cols, ", 4)")
rhs_terms     <- c(rcs_terms, categorical_vars)

# 2️⃣ nomogram + C-index + calibration plot -----------------------------------
draw_nomogram_binary <- function(B = 200, seed = 2025) {
  message("\n📌 Drawing nomogram & validation for: 0 vs (1‒2)")
  
  ## 1) 0/1/2 → 0/1 로 변환
  data_bin            <- data                                    # 전체 사용
  data_bin$label_fac  <- factor(ifelse(data_bin[[target]] == 0, 0L, 1L),
                                levels = c(0, 1))                # 모델용 factor
  data_bin$label_num  <- ifelse(data_bin[[target]] == 0, 0L, 1L) # 0/1 numeric
  
  ## 2) datadist( )
  dd <<- datadist(data_bin)
  options(datadist = "dd")
  
  ## 3) 모델 적합
  formula_obj <- as.formula(paste("label_fac ~", paste(rhs_terms, collapse = " + ")))
  fit <- lrm(formula_obj, data = data_bin, x = TRUE, y = TRUE)
  
  ## 4) Nomogram
  nom <- nomogram(
    fit,
    fun      = plogis,
    funlabel = "Predicted risk: 0 vs (1‒2)",
    lp       = TRUE,
    conf.int = FALSE,
    fun.at   = c(0.1, 0.25, 0.5, 0.75, 0.9)
  )
  plot(nom,
       xfrac = .45, cex.var = 0.85, cex.axis = 0.75,
       nint  = 2,   label.every = 3, tcl = 0.15, lmgp = .1,
       col.grid = gray(c(.85, .95)))
  
  ## 5) C-index (bootstrap-corrected)
  set.seed(seed)
  val      <- validate(fit, B = B, method = "boot")
  c_index  <- (val["Dxy", "index.corrected"] + 1) / 2
  cat("\n✔️  Bootstrap-corrected C-index:", round(c_index, 3), "\n")
  
  ## 6) Calibration plot
  cal <- calibrate(fit, B = B, method = "boot")
  plot(cal,
       xlab = "Predicted probability",
       ylab = "Observed probability",
       subtitles = FALSE,
       lwd = 2,
       legend = FALSE)
  abline(0, 1, lty = 2)
  
  ## 7) 수치 지표
  prob_hat <- predict(fit, type = "fitted")
  y_bin    <- data_bin$label_num
  
  # Brier
  brier <- mean((prob_hat - y_bin)^2)
  
  # Intercept / Slope
  logit_p <- log(prob_hat / (1 - prob_hat))
  cal_glm <- glm(y_bin ~ logit_p, family = binomial)
  cal_intercept <- coef(cal_glm)[1]
  cal_slope     <- coef(cal_glm)[2]
  
  # ECE / MCE
  n_bins <- 10
  bin_id <- cut(prob_hat,
                breaks = quantile(prob_hat,
                                  probs = seq(0, 1, length.out = n_bins + 1),
                                  na.rm = TRUE),
                include.lowest = TRUE)
  library(dplyr)
  cal_tbl <- data.frame(prob_hat, obs = y_bin, bin = bin_id) |>
    group_by(bin) |>
    summarise(mean_pred = mean(prob_hat),
              mean_obs  = mean(obs),
              n         = n(),
              .groups   = "drop")
  ece <- with(cal_tbl,
              sum(abs(mean_pred - mean_obs) * n) / nrow(data_bin))
  mce <- max(abs(cal_tbl$mean_pred - cal_tbl$mean_obs))
  
  cat(sprintf(
    "\n📊  Calibration metrics\n   • Brier score            : %.4f\n   • Intercept / Slope      : %.3f / %.3f\n   • Expected calib. error  : %.4f\n   • Max calib. error       : %.4f\n",
    brier, cal_intercept, cal_slope, ece, mce))
}


# 실행 ------------------------------------------------------------------------
#✔️  Bootstrap-corrected C-index: 0.749 
# n=7379   Mean absolute error=0.003   Mean squared error=1e-05
# 0.9 Quantile of absolute error=0.004
# draw_nomogram_vs_zero(class_label = 1)  # 0 vs 1


########################################################

# ✔️  Bootstrap-corrected C-index: 0.657 

# n=7346   Mean absolute error=0.011   Mean squared error=2e-04
# 0.9 Quantile of absolute error=0.023


# 📊  Calibration metrics
# • Brier score            : 0.2035
# • Intercept / Slope      : -0.000 / 1.000
# • Expected calib. error  : 0.0149
# • Max calib. error       : 0.0304

draw_nomogram_binary()   # 0 vs 2

#########################################################




