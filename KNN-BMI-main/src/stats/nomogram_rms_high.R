# 공통 설정 ------------------------------------------------------------------
setwd("D:/programing/AI/KNN-BMI")
# setwd("C:/projects/KNN-BMI")
library(rms)
library(Hmisc)   # cut2, rcorr.cens
library(pROC)
library(dplyr)
library(yaml)

# YAML 설정 파일 --------------------------------------------------------------
config           <- yaml.load_file("./conf/data/features.yaml")
continuous_cols  <- c("gagew", "bwei", "iarvppd28")
categorical_vars <- c("atbyn", "bdp") # c("vent_severe")
target           <- config$derived_columns$label
features         <- c(target, continuous_cols, categorical_vars)
# features         <- c(target, continuous_cols)

# --- 데이터 로딩(Train) ------------------------------------------------------
train <- read.csv("./data/split/train_dataset.csv", stringsAsFactors = FALSE)
train[[target]]        <- as.integer(train[[target]])
train[continuous_cols] <- lapply(train[continuous_cols], as.numeric)
train[categorical_vars]<- lapply(train[categorical_vars], \(x) factor(x))
train                  <- train[complete.cases(train[, features]), features]

# datadist: Train에서 한 번 정의 → 고정
dd <- datadist(train)
options(datadist = "dd")

# rcs term 생성
rcs_terms <- paste0("rcs(", continuous_cols, ", 4)")
rhs_terms <- c(rcs_terms, categorical_vars)

# ---------------------------------------------------------------------------
# 1) Nomogram + 내부 부트스트랩 검증 (0 vs class_label)
# ---------------------------------------------------------------------------
draw_nomogram_vs_zero <- function(class_label, B = 500, seed = 2025, var_labels=NULL) {
  message("\n📌 0 vs ", class_label)
  class_name <- "High"
  data_bin <- subset(train, train[[target]] %in% c(0, class_label))
  data_bin$label <- ifelse(data_bin[[target]] == class_label, 1L, 0L)
  
  # ✅ 변수명 매핑이 제공된 경우, label() 함수로 레이블 지정
  if (!is.null(var_labels)) {
    for (var_name in names(var_labels)) {
      if (var_name %in% names(data_bin)) {
        label(data_bin[[var_name]]) <- var_labels[[var_name]]
      }
    }
  }
  
  # 모델 적합
  f <- as.formula(paste("label ~", paste(rhs_terms, collapse = " + ")))
  set.seed(seed)
  fit <- lrm(f, data = data_bin, x = TRUE, y = TRUE)
  
  # Nomogram (라벨 동적)
  funlab <- sprintf("Probability of %s BMI at Discharge", class_name)
  nom <- nomogram(fit,
                  fun      = plogis,
                  funlabel = funlab,
                  lp       = TRUE,
                  conf.int = FALSE,
                  fun.at   = c(0.1, 0.25, 0.5, 0.75, 0.9))
  plot(nom, xfrac=.45, cex.var=.85, cex.axis=.75, nint=2,
       label.every=3, tcl=.15, lmgp=.1, col.grid=gray(c(.85,.95)))
  
  # C-index (bootstrap-corrected)
  val <- validate(fit, B = B, method = "boot")
  c_index <- (val["Dxy","index.corrected"] + 1)/2
  cat("\n✔️  Bootstrap-corrected C-index:", round(c_index, 3), "\n")
  
  # Calibration (내부 부트스트랩 교정)
  cal <- calibrate(fit, B = B, method = "boot")
  plot(cal, xlab="Predicted probability", ylab="Observed probability",
       subtitles=FALSE, lwd=2, legend=FALSE)
  abline(0,1,lty=2)
  
  # 예측확률 & 보조지표
  prob_hat <- predict(fit, type="fitted")
  y        <- data_bin$label
  
  # Brier + Scaled Brier
  brier  <- mean((prob_hat - y)^2)
  scaled <- 1 - brier/var(y)
  
  # Intercept/Slope
  logit_p <- qlogis(prob_hat)
  cal_glm <- glm(y ~ logit_p, family=binomial)
  cal_intercept <- coef(cal_glm)[1]; cal_slope <- coef(cal_glm)[2]
  
  # ECE/MCE (안전한 binning)
  g <- 10
  bin <- cut2(prob_hat, g=g)
  cal_tbl <- data.frame(prob_hat, y, bin) |>
    group_by(bin) |>
    summarise(mean_pred=mean(prob_hat), mean_obs=mean(y), n=dplyr::n(), .groups="drop")
  ece <- sum(abs(cal_tbl$mean_pred - cal_tbl$mean_obs) * cal_tbl$n) / length(y)
  mce <- max(abs(cal_tbl$mean_pred - cal_tbl$mean_obs))
  
  cat(sprintf(
    "\n📊  Calibration metrics (Train bootstrap)\n   • Brier / Scaled Brier : %.4f / %.4f\n   • Intercept / Slope    : %.3f / %.3f\n   • ECE / MCE            : %.4f / %.4f\n",
    brier, scaled, cal_intercept, cal_slope, ece, mce))

  invisible(list(fit=fit, dd=dd))
}

custom_labels <- c(
  "gagew" = "Gestational age",
  "bwei" = "Birth weight",
  "iarvppd28" = "IMV Duration (DOL 28)",
  "atbyn" = "Antenatal antibiotic",
  "bdp" = "Bronchopulmonary dysplasia"
)

res <- draw_nomogram_vs_zero(class_label = 2, var_labels = custom_labels)
# ✔️  Bootstrap-corrected C-index: 0.721 

# n=5357   Mean absolute error=0.003   Mean squared error=1e-05
# 0.9 Quantile of absolute error=0.007


# 📊  Calibration metrics (Train bootstrap)
# • Brier / Scaled Brier : 0.0800 / 0.0646
# • Intercept / Slope    : 0.000 / 1.000
# • ECE / MCE            : 0.0082 / 0.0207



validate_external <- function(fit, newdata, label_name = "Test") {
  nd <- newdata
  
  # label 포함 여부 먼저 확인
  if(!("label" %in% names(nd))) {
    stop("`nd` must include binary `label` (0/1) to compute metrics consistently.")
  }
  
  # 데이터 타입 변환 (label은 그대로 유지)
  nd[[target]] <- as.integer(nd[[target]])
  nd[continuous_cols] <- lapply(nd[continuous_cols], as.numeric)
  
  # ✅ categorical 변수 각각에 맞는 levels 적용
  for(var in categorical_vars) {
    if(var %in% names(nd) && var %in% names(train)) {
      nd[[var]] <- factor(nd[[var]], levels = levels(train[[var]]))
    }
  }
  
  # ✅ features에서 label 제외
  features_only <- setdiff(features, "label")
  
  # ✅ complete.cases 체크 (label 제외한 features만)
  complete_idx <- complete.cases(nd[, features_only])
  nd <- nd[complete_idx, ]
  
  cat("\nAfter complete.cases filtering:\n")
  cat("Rows remaining:", nrow(nd), "\n")
  print(table(nd$label))
  
  if(nrow(nd) == 0) {
    stop("No complete cases found in validation data!")
  }
  
  # ✅ 수정: label을 그대로 사용 (이미 올바른 factor)
  y <- nd$label
  
  # ✅ 디버깅: y 변수 확인
  cat("\nDebug - y variable:\n")
  print(table(y, useNA = "ifany"))
  print(class(y))
  print(levels(y))
  
  # 양쪽 클래스가 모두 있는지 확인
  if(length(unique(y)) < 2) {
    cat("\nClass distribution:\n")
    print(table(y, useNA = "ifany"))
    stop("Error: Only one class present in validation data after filtering!")
  }
  
  # 예측
  p <- predict(fit, newdata = nd, type = "fitted")
  
  # 예측값에 NA가 있는지 확인
  if(any(is.na(p))) {
    cat("\nWarning: NA predictions found, removing them...\n")
    valid_idx <- !is.na(p)
    p <- p[valid_idx]
    y <- y[valid_idx]
    cat("After removing NA predictions:\n")
    print(table(y))
  }
  
  # 최종 확인
  if(length(unique(y)) < 2) {
    cat("\nFinal class distribution:\n")
    print(table(y, useNA = "ifany"))
    stop("Error: Only one class remains after prediction!")
  }
  
  # AUROC / C-index
  au <- as.numeric(auc(roc(response = y, predictor = p)))
  cx <- as.numeric(rcorr.cens(p, y)["C Index"])
  
  # Calibration curve
  y_numeric <- as.numeric(as.character(y))
  val.prob(p, y_numeric, g = 10, pl = TRUE)
  
  # 보조지표
  brier  <- mean((p - y_numeric)^2)
  scaled <- 1 - brier/var(y_numeric)
  bin    <- cut2(p, g=10)
  cal_tbl <- data.frame(p, y=y_numeric, bin) |>
    group_by(bin) |>
    summarise(mean_pred=mean(p), mean_obs=mean(y), n=dplyr::n(), .groups="drop")
  ece <- sum(abs(cal_tbl$mean_pred - cal_tbl$mean_obs) * cal_tbl$n) / length(y)
  mce <- max(abs(cal_tbl$mean_pred - cal_tbl$mean_obs))
  
  cat(sprintf(
    "\n[%s]\nAUROC=%.3f, C-index=%.3f\nBrier/Scaled=%.4f/%.4f, ECE/MCE=%.4f/%.4f\n",
    label_name, au, cx, brier, scaled, ece, mce))
}

# Test set 준비
# [Test]
# AUROC=0.727, C-index=0.727
# Brier/Scaled=1.0843/-11.5120, ECE/MCE=1.0012/1.0252
test <- read.csv("./data/split/test_dataset.csv")
test[[target]] <- as.integer(test[[target]])
test[continuous_cols] <- lapply(test[continuous_cols], as.numeric)
test[categorical_vars] <- lapply(test[categorical_vars], \(x) factor(x))

# ✅ 0과 2만 남기기
test <- subset(test, test[[target]] %in% c(0, 2))

# ✅ 디버깅: target 값 확인
cat("Target values before label creation:\n")
print(table(test[[target]]))

# label 이진화 (2를 1로, 0을 0으로)
test$label <- ifelse(test[[target]] == 2, 1L, 0L)

# ✅ 디버깅: label 값 확인
cat("Label values after ifelse:\n")
print(table(test$label))
print(class(test$label))

# features에 label 추가
features_with_label <- unique(c(features, "label"))

# NA 제거 (label 포함)
test <- na.omit(test[, features_with_label])

# ✅ label을 factor로 변환
test$label <- factor(test$label, levels = c(0, 1))

# ✅ 최종 확인
cat("Final test data:\n")
print(table(test$label))
print(table(test$label, useNA = "ifany"))

# 검증 실행
validate_external(res$fit, test, label_name="Test")


# External set 준비
external <- read.csv("./data/external/external_validation_dataset.csv")
external[[target]] <- as.integer(external[[target]])
external[continuous_cols] <- lapply(external[continuous_cols], as.numeric)
external[categorical_vars] <- lapply(external[categorical_vars], \(x) factor(x))

# ✅ 0과 2만 남기기
external <- subset(external, external[[target]] %in% c(0, 2))

# ✅ 디버깅: target 값 확인
cat("Target values before label creation:\n")
print(table(external[[target]]))

# label 이진화 (2를 1로, 0을 0으로)
external$label <- ifelse(external[[target]] == 2, 1L, 0L)

# ✅ 디버깅: label 값 확인
cat("Label values after ifelse:\n")
print(table(external$label))  # ✅ external$label로 수정
print(class(external$label))  # ✅ external$label로 수정

# features에 label 추가
features_with_label <- unique(c(features, "label"))

# NA 제거 (label 포함)
external <- na.omit(external[, features_with_label])

# ✅ label을 factor로 변환
external$label <- factor(external$label, levels = c(0, 1))

# ✅ 최종 확인
cat("Final external data:\n")
print(table(external$label))
print(table(external$label, useNA = "ifany"))

# 검증 실행
validate_external(res$fit, external, label_name="External set")



# External set 준비
# [External set]
# AUROC=0.738, C-index=0.738
# Brier/Scaled=1.1610/-8.9979, ECE/MCE=1.0271/1.0715

external <- read.csv("./data/external/external_validation_dataset.csv")
external[[target]] <- as.integer(external[[target]])
external[continuous_cols] <- lapply(external[continuous_cols], as.numeric)
external[categorical_vars] <- lapply(external[categorical_vars], \(x) factor(x))

# ✅ 0과 2만 남기기
external <- subset(external, external[[target]] %in% c(0, 2))

# ✅ 디버깅: target 값 확인
cat("Target values before label creation:\n")
print(table(external[[target]]))

# label 이진화 (2를 1로, 0을 0으로)
external$label <- ifelse(external[[target]] == 2, 1L, 0L)

# ✅ 디버깅: label 값 확인
cat("Label values after ifelse:\n")
print(table(external$label))  # ✅ external$label로 수정
print(class(external$label))  # ✅ external$label로 수정

# features에 label 추가
features_with_label <- unique(c(features, "label"))

# NA 제거 (label 포함)
external <- na.omit(external[, features_with_label])

# ✅ label을 factor로 변환
external$label <- factor(external$label, levels = c(0, 1))

# ✅ 최종 확인
cat("Final external data:\n")
print(table(external$label))
print(table(external$label, useNA = "ifany"))

# 검증 실행
validate_external(res$fit, external, label_name="External set")