"""
Production workflow runner
raw ➜ preprocess ➜ split ➜ train ➜ validate ➜ visualize

Usage examples
--------------
# 전체 파이프라인
python main.py

# 특정 단계만 실행
python main.py --step train

# 전체 실행 + 로그 레벨 DEBUG
python main.py --log-level DEBUG

# 특정 데이터 파일 사용
python main.py --data-path path/to/your/data.csv

# 단계별 추가 인자 사용
python main.py --step train --params model=random_forest n_estimators=100
python main.py --step tableone --params groupby=age_group pval=True
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import logging
import sys
from pathlib import Path
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from src.utils.utils import get_korea_time

# ----------------------------------------------------------------------
# 0. 공통 설정
# ----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "conf" / "config.yaml"
FEATURES_PATH = ROOT / "conf" / "data" / "features.yaml"
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

def init_logger(level: str = "INFO", backup_days: int = 30):
    """UTC+9 기준 자정마다 새 파일로 자동 교체되는 로거 초기화"""
    # 한국 시간 기준으로 로그 파일 이름 지정
    now_kst = get_korea_time()
    logfile_today = LOG_DIR / f"{now_kst:%Y-%m-%d}.log"

    file_handler = TimedRotatingFileHandler(
        logfile_today,
        when="midnight",
        interval=1,
        backupCount=backup_days,
        encoding="utf-8",
        utc=True,  # 실제 롤오버 기준은 UTC 자정이지만 파일 이름은 KST
    )
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        handlers=[file_handler, console_handler],
        force=True,
    )

# ----------------------------------------------------------------------
# 1. 단계 레지스트리 및 의존성 정의
#    (name, module_path, function_name, dependencies)
# ----------------------------------------------------------------------
STEPS: list[tuple[str, str, str, list[str]]] = [
    ("preprocess", "src.data.make_dataset",        "make_preprocessed_data", []),
    ("filter", "src.data.make_dataset",        "make_filtered_data", ["preprocess"]),
    ("split",      "src.data.make_dataset",          "make_split_data", ["filter", "preprocess"]),
    ("compare",      "src.stats.stats_runner",       "run_compare_train_test", []),
    ("tableone",      "src.stats.stats_runner",       "run_tableone", []),
    ("fu_outcomes",      "src.stats.stats_runner",       "run_followup_outcomes", []),
    ("posthoc",      "src.stats.stats_runner",          "run_posthoc", []),
    ("chi_posthoc",      "src.stats.stats_runner",          "run_chisq_posthoc", []),
    ("corr",      "src.stats.stats_runner",       "run_corr", []),
    ("elastic",      "src.stats.stats_runner",       "run_elasticnet", []),
    ("corr_sig",      "src.stats.stats_runner",       "run_corr_significant", []),
    ("lr",      "src.stats.stats_runner",       "run_lr", []),
    ("var_screen",      "src.stats.stats_runner",          "run_screening", []),
    # ("var_screen_fu",      "src.stats.stats_runner",          "run_screening_fu"),
    ("tableone_derivation",      "src.stats.stats_runner",       "run_derivation_tableone", []),
    ("tableone_derivation_ml",      "src.stats.stats_runner",       "run_derivation_ml_tableone", []),
    ("tableone_fu",      "src.stats.stats_runner",       "run_followup_tableone", []),
    ("tableone_fu_bmi",      "src.stats.stats_runner",       "run_followup_bmi_tableone", []),
    # ("bsid",      "src.stats.stats_runner",       "run_bsid", []),
    # ("tableone_stats2",      "src.stats.stats_runner",       "run_tableone_stats2"),
    ("vif",      "src.stats.stats_runner",       "run_vif", []),
    ("demo",      "src.stats.stats_runner",       "run_demographics", []),
    ("demo_all",      "src.stats.stats_runner",       "run_all_demographics", []),
    ("multi_lr",      "src.stats.stats_runner",       "run_multi_lr", []),
    ("uni_lr",      "src.stats.stats_runner",       "run_uni_lr", []),
    ("bin_lr",      "src.stats.stats_runner",       "run_binary_lr", []),
    ("bin_ab_lr",      "src.stats.stats_runner",       "run_abnormal_lr_univariate_multivariate", []),
    ("sen",      "src.stats.stats_runner",       "run_sensitivity_analysis", []),
    ("adj",      "src.stats.stats_runner",       "run_adjusted_logit", []),
    ("feature_selection",      "src.stats.stats_runner",       "run_feature_selection", []),
    ("backward_selection",      "src.stats.stats_runner",       "run_backward", []),
    # ("train",      "src.models.train_model",       "train_and_save_models"),
    # ("validate",   "src.validation.external_val",  "run_external_validation"),
    # ("visualize",  "src.visualization.report_plot","generate_reports"),
]

STEP_NAMES = [name for name, *_ in STEPS]

# ----------------------------------------------------------------------
# 2. 의존성 해결 함수
# ----------------------------------------------------------------------
def resolve_dependencies(target_steps: list[str]) -> list[str]:
    """의존성을 고려하여 실행 순서를 결정합니다."""
    execution_order = []
    visited = set()
    
    def visit(step_name: str):
        if step_name in visited:
            return
        visited.add(step_name)
        
        # 의존성 먼저 방문
        for name, _, _, deps in STEPS:
            if name == step_name:
                for dep in deps:
                    if dep in STEP_NAMES:
                        visit(dep)
                break
        
        execution_order.append(step_name)
    
    for step in target_steps:
        if step in STEP_NAMES:
            visit(step)
    
    return execution_order

# ----------------------------------------------------------------------
# 3. 실행 유틸
# ----------------------------------------------------------------------
def _get_impl(step_name: str):
    """단계 이름에 해당하는 실제 함수를 가져옵니다."""
    for name, mod_path, fn_name, _ in STEPS:
        if name == step_name:
            try:
                return getattr(importlib.import_module(mod_path), fn_name)
            except (ImportError, AttributeError) as e:
                logging.error(f"모듈 또는 함수를 찾을 수 없습니다: {mod_path}.{fn_name}")
                logging.error(f"에러: {e}")
                raise
    raise ValueError(f"알 수 없는 단계: {step_name}")

def run_step(step: str, config_path: Path, features_path: Path):
    log = logging.getLogger(step)
    log.info("====================▶ start ◀=====================")
    func = _get_impl(step)
    sig = inspect.signature(func)
    arg_map = {
        "config_path": config_path,
        "features_path": features_path
    }
    kwargs = {k: v for k, v in arg_map.items() if k in sig.parameters}
    func(**kwargs)
    log.info("✓ done")

def run_all(config_path: Path, features_path: Path):
    for idx, (step, *_ ) in enumerate(STEPS, 1):
        logging.info("[%d/%d] ===== %s =====", idx, len(STEPS), step)
        run_step(step, config_path, features_path)

def run_steps_with_dependencies(target_steps: list[str], config_path: Path, features_path: Path):
    """의존성을 고려하여 단계들을 실행합니다."""
    execution_order = resolve_dependencies(target_steps)
    
    logging.info("실행 순서:")
    for i, step in enumerate(execution_order, 1):
        deps = next((deps for name, _, _, deps in STEPS if name == step), [])
        deps_str = f" (의존: {', '.join(deps)})" if deps else ""
        logging.info(f"  {i:2d}. {step}{deps_str}")
    
    for idx, step in enumerate(execution_order, 1):
        logging.info("[%d/%d] ===== %s =====", idx, len(execution_order), step)
        run_step(step, config_path, features_path)

# ----------------------------------------------------------------------
# 4. CLI entry-point
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser("Production ML workflow")
    parser.add_argument("--step",
                        nargs='+',
                        choices=STEP_NAMES + ["all"],
                        default="all",
                        help="실행할 단계 (default: all)")
    parser.add_argument("--log-level",
                        default="DEBUG",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--config",
                        default=CONFIG_PATH,
                        help="사용할 설정 파일 경로 (기본값: conf/config.yaml)")
    parser.add_argument("--features",
                        default=FEATURES_PATH,
                        help="사용할 특성 정의 파일 경로 (기본값: conf/data/features.yaml)")
    parser.add_argument("--data-path",
                        help="사용할 데이터 파일 경로 (지정하지 않으면 설정 파일의 기본 경로 사용)")
    parser.add_argument("--params",
                        nargs="*",
                        default=[],
                        help="단계별 매개변수 (형식: key=value)")
    args = parser.parse_args()

    init_logger(args.log_level)

    try :
        if "all" in args.step:
            run_all(CONFIG_PATH, FEATURES_PATH)
        else:
            run_steps_with_dependencies(args.step, CONFIG_PATH, FEATURES_PATH)
        logging.info("====================▶ done ◀=====================\n")
    except Exception as e:
        logging.exception("예외 발생: 프로그램 실행 중 오류가 발생했습니다.\n")
        logging.exception(e)
        sys.exit(1)
if __name__ == "__main__":
    main()
    