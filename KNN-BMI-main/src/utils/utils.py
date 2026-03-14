import os
import random
import pandas as pd
import yaml
import numpy as np
import logging
from datetime import datetime
import pytz
import matplotlib.pyplot as plt
import seaborn as sns

# =======================================================
# 공통 설정
# =======================================================
default_config_path = "conf/config.yaml"
default_feature_config_path = "conf/feature_config.yaml"


def get_korea_time() -> datetime:
    """한국 시간(서울)을 반환하는 함수"""
    kst = pytz.timezone('Asia/Seoul')
    return datetime.now(kst)


def seed_everything(seed=42):
    """Random seed 고정"""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass  # Torch가 없는 경우 무시


def create_dirs(dirs):
    """필요한 디렉토리 생성"""
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)


def load_config(config_path=default_config_path):
    with open(config_path, 'r', encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_data(path_type: str, data_type: str, config_path=default_config_path):
    """
    데이터 로드 (config_path 기본값: conf/config.yaml)
    data_type: raw, preprocessed, filtered, train, test, scaled_encoded_train, scaled_encoded_test, scaled_encoded_external_validation
    """
    config = load_config(config_path)

    file_path = config["paths"][path_type]
    file_name = config["file_name"][data_type]
    full_path = os.path.join(file_path, file_name)
    df = pd.read_csv(full_path)
    logging.info(f":> Loaded: {full_path}, shape: {df.shape}")
    if "label" in df.columns:
        logging.info(f"data shape: {df['label'].shape}")
        logging.info(f"Counts by label: {df['label'].value_counts()}")
    return df


def save_data(df, path_type: str, data_type: str, config_path=default_config_path, index=False):
    """
    데이터 저장 (config_path 기본값: conf/config.yaml)
    df: 저장할 데이터프레임
    data_type: 저장할 데이터 타입
    data_type: raw, preprocessed, filtered, train, test, scaled_encoded_train, scaled_encoded_test, scaled_encoded_external_validation
    """
    config = load_config(config_path)

    file_path = config["paths"][path_type]
    file_name = config["file_name"][data_type]
    save_path = os.path.join(file_path, file_name)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, encoding='utf-8-sig', index=index)
    logging.info(f"[{data_type}] Saved: {save_path}")
    if "label" in df.columns:
        logging.info(f"data shape: {df['label'].shape}")
        logging.info(f"Counts by label: {df['label'].value_counts()}")


def save_result_csv(df, path_type: str, result_type: str, execution_time, config_path=default_config_path,  index=False):
    """
    결과 저장 (config_path 기본값: conf/config.yaml)
    df: 저장할 데이터프레임
    data_type: 저장할 데이터 타입
    data_type: results에 구분된 데이터 타입
    CSV 저장:  results/<data_type>/<YYYYMMDD_HHMMSS>/file.csv
    """
    config = load_config(config_path)

    root_dir = config["paths"][path_type]
    base_name = config["results"][result_type]

    out_dir = os.path.join(root_dir, execution_time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)

    save_path = os.path.join(out_dir, f"{base_name}.csv")
    df.to_csv(save_path, index=index)
    logging.info(f"[{result_type}] Saved: {save_path}")


def save_heatmap(df, path_type, result_type, execution_time, config_path=default_config_path):
    """
    """
    config = load_config(config_path)

    root_dir = config["paths"][path_type]
    base_name = config["results"][result_type]

    out_dir = os.path.join(root_dir, execution_time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)

    save_path = os.path.join(out_dir, f"{base_name}.png")

    fig, ax = plt.subplots(figsize=(26, 24), constrained_layout=False)
    heatmap = sns.heatmap(
        df,
        annot=True,
        cmap='coolwarm',
        fmt=".2f",
        linewidths=0.5,
        annot_kws={"size": 20},
        cbar_kws={"pad": 0.02}
    )

    # X축 레이블: 45도 기울임
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        ha='right',
        rotation_mode='anchor',
        fontsize=22
    )

    # Y축 레이블: 기울이지 않음
    ax.set_yticklabels(
        ax.get_yticklabels(),
        rotation=0,
        fontsize=22
    )

    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=28)

    # 최대한 여백 제거
    plt.tight_layout(pad=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.2)

    logging.info(f"[{result_type}] Figure Saved: {save_path}")


def save_plot(fig, path_type, result_type, execution_time, config_path=default_config_path, dpi=300):
    config = load_config(config_path)

    root_dir = config["paths"][path_type]
    base_name = config["results"][result_type]

    out_dir = os.path.join(root_dir, execution_time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)

    save_path = os.path.join(out_dir, f"{base_name}.png")

    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"Saved: {save_path}")
