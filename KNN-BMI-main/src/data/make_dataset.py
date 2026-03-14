"""
데이터셋 생성 및 전처리 모듈
"""
import logging
from pathlib import Path
from typing import Optional, Dict, Any

from src.utils.utils import load_data, save_data
from src.data.dataset_utils import preprocess, filter_data, split_data, postprocess, dropna_subset


def make_preprocessed_data(config_path: str, features_path: Optional[str] = None, **kwargs) -> None:
    """
    원시 데이터를 전처리하여 저장합니다.

    Args:
        config_path: 설정 파일 경로
        features_path: 특성 정의 파일 경로 (사용하지 않지만 인터페이스 통일을 위해 유지)
        **kwargs: 추가 매개변수
    """
    logger = logging.getLogger(__name__)
    logger.info("=== 데이터 전처리 시작 ===")

    try:
        # 원시 데이터 로드
        raw_data = load_data("raw", "raw", config_path)
        logger.info(f"원시 데이터 로드 완료: {raw_data.shape}")

        # 전처리 수행
        preprocessed_data = preprocess(raw_data, config_path)
        logger.info(f"전처리 완료: {preprocessed_data.shape}")

        # 전처리된 데이터 저장
        save_data(preprocessed_data, "preprocess", "preprocess", config_path)
        logger.info("전처리된 데이터 저장 완료")

    except Exception as e:
        logger.error(f"데이터 전처리 중 오류 발생: {e}")
        raise


def make_filtered_data(config_path: str, features_path: str, **kwargs) -> None:
    """
    전처리된 데이터를 필터링하여 저장합니다.

    Args:
        config_path: 설정 파일 경로
        features_path: 특성 정의 파일 경로
        **kwargs: 추가 매개변수
    """
    logger = logging.getLogger(__name__)
    logger.info("=== 데이터 필터링 시작 ===")

    try:
        # 전처리된 데이터 로드
        preprocessed_data = load_data("preprocess", "preprocess", config_path)
        logger.info(f"전처리된 데이터 로드 완료: {preprocessed_data.shape}")

        # 필터링 수행
        filtered_data = filter_data(
            preprocessed_data, config_path, features_path)
        logger.info(f"필터링 완료: {filtered_data.shape}")

        # # 결측값 처리
        # filtered_data = dropna_subset(filtered_data)
        # logger.info(f"결측값 처리 완료: {filtered_data.shape}")

        # 필터링된 데이터 저장
        save_data(filtered_data, "filter", "filter", config_path)
        logger.info("필터링된 데이터 저장 완료")

    except Exception as e:
        logger.error(f"데이터 필터링 중 오류 발생: {e}")
        raise


def make_split_data(config_path: str, features_path: str, **kwargs) -> None:
    """
    필터링된 데이터를 분할하여 저장합니다.

    Args:
        config_path: 설정 파일 경로
        features_path: 특성 정의 파일 경로
        **kwargs: 추가 매개변수
    """
    logger = logging.getLogger(__name__)
    logger.info("=== 데이터 분할 시작 ===")

    try:
        # 필터링된 데이터 로드
        filtered_data = load_data("filter", "filter", config_path)
        logger.info(f"필터링된 데이터 로드 완료: {filtered_data.shape}")

        # 데이터 분할
        external_data, train_data, test_data = split_data(
            filtered_data, config_path, features_path
        )
        logger.info(
            f"데이터 분할 완료 - 외부검증: {external_data.shape}, 훈련: {train_data.shape}, 테스트: {test_data.shape}")

        # 후처리
        train_data, test_data, train_ml_data, test_ml_data, derivation_data, derivation_data_ml = postprocess(
            train_data, test_data, config_path, features_path)
        logger.info(f"후처리 완료 - 훈련: {train_data.shape}, 테스트: {test_data.shape}")

        # 분할된 데이터 저장
        save_data(derivation_data, "split", "derivation", config_path)
        save_data(train_data, "split", "train", config_path)
        save_data(test_data, "split", "test", config_path)
        save_data(train_ml_data, "split", "train_ml", config_path)
        save_data(test_ml_data, "split", "test_ml", config_path)
        save_data(derivation_data_ml, "split", "derivation_ml", config_path)
        save_data(external_data, "external", "external", config_path)
        logger.info("분할된 데이터 저장 완료")

    except Exception as e:
        logger.error(f"데이터 분할 중 오류 발생: {e}")
        raise
