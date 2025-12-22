#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
초음파 데이터 정규화 모듈
데이터 스케일링 및 정규화를 위한 기능을 제공합니다.
"""

import numpy as np
import pickle
import os
from typing import Dict, List, Tuple, Optional, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import warnings

warnings.filterwarnings("ignore")


class UltrasoundDataNormalizer:
    """초음파 데이터 정규화 클래스"""

    def __init__(self, method: str = "standard", save_path: str = "saved_models"):
        """
        Args:
            method: 정규화 방법 ('standard', 'minmax', 'robust')
            save_path: 스케일러 저장 경로
        """
        self.method = method
        self.save_path = save_path
        self.scalers = {}

        # 저장 디렉토리 생성
        os.makedirs(save_path, exist_ok=True)

        print(f"데이터 정규화 초기화: {method} 방법 사용")

    def _get_scaler(self, scaler_type: str = "voltage"):
        """스케일러 객체 생성"""
        if self.method == "standard":
            return StandardScaler()
        elif self.method == "minmax":
            return MinMaxScaler()
        elif self.method == "robust":
            return RobustScaler()
        else:
            raise ValueError(f"지원하지 않는 정규화 방법: {self.method}")

    def fit_scaler(self, data: np.ndarray, scaler_name: str = "voltage") -> None:
        """데이터에 스케일러를 학습시키고 저장"""
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        scaler = self._get_scaler()
        scaler.fit(data)
        self.scalers[scaler_name] = scaler

        # 스케일러 저장
        scaler_path = os.path.join(
            self.save_path, f"{scaler_name}_scaler_{self.method}.pkl"
        )
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)

        print(f"스케일러 학습 및 저장 완료: {scaler_name} ({scaler_path})")

    def transform_data(
        self, data: np.ndarray, scaler_name: str = "voltage"
    ) -> np.ndarray:
        """학습된 스케일러로 데이터 변환"""
        if scaler_name not in self.scalers:
            raise ValueError(f"스케일러가 학습되지 않음: {scaler_name}")

        if data.ndim == 1:
            data = data.reshape(-1, 1)

        transformed = self.scalers[scaler_name].transform(data)

        # 1D로 변환하여 반환
        return transformed.flatten()

    def fit_transform_data(
        self, data: np.ndarray, scaler_name: str = "voltage"
    ) -> np.ndarray:
        """데이터에 스케일러를 학습시키고 변환"""
        self.fit_scaler(data, scaler_name)
        return self.transform_data(data, scaler_name)

    def inverse_transform_data(
        self, data: np.ndarray, scaler_name: str = "voltage"
    ) -> np.ndarray:
        """정규화된 데이터를 원래 스케일로 복원"""
        if scaler_name not in self.scalers:
            raise ValueError(f"스케일러가 학습되지 않음: {scaler_name}")

        if data.ndim == 1:
            data = data.reshape(-1, 1)

        inverse_transformed = self.scalers[scaler_name].inverse_transform(data)

        # 1D로 변환하여 반환
        return inverse_transformed.flatten()

    def load_scaler(self, scaler_name: str = "voltage") -> None:
        """저장된 스케일러 로드"""
        scaler_path = os.path.join(
            self.save_path, f"{scaler_name}_scaler_{self.method}.pkl"
        )

        if os.path.exists(scaler_path):
            with open(scaler_path, "rb") as f:
                self.scalers[scaler_name] = pickle.load(f)
            print(f"스케일러 로드 완료: {scaler_name} ({scaler_path})")
        else:
            print(f"스케일러 파일이 존재하지 않음: {scaler_path}")

    def normalize_patient_data(self, patient_data: Dict[str, Dict]) -> Dict[str, Dict]:
        """환자 데이터를 정규화"""
        normalized_data = {}

        for measurement_id, measurement_data in patient_data.items():
            voltage_data = measurement_data["voltage_data"]

            # NaN 값 제거 및 유효한 데이터만 사용
            valid_mask = ~np.isnan(voltage_data)
            if np.sum(valid_mask) == 0:
                print(f"경고: 측정 {measurement_id}에 유효한 데이터가 없음")
                continue

            valid_voltage = voltage_data[valid_mask]

            # 전압 데이터 정규화
            normalized_voltage = self.fit_transform_data(valid_voltage, "voltage")

            # 정규화된 데이터를 원래 위치에 복원
            normalized_voltage_full = np.full_like(voltage_data, np.nan)
            normalized_voltage_full[valid_mask] = normalized_voltage

            # 정규화된 데이터로 업데이트
            normalized_measurement = measurement_data.copy()
            normalized_measurement["voltage_data"] = normalized_voltage_full
            normalized_measurement["original_voltage_data"] = voltage_data.copy()

            normalized_data[measurement_id] = normalized_measurement

        return normalized_data

    def normalize_all_data(
        self, all_data: Dict[str, Dict[str, Dict]]
    ) -> Dict[str, Dict[str, Dict]]:
        """모든 환자 데이터를 정규화 (글로벌 스케일러 사용)"""
        print("데이터 정규화 시작...")

        # 1. 모든 유효한 전압 데이터를 수집하여 글로벌 스케일러 학습
        all_voltage_data = []

        for patient_id, patient_data in all_data.items():
            for measurement_id, measurement_data in patient_data.items():
                voltage_data = measurement_data["voltage_data"]
                valid_mask = ~np.isnan(voltage_data)
                if np.sum(valid_mask) > 0:
                    all_voltage_data.extend(voltage_data[valid_mask])

        if not all_voltage_data:
            raise ValueError("정규화할 유효한 데이터가 없습니다.")

        all_voltage_array = np.array(all_voltage_data)

        # 글로벌 스케일러 학습
        self.fit_scaler(all_voltage_array, "global_voltage")

        # 2. 학습된 스케일러로 모든 데이터 변환
        normalized_all_data = {}

        for patient_id, patient_data in all_data.items():
            print(f"환자 {patient_id} 데이터 정규화 적용 중...")
            normalized_patient_data = {}

            for measurement_id, measurement_data in patient_data.items():
                voltage_data = measurement_data["voltage_data"]
                valid_mask = ~np.isnan(voltage_data)

                if np.sum(valid_mask) == 0:
                    print(f"경고: 측정 {measurement_id}에 유효한 데이터가 없음")
                    continue

                valid_voltage = voltage_data[valid_mask]

                # 글로벌 스케일러로 변환
                normalized_voltage = self.transform_data(
                    valid_voltage, "global_voltage"
                )

                # 정규화된 데이터를 원래 위치에 복원
                normalized_voltage_full = np.full_like(voltage_data, np.nan)
                normalized_voltage_full[valid_mask] = normalized_voltage

                # 정규화된 데이터로 업데이트
                normalized_measurement = measurement_data.copy()
                normalized_measurement["voltage_data"] = normalized_voltage_full
                normalized_measurement["original_voltage_data"] = voltage_data.copy()

                normalized_patient_data[measurement_id] = normalized_measurement

            normalized_all_data[patient_id] = normalized_patient_data

        print("데이터 정규화 완료")
        return normalized_all_data

    def get_normalization_stats(self, scaler_name: str = "voltage") -> Dict[str, float]:
        """정규화 통계 정보 반환"""
        if scaler_name not in self.scalers:
            return {}

        scaler = self.scalers[scaler_name]

        stats = {"method": self.method, "scaler_type": type(scaler).__name__}

        if hasattr(scaler, "mean_"):
            stats["mean"] = scaler.mean_[0] if len(scaler.mean_) > 0 else None
        if hasattr(scaler, "scale_"):
            stats["scale"] = scaler.scale_[0] if len(scaler.scale_) > 0 else None
        if hasattr(scaler, "min_"):
            stats["min"] = scaler.min_[0] if len(scaler.min_) > 0 else None
        if hasattr(scaler, "scale_") and hasattr(scaler, "min_"):
            stats["max"] = (
                (scaler.scale_[0] + scaler.min_[0])
                if len(scaler.scale_) > 0 and len(scaler.min_) > 0
                else None
            )

        return stats


def main():
    """메인 함수 - 정규화 데모"""
    from data_loader import UltrasoundDataLoader

    # 데이터 로드
    loader = UltrasoundDataLoader()
    all_data = loader.load_all_patients()

    # 정규화 적용
    normalizer = UltrasoundDataNormalizer(method="standard")
    normalized_data = normalizer.normalize_all_data(all_data)

    # 정규화 통계 출력
    stats = normalizer.get_normalization_stats("global_voltage")
    print(f"\n정규화 통계: {stats}")

    # 샘플 비교
    if normalized_data:
        first_patient = list(normalized_data.keys())[0]
        first_measurement = list(normalized_data[first_patient].keys())[0]

        original = normalized_data[first_patient][first_measurement][
            "original_voltage_data"
        ]
        normalized = normalized_data[first_patient][first_measurement]["voltage_data"]

        print(f"\n샘플 데이터 비교 ({first_patient}, 측정 {first_measurement}):")
        print(f"원본 전압 범위: {np.nanmin(original):.2f} ~ {np.nanmax(original):.2f}")
        print(
            f"정규화 전압 범위: {np.nanmin(normalized):.2f} ~ {np.nanmax(normalized):.2f}"
        )


if __name__ == "__main__":
    main()
