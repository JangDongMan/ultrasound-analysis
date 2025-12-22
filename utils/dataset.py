#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
초음파 데이터셋 및 모델 학습 준비 모듈
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Dict, List, Tuple, Optional, Union
import pickle
from datetime import datetime

from .data_loader import UltrasoundDataLoader
from .preprocessor import UltrasoundPreprocessor

class UltrasoundDataset(Dataset):
    """초음파 데이터셋 클래스"""

    def __init__(self, features: np.ndarray, labels: Optional[np.ndarray] = None,
                 transform: Optional[callable] = None):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels) if labels is not None else None
        self.transform = transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        sample = self.features[idx]

        if self.transform:
            sample = self.transform(sample)

        if self.labels is not None:
            return sample, self.labels[idx]
        else:
            return sample

class UltrasoundDataProcessor:
    """초음파 데이터 처리 및 준비 클래스"""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.loader = UltrasoundDataLoader(data_dir)
        self.preprocessor = UltrasoundPreprocessor()
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

        print("초음파 데이터 프로세서 초기화")

    def create_feature_dataset(self, patient_ids: Optional[List[str]] = None,
                             save_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """특성 데이터셋 생성"""
        print("특성 데이터셋 생성 중...")

        # 데이터 로드
        if patient_ids is None:
            all_data = self.loader.load_all_patients()
        else:
            all_data = {pid: self.loader.load_patient_data(pid) for pid in patient_ids}

        features_list = []
        labels_list = []

        for patient_id, patient_data in all_data.items():
            print(f"환자 {patient_id} 처리 중...")

            for measurement_id, (time_data, voltage_data) in patient_data.items():
                if len(voltage_data) == 0:
                    continue

                # 특성 추출
                features = self.preprocessor.extract_all_features(voltage_data)

                if features:
                    features_list.append(list(features.values()))
                    labels_list.append(patient_id)  # 환자 ID를 레이블로 사용

        # NumPy 배열로 변환
        X = np.array(features_list)
        y = np.array(labels_list)

        # 레이블 인코딩
        y_encoded = self.label_encoder.fit_transform(y)

        print(f"데이터셋 생성 완료: {X.shape[0]} 샘플, {X.shape[1]} 특성, {len(np.unique(y_encoded))} 클래스")

        # 저장
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.savez(save_path, X=X, y=y_encoded, patient_ids=y)
            print(f"데이터셋 저장됨: {save_path}")

        return X, y_encoded

    def create_signal_dataset(self, sequence_length: int = 1024,
                            patient_ids: Optional[List[str]] = None,
                            save_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """원본 신호 데이터셋 생성 (시퀀스 모델용)"""
        print("신호 데이터셋 생성 중...")

        # 데이터 로드
        if patient_ids is None:
            all_data = self.loader.load_all_patients()
        else:
            all_data = {pid: self.loader.load_patient_data(pid) for pid in patient_ids}

        signals_list = []
        labels_list = []

        for patient_id, patient_data in all_data.items():
            print(f"환자 {patient_id} 처리 중...")

            for measurement_id, (time_data, voltage_data) in patient_data.items():
                if len(voltage_data) < sequence_length:
                    continue

                # 신호 전처리
                processed_signal = self.preprocessor.preprocess_signal(voltage_data)

                # 시퀀스 길이에 맞게 자르기 또는 패딩
                if len(processed_signal) > sequence_length:
                    # 중앙 부분 추출
                    start_idx = (len(processed_signal) - sequence_length) // 2
                    signal_seq = processed_signal[start_idx:start_idx + sequence_length]
                else:
                    # 패딩
                    padding = sequence_length - len(processed_signal)
                    signal_seq = np.pad(processed_signal, (0, padding), 'constant')

                signals_list.append(signal_seq)
                labels_list.append(patient_id)

        # NumPy 배열로 변환
        X = np.array(signals_list).reshape(-1, 1, sequence_length)  # 채널 차원 추가
        y = np.array(labels_list)

        # 레이블 인코딩
        y_encoded = self.label_encoder.fit_transform(y)

        print(f"신호 데이터셋 생성 완료: {X.shape[0]} 샘플, 시퀀스 길이 {sequence_length}")

        # 저장
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.savez(save_path, X=X, y=y_encoded, patient_ids=y)
            print(f"데이터셋 저장됨: {save_path}")

        return X, y_encoded

    def split_dataset(self, X: np.ndarray, y: np.ndarray,
                     test_size: float = 0.2, val_size: float = 0.2,
                     random_state: int = 42) -> Dict[str, Union[np.ndarray, DataLoader]]:
        """데이터셋 분할 및 DataLoader 생성"""

        # Train/Val/Test 분할
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )

        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size_adjusted,
            stratify=y_train_val, random_state=random_state
        )

        print(f"데이터 분할 완료:")
        print(f"  훈련: {len(X_train)} 샘플")
        print(f"  검증: {len(X_val)} 샘플")
        print(f"  테스트: {len(X_test)} 샘플")

        # 특성 스케일링 (특성 데이터셋인 경우)
        if X.ndim == 2 and X.shape[1] > 1:  # 특성 데이터셋
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            X_test_scaled = self.scaler.transform(X_test)
        else:  # 신호 데이터셋
            X_train_scaled, X_val_scaled, X_test_scaled = X_train, X_val, X_test

        # PyTorch Dataset 생성
        train_dataset = UltrasoundDataset(X_train_scaled, y_train)
        val_dataset = UltrasoundDataset(X_val_scaled, y_val)
        test_dataset = UltrasoundDataset(X_test_scaled, y_test)

        # DataLoader 생성
        batch_size = min(32, len(X_train) // 4)  # 적절한 배치 크기

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return {
            'train_loader': train_loader,
            'val_loader': val_loader,
            'test_loader': test_loader,
            'X_train': X_train_scaled,
            'X_val': X_val_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test
        }

    def save_preprocessing_objects(self, save_dir: str = "processed_images"):
        """전처리 객체 저장"""
        os.makedirs(save_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 스케일러 저장
        scaler_path = os.path.join(save_dir, f'scaler_{timestamp}.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)

        # 레이블 인코더 저장
        encoder_path = os.path.join(save_dir, f'label_encoder_{timestamp}.pkl')
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)

        print(f"전처리 객체 저장됨: {scaler_path}, {encoder_path}")

        return scaler_path, encoder_path

def create_data_summary(data_dir: str = "data") -> pd.DataFrame:
    """데이터 요약 생성"""
    processor = UltrasoundDataProcessor(data_dir)
    all_data = processor.loader.load_all_patients()

    summary_data = []

    for patient_id, patient_data in all_data.items():
        for measurement_id, (time_data, voltage_data) in patient_data.items():
            analysis = processor.loader.analyze_signal(time_data, voltage_data)

            summary_data.append({
                'patient_id': patient_id,
                'measurement_id': measurement_id,
                'duration_ms': analysis.get('duration', 0) * 1000,
                'samples': analysis.get('samples', 0),
                'mean_voltage_mv': analysis.get('mean_voltage', 0) * 1000,
                'std_voltage_mv': analysis.get('std_voltage', 0) * 1000,
                'max_voltage_mv': analysis.get('max_voltage', 0) * 1000,
                'min_voltage_mv': analysis.get('min_voltage', 0) * 1000,
                'rms_voltage_mv': analysis.get('rms_voltage', 0) * 1000,
                'peak_to_peak_mv': analysis.get('peak_to_peak', 0) * 1000
            })

    df_summary = pd.DataFrame(summary_data)
    return df_summary

if __name__ == "__main__":
    # 데이터 요약 생성 및 표시
    print("초음파 데이터 요약 생성 중...")
    summary_df = create_data_summary()

    print("\n데이터 요약:")
    print(summary_df.head())
    print(f"\n총 샘플 수: {len(summary_df)}")
    print(f"총 환자 수: {summary_df['patient_id'].nunique()}")

    # 기본 통계
    print("\n기본 통계:")
    print(summary_df.describe())

    # 환자별 샘플 수
    print("\n환자별 샘플 수:")
    print(summary_df.groupby('patient_id').size().sort_values(ascending=False))