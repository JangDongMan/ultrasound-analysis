#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
초음파 데이터 분석 및 전처리 모듈
얼굴 초음파 신호 데이터를 로드하고 분석하는 기능을 제공합니다.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

class UltrasoundDataLoader:
    """초음파 데이터 로더 클래스"""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.sample_rate = 1e6  # 1MHz 샘플링 주파수 (추정)
        print(f"초음파 데이터 로더 초기화: {data_dir}")

    def parse_filename(self, filename: str) -> Dict[str, str]:
        """파일명에서 메타데이터 추출
        형식: {patient_name}-{probe}-{region}.csv
        예: bhjung-5M-1.csv -> patient_name: bhjung, probe: 5M, region: 1
        """
        try:
            # .csv 확장자 제거
            name_without_ext = filename.replace('.csv', '')

            # '-'로 분리
            parts = name_without_ext.split('-')

            if len(parts) >= 3:
                patient_name = parts[0]
                probe = parts[1]
                region = parts[2]
            else:
                # 예외 처리: 기본값 설정
                patient_name = parts[0] if len(parts) > 0 else 'unknown'
                probe = parts[1] if len(parts) > 1 else 'unknown'
                region = parts[2] if len(parts) > 2 else 'unknown'

            return {
                'patient_name': patient_name,
                'probe': probe,
                'region': region,
                'filename': filename
            }

        except Exception as e:
            print(f"파일명 파싱 실패 {filename}: {e}")
            return {
                'patient_name': 'unknown',
                'probe': 'unknown',
                'region': 'unknown',
                'filename': filename
            }

    def load_single_file(self, filepath: str) -> Tuple[np.ndarray, np.ndarray]:
        """단일 CSV 파일 로드"""
        try:
            # 첫 두 줄은 헤더이므로 skip
            df = pd.read_csv(filepath, header=None, skiprows=2)

            # 시간과 전압 데이터 추출
            time_data = df.iloc[:, 0].values.astype(float)
            voltage_data = df.iloc[:, 1].values.astype(float)

            return time_data, voltage_data

        except Exception as e:
            print(f"파일 로드 실패 {filepath}: {e}")
            return np.array([]), np.array([])

    def load_patient_data(self, patient_id: str) -> Dict[str, Dict]:
        """특정 환자의 모든 측정 데이터 로드"""
        patient_files = [f for f in os.listdir(self.data_dir)
                        if f.startswith(patient_id) and f.endswith('.csv')]

        patient_data = {}
        for filename in sorted(patient_files):
            filepath = os.path.join(self.data_dir, filename)
            time_data, voltage_data = self.load_single_file(filepath)

            if len(time_data) > 0:
                # 파일명에서 메타데이터 추출
                metadata = self.parse_filename(filename)

                # 측정 데이터 구성
                measurement_data = {
                    'metadata': metadata,
                    'time_data': time_data,
                    'voltage_data': voltage_data,
                    'analysis': self.analyze_signal(time_data, voltage_data)
                }

                # region을 키로 사용 (1, 2, 3, ...)
                measurement_id = metadata['region']
                patient_data[measurement_id] = measurement_data

        return patient_data

    def load_all_patients(self) -> Dict[str, Dict[str, Dict]]:
        """모든 환자 데이터 로드"""
        all_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        patient_ids = set()

        # 환자 ID 추출 (patient_name)
        for filename in all_files:
            metadata = self.parse_filename(filename)
            patient_ids.add(metadata['patient_name'])

        all_data = {}
        for patient_id in sorted(patient_ids):
            print(f"환자 데이터 로드 중: {patient_id}")
            patient_data = self.load_patient_data(patient_id)
            if patient_data:
                all_data[patient_id] = patient_data

        return all_data

    def analyze_signal(self, time_data: np.ndarray, voltage_data: np.ndarray) -> Dict[str, float]:
        """초음파 신호 기본 분석"""
        if len(voltage_data) == 0:
            return {}

        analysis = {
            'duration': time_data[-1] - time_data[0] if len(time_data) > 1 else 0,
            'samples': len(voltage_data),
            'mean_voltage': np.mean(voltage_data),
            'std_voltage': np.std(voltage_data),
            'max_voltage': np.max(voltage_data),
            'min_voltage': np.min(voltage_data),
            'rms_voltage': np.sqrt(np.mean(voltage_data**2)),
            'peak_to_peak': np.max(voltage_data) - np.min(voltage_data)
        }

        return analysis

    def plot_signal(self, time_data: np.ndarray, voltage_data: np.ndarray,
                   title: str = "초음파 신호", save_path: Optional[str] = None):
        """초음파 신호 시각화"""
        if len(voltage_data) == 0:
            print("표시할 데이터가 없습니다.")
            return

        plt.figure(figsize=(12, 6))

        plt.subplot(2, 1, 1)
        plt.plot(time_data * 1e6, voltage_data * 1e3)  # μs, mV 단위로 변환
        plt.xlabel('시간 (μs)')
        plt.ylabel('전압 (mV)')
        plt.title(f'{title} - 시간 도메인')
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 1, 2)
        # FFT 분석
        if len(voltage_data) > 1:
            fft_freq = np.fft.fftfreq(len(voltage_data), d=(time_data[1] - time_data[0]))
            fft_magnitude = np.abs(np.fft.fft(voltage_data))

            # 양의 주파수만 표시
            pos_mask = fft_freq > 0
            plt.plot(fft_freq[pos_mask] / 1e6, fft_magnitude[pos_mask])  # MHz 단위
            plt.xlabel('주파수 (MHz)')
            plt.ylabel('크기')
            plt.title(f'{title} - 주파수 도메인')
            plt.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"그래프 저장됨: {save_path}")

        plt.show()

def main():
    """메인 함수 - 데이터 분석 데모"""
    loader = UltrasoundDataLoader()

    # 모든 환자 데이터 로드
    print("모든 환자 데이터 로드 중...")
    all_data = loader.load_all_patients()

    print(f"\n총 환자 수: {len(all_data)}")

    # 각 환자별 데이터 요약
    for patient_id, patient_data in all_data.items():
        print(f"\n환자 {patient_id}: {len(patient_data)}개 측정")

        # 첫 번째 측정 데이터 분석
        if patient_data:
            first_measurement = list(patient_data.keys())[0]
            measurement_data = patient_data[first_measurement]

            metadata = measurement_data['metadata']
            analysis = measurement_data['analysis']

            print(f"  프로브: {metadata['probe']}, 부위: {metadata['region']}")
            print(f"  샘플 수: {analysis['samples']}, "
                  f"기간: {analysis['duration']*1e6:.1f}μs, "
                  f"전압 범위: {analysis['min_voltage']*1e3:.2f} ~ {analysis['max_voltage']*1e3:.2f}mV")

    # 샘플 데이터 시각화 (첫 번째 환자의 첫 번째 측정)
    if all_data:
        first_patient = list(all_data.keys())[0]
        first_measurement = list(all_data[first_patient].keys())[0]
        measurement_data = all_data[first_patient][first_measurement]

        time_data = measurement_data['time_data']
        voltage_data = measurement_data['voltage_data']
        metadata = measurement_data['metadata']

        print(f"\n샘플 데이터 시각화: {first_patient} - 프로브 {metadata['probe']}, 부위 {metadata['region']}")
        loader.plot_signal(time_data, voltage_data,
                          f"{first_patient} 초음파 신호 (프로브 {metadata['probe']}, 부위 {metadata['region']})")

if __name__ == "__main__":
    main()