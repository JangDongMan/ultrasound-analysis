#!/usr/bin/env python3
"""
초음파 피부 두께 분석 스크립트
5MHz 탐촉자 데이터를 분석하여 피부 두께를 측정합니다.
"""

import os
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from utils.preprocessor import UltrasoundPreprocessor
import matplotlib.pyplot as plt

def parse_ultrasound_csv(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """CSV 파일에서 시간과 전압 데이터를 파싱"""
    times = []
    voltages = []

    with open(file_path, 'r') as f:
        lines = f.readlines()[2:]  # 헤더 2줄 건너뛰기

        for line in lines:
            parts = line.strip().split(',')
            if len(parts) >= 2 and parts[1]:  # 전압 값이 있는 경우만
                try:
                    time_val = float(parts[0])
                    volt_val = float(parts[1])
                    times.append(time_val)
                    voltages.append(volt_val)
                except ValueError:
                    continue

    return np.array(times), np.array(voltages)

def analyze_patient_data(patient_files: List[str], patient_id: str) -> Dict:
    """단일 환자의 모든 부위 데이터를 분석"""
    results = []

    for file_path in patient_files:
        try:
            # 데이터 로드
            time_data, voltage_data = parse_ultrasound_csv(file_path)

            if len(voltage_data) < 10:
                continue

            # 샘플링 주파수 계산
            dt = time_data[1] - time_data[0]
            sample_rate = int(1.0 / dt)

            # 펄스 시작점 찾기 (진폭이 큰 신호의 시작)
            threshold = np.std(voltage_data) * 3
            pulse_start_indices = np.where(np.abs(voltage_data) > threshold)[0]

            if len(pulse_start_indices) == 0:
                continue

            pulse_start_idx = pulse_start_indices[0]

            # 펄스 시작점 이후 데이터만 사용
            analysis_time = time_data[pulse_start_idx:] - time_data[pulse_start_idx]
            analysis_voltage = voltage_data[pulse_start_idx:]

            # 전처리기 초기화
            preprocessor = UltrasoundPreprocessor(sample_rate=sample_rate)

            # 데이터 전처리
            filtered_data = preprocessor.apply_bandpass_filter(analysis_voltage)

            # 피크 검출 (펄스 시작점 이후)
            peaks, peak_info = preprocessor.detect_peaks(filtered_data, max_peaks=3)

            # 두께 계산 (첫 번째와 두 번째 피크 사이)
            thickness_result = {'thickness_mm': 0.0, 'num_layers': 0}
            if len(peaks) >= 2:
                # 피크 간 시간 차이 계산
                peak_times = analysis_time[peaks[:2]]
                time_diff = peak_times[1] - peak_times[0]

                # 두께 계산: thickness = (time_diff * speed_of_sound) / 2
                thickness = (time_diff * preprocessor.speed_of_sound) / 2
                thickness_mm = thickness * 1000  # mm 단위

                thickness_result = {
                    'thickness_mm': thickness_mm,
                    'num_layers': 1,
                    'individual_thicknesses': [thickness_mm]
                }

            # 결과 저장
            result = {
                'file_path': file_path,
                'position': os.path.basename(file_path).split('-')[-1].replace('.csv', ''),
                'sample_rate': sample_rate,
                'pulse_start_time': time_data[pulse_start_idx],
                'num_peaks': len(peaks),
                'thickness_mm': thickness_result['thickness_mm'],
                'num_layers': thickness_result['num_layers'],
                'time_data': analysis_time,
                'processed_data': filtered_data,
                'peaks': peaks
            }
            results.append(result)

        except Exception as e:
            print(f"파일 처리 오류 {file_path}: {e}")
            continue

    return {
        'patient_id': patient_id,
        'results': results,
        'avg_thickness': np.mean([r['thickness_mm'] for r in results if r['thickness_mm'] > 0]),
        'std_thickness': np.std([r['thickness_mm'] for r in results if r['thickness_mm'] > 0]),
        'num_positions': len(results)
    }

def main():
    """메인 분석 함수"""
    data_dir = 'data'
    patient_groups = {}

    # 파일들을 환자별로 그룹화
    for filename in os.listdir(data_dir):
        if filename.endswith('.csv'):
            patient_id = filename.split('-')[0]
            if patient_id not in patient_groups:
                patient_groups[patient_id] = []
            patient_groups[patient_id].append(os.path.join(data_dir, filename))

    print(f"발견된 환자 수: {len(patient_groups)}")

    all_results = []

    # 각 환자 데이터 분석
    for patient_id, files in patient_groups.items():
        print(f"\n{patient_id} 환자 분석 중... ({len(files)}개 파일)")
        patient_result = analyze_patient_data(files, patient_id)
        all_results.append(patient_result)

        print(".4f"
              ".4f"
              ".1f")

    # 전체 통계
    valid_thicknesses = [r['avg_thickness'] for r in all_results if r['avg_thickness'] > 0]
    if valid_thicknesses:
        overall_mean = np.mean(valid_thicknesses)
        overall_std = np.std(valid_thicknesses)
        cv = (overall_std / overall_mean) * 100 if overall_mean > 0 else 0

        print(f"\n=== 전체 분석 결과 ===")
        print(f"분석된 환자 수: {len(valid_thicknesses)}")
        print(f"평균 피부 두께: {overall_mean:.4f} mm")
        print(f"표준편차: {overall_std:.4f} mm")
        print(f"변동계수: {cv:.1f}%")

        # 정상 범위 확인 (0.5-2.0mm)
        normal_range = [t for t in valid_thicknesses if 0.5 <= t <= 2.0]
        print(f"정상 범위(0.5-2.0mm) 내 데이터: {len(normal_range)}/{len(valid_thicknesses)} ({len(normal_range)/len(valid_thicknesses)*100:.1f}%)")

if __name__ == "__main__":
    main()