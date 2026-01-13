#!/usr/bin/env python3
"""
C로 구현된 경계 검출 알고리즘 테스트
"""

import json
import numpy as np
from utils2.boundary_detector_wrapper import BoundaryDetector


def parse_ultrasound_csv(file_path):
    """CSV 파일에서 시간과 전압 데이터 파싱"""
    times = []
    voltages = []

    with open(file_path, 'r') as f:
        lines = f.readlines()[2:]  # 헤더 2줄 건너뛰기

        for line in lines:
            parts = line.strip().split(',')
            if len(parts) >= 2 and parts[1]:
                try:
                    time_val = float(parts[0])
                    volt_val = float(parts[1])
                    times.append(time_val)
                    voltages.append(volt_val)
                except ValueError:
                    continue

    return np.array(times), np.array(voltages)


def main():
    # 테스트 데이터
    test_cases = [
        ('bhjung', 1),
        ('bhjung', 7),
        ('cmkim', 1),
    ]

    # 검출기 초기화
    detector = BoundaryDetector()

    print(f"\n{'='*80}")
    print(f"Testing C Boundary Detector with Real Data")
    print(f"{'='*80}\n")

    for patient_id, position in test_cases:
        file_path = f'data/{patient_id}-5M-{position}.csv'
        json_path = f'manual_boundaries/{patient_id}-5M-{position}_positions.json'

        # 데이터 로드
        time_data, voltage_data = parse_ultrasound_csv(file_path)
        time_us = time_data * 1e6  # μs로 변환

        # 수동 레이블 로드
        with open(json_path, 'r') as f:
            manual_label = json.load(f)

        manual_start_us = manual_label['start_point_us']
        manual_dermis_depth = manual_label['positions'][0]['thickness_mm']
        manual_fascia_depth = manual_label['positions'][1]['thickness_mm']

        # C 검출
        dermis_idx, fascia_idx, success = detector.detect(
            time_us, voltage_data, manual_start_us
        )

        print(f"\n{'─'*80}")
        print(f"{patient_id} Position {position}")
        print(f"{'─'*80}")

        if not success:
            print("✗ Detection failed")
            continue

        # 시작점 인덱스
        start_idx = np.argmin(np.abs(time_us - manual_start_us))
        analysis_time_us = time_us[start_idx:] - manual_start_us

        # 자동 검출 결과
        speed_of_sound = 1540  # m/s
        auto_dermis_time = analysis_time_us[dermis_idx]
        auto_fascia_time = analysis_time_us[fascia_idx]
        auto_dermis_depth = auto_dermis_time * 1e-6 * speed_of_sound / 2 * 1000
        auto_fascia_depth = auto_fascia_time * 1e-6 * speed_of_sound / 2 * 1000

        # 오차 계산
        dermis_error = abs(auto_dermis_depth - manual_dermis_depth)
        fascia_error = abs(auto_fascia_depth - manual_fascia_depth)

        print(f"\nManual Labels:")
        print(f"  Dermis: {manual_dermis_depth:.2f} mm")
        print(f"  Fascia: {manual_fascia_depth:.2f} mm")

        print(f"\nC Detection:")
        print(f"  Dermis: {auto_dermis_depth:.2f} mm (error: {dermis_error:.2f} mm)")
        print(f"  Fascia: {auto_fascia_depth:.2f} mm (error: {fascia_error:.2f} mm)")

        status = '✓' if (dermis_error < 0.5 and fascia_error < 0.5) else '△'
        print(f"\nResult: {status}")

    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
