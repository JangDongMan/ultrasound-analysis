#!/usr/bin/env python3
"""
자동 검출 알고리즘의 정확도 평가
"""

import numpy as np
import json
import os
from visualize_signal_improved import (
    parse_ultrasound_csv,
    load_manual_label_json,
    detect_positions_with_reference
)


def evaluate_all_samples():
    """모든 샘플에 대해 검출 정확도 평가"""

    patient_ids = ['bhjung', 'cmkim', 'Drpark']

    all_results = []

    print(f"\n{'='*80}")
    print(f"Detection Algorithm Accuracy Evaluation")
    print(f"{'='*80}\n")

    for patient_id in patient_ids:
        print(f"\n{'─'*80}")
        print(f"Patient: {patient_id}")
        print(f"{'─'*80}")

        for position in range(1, 9):
            file_path = f'data/{patient_id}-5M-{position}.csv'

            if not os.path.exists(file_path):
                continue

            # 수동 레이블 로드
            manual_label = load_manual_label_json(patient_id, position)
            if manual_label is None:
                continue

            # 데이터 로드
            time_data, voltage_data = parse_ultrasound_csv(file_path)

            # 수동 레이블 값
            manual_start_us = manual_label['start_point_us']
            manual_dermis_depth = manual_label['positions'][0]['thickness_mm']
            manual_fascia_depth = manual_label['positions'][1]['thickness_mm']
            manual_dermis_time = manual_label['positions'][0]['time_us'] - manual_start_us
            manual_fascia_time = manual_label['positions'][1]['time_us'] - manual_start_us

            # 자동 검출
            sample_rate = 100e6  # 100MHz
            detected = detect_positions_with_reference(
                time_data, voltage_data, manual_start_us, sample_rate
            )

            if detected[0] is None:
                print(f"  Position {position}: ✗ Detection failed")
                all_results.append({
                    'patient': patient_id,
                    'position': position,
                    'success': False
                })
                continue

            auto_positions, _, analysis_time_us = detected

            # 자동 검출 값
            speed_of_sound = 1540  # m/s
            auto_dermis_time = analysis_time_us[auto_positions[0]]
            auto_fascia_time = analysis_time_us[auto_positions[1]]
            auto_dermis_depth = auto_dermis_time * 1e-6 * speed_of_sound / 2 * 1000
            auto_fascia_depth = auto_fascia_time * 1e-6 * speed_of_sound / 2 * 1000

            # 오차 계산
            dermis_time_error = abs(auto_dermis_time - manual_dermis_time)
            fascia_time_error = abs(auto_fascia_time - manual_fascia_time)
            dermis_depth_error = abs(auto_dermis_depth - manual_dermis_depth)
            fascia_depth_error = abs(auto_fascia_depth - manual_fascia_depth)

            result = {
                'patient': patient_id,
                'position': position,
                'success': True,
                'manual_dermis_depth': manual_dermis_depth,
                'auto_dermis_depth': auto_dermis_depth,
                'dermis_depth_error': dermis_depth_error,
                'manual_fascia_depth': manual_fascia_depth,
                'auto_fascia_depth': auto_fascia_depth,
                'fascia_depth_error': fascia_depth_error,
                'dermis_time_error_us': dermis_time_error,
                'fascia_time_error_us': fascia_time_error,
            }

            all_results.append(result)

            # 출력
            status = '✓' if (dermis_depth_error < 0.5 and fascia_depth_error < 0.5) else '△'
            print(f"  Position {position} {status}")
            print(f"    Dermis:  Manual={manual_dermis_depth:.2f}mm, Auto={auto_dermis_depth:.2f}mm, Error={dermis_depth_error:.2f}mm ({dermis_time_error:.2f}μs)")
            print(f"    Fascia:  Manual={manual_fascia_depth:.2f}mm, Auto={auto_fascia_depth:.2f}mm, Error={fascia_depth_error:.2f}mm ({fascia_time_error:.2f}μs)")

    # 통계 계산
    successful_results = [r for r in all_results if r['success']]

    if len(successful_results) == 0:
        print("\n⚠ No successful detections")
        return

    dermis_errors = [r['dermis_depth_error'] for r in successful_results]
    fascia_errors = [r['fascia_depth_error'] for r in successful_results]
    dermis_time_errors = [r['dermis_time_error_us'] for r in successful_results]
    fascia_time_errors = [r['fascia_time_error_us'] for r in successful_results]

    print(f"\n{'='*80}")
    print(f"Overall Statistics (N={len(successful_results)})")
    print(f"{'='*80}")

    print(f"\nDermis Detection:")
    print(f"  Mean Error:   {np.mean(dermis_errors):.3f} ± {np.std(dermis_errors):.3f} mm")
    print(f"  Median Error: {np.median(dermis_errors):.3f} mm")
    print(f"  Max Error:    {np.max(dermis_errors):.3f} mm")
    print(f"  Time Error:   {np.mean(dermis_time_errors):.3f} ± {np.std(dermis_time_errors):.3f} μs")

    print(f"\nFascia Detection:")
    print(f"  Mean Error:   {np.mean(fascia_errors):.3f} ± {np.std(fascia_errors):.3f} mm")
    print(f"  Median Error: {np.median(fascia_errors):.3f} mm")
    print(f"  Max Error:    {np.max(fascia_errors):.3f} mm")
    print(f"  Time Error:   {np.mean(fascia_time_errors):.3f} ± {np.std(fascia_time_errors):.3f} μs")

    # 정확도 기준별 성공률
    thresholds = [0.2, 0.3, 0.5, 1.0]

    print(f"\nAccuracy by Threshold:")
    print(f"{'Threshold (mm)':<20} {'Dermis':<15} {'Fascia':<15} {'Both':<15}")
    print(f"{'-'*65}")

    for thresh in thresholds:
        dermis_count = sum(1 for e in dermis_errors if e < thresh)
        fascia_count = sum(1 for e in fascia_errors if e < thresh)
        both_count = sum(1 for r in successful_results
                        if r['dermis_depth_error'] < thresh and r['fascia_depth_error'] < thresh)

        total = len(successful_results)
        print(f"< {thresh} mm{' '*13} {dermis_count}/{total} ({dermis_count/total*100:.1f}%)    "
              f"{fascia_count}/{total} ({fascia_count/total*100:.1f}%)    "
              f"{both_count}/{total} ({both_count/total*100:.1f}%)")

    print(f"\n{'='*80}\n")

    # 가장 큰 오차 케이스 출력
    print(f"Top 5 Worst Cases (Dermis):")
    print(f"{'Patient':<10} {'Pos':<5} {'Manual':<10} {'Auto':<10} {'Error':<10}")
    print(f"{'-'*50}")

    sorted_by_dermis = sorted(successful_results, key=lambda x: x['dermis_depth_error'], reverse=True)
    for r in sorted_by_dermis[:5]:
        print(f"{r['patient']:<10} {r['position']:<5} "
              f"{r['manual_dermis_depth']:<10.2f} {r['auto_dermis_depth']:<10.2f} "
              f"{r['dermis_depth_error']:<10.2f}")

    print(f"\nTop 5 Worst Cases (Fascia):")
    print(f"{'Patient':<10} {'Pos':<5} {'Manual':<10} {'Auto':<10} {'Error':<10}")
    print(f"{'-'*50}")

    sorted_by_fascia = sorted(successful_results, key=lambda x: x['fascia_depth_error'], reverse=True)
    for r in sorted_by_fascia[:5]:
        print(f"{r['patient']:<10} {r['position']:<5} "
              f"{r['manual_fascia_depth']:<10.2f} {r['auto_fascia_depth']:<10.2f} "
              f"{r['fascia_depth_error']:<10.2f}")


if __name__ == "__main__":
    evaluate_all_samples()
