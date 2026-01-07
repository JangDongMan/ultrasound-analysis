#!/usr/bin/env python3
"""
검출 알고리즘 디버깅 - 실제 신호와 수동 레이블 비교
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
from visualize_signal_improved import parse_ultrasound_csv, load_manual_label_json
from scipy.signal import find_peaks

def analyze_single_sample(patient_id, position):
    """단일 샘플 상세 분석"""

    # 데이터 로드
    file_path = f'data/{patient_id}-5M-{position}.csv'
    manual_label = load_manual_label_json(patient_id, position)

    if manual_label is None:
        print(f"No manual label for {patient_id}-{position}")
        return

    time_data, voltage_data = parse_ultrasound_csv(file_path)
    time_us = time_data * 1e6

    manual_start_us = manual_label['start_point_us']
    manual_positions = manual_label['positions']

    # 시작점 기준
    start_idx = np.argmin(np.abs(time_us - manual_start_us))
    analysis_time_us = time_us[start_idx:] - manual_start_us
    analysis_voltage = voltage_data[start_idx:]

    # 6mm 범위
    speed_of_sound = 1540
    max_time_us = (6.0 / 1000) * 2 / speed_of_sound * 1e6
    skin_end_idx = np.argmin(np.abs(analysis_time_us - max_time_us))

    # 플롯
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))

    # 1. 원본 신호
    ax1.plot(analysis_time_us[:skin_end_idx], analysis_voltage[:skin_end_idx], 'b-', linewidth=1)
    ax1.axvline(x=0, color='purple', linestyle='--', label='T0 (Start)')

    for i, pos in enumerate(manual_positions):
        pos_time = pos['time_us'] - manual_start_us
        ax1.axvline(x=pos_time, color='red', linestyle='-', linewidth=2,
                   label=f"Manual {pos['position_name']}: {pos['thickness_mm']:.2f}mm")

    ax1.set_xlabel('Time (μs)')
    ax1.set_ylabel('Voltage (V)')
    ax1.set_title(f'{patient_id} Position {position} - Raw Signal')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. 절댓값
    abs_voltage = np.abs(analysis_voltage[:skin_end_idx])
    ax2.plot(analysis_time_us[:skin_end_idx], abs_voltage, 'g-', linewidth=1)
    ax2.axvline(x=0, color='purple', linestyle='--', label='T0')

    for i, pos in enumerate(manual_positions):
        pos_time = pos['time_us'] - manual_start_us
        ax2.axvline(x=pos_time, color='red', linestyle='-', linewidth=2,
                   label=f"Manual {pos['position_name']}")

    # 피크 검출
    peaks, properties = find_peaks(abs_voltage, distance=10, prominence=np.std(abs_voltage)*0.2)
    ax2.plot(analysis_time_us[peaks], abs_voltage[peaks], 'bx', markersize=10, label='Detected Peaks')

    ax2.set_xlabel('Time (μs)')
    ax2.set_ylabel('|Voltage| (V)')
    ax2.set_title('Absolute Value + Peak Detection')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. 1차 미분 (기울기)
    gradient = np.gradient(abs_voltage)
    ax3.plot(analysis_time_us[:skin_end_idx], gradient, 'orange', linewidth=1)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.axvline(x=0, color='purple', linestyle='--', label='T0')

    for i, pos in enumerate(manual_positions):
        pos_time = pos['time_us'] - manual_start_us
        ax3.axvline(x=pos_time, color='red', linestyle='-', linewidth=2,
                   label=f"Manual {pos['position_name']}")

    ax3.set_xlabel('Time (μs)')
    ax3.set_ylabel('Gradient')
    ax3.set_title('1st Derivative (Gradient)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. 2차 미분 (변곡점)
    gradient2 = np.gradient(gradient)
    ax4.plot(analysis_time_us[:skin_end_idx], gradient2, 'brown', linewidth=1)
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax4.axvline(x=0, color='purple', linestyle='--', label='T0')

    for i, pos in enumerate(manual_positions):
        pos_time = pos['time_us'] - manual_start_us
        ax4.axvline(x=pos_time, color='red', linestyle='-', linewidth=2,
                   label=f"Manual {pos['position_name']}")

    # 변곡점 표시
    inflection_points = []
    for i in range(1, len(gradient2) - 1):
        if gradient2[i-1] < 0 and gradient2[i] > 0:
            inflection_points.append(i)

    if len(inflection_points) > 0:
        ax4.plot(analysis_time_us[inflection_points], gradient2[inflection_points],
                'go', markersize=8, label='Inflection Points')

    ax4.set_xlabel('Time (μs)')
    ax4.set_ylabel('2nd Derivative')
    ax4.set_title('2nd Derivative (Inflection Points)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    save_path = f'results/debug_{patient_id}_pos{position}.png'
    os.makedirs('results', exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")

    # 분석 결과 출력
    print(f"\n{'='*60}")
    print(f"{patient_id} Position {position} Analysis")
    print(f"{'='*60}")

    dermis_manual_time = manual_positions[0]['time_us'] - manual_start_us
    fascia_manual_time = manual_positions[1]['time_us'] - manual_start_us

    print(f"\nManual Labels:")
    print(f"  Dermis: {dermis_manual_time:.2f} μs ({manual_positions[0]['thickness_mm']:.2f} mm)")
    print(f"  Fascia: {fascia_manual_time:.2f} μs ({manual_positions[1]['thickness_mm']:.2f} mm)")

    print(f"\nDetected Peaks (top 5 by height):")
    if len(peaks) > 0:
        peak_heights = abs_voltage[peaks]
        sorted_indices = np.argsort(peak_heights)[::-1]
        for i in range(min(5, len(peaks))):
            idx = sorted_indices[i]
            peak_time = analysis_time_us[peaks[idx]]
            peak_depth = peak_time * 1e-6 * speed_of_sound / 2 * 1000
            print(f"  Peak {i+1}: {peak_time:.2f} μs ({peak_depth:.2f} mm), height={peak_heights[idx]:.4f}V")

    print(f"\nInflection Points (first 5):")
    for i in range(min(5, len(inflection_points))):
        ip_time = analysis_time_us[inflection_points[i]]
        ip_depth = ip_time * 1e-6 * speed_of_sound / 2 * 1000
        print(f"  IP {i+1}: {ip_time:.2f} μs ({ip_depth:.2f} mm)")

    plt.close()

if __name__ == "__main__":
    # 여러 샘플 분석
    samples = [
        ('bhjung', 1),
        ('bhjung', 5),
        ('cmkim', 1),
        ('cmkim', 5),
        ('Drpark', 1),
        ('Drpark', 5),
    ]

    for patient_id, position in samples:
        analyze_single_sample(patient_id, position)
