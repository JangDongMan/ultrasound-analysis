#!/usr/bin/env python3
"""
개선된 초음파 신호 시각화 스크립트
수동 레이블 데이터를 참조하여 자동 검출 알고리즘 개선
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import json
from utils.preprocessor import UltrasoundPreprocessor
from scipy.signal import find_peaks

# Matplotlib font configuration
import matplotlib as mpl
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['font.family'] = 'DejaVu Sans'


def load_manual_label_json(patient_id: str, position: int, label_dir: str = './manual_boundaries') -> dict:
    """
    JSON 레이블 파일 로드

    Args:
        patient_id: 환자 ID (예: 'bhjung')
        position: 포지션 번호 (1-8)
        label_dir: 레이블 디렉토리 경로

    Returns:
        레이블 데이터 딕셔너리 또는 None
    """
    file_pattern = f"{patient_id}-5M-{position}"
    json_file = os.path.join(label_dir, f"{file_pattern}_positions.json")

    if os.path.exists(json_file):
        with open(json_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def parse_ultrasound_csv(file_path: str) -> tuple:
    """CSV 파일에서 시간과 전압 데이터를 파싱"""
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


def detect_positions_with_reference(time_data, voltage_data, reference_start_us, sample_rate):
    """
    통계 기반 개선된 피부층 경계 검출 알고리즘

    234개 수동 레이블 분석 결과:
    - 진피 (Thickness1): T0 + 2.33±0.33 μs (1.80±0.25 mm)
    - 근막 (Thickness2): T0 + 5.16±0.79 μs (3.97±0.61 mm)
    - 두께 차이: 2.18±0.59 mm

    검출 전략:
    1. 표피: 초기 큰 피크 (±2.5V)
    2. 진피: 표피 지나 0.5V 이하 후 변곡점, 예상 범위 2.0~2.7 μs에서 탐색
    3. 근막: 진피 +2.18mm 위치에서 피크 탐색, 예상 범위 4.4~5.9 μs

    Args:
        time_data: 시간 데이터 (seconds)
        voltage_data: 전압 데이터
        reference_start_us: 수동 레이블의 시작점 (μs)
        sample_rate: 샘플링 주파수

    Returns:
        positions: [진피 인덱스, 근막 인덱스] 또는 None
    """
    # 통계 기반 파라미터
    DERMIS_EXPECTED_TIME_US = 2.33
    DERMIS_TIME_STD_US = 0.33
    FASCIA_EXPECTED_TIME_US = 5.16
    FASCIA_TIME_STD_US = 0.79
    THICKNESS_GAP_MM = 2.18
    THICKNESS_GAP_STD_MM = 0.59
    time_us = time_data * 1e6

    # 레퍼런스 시작점 찾기
    start_idx = np.argmin(np.abs(time_us - reference_start_us))

    # 시작점 이후 데이터
    analysis_time_us = time_us[start_idx:] - reference_start_us
    analysis_voltage = voltage_data[start_idx:]

    # 6mm 이내 피부층 분석
    speed_of_sound = 1540  # m/s
    max_distance_mm = 6.0
    max_time_us = (max_distance_mm / 1000) * 2 / speed_of_sound * 1e6

    skin_end_idx = np.argmin(np.abs(analysis_time_us - max_time_us))
    skin_voltage = analysis_voltage[:skin_end_idx]

    if len(skin_voltage) < 100:
        return None, analysis_voltage, analysis_time_us

    # Step 1: 표피 영역 제외
    abs_voltage = np.abs(skin_voltage)
    max_voltage = np.max(abs_voltage)

    # 표피 영역: 시간 기반 제한 (1.5μs 이전은 표피로 간주)
    # 통계적으로 진피는 2.0μs 이후에 나타나므로, 안전하게 1.5μs 이전을 제외
    epidermis_cutoff_time_us = 1.5

    epidermis_cutoff_idx = np.argmin(np.abs(analysis_time_us[:len(skin_voltage)] - epidermis_cutoff_time_us))
    epidermis_end = min(epidermis_cutoff_idx, len(skin_voltage) - 100)

    if epidermis_end < 10:
        # 표피 영역이 너무 짧으면 폴백
        return detect_positions_fallback(skin_voltage, analysis_time_us)

    # Step 2: 진피 검출 - 통계 기반 예상 범위 내에서 탐색
    # 예상 시간 범위: 2.0 ~ 2.7 μs
    dermis_min_time_us = DERMIS_EXPECTED_TIME_US - DERMIS_TIME_STD_US
    dermis_max_time_us = DERMIS_EXPECTED_TIME_US + DERMIS_TIME_STD_US

    # 인덱스로 변환
    dermis_min_idx = np.argmin(np.abs(analysis_time_us[:len(skin_voltage)] - dermis_min_time_us))
    dermis_max_idx = np.argmin(np.abs(analysis_time_us[:len(skin_voltage)] - dermis_max_time_us))

    # 탐색 범위 확보
    search_start = max(epidermis_end, dermis_min_idx - 50)
    search_end = min(len(skin_voltage), dermis_max_idx + 100)

    if search_start >= search_end - 20:
        return detect_positions_fallback(skin_voltage, analysis_time_us)

    # 진피 탐색 영역: 예상 시간 범위를 우선적으로 사용
    # 표피 이후이면서 예상 범위(2.0~2.7μs) 내에서 탐색
    dermis_search_start = max(epidermis_end, search_start)
    dermis_search_end = search_end

    # 탐색 영역이 예상 범위와 겹치는지 확인
    # 예상 범위 내에서만 탐색하도록 제한
    expected_start_idx = np.argmin(np.abs(analysis_time_us[:len(skin_voltage)] - dermis_min_time_us))
    expected_end_idx = np.argmin(np.abs(analysis_time_us[:len(skin_voltage)] - dermis_max_time_us))

    # 표피 끝과 예상 시작 중 더 늦은 시점부터 시작
    dermis_search_start = max(dermis_search_start, expected_start_idx - 20)
    dermis_search_end = min(dermis_search_end, expected_end_idx + 20)

    if dermis_search_start >= dermis_search_end - 20:
        return detect_positions_fallback(skin_voltage, analysis_time_us)

    search_region = skin_voltage[dermis_search_start:dermis_search_end]

    if len(search_region) < 20:
        return detect_positions_fallback(skin_voltage, analysis_time_us)

    # 절댓값의 기울기 변화 (변곡점)
    abs_search = np.abs(search_region)
    gradient = np.gradient(abs_search)
    gradient2 = np.gradient(gradient)

    # 상승 변곡점 찾기 (음수에서 양수로 변하는 지점)
    inflection_points = []
    for i in range(1, len(gradient2) - 1):
        if gradient2[i-1] < 0 and gradient2[i] > 0 and gradient[i] > 0:
            inflection_points.append(i)

    # 예상 범위 내 변곡점 우선 선택
    if len(inflection_points) > 0:
        # 예상 시간에 가장 가까운 변곡점 선택
        best_inflection = inflection_points[0]
        best_diff = abs(analysis_time_us[dermis_search_start + best_inflection] - DERMIS_EXPECTED_TIME_US)

        for ip in inflection_points:
            ip_time = analysis_time_us[dermis_search_start + ip]
            if dermis_min_time_us <= ip_time <= dermis_max_time_us:
                time_diff = abs(ip_time - DERMIS_EXPECTED_TIME_US)
                if time_diff < best_diff:
                    best_diff = time_diff
                    best_inflection = ip

        dermis_idx = dermis_search_start + best_inflection
    else:
        # 변곡점을 못 찾으면 예상 시간에 가장 가까운 피크 사용
        peaks, _ = find_peaks(abs_search, distance=10)
        if len(peaks) > 0:
            # 예상 시간에 가장 가까운 피크 선택
            peak_times = analysis_time_us[dermis_search_start + peaks]
            time_diffs = np.abs(peak_times - DERMIS_EXPECTED_TIME_US)
            best_peak_idx = np.argmin(time_diffs)
            dermis_idx = dermis_search_start + peaks[best_peak_idx]
        else:
            # 피크도 없으면 예상 시간 위치 사용
            dermis_idx = np.argmin(np.abs(analysis_time_us[:len(skin_voltage)] - DERMIS_EXPECTED_TIME_US))

    # Step 3: 근막 검출 - 통계 기반 예상 범위 내에서 탐색
    # 예상 시간 범위: 4.4 ~ 5.9 μs (T0 기준)
    # 또는 진피 + 2.18mm (통계적 두께 차이)
    fascia_min_time_us = FASCIA_EXPECTED_TIME_US - FASCIA_TIME_STD_US
    fascia_max_time_us = FASCIA_EXPECTED_TIME_US + FASCIA_TIME_STD_US

    # 진피로부터의 예상 거리도 고려
    speed_of_sound = 1540  # m/s
    expected_gap_time_us = (THICKNESS_GAP_MM / 1000) * 2 / speed_of_sound * 1e6
    dermis_time_us = analysis_time_us[dermis_idx]
    fascia_from_dermis_us = dermis_time_us + expected_gap_time_us

    # 두 가지 예상값의 평균 사용
    fascia_expected_us = (FASCIA_EXPECTED_TIME_US + fascia_from_dermis_us) / 2

    # 탐색 범위 설정
    fascia_search_start = dermis_idx + 20
    fascia_min_idx = np.argmin(np.abs(analysis_time_us[:len(skin_voltage)] - fascia_min_time_us))
    fascia_max_idx = np.argmin(np.abs(analysis_time_us[:len(skin_voltage)] - fascia_max_time_us))

    # 범위 조정
    fascia_search_start = max(fascia_search_start, fascia_min_idx - 30)
    fascia_search_end = min(len(skin_voltage), fascia_max_idx + 50)

    if fascia_search_start >= fascia_search_end - 10:
        # 범위가 너무 좁으면 진피 기반 추정값 사용
        fascia_idx = dermis_idx + int(expected_gap_time_us / (analysis_time_us[1] - analysis_time_us[0]))
        if fascia_idx >= len(skin_voltage):
            return None, analysis_voltage, analysis_time_us
    else:
        fascia_search_region = skin_voltage[fascia_search_start:fascia_search_end]
        abs_fascia_search = np.abs(fascia_search_region)

        # 피크 검출
        std_val = np.std(abs_fascia_search)
        peaks, properties = find_peaks(abs_fascia_search,
                                       prominence=std_val * 0.3,
                                       distance=10)

        if len(peaks) > 0:
            # 예상 시간에 가장 가까운 피크 선택
            peak_times = analysis_time_us[fascia_search_start + peaks]
            time_diffs = np.abs(peak_times - fascia_expected_us)
            best_peak_idx = np.argmin(time_diffs)
            fascia_idx = fascia_search_start + peaks[best_peak_idx]
        else:
            # 피크를 못 찾으면 예상 위치 사용
            fascia_idx = np.argmin(np.abs(analysis_time_us[:len(skin_voltage)] - fascia_expected_us))

    positions = np.array([dermis_idx, fascia_idx])
    return positions, analysis_voltage, analysis_time_us


def detect_positions_fallback(skin_voltage, analysis_time_us):
    """기존 방식으로 폴백"""
    skin_filtered_abs = np.abs(skin_voltage)
    std_val = np.std(skin_filtered_abs)

    all_peaks, _ = find_peaks(skin_filtered_abs,
                              prominence=std_val * 0.3,
                              distance=5)

    if len(all_peaks) >= 2:
        heights = skin_filtered_abs[all_peaks]
        sorted_indices = np.argsort(heights)[::-1]
        top_2_indices = sorted_indices[:2]
        positions = np.sort(all_peaks[top_2_indices])
        return positions, skin_voltage, analysis_time_us

    return None, skin_voltage, analysis_time_us


def visualize_ultrasound_signal_improved(file_path: str, save_path: str = None,
                                        manual_label: dict = None):
    """개선된 초음파 신호 시각화"""

    # 데이터 로드
    time_data, voltage_data = parse_ultrasound_csv(file_path)

    if len(voltage_data) < 10:
        print("Insufficient data.")
        return

    # 샘플링 주파수 계산
    dt = time_data[1] - time_data[0]
    sample_rate = int(1.0 / dt)

    # 시간 데이터를 μs로 변환
    time_us = time_data * 1e6

    # 수동 레이블 사용
    if manual_label is None:
        print("⚠ No manual label provided. Using automatic detection.")
        return

    manual_start_us = manual_label['start_point_us']
    manual_positions = manual_label['positions']
    speed_of_sound = manual_label['speed_of_sound']

    # 자동 검출 (수동 시작점 참조)
    detected = detect_positions_with_reference(time_data, voltage_data,
                                               manual_start_us, sample_rate)

    if detected[0] is None:
        print("⚠ Failed to detect positions automatically")
        auto_positions = None
        skin_filtered_data = detected[1]
        analysis_time_us = detected[2]
    else:
        auto_positions, skin_filtered_data, analysis_time_us = detected

    # 플롯 생성 (2x2 그리드)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # 시작점 인덱스 찾기
    start_idx = np.argmin(np.abs(time_us - manual_start_us))

    # 펄스 시작점 자동 검출 (첫 번째 그래프 표시용)
    abs_voltage = np.abs(voltage_data)
    max_voltage = np.max(abs_voltage)
    threshold = max_voltage * 0.1  # 10% 임계값
    pulse_start_indices = np.where(abs_voltage > threshold)[0]

    if len(pulse_start_indices) > 0:
        pulse_start_idx = pulse_start_indices[0]
        pulse_start_us = time_us[pulse_start_idx]
    else:
        pulse_start_us = None

    # 1. 전체 신호 (원본 데이터)
    ax1.plot(time_us, voltage_data, 'b-', linewidth=1)

    # 펄스 시작점 표시 (자동 검출)
    if pulse_start_us is not None:
        ax1.axvline(x=pulse_start_us, color='purple', linestyle='--', alpha=0.7,
                   linewidth=1.5, label=f'Pulse Start: {pulse_start_us:.1f}μs (Auto)')

    # 수동 시작점 표시 (T0)
    ax1.axvline(x=manual_start_us, color='r', linestyle='--', alpha=0.7,
               linewidth=2, label=f'Manual Start (T0): {manual_start_us:.1f}μs')

    ax1.set_title('Full Ultrasound Signal')
    ax1.set_xlabel('Time (μs)')
    ax1.set_ylabel('Voltage (V)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 2. 시작점 이후 신호
    post_start_time = time_us[start_idx:] - manual_start_us
    post_start_voltage = voltage_data[start_idx:]
    ax2.plot(post_start_time, post_start_voltage, 'g-', linewidth=1)
    ax2.set_title('Signal After Manual Start Point')
    ax2.set_xlabel('Time After Start (μs)')
    ax2.set_ylabel('Voltage (V)')
    ax2.grid(True, alpha=0.3)

    # 6mm 제한선
    max_time_us = (6.0 / 1000) * 2 / speed_of_sound * 1e6
    ax2.axvline(x=max_time_us, color='red', linestyle='--', alpha=0.7,
               label=f'6mm Limit ({max_time_us:.1f}μs)')
    ax2.legend()

    # 3. 원본 신호 (시작점 이후) + 포지션 마커 - 6mm 범위
    ax3.plot(post_start_time, post_start_voltage, 'b-', linewidth=1, alpha=0.8)

    # 6mm 제한선 표시
    ax3.axvline(x=0, color='purple', linestyle='--', alpha=0.7, linewidth=1.5,
               label='Manual Start')
    ax3.axvline(x=max_time_us, color='red', linestyle='--', alpha=0.7, linewidth=1.5,
               label=f'6mm Limit')

    # 수동 레이블 위치 표시 (빨간색 삼각형)
    for i, pos in enumerate(manual_positions):
        pos_time_offset = (pos['time_us'] - manual_start_us)
        depth_mm = pos["thickness_mm"]
        # 원본 데이터에서 해당 시간의 전압값 찾기
        idx = np.argmin(np.abs(post_start_time - pos_time_offset))
        if idx < len(post_start_voltage):
            voltage_val = post_start_voltage[idx]
            ax3.scatter(pos_time_offset, voltage_val, color='red', s=150,
                      marker='v', zorder=11, edgecolors='darkred', linewidths=2.5,
                      label=f'Manual {i+1}: {pos["position_name"]} (Depth: {depth_mm:.2f}mm)')

    # 자동 검출 위치 표시 (파란색 삼각형)
    if auto_positions is not None:
        for i, auto_pos_idx in enumerate(auto_positions):
            auto_time = analysis_time_us[auto_pos_idx]
            auto_depth = auto_time * 1e-6 * speed_of_sound / 2 * 1000
            # 원본 데이터에서 해당 시간의 전압값 찾기
            idx = np.argmin(np.abs(post_start_time - auto_time))
            if idx < len(post_start_voltage):
                voltage_val = post_start_voltage[idx]
                name = 'Dermis' if i == 0 else 'Fascia'
                ax3.scatter(auto_time, voltage_val, color='blue', s=120,
                          marker='v', zorder=10, edgecolors='darkblue', linewidths=2,
                          label=f'Auto {i+1}: {name} ({auto_depth:.2f}mm)')

    ax3.set_title('Raw Signal (6mm Range) with Detection Markers')
    ax3.set_xlabel('Time After Start (μs)')
    ax3.set_ylabel('Voltage (V)')
    ax3.set_xlim(0, max_time_us)  # 6mm 범위로 제한
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper right', fontsize=8)

    # 4. 원본 신호 (전체 범위) + 포지션 마커
    ax4.plot(post_start_time, post_start_voltage, 'b-', linewidth=0.5, alpha=0.7)

    # 수동 레이블 위치 표시 (빨간색 삼각형)
    for i, pos in enumerate(manual_positions):
        pos_time_offset = (pos['time_us'] - manual_start_us)
        depth_mm = pos["thickness_mm"]
        # 원본 데이터에서 해당 시간의 전압값 찾기
        idx = np.argmin(np.abs(post_start_time - pos_time_offset))
        if idx < len(post_start_voltage):
            voltage_val = post_start_voltage[idx]
            ax4.scatter(pos_time_offset, voltage_val, color='red', s=150,
                      marker='v', zorder=11, edgecolors='darkred', linewidths=2.5,
                      label=f'Manual {i+1}: {pos["position_name"]} (Depth: {depth_mm:.2f}mm)')

    # 자동 검출 위치 표시 (파란색 삼각형)
    if auto_positions is not None:
        for i, auto_pos_idx in enumerate(auto_positions):
            auto_time = analysis_time_us[auto_pos_idx]
            auto_depth = auto_time * 1e-6 * speed_of_sound / 2 * 1000
            # 원본 데이터에서 해당 시간의 전압값 찾기
            idx = np.argmin(np.abs(post_start_time - auto_time))
            if idx < len(post_start_voltage):
                voltage_val = post_start_voltage[idx]
                name = 'Dermis' if i == 0 else 'Fascia'
                ax4.scatter(auto_time, voltage_val, color='blue', s=120,
                          marker='v', zorder=10, edgecolors='darkblue', linewidths=2,
                          label=f'Auto {i+1}: {name} ({auto_depth:.2f}mm)')

    ax4.set_title('Full Signal (Raw Data) with Detection Markers')
    ax4.set_xlabel('Time After Start (μs)')
    ax4.set_ylabel('Voltage (V)')
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='upper right', fontsize=7)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved: {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    import os

    # 모든 환자 처리
    patient_ids = ['bhjung', 'cmkim', 'Drpark']
    data_dir = 'data'
    results_dir = 'results/improved'

    # 결과 디렉토리 생성
    os.makedirs(results_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Processing All Patients with Improved Algorithm")
    print(f"{'='*70}\n")

    total_processed = 0
    total_failed = 0

    for patient_id in patient_ids:
        print(f"\n{'='*70}")
        print(f"Processing patient: {patient_id}")
        print(f"{'='*70}\n")

        # 8개 부위 처리
        for position in range(1, 9):
            file_name = f'{patient_id}-5M-{position}.csv'
            file_path = os.path.join(data_dir, file_name)

            if not os.path.exists(file_path):
                print(f"⚠ File not found: {file_path}")
                total_failed += 1
                continue

            save_path = os.path.join(results_dir, f'{patient_id}_position_{position}_layer_analysis.png')

            print(f"\n{'─'*70}")
            print(f"Processing Position {position}: {file_name}")
            print(f"{'─'*70}")

            # JSON 레이블 로드
            manual_label = load_manual_label_json(patient_id, position)

            if manual_label is None:
                print(f"⚠ No manual label found for {file_name}")
                total_failed += 1
                continue

            print(f"  Manual label loaded:")
            print(f"    Start point: {manual_label['start_point_us']:.2f}μs")
            for pos in manual_label['positions']:
                print(f"    {pos['position_name']}: {pos['time_us']:.2f}μs, Depth: {pos['thickness_mm']:.2f}mm")

            try:
                visualize_ultrasound_signal_improved(file_path, save_path, manual_label)
                print(f"✓ Successfully saved: {save_path}")
                total_processed += 1
            except Exception as e:
                print(f"✗ Error processing {file_name}: {e}")
                import traceback
                traceback.print_exc()
                total_failed += 1

    print(f"\n{'='*70}")
    print(f"All patients processed!")
    print(f"  Total processed: {total_processed}")
    print(f"  Total failed: {total_failed}")
    print(f"{'='*70}\n")
