#!/usr/bin/env python3
"""
Ultrasound data visualization and debugging script
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from utils.preprocessor import UltrasoundPreprocessor

# Matplotlib font configuration - use default fonts to avoid Korean text corruption
import matplotlib as mpl
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['font.family'] = 'DejaVu Sans'

def load_manual_labels(excel_path: str, patient_id: str, position: int) -> dict:
    """
    Load manual thickness measurements from Excel or CSV file

    Args:
        excel_path: Path to Excel/CSV file with manual measurements
        patient_id: Patient ID (e.g., 'bhjung')
        position: Position number (1-8)

    Returns:
        Dictionary with 'thickness1' and 'thickness2' in meters, or None if not found
    """
    try:
        # CSV 또는 Excel 파일 읽기
        if excel_path.endswith('.csv'):
            df = pd.read_csv(excel_path)
        else:
            df = pd.read_excel(excel_path)

        # 파일명 패턴: {patient_id}-5M-{position}.csv
        file_pattern = f"{patient_id}-5M-{position}"

        # 해당 행 찾기 (파일명 열에서)
        # Excel 파일의 구조에 따라 열 이름이 다를 수 있음
        for col in df.columns:
            if 'file' in col.lower() or 'name' in col.lower():
                matching_rows = df[df[col].astype(str).str.contains(file_pattern, na=False)]
                if not matching_rows.empty:
                    row = matching_rows.iloc[0]
                    result = {}

                    # thickness1과 thickness2 찾기
                    for col_name in df.columns:
                        if 'thickness1' in col_name.lower():
                            result['thickness1'] = row[col_name] if pd.notna(row[col_name]) else None
                        elif 'thickness2' in col_name.lower():
                            result['thickness2'] = row[col_name] if pd.notna(row[col_name]) else None

                    if result:
                        return result

        # 열 이름이 없는 경우 인덱스로 찾기 시도
        # 첫 번째 열이 파일명이라고 가정
        if len(df.columns) >= 3:
            for idx, row in df.iterrows():
                if file_pattern in str(row.iloc[0]):
                    return {
                        'thickness1': row.iloc[1] if pd.notna(row.iloc[1]) else None,
                        'thickness2': row.iloc[2] if pd.notna(row.iloc[2]) else None
                    }

        return None
    except Exception as e:
        print(f"Warning: Could not load manual labels: {e}")
        return None


def parse_ultrasound_csv(file_path: str) -> tuple[np.ndarray, np.ndarray]:
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

def visualize_ultrasound_signal(file_path: str, save_path: str = None, manual_labels: dict = None):
    """Visualize ultrasound signal

    Args:
        file_path: Path to ultrasound CSV file
        save_path: Path to save the visualization
        manual_labels: Dict with manual measurements, e.g., {'thickness1': 0.001, 'thickness2': 0.003} in meters
    """
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

    # 펄스 시작점 찾기 (첫 번째 큰 전압 기반 - 2V 이상)
    abs_voltage = np.abs(voltage_data)
    max_voltage = np.max(abs_voltage)

    # 임계값 설정: 2V를 기본값으로, 최대값이 2V보다 작으면 70%로 조정
    if max_voltage >= 2.0:
        threshold = 2.0
    else:
        threshold = max_voltage * 0.7

    # 임계값을 넘는 첫 번째 지점 찾기
    pulse_start_indices = np.where(abs_voltage > threshold)[0]

    if len(pulse_start_indices) == 0:
        threshold = max_voltage * 0.5
        pulse_start_indices = np.where(abs_voltage > threshold)[0]

    # 플롯 생성
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # 1. 전체 신호
    ax1.plot(time_us, voltage_data, 'b-', linewidth=1)
    ax1.set_title('Full Ultrasound Signal')
    ax1.set_xlabel('Time (μs)')
    ax1.set_ylabel('Voltage (V)')
    ax1.grid(True, alpha=0.3)

    if len(pulse_start_indices) > 0:
        pulse_start_time = time_us[pulse_start_indices[0]]
        ax1.axvline(x=pulse_start_time, color='r', linestyle='--', alpha=0.7,
                   label=f'Pulse Start: {pulse_start_time:.1f} μs')
        ax1.legend()

        # 펄스 시작점 이후 데이터
        analysis_time = time_us[pulse_start_indices[0]:] - pulse_start_time
        analysis_voltage = voltage_data[pulse_start_indices[0]:]

        # 2. 펄스 시작 후 신호
        ax2.plot(analysis_time, analysis_voltage, 'g-', linewidth=1)
        ax2.set_title('Signal After Pulse Start')
        ax2.set_xlabel('Time After Pulse Start (μs)')
        ax2.set_ylabel('Voltage (V)')
        ax2.grid(True, alpha=0.3)

        # 전처리
        preprocessor = UltrasoundPreprocessor(sample_rate=sample_rate)
        filtered_data = preprocessor.apply_bandpass_filter(analysis_voltage)

        # 피부층 분석 영역 설정 (6mm 이내)
        speed_of_sound = 1540  # m/s
        max_distance_mm = 6.0  # mm
        max_time_us = (max_distance_mm / 1000) * 2 / speed_of_sound * 1e6  # μs

        # 분석 시작점: 펄스 시작점이 이미 표피 시작점이므로 0부터 시작
        skin_start_idx = 0
        skin_start_time = 0

        # 6mm 이내 데이터만 추출
        skin_end_time = skin_start_time + max_time_us
        skin_end_idx = np.argmin(np.abs(analysis_time - skin_end_time))

        # 피부층 데이터 추출
        skin_analysis_time = analysis_time[skin_start_idx:skin_end_idx]
        skin_filtered_data = filtered_data[skin_start_idx:skin_end_idx]

        # 3. 필터링된 신호 (6mm 이내 피부층 분석)
        ax3.plot(analysis_time, filtered_data, 'gray', linewidth=0.5, alpha=0.3, label='Full Signal')
        ax3.axvline(x=skin_start_time, color='purple', linestyle='--', alpha=0.7,
                   label=f'Skin Start ({skin_start_time:.1f}μs)')
        ax3.axvline(x=skin_end_time, color='red', linestyle='--', alpha=0.7,
                   label=f'6mm Limit ({skin_end_time:.1f}μs)')
        ax3.set_title(f'Skin Layer Analysis (0-{max_distance_mm}mm depth)')
        ax3.set_xlabel('Time After Pulse Start (μs)')
        ax3.set_ylabel('Voltage (V)')
        ax3.grid(True, alpha=0.3)

        # 피부층 데이터 강조 표시
        ax3.plot(skin_analysis_time, skin_filtered_data, 'b-', linewidth=2, label='Skin Layer (0-6mm)')

        # 포지션 검출 (2개: 진피층 시작, 근막층 시작)
        # 절댓값을 사용하여 양수/음수 피크 모두 검출
        skin_filtered_abs = np.abs(skin_filtered_data)

        # 파라미터 설정
        std_val = np.std(skin_filtered_abs)
        mean_val = np.mean(skin_filtered_abs)

        # scipy의 find_peaks를 직접 사용
        from scipy.signal import find_peaks as scipy_find_peaks

        # 피크 검출 - 주요 경계를 찾기 위해 강한 피크만 검출
        all_peaks, properties = scipy_find_peaks(skin_filtered_abs,
                                                 prominence=std_val * 0.5,
                                                 distance=10,
                                                 height=std_val * 0.7)

        # 피크가 충분하지 않으면 낮은 임계값으로 재시도
        if len(all_peaks) < 2:
            all_peaks2, properties2 = scipy_find_peaks(skin_filtered_abs,
                                                       prominence=std_val * 0.3,
                                                       distance=5,
                                                       height=std_val * 0.5)
            if len(all_peaks2) >= len(all_peaks):
                all_peaks = all_peaks2
                properties = properties2

        # 그래도 부족하면 더 낮은 임계값
        if len(all_peaks) < 2:
            all_peaks3, properties3 = scipy_find_peaks(skin_filtered_abs,
                                                       prominence=std_val * 0.2,
                                                       distance=5,
                                                       height=mean_val * 0.1)
            if len(all_peaks3) >= len(all_peaks):
                all_peaks = all_peaks3
                properties = properties3

        # 정확히 2개 포지션만 선택 (가장 큰 2개 피크)
        if len(all_peaks) >= 2:
            if 'peak_heights' in properties:
                heights = properties['peak_heights']
                sorted_indices = np.argsort(heights)[::-1]
                top_2_indices = sorted_indices[:2]
                # 시간순으로 정렬 (첫 번째가 진피층, 두 번째가 근막층)
                positions = np.sort(all_peaks[top_2_indices])
            else:
                positions = all_peaks[:2]
        elif len(all_peaks) == 1:
            # 피크가 1개만 있으면 진피층 시작으로 간주
            positions = all_peaks
        else:
            positions = np.array([])

        if len(positions) > 0:
            # 포지션을 실제 시간축으로 변환
            positions_time = skin_analysis_time[positions]
            positions_voltage = skin_filtered_data[positions]

            # 포지션 마커 표시
            ax3.plot(positions_time, positions_voltage, 'ko', markersize=14,
                    markerfacecolor='red', markeredgecolor='black', linewidth=2,
                    label=f'Positions ({len(positions)})', zorder=10)

            # Layer colors (3 layers: Epidermis, Dermis, Fascia)
            layer_color_map = {
                'Epidermis': 'lightcoral',
                'Dermis': 'lightskyblue',
                'Fascia': 'lightgreen',
                'Dermis+Fascia': 'lightyellow'
            }

            # 계단식 Y 위치
            max_voltage = np.max(skin_filtered_data)
            min_voltage = np.min(skin_filtered_data)
            voltage_range = max_voltage - min_voltage

            # 층 정보 계산
            layers = []

            if len(positions) >= 2:
                # Layer 1: Epidermis (0 → Position 1: Dermis Start)
                epidermis_depth = (positions_time[0] - skin_start_time) * 1e-6 * speed_of_sound / 2 * 1000
                layers.append({
                    'name': 'Epidermis',
                    'thickness': epidermis_depth,
                    'start': 0.0,
                    'end': epidermis_depth,
                    'time_start': skin_start_time,
                    'time_end': positions_time[0]
                })

                # Layer 2: Dermis (Position 1 → Position 2: Fascia Start)
                dermis_end = (positions_time[1] - skin_start_time) * 1e-6 * speed_of_sound / 2 * 1000
                dermis_thickness = dermis_end - epidermis_depth
                layers.append({
                    'name': 'Dermis',
                    'thickness': dermis_thickness,
                    'start': epidermis_depth,
                    'end': dermis_end,
                    'time_start': positions_time[0],
                    'time_end': positions_time[1]
                })

                # Layer 3: Fascia (Position 2 → End)
                fascia_thickness = max_distance_mm - dermis_end
                layers.append({
                    'name': 'Fascia',
                    'thickness': fascia_thickness,
                    'start': dermis_end,
                    'end': max_distance_mm,
                    'time_start': positions_time[1],
                    'time_end': skin_end_time
                })

            elif len(positions) == 1:
                # Only 1 position detected
                epidermis_depth = (positions_time[0] - skin_start_time) * 1e-6 * speed_of_sound / 2 * 1000
                layers.append({
                    'name': 'Epidermis',
                    'thickness': epidermis_depth,
                    'start': 0.0,
                    'end': epidermis_depth,
                    'time_start': skin_start_time,
                    'time_end': positions_time[0]
                })

                remaining_thickness = max_distance_mm - epidermis_depth
                layers.append({
                    'name': 'Dermis+Fascia',
                    'thickness': remaining_thickness,
                    'start': epidermis_depth,
                    'end': max_distance_mm,
                    'time_start': positions_time[0],
                    'time_end': skin_end_time
                })

            # Y 위치 계산
            layer_y_positions = []
            for i in range(len(layers)):
                y_pos = max_voltage - (voltage_range * 0.15) - (i * voltage_range * 0.2)
                layer_y_positions.append(y_pos)

            # 층 표시
            for idx, layer in enumerate(layers):
                color = layer_color_map.get(layer['name'], 'lightgray')

                # 영역 색상
                ax3.axvspan(layer['time_start'], layer['time_end'], alpha=0.2, color=color)

                # 중간 위치
                mid_time = (layer['time_start'] + layer['time_end']) / 2

                # 두께 표시
                ax3.annotate(f'{layer["name"]}\n{layer["thickness"]:.3f}mm',
                           xy=(mid_time, layer_y_positions[idx]),
                           xytext=(0, 0), textcoords='offset points',
                           ha='center', va='center',
                           bbox=dict(boxstyle='round,pad=0.4',
                                   facecolor=color,
                                   edgecolor='black', linewidth=1.5, alpha=0.8),
                           fontsize=8, weight='bold')

            # 포지션 마커
            position_names = ['P1: Dermis Start', 'P2: Fascia Start']
            for i, (pos_time, pos_voltage) in enumerate(zip(positions_time, positions_voltage)):
                depth = (pos_time - skin_start_time) * 1e-6 * speed_of_sound / 2 * 1000
                pos_label = position_names[i] if i < len(position_names) else f'P{i+1}'
                ax3.annotate(f'{pos_label}\n{depth:.3f}mm',
                           xy=(pos_time, pos_voltage),
                           xytext=(0, 20), textcoords='offset points',
                           ha='center', va='bottom',
                           bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow',
                                   edgecolor='red', linewidth=2, alpha=0.9),
                           fontsize=8, weight='bold',
                           arrowprops=dict(arrowstyle='->', color='red', lw=2))

            # Manual t0 (시작점) 표시 - 파란색 수직선
            if manual_labels is not None:
                # t0는 skin_start_time (펄스 시작점)
                ax3.axvline(x=skin_start_time, color='blue', linestyle='-', linewidth=2,
                          alpha=0.7, label='Manual t0 (Start)', zorder=9)

                manual_positions = []
                if 'thickness1' in manual_labels and manual_labels['thickness1'] is not None:
                    # thickness1은 t0부터의 깊이 (미터 단위)
                    thickness1_mm = manual_labels['thickness1'] * 1000
                    # t0부터의 시간 변환: time = (depth * 2) / speed_of_sound
                    time1_offset_us = (thickness1_mm / 1000) * 2 / speed_of_sound * 1e6
                    # 실제 시간 = t0 + offset
                    time1_us = skin_start_time + time1_offset_us
                    manual_positions.append(('Manual P1', time1_us, thickness1_mm))

                if 'thickness2' in manual_labels and manual_labels['thickness2'] is not None:
                    # thickness2는 t0부터의 깊이 (미터 단위)
                    thickness2_mm = manual_labels['thickness2'] * 1000
                    time2_offset_us = (thickness2_mm / 1000) * 2 / speed_of_sound * 1e6
                    # 실제 시간 = t0 + offset
                    time2_us = skin_start_time + time2_offset_us
                    manual_positions.append(('Manual P2', time2_us, thickness2_mm))

                # 파란 점으로 수동 측정값 표시
                for i, (label, time_us_val, depth_mm) in enumerate(manual_positions):
                    # 해당 시간에서의 전압값 찾기
                    idx = np.argmin(np.abs(skin_analysis_time - time_us_val))
                    if idx < len(skin_filtered_data):
                        voltage_val = skin_filtered_data[idx]
                        # 첫 번째는 원형, 두 번째는 사각형
                        marker = 'o' if i == 0 else 's'
                        ax3.plot(time_us_val, voltage_val, marker, markersize=12,
                                markerfacecolor='blue', markeredgecolor='darkblue',
                                linewidth=2, zorder=11, label=label)
                        # 수동 측정값 주석
                        ax3.annotate(f'{label}\n{depth_mm:.3f}mm',
                                   xy=(time_us_val, voltage_val),
                                   xytext=(15, -15 - i*10), textcoords='offset points',
                                   ha='left', va='top',
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue',
                                           edgecolor='blue', linewidth=1.5, alpha=0.8),
                                   fontsize=7, weight='bold',
                                   arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))

            # 층 정보 출력
            print(f"\n=== Detected Skin Layers ===")
            print(f"Total positions detected: {len(positions)}")
            print(f"Total layers: {len(layers)}")
            for layer in layers:
                print(f"{layer['name']}: {layer['start']:.3f}mm - {layer['end']:.3f}mm (thickness: {layer['thickness']:.3f}mm)")

            ax3.legend(loc='upper right', fontsize=8)

        # 4. 전체 신호 원본 데이터 (필터링 전)
        ax4.plot(analysis_time, analysis_voltage, 'b-', linewidth=0.5, alpha=0.7)

        # 6mm 제한선 표시
        ax4.axvline(x=skin_start_time, color='purple', linestyle='--', alpha=0.7, linewidth=1.5,
                   label=f'Skin Start ({skin_start_time:.1f}μs)')
        ax4.axvline(x=skin_end_time, color='red', linestyle='--', alpha=0.7, linewidth=1.5,
                   label=f'6mm Limit ({skin_end_time:.1f}μs)')

        # Manual t0 표시 (파란색 수직선)
        if manual_labels is not None:
            ax4.axvline(x=skin_start_time, color='blue', linestyle='-', linewidth=2,
                      alpha=0.7, label='Manual t0', zorder=9)

        # 포지션 위치 표시
        if len(positions) > 0:
            positions_time_full = skin_analysis_time[positions]
            # 원본 데이터에서 해당 시간의 전압값 찾기
            positions_voltage_original = []
            for pos_time in positions_time_full:
                idx = np.argmin(np.abs(analysis_time - pos_time))
                positions_voltage_original.append(analysis_voltage[idx])
            ax4.scatter(positions_time_full, positions_voltage_original, color='red', s=100,
                       marker='D', zorder=5, label=f'{len(positions)} Positions')

        # Manual labels 표시 (파란 점)
        if manual_labels is not None:
            manual_positions = []
            if 'thickness1' in manual_labels and manual_labels['thickness1'] is not None:
                thickness1_mm = manual_labels['thickness1'] * 1000
                time1_offset_us = (thickness1_mm / 1000) * 2 / speed_of_sound * 1e6
                time1_us = skin_start_time + time1_offset_us
                manual_positions.append(('Manual P1', time1_us, thickness1_mm))

            if 'thickness2' in manual_labels and manual_labels['thickness2'] is not None:
                thickness2_mm = manual_labels['thickness2'] * 1000
                time2_offset_us = (thickness2_mm / 1000) * 2 / speed_of_sound * 1e6
                time2_us = skin_start_time + time2_offset_us
                manual_positions.append(('Manual P2', time2_us, thickness2_mm))

            for i, (label, time_us_val, depth_mm) in enumerate(manual_positions):
                # 원본 데이터에서 해당 시간의 전압값 찾기
                idx = np.argmin(np.abs(analysis_time - time_us_val))
                if idx < len(analysis_voltage):
                    voltage_val = analysis_voltage[idx]
                    marker = 'o' if i == 0 else 's'
                    ax4.scatter(time_us_val, voltage_val, color='blue', s=100,
                              marker=marker, zorder=11, edgecolors='darkblue', linewidths=2)

        ax4.set_title('Full Signal (Raw Data) with 6mm Analysis Window')
        ax4.set_xlabel('Time After Pulse Start (μs)')
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

    # bhjung 환자의 8개 파일 모두 처리
    patient_id = 'bhjung'
    data_dir = 'data'
    results_dir = 'results'

    # 수동 측정 라벨 파일 경로 (있는 경우)
    label_file = 'label.csv'  # 또는 'label.xlsx'
    if not os.path.exists(label_file):
        # 다른 가능한 위치들 확인
        possible_paths = [
            'labels.csv', 'labels.xlsx',
            os.path.join(data_dir, 'label.csv'),
            os.path.join(data_dir, 'labels.csv'),
            os.path.join(data_dir, 'label.xlsx'),
            os.path.join(data_dir, 'labels.xlsx')
        ]
        label_file = None
        for path in possible_paths:
            if os.path.exists(path):
                label_file = path
                break

    if label_file:
        print(f"✓ Found manual label file: {label_file}")
    else:
        print("⚠ No manual label file found. Proceeding without manual measurements.")

    # 결과 디렉토리 생성
    os.makedirs(results_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Processing patient: {patient_id}")
    print(f"{'='*60}\n")

    # 8개 부위 처리
    for position in range(1, 9):
        file_name = f'{patient_id}-5M-{position}.csv'
        file_path = os.path.join(data_dir, file_name)

        if not os.path.exists(file_path):
            print(f"⚠ File not found: {file_path}")
            continue

        save_path = os.path.join(results_dir, f'{patient_id}_pos{position}_layers.png')

        print(f"\n{'─'*60}")
        print(f"Processing Position {position}: {file_name}")
        print(f"{'─'*60}")

        # 수동 라벨 로드 (있는 경우)
        manual_labels = None
        if label_file:
            manual_labels = load_manual_labels(label_file, patient_id, position)
            if manual_labels:
                print(f"  Manual labels loaded: thickness1={manual_labels.get('thickness1')*1000:.3f}mm, "
                      f"thickness2={manual_labels.get('thickness2')*1000:.3f}mm")

        try:
            visualize_ultrasound_signal(file_path, save_path, manual_labels)
            print(f"✓ Successfully saved: {save_path}")
        except Exception as e:
            print(f"✗ Error processing {file_name}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"All positions processed!")
    print(f"{'='*60}\n")