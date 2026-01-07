#!/usr/bin/env python3
"""
모든 환자의 초음파 데이터를 분석하고 엑셀 리포트 생성
"""

import os
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from utils.preprocessor import UltrasoundPreprocessor
import matplotlib.pyplot as plt
from scipy.signal import find_peaks as scipy_find_peaks

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

def analyze_single_file(file_path: str, patient_id: str, position: int) -> Dict:
    """단일 파일을 분석하여 층 정보 반환"""
    try:
        # 데이터 로드
        time_data, voltage_data = parse_ultrasound_csv(file_path)

        if len(voltage_data) < 10:
            return None

        # 샘플링 주파수 계산
        dt = time_data[1] - time_data[0]
        sample_rate = int(1.0 / dt)

        # 시간 데이터를 μs로 변환
        time_us = time_data * 1e6

        # 펄스 시작점 찾기 (첫 번째 큰 전압 기반 - 2V 이상)
        # 표피 시작점은 보통 2V 이상의 큰 신호에서 시작
        abs_voltage = np.abs(voltage_data)
        max_voltage = np.max(abs_voltage)

        # 임계값 설정: 2V를 기본값으로, 최대값이 2V보다 작으면 70%로 조정
        if max_voltage >= 2.0:
            threshold = 2.0  # 2V 이상의 첫 지점
        else:
            threshold = max_voltage * 0.7  # 최대값의 70%

        # 임계값을 넘는 첫 번째 지점 찾기
        threshold_indices = np.where(abs_voltage > threshold)[0]

        if len(threshold_indices) == 0:
            # 임계값을 넘는 지점이 없으면 최대값의 50%로 재시도
            threshold = max_voltage * 0.5
            threshold_indices = np.where(abs_voltage > threshold)[0]

        if len(threshold_indices) == 0:
            # 그래도 없으면 최대값 위치 사용
            pulse_start_idx = np.argmax(abs_voltage)
        else:
            # 첫 번째 임계값 초과 지점 사용
            pulse_start_idx = threshold_indices[0]

        # 펄스 시작점 이후 데이터
        analysis_time = time_us[pulse_start_idx:] - time_us[pulse_start_idx]
        analysis_voltage = voltage_data[pulse_start_idx:]

        # 전처리
        preprocessor = UltrasoundPreprocessor(sample_rate=sample_rate)
        filtered_data = preprocessor.apply_bandpass_filter(analysis_voltage)

        # 피부층 분석 영역 설정 (6mm 이내)
        speed_of_sound = 1540  # m/s
        max_distance_mm = 6.0  # mm
        max_time_us = (max_distance_mm / 1000) * 2 / speed_of_sound * 1e6  # μs

        # 분석 시작점: 펄스 시작점이 이미 표피 시작점이므로 0부터 시작
        # 펄스 시작점 검출에서 2V 이상을 사용했으므로 이미 정확함
        skin_start_idx = 0
        skin_start_time = 0

        # 5mm 이내 데이터만 추출
        skin_end_time = skin_start_time + max_time_us
        skin_end_idx = np.argmin(np.abs(analysis_time - skin_end_time))

        # 피부층 데이터 추출
        skin_analysis_time = analysis_time[skin_start_idx:skin_end_idx]
        skin_filtered_data = filtered_data[skin_start_idx:skin_end_idx]

        # 포지션 검출 (2개: 진피층 시작, 근막층 시작)
        skin_filtered_abs = np.abs(skin_filtered_data)
        std_val = np.std(skin_filtered_abs)
        mean_val = np.mean(skin_filtered_abs)

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

        # 포지션을 실제 시간축으로 변환
        positions_time = skin_analysis_time[positions]

        # 층 정보 계산 (3개 층: 표피, 진피, 근막)
        layers = []

        if len(positions) >= 2:
            # Layer 1: Epidermis (0 → Position 1: Dermis Start)
            epidermis_depth = (positions_time[0] - skin_start_time) * 1e-6 * speed_of_sound / 2 * 1000
            layers.append({
                'layer_num': 1,
                'layer_name': 'Epidermis',
                'thickness_mm': epidermis_depth,
                'depth_start_mm': 0.0,
                'depth_end_mm': epidermis_depth
            })

            # Layer 2: Dermis (Position 1 → Position 2: Fascia Start)
            dermis_start = epidermis_depth
            dermis_end = (positions_time[1] - skin_start_time) * 1e-6 * speed_of_sound / 2 * 1000
            dermis_thickness = dermis_end - dermis_start
            layers.append({
                'layer_num': 2,
                'layer_name': 'Dermis',
                'thickness_mm': dermis_thickness,
                'depth_start_mm': dermis_start,
                'depth_end_mm': dermis_end
            })

            # Layer 3: Fascia (Position 2 → End)
            fascia_start = dermis_end
            fascia_end = max_distance_mm
            fascia_thickness = fascia_end - fascia_start
            layers.append({
                'layer_num': 3,
                'layer_name': 'Fascia',
                'thickness_mm': fascia_thickness,
                'depth_start_mm': fascia_start,
                'depth_end_mm': fascia_end
            })

        elif len(positions) == 1:
            # Only 1 position detected (Dermis start only)
            epidermis_depth = (positions_time[0] - skin_start_time) * 1e-6 * speed_of_sound / 2 * 1000
            layers.append({
                'layer_num': 1,
                'layer_name': 'Epidermis',
                'thickness_mm': epidermis_depth,
                'depth_start_mm': 0.0,
                'depth_end_mm': epidermis_depth
            })

            # Remaining as Dermis+Fascia combined
            remaining_start = epidermis_depth
            remaining_end = max_distance_mm
            remaining_thickness = remaining_end - remaining_start
            layers.append({
                'layer_num': 2,
                'layer_name': 'Dermis+Fascia',
                'thickness_mm': remaining_thickness,
                'depth_start_mm': remaining_start,
                'depth_end_mm': remaining_end
            })

        return {
            'patient_id': patient_id,
            'position': position,
            'file_path': file_path,
            'num_positions': len(positions),
            'num_layers': len(layers),
            'layers': layers,
            'time_data': analysis_time,
            'voltage_data': analysis_voltage,  # 원본 전압 데이터 추가
            'filtered_data': filtered_data,
            'skin_analysis_time': skin_analysis_time,
            'skin_filtered_data': skin_filtered_data,
            'positions': positions,
            'skin_start_time': skin_start_time,
            'skin_end_time': skin_end_time
        }

    except Exception as e:
        print(f"Error analyzing {file_path}: {e}")
        return None

def visualize_patient_file(result: Dict, save_path: str):
    """개별 파일 시각화 (visualize_signal.py와 동일한 방식)"""
    if result is None:
        return

    # 기본 데이터 추출
    time_data = result['time_data']
    voltage_data = result['voltage_data']  # 원본 전압 데이터
    filtered_data = result['filtered_data']
    skin_analysis_time = result['skin_analysis_time']
    skin_filtered_data = result['skin_filtered_data']
    positions = result['positions']
    skin_start_time = result['skin_start_time']
    skin_end_time = result['skin_end_time']

    # 시각화
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    speed_of_sound = 1540
    max_distance_mm = 6.0

    # 전체 신호 (간단히)
    ax1.plot(time_data[:1000], filtered_data[:1000], 'b-', linewidth=1)
    ax1.set_title(f'{result["patient_id"]} - Position {result["position"]} - Full Signal')
    ax1.set_xlabel('Time (μs)')
    ax1.set_ylabel('Voltage (V)')
    ax1.grid(True, alpha=0.3)

    # 펄스 후 신호
    ax2.plot(time_data, filtered_data, 'g-', linewidth=1)
    ax2.set_title('Signal After Pulse Start')
    ax2.set_xlabel('Time (μs)')
    ax2.set_ylabel('Voltage (V)')
    ax2.grid(True, alpha=0.3)

    # 피부층 분석 (메인)
    ax3.plot(time_data, filtered_data, 'gray', linewidth=0.5, alpha=0.3, label='Full Signal')
    ax3.axvline(x=skin_start_time, color='purple', linestyle='--', alpha=0.7,
               label=f'Skin Start ({skin_start_time:.1f}μs)')
    ax3.axvline(x=skin_end_time, color='red', linestyle='--', alpha=0.7,
               label=f'6mm Limit ({skin_end_time:.1f}μs)')
    ax3.set_title(f'Skin Layer Analysis (0-{max_distance_mm}mm depth)')
    ax3.set_xlabel('Time (μs)')
    ax3.set_ylabel('Voltage (V)')
    ax3.grid(True, alpha=0.3)

    ax3.plot(skin_analysis_time, skin_filtered_data, 'b-', linewidth=2, label='Skin Layer (0-6mm)')

    if len(positions) > 0:
        positions_time = skin_analysis_time[positions]
        positions_voltage = skin_filtered_data[positions]

        ax3.plot(positions_time, positions_voltage, 'ko', markersize=14,
                markerfacecolor='red', markeredgecolor='black', linewidth=2,
                label=f'Positions ({len(positions)})', zorder=10)

        # Layer colors (3 layers: Epidermis, Dermis, Fascia)
        layer_colors = ['lightcoral', 'lightskyblue', 'lightgreen']
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

        num_layers = len(result['layers'])
        layer_y_positions = []
        for i in range(num_layers):
            y_pos = max_voltage - (voltage_range * 0.15) - (i * voltage_range * 0.2)
            layer_y_positions.append(y_pos)

        # 층 표시
        for idx, layer in enumerate(result['layers']):
            layer_name = layer['layer_name']
            thickness = layer['thickness_mm']
            layer_num = layer['layer_num']

            # 층 색상
            color = layer_color_map.get(layer_name, layer_colors[idx % len(layer_colors)])

            # 영역 색상 및 중간 위치
            if layer_num == 1:
                # 표피: 시작 → Position 1
                if len(positions) > 0:
                    ax3.axvspan(skin_start_time, positions_time[0], alpha=0.2, color=color)
                    mid_time = (skin_start_time + positions_time[0]) / 2
                else:
                    ax3.axvspan(skin_start_time, skin_end_time, alpha=0.2, color=color)
                    mid_time = (skin_start_time + skin_end_time) / 2
            elif layer_num == 2 and len(positions) >= 2:
                # 진피: Position 1 → Position 2
                ax3.axvspan(positions_time[0], positions_time[1], alpha=0.2, color=color)
                mid_time = (positions_time[0] + positions_time[1]) / 2
            elif layer_num == 2 and len(positions) == 1:
                # 진피+근막: Position 1 → 끝
                ax3.axvspan(positions_time[0], skin_end_time, alpha=0.2, color=color)
                mid_time = (positions_time[0] + skin_end_time) / 2
            elif layer_num == 3:
                # 근막: Position 2 → 끝
                ax3.axvspan(positions_time[1], skin_end_time, alpha=0.2, color=color)
                mid_time = (positions_time[1] + skin_end_time) / 2
            else:
                mid_time = (skin_start_time + skin_end_time) / 2

            # 두께 표시
            ax3.annotate(f'{layer_name}\n{thickness:.3f}mm',
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

        ax3.legend(loc='upper right', fontsize=8)

    # 4. 전체 신호 원본 데이터 (필터링 전, 스무스하지 않은 신호)
    ax4.plot(time_data, voltage_data, 'b-', linewidth=0.5, alpha=0.7)

    # 5mm 제한선 표시
    ax4.axvline(x=skin_start_time, color='purple', linestyle='--', alpha=0.7, linewidth=1.5,
               label=f'Skin Start ({skin_start_time:.1f}μs)')
    ax4.axvline(x=skin_end_time, color='red', linestyle='--', alpha=0.7, linewidth=1.5,
               label=f'6mm Limit ({skin_end_time:.1f}μs)')

    # 포지션 위치 표시 (필터링된 데이터에서 찾은 포지션을 원본에 표시)
    if len(positions) > 0:
        positions_time_full = skin_analysis_time[positions]
        # 원본 데이터에서 해당 시간의 전압값 찾기
        positions_voltage_original = []
        for pos_time in positions_time_full:
            idx = np.argmin(np.abs(time_data - pos_time))
            positions_voltage_original.append(voltage_data[idx])
        ax4.scatter(positions_time_full, positions_voltage_original, color='red', s=100,
                   marker='D', zorder=5, label=f'{len(positions)} Positions')

    ax4.set_title('Full Signal (Raw Data) with 6mm Analysis Window')
    ax4.set_xlabel('Time (μs)')
    ax4.set_ylabel('Voltage (V)')
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='upper right', fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """메인 함수: 모든 환자 분석 및 엑셀 생성"""
    data_dir = 'data'
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)

    # 환자 목록
    patient_ids = ['bhjung', 'Drpark', 'cmkim']

    all_results = []
    excel_data = []

    print(f"\n{'='*70}")
    print(f"Starting analysis for all patients")
    print(f"{'='*70}\n")

    for patient_id in patient_ids:
        print(f"\n{'─'*70}")
        print(f"Processing Patient: {patient_id}")
        print(f"{'─'*70}")

        for position in range(1, 9):
            file_name = f'{patient_id}-5M-{position}.csv'
            file_path = os.path.join(data_dir, file_name)

            if not os.path.exists(file_path):
                print(f"  ⚠ Position {position}: File not found")
                continue

            print(f"  Processing Position {position}...", end=' ')

            # 분석
            result = analyze_single_file(file_path, patient_id, position)

            if result is None:
                print("✗ Failed")
                continue

            all_results.append(result)

            # 시각화 생성
            save_path = os.path.join(results_dir, f'{patient_id}_pos{position}_layers.png')
            visualize_patient_file(result, save_path)

            # 엑셀 데이터 수집
            for layer in result['layers']:
                excel_data.append({
                    'Patient_ID': patient_id,
                    'Position': position,
                    'Layer_Number': layer['layer_num'],
                    'Layer_Name': layer['layer_name'],
                    'Thickness_mm': round(layer['thickness_mm'], 3),
                    'Depth_Start_mm': round(layer['depth_start_mm'], 3),
                    'Depth_End_mm': round(layer['depth_end_mm'], 3),
                    'Total_Positions': result['num_positions'],
                    'Total_Layers': result['num_layers']
                })

            print(f"✓ {result['num_positions']} positions, {result['num_layers']} layers detected")

    # 엑셀 파일 생성
    if excel_data:
        df = pd.DataFrame(excel_data)

        # 컬럼 순서 정리
        df = df[['Patient_ID', 'Position', 'Layer_Number', 'Layer_Name',
                'Thickness_mm', 'Depth_Start_mm', 'Depth_End_mm',
                'Total_Positions', 'Total_Layers']]

        # 정렬
        df = df.sort_values(['Patient_ID', 'Position', 'Layer_Number'])

        excel_path = os.path.join(results_dir, 'skin_layer_analysis_summary.xlsx')
        df.to_excel(excel_path, index=False, sheet_name='Layer Analysis')

        print(f"\n{'='*70}")
        print(f"✓ Excel report created: {excel_path}")
        print(f"  Total records: {len(excel_data)}")
        print(f"  Patients analyzed: {len(patient_ids)}")
        print(f"  Total images: {len(all_results)}")
        print(f"{'='*70}\n")

        # 요약 통계
        print("\nSummary Statistics:")
        summary = df.groupby('Patient_ID').agg({
            'Position': 'count',
            'Thickness_mm': ['mean', 'std'],
            'Total_Layers': 'mean'
        }).round(3)
        print(summary)

    else:
        print("\n⚠ No data to export to Excel")

if __name__ == "__main__":
    main()
