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

        # 펄스 시작점 찾기
        threshold = np.std(voltage_data) * 3
        pulse_start_indices = np.where(np.abs(voltage_data) > threshold)[0]

        if len(pulse_start_indices) == 0:
            return None

        # 펄스 시작점 이후 데이터
        analysis_time = time_us[pulse_start_indices[0]:] - time_us[pulse_start_indices[0]]
        analysis_voltage = voltage_data[pulse_start_indices[0]:]

        # 전처리
        preprocessor = UltrasoundPreprocessor(sample_rate=sample_rate)
        filtered_data = preprocessor.apply_bandpass_filter(analysis_voltage)

        # 피부층 분석 영역 설정 (5mm 이내)
        speed_of_sound = 1540  # m/s
        max_distance_mm = 5.0  # mm
        max_time_us = (max_distance_mm / 1000) * 2 / speed_of_sound * 1e6  # μs

        # 분석 시작점: 펄스 시작 후 첫 피크 찾기
        initial_threshold = np.std(filtered_data) * 2
        initial_peaks = np.where(np.abs(filtered_data) > initial_threshold)[0]

        if len(initial_peaks) > 0:
            skin_start_idx = initial_peaks[0]
            skin_start_time = analysis_time[skin_start_idx]
        else:
            skin_start_idx = 0
            skin_start_time = 0

        # 5mm 이내 데이터만 추출
        skin_end_time = skin_start_time + max_time_us
        skin_end_idx = np.argmin(np.abs(analysis_time - skin_end_time))

        # 피부층 데이터 추출
        skin_analysis_time = analysis_time[skin_start_idx:skin_end_idx]
        skin_filtered_data = filtered_data[skin_start_idx:skin_end_idx]

        # 피크 검출 (최대 4개 층 경계)
        skin_filtered_abs = np.abs(skin_filtered_data)
        std_val = np.std(skin_filtered_abs)
        mean_val = np.mean(skin_filtered_abs)

        # 피크 검출
        all_peaks, properties = scipy_find_peaks(skin_filtered_abs,
                                                 prominence=None,
                                                 distance=5,
                                                 height=mean_val * 0.1)

        if len(all_peaks) < 4:
            all_peaks2, properties2 = scipy_find_peaks(skin_filtered_abs,
                                                       distance=5,
                                                       height=mean_val * 0.05)
            if len(all_peaks2) >= len(all_peaks):
                all_peaks = all_peaks2
                properties = properties2

        # 최대 4개 피크만 선택
        if len(all_peaks) > 4:
            if 'peak_heights' in properties:
                heights = properties['peak_heights']
                sorted_indices = np.argsort(heights)[::-1]
                top_peaks_indices = sorted_indices[:4]
                peaks = np.sort(all_peaks[top_peaks_indices])
            else:
                peaks = all_peaks[:4]
        else:
            peaks = all_peaks

        # 피크를 실제 시간축으로 변환
        peaks_time = skin_analysis_time[peaks]

        # 층 정보 계산
        layers = []

        if len(peaks) > 0:
            # Layer 1: 표면부터 첫 번째 경계까지
            first_depth = (peaks_time[0] - skin_start_time) * 1e-6 * speed_of_sound / 2 * 1000
            layers.append({
                'layer_num': 1,
                'thickness_mm': first_depth,
                'depth_start_mm': 0.0,
                'depth_end_mm': first_depth
            })

            # 중간 층들
            for i in range(len(peaks) - 1):
                time_diff = peaks_time[i+1] - peaks_time[i]
                layer_thickness = (time_diff * 1e-6 * speed_of_sound / 2) * 1000

                depth_start = (peaks_time[i] - skin_start_time) * 1e-6 * speed_of_sound / 2 * 1000
                depth_end = (peaks_time[i+1] - skin_start_time) * 1e-6 * speed_of_sound / 2 * 1000

                layers.append({
                    'layer_num': i + 2,
                    'thickness_mm': layer_thickness,
                    'depth_start_mm': depth_start,
                    'depth_end_mm': depth_end
                })

            # 마지막 층 (마지막 피크부터 5mm 끝까지 또는 4개 제한)
            if len(peaks) < 4 and peaks_time[-1] < skin_end_time:
                last_depth_start = (peaks_time[-1] - skin_start_time) * 1e-6 * speed_of_sound / 2 * 1000
                last_depth_end = max_distance_mm
                last_thickness = last_depth_end - last_depth_start

                layers.append({
                    'layer_num': len(peaks) + 1,
                    'thickness_mm': last_thickness,
                    'depth_start_mm': last_depth_start,
                    'depth_end_mm': last_depth_end
                })

        return {
            'patient_id': patient_id,
            'position': position,
            'file_path': file_path,
            'num_boundaries': len(peaks),
            'num_layers': len(layers),
            'layers': layers,
            'time_data': analysis_time,
            'voltage_data': analysis_voltage,  # 원본 전압 데이터 추가
            'filtered_data': filtered_data,
            'skin_analysis_time': skin_analysis_time,
            'skin_filtered_data': skin_filtered_data,
            'peaks': peaks,
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
    peaks = result['peaks']
    skin_start_time = result['skin_start_time']
    skin_end_time = result['skin_end_time']

    # 시각화
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    speed_of_sound = 1540
    max_distance_mm = 5.0

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
               label=f'5mm Limit ({skin_end_time:.1f}μs)')
    ax3.set_title(f'Skin Layer Analysis (0-{max_distance_mm}mm depth)')
    ax3.set_xlabel('Time (μs)')
    ax3.set_ylabel('Voltage (V)')
    ax3.grid(True, alpha=0.3)

    ax3.plot(skin_analysis_time, skin_filtered_data, 'b-', linewidth=2, label='Skin Layer (0-5mm)')

    if len(peaks) > 0:
        peaks_time = skin_analysis_time[peaks]
        peaks_voltage = skin_filtered_data[peaks]

        ax3.plot(peaks_time, peaks_voltage, 'ko', markersize=12,
                markerfacecolor='yellow', markeredgecolor='black', linewidth=2,
                label=f'Layer Boundaries ({len(peaks)})', zorder=10)

        # 층 색상
        layer_colors = ['lightcoral', 'lightskyblue', 'lightgreen', 'lightyellow', 'lightpink']

        # 계단식 Y 위치
        max_voltage = np.max(skin_filtered_data)
        min_voltage = np.min(skin_filtered_data)
        voltage_range = max_voltage - min_voltage

        num_layers = len(result['layers'])
        layer_y_positions = []
        for i in range(num_layers):
            y_pos = max_voltage - (voltage_range * 0.15) - (i * voltage_range * 0.15)
            layer_y_positions.append(y_pos)

        # 층 표시
        for idx, layer in enumerate(result['layers']):
            layer_num = layer['layer_num']
            thickness = layer['thickness_mm']

            # 영역 색상
            if layer_num == 1:
                ax3.axvspan(skin_start_time, peaks_time[0], alpha=0.2, color=layer_colors[0])
                mid_time = (skin_start_time + peaks_time[0]) / 2
            elif layer_num <= len(peaks):
                ax3.axvspan(peaks_time[layer_num-2], peaks_time[layer_num-1],
                           alpha=0.2, color=layer_colors[(layer_num-1) % len(layer_colors)])
                mid_time = (peaks_time[layer_num-2] + peaks_time[layer_num-1]) / 2
            else:
                ax3.axvspan(peaks_time[-1], skin_end_time,
                           alpha=0.2, color=layer_colors[(layer_num-1) % len(layer_colors)])
                mid_time = (peaks_time[-1] + skin_end_time) / 2

            # 두께 표시
            ax3.annotate(f'Layer {layer_num}\n{thickness:.3f}mm',
                       xy=(mid_time, layer_y_positions[idx]),
                       xytext=(0, 0), textcoords='offset points',
                       ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=0.3',
                               facecolor=layer_colors[(layer_num-1) % len(layer_colors)],
                               edgecolor='black', linewidth=1.5, alpha=0.8),
                       fontsize=9, weight='bold')

        # 경계 마커
        for i, (peak_time, peak_voltage) in enumerate(zip(peaks_time, peaks_voltage)):
            depth = (peak_time - skin_start_time) * 1e-6 * speed_of_sound / 2 * 1000
            ax3.annotate(f'B{i+1}\n{depth:.3f}mm',
                       xy=(peak_time, peak_voltage),
                       xytext=(0, 15), textcoords='offset points',
                       ha='center', va='bottom',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow',
                               edgecolor='red', linewidth=2, alpha=0.9),
                       fontsize=8, weight='bold',
                       arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

        ax3.legend(loc='upper right', fontsize=8)

    # 4. 전체 신호 원본 데이터 (필터링 전, 스무스하지 않은 신호)
    ax4.plot(time_data, voltage_data, 'b-', linewidth=0.5, alpha=0.7)

    # 5mm 제한선 표시
    ax4.axvline(x=skin_start_time, color='purple', linestyle='--', alpha=0.7, linewidth=1.5,
               label=f'Skin Start ({skin_start_time:.1f}μs)')
    ax4.axvline(x=skin_end_time, color='red', linestyle='--', alpha=0.7, linewidth=1.5,
               label=f'5mm Limit ({skin_end_time:.1f}μs)')

    # 피크 위치 표시 (필터링된 데이터에서 찾은 피크를 원본에 표시)
    if len(peaks) > 0:
        peaks_time_full = skin_analysis_time[peaks]
        # 원본 데이터에서 해당 시간의 전압값 찾기
        peaks_voltage_original = []
        for peak_time in peaks_time_full:
            idx = np.argmin(np.abs(time_data - peak_time))
            peaks_voltage_original.append(voltage_data[idx])
        ax4.scatter(peaks_time_full, peaks_voltage_original, color='red', s=50, zorder=5,
                   label=f'{len(peaks)} Boundaries')

    ax4.set_title('Full Signal (Raw Data) with 5mm Analysis Window')
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
                    'Thickness_mm': round(layer['thickness_mm'], 3),
                    'Depth_Start_mm': round(layer['depth_start_mm'], 3),
                    'Depth_End_mm': round(layer['depth_end_mm'], 3),
                    'Total_Boundaries': result['num_boundaries'],
                    'Total_Layers': result['num_layers']
                })

            print(f"✓ {result['num_layers']} layers detected")

    # 엑셀 파일 생성
    if excel_data:
        df = pd.DataFrame(excel_data)

        # 컬럼 순서 정리
        df = df[['Patient_ID', 'Position', 'Layer_Number',
                'Thickness_mm', 'Depth_Start_mm', 'Depth_End_mm',
                'Total_Boundaries', 'Total_Layers']]

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
