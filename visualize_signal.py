#!/usr/bin/env python3
"""
Ultrasound data visualization and debugging script
"""

import numpy as np
import matplotlib.pyplot as plt
from utils.preprocessor import UltrasoundPreprocessor

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

def visualize_ultrasound_signal(file_path: str, save_path: str = None):
    """Visualize ultrasound signal"""
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

    # 펄스 시작점 찾기
    threshold = np.std(voltage_data) * 3
    pulse_start_indices = np.where(np.abs(voltage_data) > threshold)[0]

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

        # 피부층 분석 영역 설정 (5mm 이내)
        # 5mm = 5mm * 2 / 1540 m/s = 6.49 μs (왕복 시간)
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
            # 피크를 못 찾으면 펄스 시작점 사용
            skin_start_idx = 0
            skin_start_time = 0

        # 5mm 이내 데이터만 추출
        skin_end_time = skin_start_time + max_time_us
        skin_end_idx = np.argmin(np.abs(analysis_time - skin_end_time))

        # 피부층 데이터 추출
        skin_analysis_time = analysis_time[skin_start_idx:skin_end_idx]
        skin_filtered_data = filtered_data[skin_start_idx:skin_end_idx]

        # 3. 필터링된 신호 (5mm 이내 피부층 분석)
        ax3.plot(analysis_time, filtered_data, 'gray', linewidth=0.5, alpha=0.3, label='Full Signal')
        ax3.axvline(x=skin_start_time, color='purple', linestyle='--', alpha=0.7,
                   label=f'Skin Start ({skin_start_time:.1f}μs)')
        ax3.axvline(x=skin_end_time, color='red', linestyle='--', alpha=0.7,
                   label=f'5mm Limit ({skin_end_time:.1f}μs)')
        ax3.set_title(f'Skin Layer Analysis (0-{max_distance_mm}mm depth)')
        ax3.set_xlabel('Time After Pulse Start (μs)')
        ax3.set_ylabel('Voltage (V)')
        ax3.grid(True, alpha=0.3)

        # 피부층 데이터 강조 표시
        ax3.plot(skin_analysis_time, skin_filtered_data, 'b-', linewidth=2, label='Skin Layer (0-5mm)')

        # 피크 검출 (최대 4개 층 경계)
        # 절댓값을 사용하여 양수/음수 피크 모두 검출
        skin_filtered_abs = np.abs(skin_filtered_data)

        # 더 민감한 파라미터로 설정
        std_val = np.std(skin_filtered_abs)
        mean_val = np.mean(skin_filtered_abs)

        # scipy의 find_peaks를 직접 사용하여 더 많은 피크 찾기
        from scipy.signal import find_peaks as scipy_find_peaks

        # 여러 민감도로 피크 검출 시도
        all_peaks, properties = scipy_find_peaks(skin_filtered_abs,
                                                 prominence=None,  # prominence 제한 없음
                                                 distance=5,  # 최소 거리 줄임
                                                 height=mean_val * 0.1)  # 매우 낮은 임계값

        # 피크가 4개 미만이면 더 낮은 임계값으로 다시 검출
        if len(all_peaks) < 4:
            all_peaks2, properties2 = scipy_find_peaks(skin_filtered_abs,
                                                       distance=5,
                                                       height=mean_val * 0.05)  # 매우매우 낮은 임계값
            # 더 많이 찾았으면 업데이트
            if len(all_peaks2) >= len(all_peaks):
                all_peaks = all_peaks2
                properties = properties2

        # peak_heights가 있으면 height 순으로, 없으면 전부 사용
        if len(all_peaks) > 4:
            if 'peak_heights' in properties:
                heights = properties['peak_heights']
                sorted_indices = np.argsort(heights)[::-1]  # 내림차순
                top_peaks_indices = sorted_indices[:4]
                peaks = np.sort(all_peaks[top_peaks_indices])  # 시간 순서대로 재정렬
            else:
                # 단순히 앞쪽 4개 선택
                peaks = all_peaks[:4]
        else:
            peaks = all_peaks

        # peak_info 딕셔너리 생성 (호환성)
        peak_info = {'num_peaks': len(peaks)}

        if len(peaks) > 0:
            # 피크를 실제 시간축으로 변환
            peaks_time = skin_analysis_time[peaks]
            peaks_voltage = skin_filtered_data[peaks]  # 원래 신호의 전압값 (부호 포함)
            peaks_voltage_abs = skin_filtered_abs[peaks]  # 절댓값 전압

            # 피크(층 경계) 표시
            ax3.plot(peaks_time, peaks_voltage, 'ko', markersize=12,
                    markerfacecolor='yellow', markeredgecolor='black', linewidth=2,
                    label=f'Layer Boundaries ({len(peaks)})', zorder=10)

            # 층별 영역 색상으로 구분
            layer_colors = ['lightcoral', 'lightskyblue', 'lightgreen', 'lightyellow', 'lightpink']

            # 시작점부터 첫 피크까지 (Layer 1)
            layer_depths = []  # 각 층의 깊이 정보 저장

            # Layer 1: 표면부터 첫 번째 경계까지
            first_depth = (peaks_time[0] - skin_start_time) * 1e-6 * speed_of_sound / 2 * 1000
            layer_depths.append(('Layer 1', 0, first_depth))
            ax3.axvspan(skin_start_time, peaks_time[0], alpha=0.2, color=layer_colors[0])

            # 계단식 배치를 위한 Y 위치 계산
            max_voltage = np.max(skin_filtered_data)
            min_voltage = np.min(skin_filtered_data)
            voltage_range = max_voltage - min_voltage

            # 각 층마다 다른 높이에 배치 (계단식)
            layer_y_positions = []
            num_total_layers = len(peaks) + 1  # 마지막 층 포함
            if num_total_layers <= 4:
                # 4개 이하면 상단부터 균등하게 배치
                for i in range(num_total_layers):
                    y_pos = max_voltage - (voltage_range * 0.15) - (i * voltage_range * 0.15)
                    layer_y_positions.append(y_pos)
            else:
                # 4개 초과면 더 촘촘하게
                for i in range(num_total_layers):
                    y_pos = max_voltage - (voltage_range * 0.1) - (i * voltage_range * 0.12)
                    layer_y_positions.append(y_pos)

            # Layer 1 두께 표시 (최상단)
            mid_time_1 = (skin_start_time + peaks_time[0]) / 2
            ax3.annotate(f'Layer 1\n{first_depth:.3f}mm',
                       xy=(mid_time_1, layer_y_positions[0]),
                       xytext=(0, 0), textcoords='offset points',
                       ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=layer_colors[0],
                               edgecolor='black', linewidth=1.5, alpha=0.8),
                       fontsize=9, weight='bold')

            # 층 영역 표시 (Layer 1)
            ax3.axvspan(skin_start_time, peaks_time[0], alpha=0.2, color=layer_colors[0])

            # 층 사이 거리 계산 및 표시 (계단식)
            for i in range(len(peaks) - 1):
                # 층 두께 계산 (왕복 시간을 고려)
                time_diff = peaks_time[i+1] - peaks_time[i]
                layer_thickness = (time_diff * 1e-6 * speed_of_sound / 2) * 1000  # mm

                # 누적 깊이
                cumulative_depth_start = (peaks_time[i] - skin_start_time) * 1e-6 * speed_of_sound / 2 * 1000
                cumulative_depth_end = (peaks_time[i+1] - skin_start_time) * 1e-6 * speed_of_sound / 2 * 1000

                layer_depths.append((f'Layer {i+2}', cumulative_depth_start, cumulative_depth_end))

                # 층 영역 표시
                ax3.axvspan(peaks_time[i], peaks_time[i+1], alpha=0.2, color=layer_colors[(i+1) % len(layer_colors)])

                # 층 중간에 두께 표시 (계단식)
                mid_time = (peaks_time[i] + peaks_time[i+1]) / 2
                ax3.annotate(f'Layer {i+2}\n{layer_thickness:.3f}mm',
                           xy=(mid_time, layer_y_positions[i+1]),
                           xytext=(0, 0), textcoords='offset points',
                           ha='center', va='center',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor=layer_colors[(i+1) % len(layer_colors)],
                                   edgecolor='black', linewidth=1.5, alpha=0.8),
                           fontsize=9, weight='bold')

            # 마지막 층 (마지막 피크부터 5mm 끝까지)
            if len(peaks) < 4:
                last_depth_start = (peaks_time[-1] - skin_start_time) * 1e-6 * speed_of_sound / 2 * 1000
                last_depth_end = max_distance_mm
                if peaks_time[-1] < skin_end_time:
                    layer_depths.append((f'Layer {len(peaks)+1}', last_depth_start, last_depth_end))
                    ax3.axvspan(peaks_time[-1], skin_end_time, alpha=0.2, color=layer_colors[len(peaks) % len(layer_colors)])

                    # 마지막 층 두께 (계단식)
                    last_thickness = last_depth_end - last_depth_start
                    mid_time = (peaks_time[-1] + skin_end_time) / 2
                    ax3.annotate(f'Layer {len(peaks)+1}\n{last_thickness:.3f}mm',
                               xy=(mid_time, layer_y_positions[len(peaks)]),
                               xytext=(0, 0), textcoords='offset points',
                               ha='center', va='center',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor=layer_colors[len(peaks) % len(layer_colors)],
                                       edgecolor='black', linewidth=1.5, alpha=0.8),
                               fontsize=9, weight='bold')

            # 층 경계 정보 표시 (피크 위치)
            for i, (peak_time, peak_voltage) in enumerate(zip(peaks_time, peaks_voltage)):
                # 표면부터의 깊이
                depth_from_surface = (peak_time - skin_start_time) * 1e-6 * speed_of_sound / 2 * 1000  # mm
                ax3.annotate(f'B{i+1}\n{depth_from_surface:.3f}mm',
                           xy=(peak_time, peak_voltage),
                           xytext=(0, 15), textcoords='offset points',
                           ha='center', va='bottom',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow',
                                   edgecolor='red', linewidth=2, alpha=0.9),
                           fontsize=8, weight='bold',
                           arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

            # 층 정보 출력
            print(f"\n=== Detected Skin Layers ===")
            print(f"Total boundaries detected: {len(peaks)}")
            print(f"Total layers: {len(layer_depths)}")
            for layer_name, depth_start, depth_end in layer_depths:
                thickness = depth_end - depth_start
                print(f"{layer_name}: {depth_start:.3f}mm - {depth_end:.3f}mm (thickness: {thickness:.3f}mm)")

            ax3.legend(loc='upper right', fontsize=8)

        # 4. 주파수 스펙트럼
        if len(analysis_voltage) > 64:
            from scipy import signal
            freqs, psd = signal.welch(analysis_voltage, fs=sample_rate, nperseg=256)

            # MHz 단위로 변환
            freqs_mhz = freqs / 1e6
            ax4.semilogy(freqs_mhz, psd, 'b-')
            ax4.set_title('Frequency Spectrum')
            ax4.set_xlabel('Frequency (MHz)')
            ax4.set_ylabel('Power Spectral Density')
            ax4.grid(True, alpha=0.3)
            ax4.set_xlim(0, 50)  # 50MHz까지 표시

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

        save_path = os.path.join(results_dir, f'{patient_id}_position_{position}_layer_analysis.png')

        print(f"\n{'─'*60}")
        print(f"Processing Position {position}: {file_name}")
        print(f"{'─'*60}")

        try:
            visualize_ultrasound_signal(file_path, save_path)
            print(f"✓ Successfully saved: {save_path}")
        except Exception as e:
            print(f"✗ Error processing {file_name}: {e}")

    print(f"\n{'='*60}")
    print(f"All positions processed!")
    print(f"{'='*60}\n")