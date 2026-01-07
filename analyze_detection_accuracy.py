#!/usr/bin/env python3
"""
자동 검출 알고리즘과 수동 마킹 데이터 비교 분석
228개 데이터셋에서 최적의 검출 파라미터 찾기
"""

import numpy as np
import json
import glob
import os
from utils.preprocessor import UltrasoundPreprocessor
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

def load_manual_label_json(json_path: str) -> dict:
    """JSON 레이블 파일 로드"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def parse_ultrasound_csv(file_path: str) -> tuple:
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

def detect_positions_auto(time_data, voltage_data, params):
    """
    자동 검출 알고리즘

    Args:
        time_data: 시간 데이터 (seconds)
        voltage_data: 전압 데이터
        params: 검출 파라미터 딕셔너리
            - pulse_threshold: 펄스 시작점 임계값
            - prominence_factor: 피크 prominence 계수
            - distance: 피크 간 최소 거리
            - height_factor: 피크 높이 임계값 계수

    Returns:
        detected: {'start_point_us', 'dermis_time_us', 'fascia_time_us'}
    """
    # 데이터 길이 확인
    if len(time_data) < 10 or len(voltage_data) < 10:
        return None

    # 샘플링 주파수 계산
    dt = time_data[1] - time_data[0]
    if dt <= 0 or dt > 1:  # 비정상적인 dt 값 체크
        return None

    sample_rate = int(1.0 / dt)

    # 시간 데이터를 μs로 변환
    time_us = time_data * 1e6

    # 펄스 시작점 찾기
    abs_voltage = np.abs(voltage_data)
    max_voltage = np.max(abs_voltage)

    threshold = params.get('pulse_threshold', 2.0)
    if max_voltage < threshold:
        threshold = max_voltage * 0.7

    pulse_start_indices = np.where(abs_voltage > threshold)[0]

    if len(pulse_start_indices) == 0:
        threshold = max_voltage * 0.5
        pulse_start_indices = np.where(abs_voltage > threshold)[0]

    if len(pulse_start_indices) == 0:
        return None

    pulse_start_idx = pulse_start_indices[0]
    pulse_start_time_us = time_us[pulse_start_idx]

    # 펄스 시작점 이후 데이터
    analysis_time_us = time_us[pulse_start_idx:] - pulse_start_time_us
    analysis_voltage = voltage_data[pulse_start_idx:]

    # 전처리
    preprocessor = UltrasoundPreprocessor(sample_rate=sample_rate)
    filtered_data = preprocessor.apply_bandpass_filter(analysis_voltage)

    # 6mm 이내 피부층 분석
    speed_of_sound = 1540  # m/s
    max_distance_mm = 6.0
    max_time_us = (max_distance_mm / 1000) * 2 / speed_of_sound * 1e6

    skin_end_idx = np.argmin(np.abs(analysis_time_us - max_time_us))
    skin_filtered_data = filtered_data[:skin_end_idx]
    skin_analysis_time_us = analysis_time_us[:skin_end_idx]

    # 피크 검출
    skin_filtered_abs = np.abs(skin_filtered_data)
    std_val = np.std(skin_filtered_abs)
    mean_val = np.mean(skin_filtered_abs)

    prominence = std_val * params.get('prominence_factor', 0.5)
    distance = params.get('distance', 10)
    height = std_val * params.get('height_factor', 0.7)

    all_peaks, properties = find_peaks(skin_filtered_abs,
                                       prominence=prominence,
                                       distance=distance,
                                       height=height)

    # 피크가 부족하면 낮은 임계값으로 재시도
    if len(all_peaks) < 2:
        all_peaks2, _ = find_peaks(skin_filtered_abs,
                                   prominence=std_val * 0.3,
                                   distance=5,
                                   height=std_val * 0.5)
        if len(all_peaks2) >= 2:
            all_peaks = all_peaks2

    if len(all_peaks) < 2:
        all_peaks3, _ = find_peaks(skin_filtered_abs,
                                   prominence=std_val * 0.2,
                                   distance=5,
                                   height=mean_val * 0.1)
        if len(all_peaks3) >= 2:
            all_peaks = all_peaks3

    # 가장 큰 2개 피크 선택
    if len(all_peaks) >= 2:
        heights = skin_filtered_abs[all_peaks]
        sorted_indices = np.argsort(heights)[::-1]
        top_2_indices = sorted_indices[:2]
        positions = np.sort(all_peaks[top_2_indices])
    else:
        return None

    if len(positions) < 2:
        return None

    # 실제 시간으로 변환
    dermis_time_us = pulse_start_time_us + skin_analysis_time_us[positions[0]]
    fascia_time_us = pulse_start_time_us + skin_analysis_time_us[positions[1]]

    return {
        'start_point_us': pulse_start_time_us,
        'dermis_time_us': dermis_time_us,
        'fascia_time_us': fascia_time_us
    }

def calculate_error(manual, detected):
    """
    수동 레이블과 자동 검출 결과 비교

    Returns:
        errors: {'start_error_us', 'dermis_error_us', 'fascia_error_us',
                 'dermis_thickness_error_mm', 'fascia_thickness_error_mm'}
    """
    if detected is None:
        return None

    errors = {}

    # 시작점 오차
    errors['start_error_us'] = abs(detected['start_point_us'] - manual['start_point_us'])

    # 진피 시작점 오차
    manual_dermis = manual['positions'][0]['time_us']
    errors['dermis_error_us'] = abs(detected['dermis_time_us'] - manual_dermis)

    # 근막 시작점 오차
    manual_fascia = manual['positions'][1]['time_us']
    errors['fascia_error_us'] = abs(detected['fascia_time_us'] - manual_fascia)

    # 두께 오차 (피부 시작점으로부터의 거리)
    speed_of_sound = 1540  # m/s

    # 자동 검출 두께
    auto_dermis_thickness = (detected['dermis_time_us'] - detected['start_point_us']) * 1e-6 * speed_of_sound / 2 * 1000
    auto_fascia_thickness = (detected['fascia_time_us'] - detected['start_point_us']) * 1e-6 * speed_of_sound / 2 * 1000

    # 수동 레이블 두께
    manual_dermis_thickness = manual['positions'][0]['thickness_mm']
    manual_fascia_thickness = manual['positions'][1]['thickness_mm']

    errors['dermis_thickness_error_mm'] = abs(auto_dermis_thickness - manual_dermis_thickness)
    errors['fascia_thickness_error_mm'] = abs(auto_fascia_thickness - manual_fascia_thickness)

    return errors

def analyze_all_samples(params):
    """228개 샘플 모두 분석"""
    label_dir = './manual_boundaries'
    data_dir = './data'

    label_files = glob.glob(os.path.join(label_dir, '*_positions.json'))

    results = []
    failed = 0

    for label_file in label_files:
        # 레이블 로드
        manual_label = load_manual_label_json(label_file)

        # 해당 CSV 파일 찾기
        base_name = os.path.basename(label_file).replace('_positions.json', '')
        csv_file = os.path.join(data_dir, f'{base_name}.csv')

        if not os.path.exists(csv_file):
            continue

        # 데이터 로드
        time_data, voltage_data = parse_ultrasound_csv(csv_file)

        # 자동 검출
        detected = detect_positions_auto(time_data, voltage_data, params)

        if detected is None:
            failed += 1
            continue

        # 오차 계산
        errors = calculate_error(manual_label, detected)

        if errors is not None:
            errors['file'] = base_name
            results.append(errors)

    return results, failed

def optimize_parameters():
    """
    다양한 파라미터 조합을 테스트하여 최적의 파라미터 찾기
    """
    print("=== 자동 검출 파라미터 최적화 ===\n")

    # 테스트할 파라미터 범위
    prominence_factors = [0.3, 0.4, 0.5, 0.6, 0.7]
    distances = [5, 8, 10, 12, 15]
    height_factors = [0.5, 0.6, 0.7, 0.8]

    best_params = None
    best_score = float('inf')
    best_results = None

    total_tests = len(prominence_factors) * len(distances) * len(height_factors)
    test_count = 0

    print(f"총 {total_tests}개 파라미터 조합 테스트 중...\n")

    for prom in prominence_factors:
        for dist in distances:
            for height in height_factors:
                test_count += 1
                params = {
                    'pulse_threshold': 2.0,
                    'prominence_factor': prom,
                    'distance': dist,
                    'height_factor': height
                }

                results, failed = analyze_all_samples(params)

                if len(results) == 0:
                    continue

                # 평균 오차 계산
                avg_dermis_error = np.mean([r['dermis_thickness_error_mm'] for r in results])
                avg_fascia_error = np.mean([r['fascia_thickness_error_mm'] for r in results])
                total_error = avg_dermis_error + avg_fascia_error

                if test_count % 10 == 0:
                    print(f"진행: {test_count}/{total_tests} - 현재 최고: {best_score:.4f}mm")

                if total_error < best_score:
                    best_score = total_error
                    best_params = params
                    best_results = results
                    print(f"✓ 새로운 최적 파라미터 발견!")
                    print(f"  prominence_factor={prom}, distance={dist}, height_factor={height}")
                    print(f"  평균 진피 오차: {avg_dermis_error:.4f}mm")
                    print(f"  평균 근막 오차: {avg_fascia_error:.4f}mm")
                    print(f"  총 오차: {total_error:.4f}mm")
                    print(f"  성공: {len(results)}개, 실패: {failed}개\n")

    return best_params, best_results

def print_detailed_statistics(results):
    """상세 통계 출력"""
    print("\n=== 상세 통계 ===")
    print(f"총 샘플 수: {len(results)}")

    # 시작점 오차
    start_errors = [r['start_error_us'] for r in results]
    print(f"\n피부 시작점 오차:")
    print(f"  평균: {np.mean(start_errors):.2f}μs")
    print(f"  표준편차: {np.std(start_errors):.2f}μs")
    print(f"  최대: {np.max(start_errors):.2f}μs")
    print(f"  최소: {np.min(start_errors):.2f}μs")

    # 진피 오차
    dermis_errors = [r['dermis_error_us'] for r in results]
    dermis_thickness_errors = [r['dermis_thickness_error_mm'] for r in results]
    print(f"\n진피 검출 오차:")
    print(f"  시간 오차 평균: {np.mean(dermis_errors):.2f}μs")
    print(f"  두께 오차 평균: {np.mean(dermis_thickness_errors):.4f}mm")
    print(f"  두께 오차 표준편차: {np.std(dermis_thickness_errors):.4f}mm")
    print(f"  두께 오차 최대: {np.max(dermis_thickness_errors):.4f}mm")

    # 근막 오차
    fascia_errors = [r['fascia_error_us'] for r in results]
    fascia_thickness_errors = [r['fascia_thickness_error_mm'] for r in results]
    print(f"\n근막 검출 오차:")
    print(f"  시간 오차 평균: {np.mean(fascia_errors):.2f}μs")
    print(f"  두께 오차 평균: {np.mean(fascia_thickness_errors):.4f}mm")
    print(f"  두께 오차 표준편차: {np.std(fascia_thickness_errors):.4f}mm")
    print(f"  두께 오차 최대: {np.max(fascia_thickness_errors):.4f}mm")

    # 정확도 분석 (0.1mm 이내, 0.2mm 이내, 0.5mm 이내)
    print(f"\n정확도 분석:")
    for threshold in [0.05, 0.1, 0.2, 0.5]:
        dermis_acc = sum(1 for e in dermis_thickness_errors if e <= threshold) / len(results) * 100
        fascia_acc = sum(1 for e in fascia_thickness_errors if e <= threshold) / len(results) * 100
        print(f"  {threshold}mm 이내:")
        print(f"    진피: {dermis_acc:.1f}%")
        print(f"    근막: {fascia_acc:.1f}%")

def plot_error_distribution(results, save_path='results/error_distribution.png'):
    """오차 분포 시각화"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 진피 두께 오차
    dermis_errors = [r['dermis_thickness_error_mm'] for r in results]
    axes[0, 0].hist(dermis_errors, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].set_title('Dermis Thickness Error Distribution')
    axes[0, 0].set_xlabel('Error (mm)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].axvline(np.mean(dermis_errors), color='r', linestyle='--',
                       label=f'Mean: {np.mean(dermis_errors):.4f}mm')
    axes[0, 0].legend()

    # 근막 두께 오차
    fascia_errors = [r['fascia_thickness_error_mm'] for r in results]
    axes[0, 1].hist(fascia_errors, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 1].set_title('Fascia Thickness Error Distribution')
    axes[0, 1].set_xlabel('Error (mm)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].axvline(np.mean(fascia_errors), color='r', linestyle='--',
                       label=f'Mean: {np.mean(fascia_errors):.4f}mm')
    axes[0, 1].legend()

    # 진피 시간 오차
    dermis_time_errors = [r['dermis_error_us'] for r in results]
    axes[1, 0].hist(dermis_time_errors, bins=50, edgecolor='black', alpha=0.7)
    axes[1, 0].set_title('Dermis Time Error Distribution')
    axes[1, 0].set_xlabel('Error (μs)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].axvline(np.mean(dermis_time_errors), color='r', linestyle='--',
                       label=f'Mean: {np.mean(dermis_time_errors):.2f}μs')
    axes[1, 0].legend()

    # 근막 시간 오차
    fascia_time_errors = [r['fascia_error_us'] for r in results]
    axes[1, 1].hist(fascia_time_errors, bins=50, edgecolor='black', alpha=0.7)
    axes[1, 1].set_title('Fascia Time Error Distribution')
    axes[1, 1].set_xlabel('Error (μs)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].axvline(np.mean(fascia_time_errors), color='r', linestyle='--',
                       label=f'Mean: {np.mean(fascia_time_errors):.2f}μs')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n오차 분포 그래프 저장: {save_path}")

if __name__ == '__main__':
    import os

    os.makedirs('results', exist_ok=True)

    # 파라미터 최적화
    best_params, best_results = optimize_parameters()

    print("\n" + "="*60)
    print("최적 파라미터:")
    print("="*60)
    for key, value in best_params.items():
        print(f"{key}: {value}")

    # 상세 통계
    print_detailed_statistics(best_results)

    # 오차 분포 시각화
    plot_error_distribution(best_results)

    # 최적 파라미터 저장
    import json
    with open('results/optimal_params.json', 'w') as f:
        json.dump(best_params, f, indent=2)
    print("\n최적 파라미터 저장: results/optimal_params.json")
