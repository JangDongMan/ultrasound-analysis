#!/usr/bin/env python3
"""
수동 레이블 데이터 통계 분석
234개의 수동 레이블에서 패턴을 찾아 알고리즘 개선에 활용
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
import glob
from scipy import stats

def load_all_manual_labels(label_dir='./manual_boundaries'):
    """모든 수동 레이블 JSON 파일 로드"""
    json_files = glob.glob(os.path.join(label_dir, '*_positions.json'))

    labels_data = []

    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            labels_data.append(data)

    return labels_data

def analyze_label_statistics(labels_data):
    """레이블 데이터 통계 분석"""

    # 데이터 수집
    start_points = []
    dermis_times = []
    fascia_times = []
    dermis_depths = []
    fascia_depths = []

    # 상대 시간 (시작점 대비)
    dermis_relative_times = []
    fascia_relative_times = []

    # 두께 차이
    thickness_differences = []

    for label in labels_data:
        start_point = label['start_point_us']
        start_points.append(start_point)

        positions = label['positions']

        if len(positions) >= 2:
            # Dermis (첫 번째)
            dermis = positions[0]
            dermis_time = dermis['time_us']
            dermis_depth = dermis['thickness_mm']

            dermis_times.append(dermis_time)
            dermis_depths.append(dermis_depth)
            dermis_relative_times.append(dermis_time - start_point)

            # Fascia (두 번째)
            fascia = positions[1]
            fascia_time = fascia['time_us']
            fascia_depth = fascia['thickness_mm']

            fascia_times.append(fascia_time)
            fascia_depths.append(fascia_depth)
            fascia_relative_times.append(fascia_time - start_point)

            # 두께 차이
            thickness_differences.append(fascia_depth - dermis_depth)

    # NumPy 배열로 변환
    start_points = np.array(start_points)
    dermis_relative_times = np.array(dermis_relative_times)
    fascia_relative_times = np.array(fascia_relative_times)
    dermis_depths = np.array(dermis_depths)
    fascia_depths = np.array(fascia_depths)
    thickness_differences = np.array(thickness_differences)

    # 통계 계산
    stats_dict = {
        'total_samples': len(labels_data),
        'start_point': {
            'mean': np.mean(start_points),
            'std': np.std(start_points),
            'min': np.min(start_points),
            'max': np.max(start_points),
        },
        'dermis_relative_time_us': {
            'mean': np.mean(dermis_relative_times),
            'std': np.std(dermis_relative_times),
            'min': np.min(dermis_relative_times),
            'max': np.max(dermis_relative_times),
            'median': np.median(dermis_relative_times),
            'q25': np.percentile(dermis_relative_times, 25),
            'q75': np.percentile(dermis_relative_times, 75),
        },
        'fascia_relative_time_us': {
            'mean': np.mean(fascia_relative_times),
            'std': np.std(fascia_relative_times),
            'min': np.min(fascia_relative_times),
            'max': np.max(fascia_relative_times),
            'median': np.median(fascia_relative_times),
            'q25': np.percentile(fascia_relative_times, 25),
            'q75': np.percentile(fascia_relative_times, 75),
        },
        'dermis_depth_mm': {
            'mean': np.mean(dermis_depths),
            'std': np.std(dermis_depths),
            'min': np.min(dermis_depths),
            'max': np.max(dermis_depths),
            'median': np.median(dermis_depths),
            'q25': np.percentile(dermis_depths, 25),
            'q75': np.percentile(dermis_depths, 75),
        },
        'fascia_depth_mm': {
            'mean': np.mean(fascia_depths),
            'std': np.std(fascia_depths),
            'min': np.min(fascia_depths),
            'max': np.max(fascia_depths),
            'median': np.median(fascia_depths),
            'q25': np.percentile(fascia_depths, 25),
            'q75': np.percentile(fascia_depths, 75),
        },
        'thickness_difference_mm': {
            'mean': np.mean(thickness_differences),
            'std': np.std(thickness_differences),
            'min': np.min(thickness_differences),
            'max': np.max(thickness_differences),
            'median': np.median(thickness_differences),
        }
    }

    return stats_dict, {
        'dermis_relative_times': dermis_relative_times,
        'fascia_relative_times': fascia_relative_times,
        'dermis_depths': dermis_depths,
        'fascia_depths': fascia_depths,
        'thickness_differences': thickness_differences,
    }

def print_statistics(stats_dict):
    """통계 결과 출력"""
    print(f"\n{'='*70}")
    print(f"Manual Label Statistics Analysis")
    print(f"{'='*70}\n")

    print(f"Total Samples: {stats_dict['total_samples']}\n")

    print(f"시작점 (Start Point):")
    print(f"  Mean: {stats_dict['start_point']['mean']:.2f} μs")
    print(f"  Std:  {stats_dict['start_point']['std']:.2f} μs")
    print(f"  Range: {stats_dict['start_point']['min']:.2f} ~ {stats_dict['start_point']['max']:.2f} μs\n")

    print(f"진피 상대 시간 (Dermis Relative Time from T0):")
    print(f"  Mean:   {stats_dict['dermis_relative_time_us']['mean']:.2f} μs")
    print(f"  Median: {stats_dict['dermis_relative_time_us']['median']:.2f} μs")
    print(f"  Std:    {stats_dict['dermis_relative_time_us']['std']:.2f} μs")
    print(f"  Range:  {stats_dict['dermis_relative_time_us']['min']:.2f} ~ {stats_dict['dermis_relative_time_us']['max']:.2f} μs")
    print(f"  Q25-Q75: {stats_dict['dermis_relative_time_us']['q25']:.2f} ~ {stats_dict['dermis_relative_time_us']['q75']:.2f} μs\n")

    print(f"근막 상대 시간 (Fascia Relative Time from T0):")
    print(f"  Mean:   {stats_dict['fascia_relative_time_us']['mean']:.2f} μs")
    print(f"  Median: {stats_dict['fascia_relative_time_us']['median']:.2f} μs")
    print(f"  Std:    {stats_dict['fascia_relative_time_us']['std']:.2f} μs")
    print(f"  Range:  {stats_dict['fascia_relative_time_us']['min']:.2f} ~ {stats_dict['fascia_relative_time_us']['max']:.2f} μs")
    print(f"  Q25-Q75: {stats_dict['fascia_relative_time_us']['q25']:.2f} ~ {stats_dict['fascia_relative_time_us']['q75']:.2f} μs\n")

    print(f"진피 깊이 (Dermis Depth - Thickness1):")
    print(f"  Mean:   {stats_dict['dermis_depth_mm']['mean']:.2f} mm")
    print(f"  Median: {stats_dict['dermis_depth_mm']['median']:.2f} mm")
    print(f"  Std:    {stats_dict['dermis_depth_mm']['std']:.2f} mm")
    print(f"  Range:  {stats_dict['dermis_depth_mm']['min']:.2f} ~ {stats_dict['dermis_depth_mm']['max']:.2f} mm")
    print(f"  Q25-Q75: {stats_dict['dermis_depth_mm']['q25']:.2f} ~ {stats_dict['dermis_depth_mm']['q75']:.2f} mm\n")

    print(f"근막 깊이 (Fascia Depth - Thickness2):")
    print(f"  Mean:   {stats_dict['fascia_depth_mm']['mean']:.2f} mm")
    print(f"  Median: {stats_dict['fascia_depth_mm']['median']:.2f} mm")
    print(f"  Std:    {stats_dict['fascia_depth_mm']['std']:.2f} mm")
    print(f"  Range:  {stats_dict['fascia_depth_mm']['min']:.2f} ~ {stats_dict['fascia_depth_mm']['max']:.2f} mm")
    print(f"  Q25-Q75: {stats_dict['fascia_depth_mm']['q25']:.2f} ~ {stats_dict['fascia_depth_mm']['q75']:.2f} mm\n")

    print(f"두께 차이 (Thickness2 - Thickness1):")
    print(f"  Mean:   {stats_dict['thickness_difference_mm']['mean']:.2f} mm")
    print(f"  Median: {stats_dict['thickness_difference_mm']['median']:.2f} mm")
    print(f"  Std:    {stats_dict['thickness_difference_mm']['std']:.2f} mm")
    print(f"  Range:  {stats_dict['thickness_difference_mm']['min']:.2f} ~ {stats_dict['thickness_difference_mm']['max']:.2f} mm\n")

def visualize_distributions(data_arrays, save_path='results/manual_label_statistics.png'):
    """분포 시각화"""

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. 진피 상대 시간 분포
    ax = axes[0, 0]
    ax.hist(data_arrays['dermis_relative_times'], bins=30, edgecolor='black', alpha=0.7, color='blue')
    ax.axvline(np.mean(data_arrays['dermis_relative_times']), color='red', linestyle='--',
               linewidth=2, label=f"Mean: {np.mean(data_arrays['dermis_relative_times']):.2f}μs")
    ax.axvline(np.median(data_arrays['dermis_relative_times']), color='green', linestyle='--',
               linewidth=2, label=f"Median: {np.median(data_arrays['dermis_relative_times']):.2f}μs")
    ax.set_xlabel('Dermis Relative Time (μs)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Dermis Relative Time Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. 근막 상대 시간 분포
    ax = axes[0, 1]
    ax.hist(data_arrays['fascia_relative_times'], bins=30, edgecolor='black', alpha=0.7, color='orange')
    ax.axvline(np.mean(data_arrays['fascia_relative_times']), color='red', linestyle='--',
               linewidth=2, label=f"Mean: {np.mean(data_arrays['fascia_relative_times']):.2f}μs")
    ax.axvline(np.median(data_arrays['fascia_relative_times']), color='green', linestyle='--',
               linewidth=2, label=f"Median: {np.median(data_arrays['fascia_relative_times']):.2f}μs")
    ax.set_xlabel('Fascia Relative Time (μs)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Fascia Relative Time Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. 진피 깊이 분포
    ax = axes[0, 2]
    ax.hist(data_arrays['dermis_depths'], bins=30, edgecolor='black', alpha=0.7, color='blue')
    ax.axvline(np.mean(data_arrays['dermis_depths']), color='red', linestyle='--',
               linewidth=2, label=f"Mean: {np.mean(data_arrays['dermis_depths']):.2f}mm")
    ax.axvline(np.median(data_arrays['dermis_depths']), color='green', linestyle='--',
               linewidth=2, label=f"Median: {np.median(data_arrays['dermis_depths']):.2f}mm")
    ax.set_xlabel('Dermis Depth (mm)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Dermis Depth (Thickness1) Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. 근막 깊이 분포
    ax = axes[1, 0]
    ax.hist(data_arrays['fascia_depths'], bins=30, edgecolor='black', alpha=0.7, color='orange')
    ax.axvline(np.mean(data_arrays['fascia_depths']), color='red', linestyle='--',
               linewidth=2, label=f"Mean: {np.mean(data_arrays['fascia_depths']):.2f}mm")
    ax.axvline(np.median(data_arrays['fascia_depths']), color='green', linestyle='--',
               linewidth=2, label=f"Median: {np.median(data_arrays['fascia_depths']):.2f}mm")
    ax.set_xlabel('Fascia Depth (mm)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Fascia Depth (Thickness2) Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. 두께 차이 분포
    ax = axes[1, 1]
    ax.hist(data_arrays['thickness_differences'], bins=30, edgecolor='black', alpha=0.7, color='green')
    ax.axvline(np.mean(data_arrays['thickness_differences']), color='red', linestyle='--',
               linewidth=2, label=f"Mean: {np.mean(data_arrays['thickness_differences']):.2f}mm")
    ax.axvline(np.median(data_arrays['thickness_differences']), color='blue', linestyle='--',
               linewidth=2, label=f"Median: {np.median(data_arrays['thickness_differences']):.2f}mm")
    ax.set_xlabel('Thickness Difference (mm)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Thickness Difference (Thickness2 - Thickness1)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. 상관관계 (Dermis vs Fascia 깊이)
    ax = axes[1, 2]
    ax.scatter(data_arrays['dermis_depths'], data_arrays['fascia_depths'],
               alpha=0.6, s=30, edgecolors='black', linewidths=0.5)

    # 선형 회귀선
    z = np.polyfit(data_arrays['dermis_depths'], data_arrays['fascia_depths'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(data_arrays['dermis_depths'].min(), data_arrays['dermis_depths'].max(), 100)
    ax.plot(x_line, p(x_line), "r--", linewidth=2, label=f'y={z[0]:.2f}x+{z[1]:.2f}')

    # 상관계수 계산
    corr = np.corrcoef(data_arrays['dermis_depths'], data_arrays['fascia_depths'])[0, 1]
    ax.text(0.05, 0.95, f'Correlation: {corr:.3f}',
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel('Dermis Depth (mm)', fontsize=11)
    ax.set_ylabel('Fascia Depth (mm)', fontsize=11)
    ax.set_title('Dermis vs Fascia Depth Correlation', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n통계 시각화 저장: {save_path}")
    plt.close()

def generate_recommendations(stats_dict):
    """알고리즘 개선 권장사항 생성"""
    print(f"\n{'='*70}")
    print(f"Algorithm Improvement Recommendations")
    print(f"{'='*70}\n")

    dermis_mean = stats_dict['dermis_relative_time_us']['mean']
    dermis_std = stats_dict['dermis_relative_time_us']['std']
    fascia_mean = stats_dict['fascia_relative_time_us']['mean']
    fascia_std = stats_dict['fascia_relative_time_us']['std']

    dermis_depth_mean = stats_dict['dermis_depth_mm']['mean']
    dermis_depth_std = stats_dict['dermis_depth_mm']['std']
    fascia_depth_mean = stats_dict['fascia_depth_mm']['mean']
    fascia_depth_std = stats_dict['fascia_depth_mm']['std']

    thickness_diff_mean = stats_dict['thickness_difference_mm']['mean']
    thickness_diff_std = stats_dict['thickness_difference_mm']['std']

    print(f"1. 진피 검출 범위 (Dermis Detection Range):")
    print(f"   - 예상 시간: {dermis_mean:.2f} ± {dermis_std:.2f} μs (T0 기준)")
    print(f"   - 권장 검색 범위: {dermis_mean - 2*dermis_std:.2f} ~ {dermis_mean + 2*dermis_std:.2f} μs")
    print(f"   - 예상 깊이: {dermis_depth_mean:.2f} ± {dermis_depth_std:.2f} mm")
    print(f"   - 권장 깊이 범위: {dermis_depth_mean - 2*dermis_depth_std:.2f} ~ {dermis_depth_mean + 2*dermis_depth_std:.2f} mm\n")

    print(f"2. 근막 검출 범위 (Fascia Detection Range):")
    print(f"   - 예상 시간: {fascia_mean:.2f} ± {fascia_std:.2f} μs (T0 기준)")
    print(f"   - 권장 검색 범위: {fascia_mean - 2*fascia_std:.2f} ~ {fascia_mean + 2*fascia_std:.2f} μs")
    print(f"   - 예상 깊이: {fascia_depth_mean:.2f} ± {fascia_depth_std:.2f} mm")
    print(f"   - 권장 깊이 범위: {fascia_depth_mean - 2*fascia_depth_std:.2f} ~ {fascia_depth_mean + 2*fascia_depth_std:.2f} mm\n")

    print(f"3. 두께 차이 (Thickness Difference):")
    print(f"   - 평균 간격: {thickness_diff_mean:.2f} ± {thickness_diff_std:.2f} mm")
    print(f"   - 권장: 진피 검출 후 {thickness_diff_mean:.2f}mm 떨어진 위치에서 근막 탐색\n")

    print(f"4. 알고리즘 파라미터 권장값:")
    print(f"   - dermis_search_window: {dermis_mean - dermis_std:.1f} ~ {dermis_mean + dermis_std:.1f} μs")
    print(f"   - fascia_search_window: {fascia_mean - fascia_std:.1f} ~ {fascia_mean + fascia_std:.1f} μs")
    print(f"   - expected_thickness_gap: {thickness_diff_mean:.2f} mm")
    print(f"   - thickness_gap_tolerance: ± {thickness_diff_std:.2f} mm\n")

if __name__ == "__main__":
    # 모든 레이블 로드
    labels_data = load_all_manual_labels()

    print(f"Loaded {len(labels_data)} manual label files")

    # 통계 분석
    stats_dict, data_arrays = analyze_label_statistics(labels_data)

    # 통계 출력
    print_statistics(stats_dict)

    # 시각화
    visualize_distributions(data_arrays)

    # 권장사항 생성
    generate_recommendations(stats_dict)

    # JSON으로 저장
    stats_file = 'results/manual_label_statistics.json'
    os.makedirs(os.path.dirname(stats_file), exist_ok=True)
    with open(stats_file, 'w') as f:
        json.dump(stats_dict, f, indent=2)
    print(f"통계 결과 저장: {stats_file}")
