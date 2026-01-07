#!/usr/bin/env python3
"""
원본 label_org.xlsx 파일을 JSON 형식으로 변환
"""

import pandas as pd
import numpy as np
import json
import os


def convert_original_labels_to_json(
    label_file='label_org.xlsx',
    data_dir='data',
    output_dir='manual_boundaries',
    speed_of_sound=1540
):
    """
    원본 Excel 파일을 JSON 레이블로 변환

    원본 파일 구조:
    - Row 0: 헤더 (시간, 거리 등)
    - Row 1: 컬럼명 (filename, TS (10ns), T0 (10ns), T1 (10ns), T2 (10ns), ...)
    - Row 2+: 데이터

    컬럼:
    - A: filename
    - B: TS (10ns) - 신호 시작 시간 (10ns = 1e-8초 단위)
    - C: T0 (10ns) - 피부 시작 (T0)
    - D: T1 (10ns) - 진피 시작 (T1)
    - E: T2 (10ns) - 근막 시작 (T2)
    """
    print("="*70)
    print("Converting Original Label File to JSON")
    print("="*70)
    print(f"Input: {label_file}")
    print(f"Output directory: {output_dir}")
    print()

    # Excel 파일 읽기 (헤더 없이, Row 1을 헤더로 사용)
    df = pd.read_excel(label_file, engine='openpyxl', header=1)

    print(f"Loaded {len(df)} rows")
    print(f"Columns: {list(df.columns)}")
    print()

    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)

    successful = 0
    skipped = 0

    for idx, row in df.iterrows():
        filename = row['filename']

        # NaN 체크
        if pd.isna(filename):
            continue

        # 시간 데이터 읽기 (10ns 단위 = 1e-8초)
        ts_10ns = row['TS (10ns)']  # 신호 시작
        t0_10ns = row['T0 (10ns)']  # 피부 시작
        t1_10ns = row['T1 (10ns)']  # 진피 시작
        t2_10ns = row['T2 (10ns)']  # 근막 시작

        # NaN 체크
        if pd.isna(ts_10ns) or pd.isna(t0_10ns) or pd.isna(t1_10ns) or pd.isna(t2_10ns):
            print(f"⚠ Skipping {filename}: Missing time data")
            skipped += 1
            continue

        # 10ns 단위를 초(s)로 변환 (1e-8초 단위)
        # 원본 데이터는 이미 초 단위로 저장되어 있음 (예: 1.528e-05)
        signal_start_s = ts_10ns
        t0_time_s = t0_10ns
        t1_time_s = t1_10ns
        t2_time_s = t2_10ns

        # μs로 변환
        signal_start_us = signal_start_s * 1e6
        t0_time_us = t0_time_s * 1e6
        t1_time_us = t1_time_s * 1e6
        t2_time_us = t2_time_s * 1e6

        # 깊이 계산 (피부 시작점 T0부터의 거리)
        # 깊이(mm) = (시간차(s) × 음속(m/s) × 1000) / 2

        # T0 깊이 (신호 시작부터)
        t0_depth_mm = ((t0_time_s - signal_start_s) * speed_of_sound * 1000) / 2

        # T1 깊이 (T0부터)
        t1_depth_from_skin_mm = ((t1_time_s - t0_time_s) * speed_of_sound * 1000) / 2

        # T2 깊이 (T0부터)
        t2_depth_from_skin_mm = ((t2_time_s - t0_time_s) * speed_of_sound * 1000) / 2

        # JSON 데이터 구조
        layers = []

        # Position 1: Dermis (T1)
        layers.append({
            'position_number': 1,
            'position_name': 'Dermis',
            'time_us': float(t1_time_us),
            'thickness_mm': float(t1_depth_from_skin_mm),  # T0부터 T1까지
            'depth_start_mm': 0.0,
            'depth_end_mm': float(t1_depth_from_skin_mm)
        })

        # Position 2: Fascia (T2)
        layers.append({
            'position_number': 2,
            'position_name': 'Fascia',
            'time_us': float(t2_time_us),
            'thickness_mm': float(t2_depth_from_skin_mm),  # T0부터 T2까지
            'depth_start_mm': float(t1_depth_from_skin_mm),
            'depth_end_mm': float(t2_depth_from_skin_mm)
        })

        # JSON 구조
        source_file = os.path.join(data_dir, f"{filename}.csv")

        data = {
            'source_file': source_file,
            'start_point_us': float(t0_time_us),  # T0 (피부 시작)
            'num_positions': 2,
            'speed_of_sound': speed_of_sound,
            'positions': layers
        }

        # JSON 파일 저장
        output_file = os.path.join(output_dir, f"{filename}_positions.json")

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        successful += 1

        if successful <= 3:  # 처음 3개만 출력
            print(f"✓ {filename}")
            print(f"  Signal Start: {signal_start_us:.2f}μs")
            print(f"  T0 (Skin):    {t0_time_us:.2f}μs")
            print(f"  T1 (Dermis):  {t1_time_us:.2f}μs (Depth: {t1_depth_from_skin_mm:.2f}mm)")
            print(f"  T2 (Fascia):  {t2_time_us:.2f}μs (Depth: {t2_depth_from_skin_mm:.2f}mm)")
            print(f"  → {output_file}")
            print()

    print("="*70)
    print(f"Conversion Complete!")
    print(f"  Successful: {successful}")
    print(f"  Skipped: {skipped}")
    print("="*70)


if __name__ == "__main__":
    convert_original_labels_to_json(
        label_file='label_org.xlsx',
        data_dir='data',
        output_dir='manual_boundaries',
        speed_of_sound=1540
    )
