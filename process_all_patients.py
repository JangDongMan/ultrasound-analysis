#!/usr/bin/env python3
"""
모든 환자 데이터에 대해 피부층 경계 검출 및 시각화 수행
"""

import os
import glob
from visualize_signal_improved import visualize_ultrasound_signal_improved, load_manual_label_json


def get_all_data_files():
    """data 디렉토리의 모든 CSV 파일 찾기"""
    data_files = glob.glob('data/*-5M-*.csv')
    return sorted(data_files)


def extract_patient_and_position(filename):
    """파일명에서 환자 ID와 위치 추출"""
    # 예: data/bhjung-5M-1.csv -> ('bhjung', '1')
    # 예: data/sicho-5M-1-01.csv -> ('sicho-5M-1-01', None) - 특수 케이스

    basename = os.path.basename(filename)  # bhjung-5M-1.csv
    name_without_ext = basename.replace('.csv', '')  # bhjung-5M-1

    parts = name_without_ext.split('-5M-')
    if len(parts) != 2:
        return None, None

    patient_id = parts[0]
    position_str = parts[1]

    return patient_id, position_str


def main():
    """모든 환자 데이터 처리"""

    data_files = get_all_data_files()
    results_dir = 'results/all_patients'

    # 결과 디렉토리 생성
    os.makedirs(results_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"Processing All Patients - Total {len(data_files)} files")
    print(f"{'='*80}\n")

    total_processed = 0
    total_failed = 0
    total_no_label = 0

    for file_path in data_files:
        basename = os.path.basename(file_path)
        patient_id, position_str = extract_patient_and_position(file_path)

        if patient_id is None:
            print(f"⚠ Skipping invalid filename: {basename}")
            total_failed += 1
            continue

        # JSON 레이블 파일명 생성
        label_filename = basename.replace('.csv', '')

        # manual_boundaries 디렉토리에서 JSON 찾기
        json_path = f'manual_boundaries/{label_filename}_positions.json'

        if not os.path.exists(json_path):
            print(f"⚠ No label for {basename}")
            total_no_label += 1
            continue

        # 수동 레이블 로드
        import json
        with open(json_path, 'r') as f:
            manual_label = json.load(f)

        # 결과 파일명 생성
        save_filename = label_filename.replace('-5M-', '_position_') + '_layer_analysis.png'
        save_path = os.path.join(results_dir, save_filename)

        print(f"Processing: {basename}")
        print(f"  Patient: {patient_id}, Position: {position_str}")
        print(f"  Label: {json_path}")
        print(f"  Output: {save_path}")

        try:
            visualize_ultrasound_signal_improved(file_path, save_path, manual_label)
            print(f"✓ Success\n")
            total_processed += 1
        except Exception as e:
            print(f"✗ Error: {e}\n")
            total_failed += 1

    print(f"\n{'='*80}")
    print(f"Processing Complete!")
    print(f"{'='*80}")
    print(f"  Total files: {len(data_files)}")
    print(f"  Successfully processed: {total_processed}")
    print(f"  No label found: {total_no_label}")
    print(f"  Failed: {total_failed}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
