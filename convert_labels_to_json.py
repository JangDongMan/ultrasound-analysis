#!/usr/bin/env python3
"""
Convert label Excel file to JSON format for training data
Reads label.xlsx and generates JSON files in manual_boundaries folder
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path


def calculate_distance(start_time_us, end_time_us, speed_of_sound=1540):
    """
    Calculate distance from time difference

    Args:
        start_time_us: Start time in microseconds
        end_time_us: End time in microseconds
        speed_of_sound: Speed of sound in m/s (default: 1540)

    Returns:
        Distance in mm
    """
    time_diff_s = (end_time_us - start_time_us) * 1e-6
    distance_mm = (time_diff_s * speed_of_sound / 2) * 1000
    return distance_mm


def convert_label_to_json(label_file, data_dir, output_dir, speed_of_sound=1540):
    """
    Convert label file to JSON format

    Args:
        label_file: Path to label Excel/CSV file
        data_dir: Directory containing source CSV data files
        output_dir: Directory to save JSON files
        speed_of_sound: Speed of sound in m/s (default: 1540)
    """
    # Read label file
    try:
        if label_file.endswith('.xlsx'):
            df = pd.read_excel(label_file, engine='openpyxl')
        else:
            df = pd.read_csv(label_file)
    except Exception as e:
        # If xlsx fails, try reading as CSV
        print(f"Warning: Failed to read as Excel, trying CSV format...")
        df = pd.read_csv(label_file)

    print(f"Reading label file: {label_file}")
    print(f"Columns: {list(df.columns)}")
    print(f"Total rows: {len(df)}")
    print()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Check if detailed format (B, C, D, E columns)
    if len(df.columns) >= 5:
        print("Using detailed Excel format (columns B, C, D, E)")
        process_detailed_format(df, data_dir, output_dir, speed_of_sound)
    else:
        print("Using simple CSV format (columns: filename, thickness1, thickness2)")
        process_simple_format(df, data_dir, output_dir, speed_of_sound)


def process_detailed_format(df, data_dir, output_dir, speed_of_sound):
    """
    Process detailed Excel format with B, C, D, E columns

    Columns:
    - A (index 0): filename
    - B (index 1): Signal start time (seconds)
    - C (index 2): T0 skin start time (seconds)
    - D (index 3): T1 dermis start time (seconds)
    - E (index 4): T2 fascia start time (seconds)
    """
    converted_count = 0

    for idx, row in df.iterrows():
        filename = str(row.iloc[0])

        # Skip if filename is empty or NaN
        if pd.isna(filename) or not filename.strip():
            continue

        # Extract base name (remove .csv extension if present)
        base_name = filename.replace('.csv', '')

        # Get time values
        try:
            signal_start_s = float(row.iloc[1]) if pd.notna(row.iloc[1]) else None
            t0_time_s = float(row.iloc[2]) if pd.notna(row.iloc[2]) else None
            t1_time_s = float(row.iloc[3]) if pd.notna(row.iloc[3]) else None
            t2_time_s = float(row.iloc[4]) if pd.notna(row.iloc[4]) else None
        except (ValueError, TypeError, IndexError) as e:
            print(f"⚠ Skipping {filename}: Error reading values - {e}")
            continue

        # Check if we have required data
        if signal_start_s is None or t0_time_s is None:
            print(f"⚠ Skipping {filename}: Missing signal_start or t0_time")
            continue

        # Convert to microseconds
        signal_start_us = signal_start_s * 1e6
        t0_time_us = t0_time_s * 1e6
        t1_time_us = t1_time_s * 1e6 if t1_time_s is not None else None
        t2_time_us = t2_time_s * 1e6 if t2_time_s is not None else None

        # Calculate depths using formulas:
        # F = ((C-B) × 1540 × 1000) / 2
        # G = (((D-B) × 1540 × 1000) / 2) - F
        # H = (((E-B) × 1540 × 1000) / 2) - F

        # T0 depth (F)
        t0_depth_mm = ((t0_time_s - signal_start_s) * speed_of_sound * 1000) / 2

        # Build positions array
        layers = []

        # Position 1: Dermis (T1)
        if t1_time_us is not None:
            absolute_depth1_mm = ((t1_time_s - signal_start_s) * speed_of_sound * 1000) / 2
            depth_from_skin_mm = absolute_depth1_mm - t0_depth_mm  # G: T0부터 T1까지의 거리

            layers.append({
                'position_number': 1,
                'position_name': 'Dermis',
                'time_us': float(t1_time_us),
                'thickness_mm': float(depth_from_skin_mm),  # 피부 시작점부터의 거리 (G)
                'depth_start_mm': 0.0,  # From skin start
                'depth_end_mm': float(depth_from_skin_mm)
            })

        # Position 2: Fascia (T2)
        if t2_time_us is not None:
            absolute_depth2_mm = ((t2_time_s - signal_start_s) * speed_of_sound * 1000) / 2
            depth_from_skin_mm = absolute_depth2_mm - t0_depth_mm  # T0부터 T2까지의 거리

            # depth_start is the end of previous layer
            depth_start_mm = layers[0]['depth_end_mm'] if layers else 0.0

            layers.append({
                'position_number': 2,
                'position_name': 'Fascia',
                'time_us': float(t2_time_us),
                'thickness_mm': float(depth_from_skin_mm),  # 피부 시작점부터의 거리 (G+H)
                'depth_start_mm': float(depth_start_mm),
                'depth_end_mm': float(depth_from_skin_mm)
            })

        # Create JSON data structure
        source_file = os.path.join(data_dir, f"{base_name}.csv")

        data = {
            'source_file': source_file,
            'start_point_us': float(t0_time_us),
            'num_positions': len(layers),
            'speed_of_sound': speed_of_sound,
            'positions': layers
        }

        # Save JSON file
        output_file = os.path.join(output_dir, f"{base_name}_positions.json")

        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            print(f"✓ Created: {output_file}")
            print(f"  Start: {t0_time_us:.2f}μs")
            if layers:
                for layer in layers:
                    print(f"  {layer['position_name']}: {layer['time_us']:.2f}μs, Thickness: {layer['thickness_mm']:.2f}mm")
            print()

            converted_count += 1

        except Exception as e:
            print(f"✗ Error saving {output_file}: {e}")
            print()

    print(f"\n{'='*60}")
    print(f"Conversion complete: {converted_count} files created")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")


def process_simple_format(df, data_dir, output_dir, speed_of_sound):
    """
    Process simple CSV format with filename, thickness1, thickness2 columns

    Note: This format doesn't have time information, so we can't generate proper JSON
    """
    print("⚠ Warning: Simple format doesn't contain time information")
    print("⚠ Cannot generate JSON files without time values")
    print("⚠ Please use detailed Excel format with B, C, D, E columns")


def print_usage():
    """Print usage information"""
    print("=" * 70)
    print("Label to JSON Converter - Usage Guide")
    print("=" * 70)
    print()
    print("DESCRIPTION:")
    print("  Converts label Excel file to JSON format for training data")
    print("  Reads label.xlsx and generates JSON files in manual_boundaries/")
    print()
    print("USAGE:")
    print("  python3 convert_labels_to_json.py [OPTIONS]")
    print()
    print("OPTIONS:")
    print("  -h, --help          Show this help message")
    print("  -i, --input FILE    Input label file (default: label.xlsx)")
    print("  -d, --data DIR      Data directory (default: data/)")
    print("  -o, --output DIR    Output directory (default: manual_boundaries/)")
    print()
    print("REQUIRED EXCEL FORMAT:")
    print("  Column A (filename):      Patient file name (e.g., bhjung-5M-1)")
    print("  Column B (signal_start):  Signal start time in seconds")
    print("  Column C (skin_start):    T0 skin start time in seconds")
    print("  Column D (dermis_start):  T1 dermis start time in seconds")
    print("  Column E (fascia_start):  T2 fascia start time in seconds")
    print()
    print("FORMULAS USED:")
    print("  F = ((C-B) × 1540 × 1000) / 2         (T0 depth in mm)")
    print("  G = (((D-B) × 1540 × 1000) / 2) - F   (T1 thickness in mm)")
    print("  H = (((E-B) × 1540 × 1000) / 2) - F   (T2 thickness in mm)")
    print()
    print("OUTPUT JSON FORMAT:")
    print("  {")
    print('    "source_file": "data/patient.csv",')
    print('    "start_point_us": 17.26,')
    print('    "num_positions": 2,')
    print('    "speed_of_sound": 1540,')
    print('    "positions": [')
    print('      {')
    print('        "position_number": 1,')
    print('        "position_name": "Dermis",')
    print('        "time_us": 19.6,')
    print('        "thickness_mm": 1.80,')
    print('        "depth_start_mm": 0.0,')
    print('        "depth_end_mm": 1.80')
    print('      },')
    print('      ...')
    print('    ]')
    print('  }')
    print()
    print("EXAMPLES:")
    print("  # Convert using default paths")
    print("  python3 convert_labels_to_json.py")
    print()
    print("  # Convert with custom input file")
    print("  python3 convert_labels_to_json.py -i my_labels.xlsx")
    print()
    print("  # Convert with custom output directory")
    print("  python3 convert_labels_to_json.py -o output/json_files/")
    print()
    print("=" * 70)


def main():
    """Main function"""
    import sys

    # Parse command line arguments
    label_file = None
    data_dir = None
    output_dir = None

    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg in ['-h', '--help']:
            print_usage()
            return
        elif arg in ['-i', '--input']:
            if i + 1 < len(sys.argv):
                label_file = sys.argv[i + 1]
                i += 2
            else:
                print("Error: --input requires a file path")
                return
        elif arg in ['-d', '--data']:
            if i + 1 < len(sys.argv):
                data_dir = sys.argv[i + 1]
                i += 2
            else:
                print("Error: --data requires a directory path")
                return
        elif arg in ['-o', '--output']:
            if i + 1 < len(sys.argv):
                output_dir = sys.argv[i + 1]
                i += 2
            else:
                print("Error: --output requires a directory path")
                return
        else:
            print(f"Unknown option: {arg}")
            print("Use -h or --help for usage information")
            return

    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Set default paths if not specified
    if label_file is None:
        label_file = os.path.join(script_dir, "label.xlsx")
        if not os.path.exists(label_file):
            label_file = os.path.join(script_dir, "label.csv")

    if data_dir is None:
        data_dir = os.path.join(script_dir, "data")

    if output_dir is None:
        output_dir = os.path.join(script_dir, "manual_boundaries")

    # Check if label file exists
    if not os.path.exists(label_file):
        print("=" * 70)
        print("ERROR: Label file not found!")
        print("=" * 70)
        print(f"Expected location: {label_file}")
        print()
        print("Please create a label.xlsx file with the following format:")
        print("  Column A: filename (e.g., bhjung-5M-1)")
        print("  Column B: signal_start (seconds)")
        print("  Column C: skin_start (seconds)")
        print("  Column D: dermis_start (seconds)")
        print("  Column E: fascia_start (seconds)")
        print()
        print("Use -h or --help for more information")
        print("=" * 70)
        return

    # Print conversion info
    print()
    print("=" * 70)
    print("Starting Label to JSON Conversion")
    print("=" * 70)
    print(f"Input file:      {label_file}")
    print(f"Data directory:  {data_dir}")
    print(f"Output directory: {output_dir}")
    print("=" * 70)
    print()

    # Convert
    convert_label_to_json(
        label_file=label_file,
        data_dir=data_dir,
        output_dir=output_dir,
        speed_of_sound=1540
    )


if __name__ == "__main__":
    main()
