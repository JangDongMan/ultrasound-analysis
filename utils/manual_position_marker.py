#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ultrasound Skin Layer Position Manual Marking Tool
Position 1: Dermis Start
Position 2: Fascia Start
"""

import os
import sys
import json
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure


class UltrasoundPositionMarker:
    """GUI for manually marking 2 positions in ultrasound signal"""

    def __init__(self, root):
        self.root = root
        self.root.title("Ultrasound Skin Layer Position Marker")
        self.root.geometry("1400x900")

        # 데이터 변수
        self.file_path = None
        self.time_data = None
        self.voltage_data = None
        self.start_point = None
        self.positions = []  # Position 1 (Dermis), Position 2 (Fascia)
        self.max_positions = 2
        self.position_names = ["Dermis Start", "Fascia Start"]
        self.label_file_path = None  # Excel label file path
        self.current_manual_labels = None  # Current file's manual labels

        # 마우스 추적
        self.mouse_vline = None
        self.distance_text = None
        self.crosshair_h = None
        self.crosshair_v = None

        # UI 구성
        self.setup_ui()

    def setup_ui(self):
        """UI 구성"""
        # 상단 컨트롤 패널
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(control_frame, text="Open CSV",
                  command=self.load_file).pack(side=tk.LEFT, padx=5)

        ttk.Button(control_frame, text="Load Label Excel",
                  command=self.load_label_file).pack(side=tk.LEFT, padx=5)

        self.file_label = ttk.Label(control_frame, text="Select a file",
                                   foreground="gray")
        self.file_label.pack(side=tk.LEFT, padx=10)

        ttk.Separator(control_frame, orient=tk.VERTICAL).pack(side=tk.LEFT,
                                                              fill=tk.Y, padx=10)

        ttk.Button(control_frame, text="Reset Start",
                  command=self.reset_start).pack(side=tk.LEFT, padx=5)

        ttk.Button(control_frame, text="Reset All",
                  command=self.reset_all).pack(side=tk.LEFT, padx=5)

        ttk.Button(control_frame, text="Save Positions",
                  command=self.save_positions,
                  style="Accent.TButton").pack(side=tk.LEFT, padx=5)

        ttk.Button(control_frame, text="Load Positions",
                  command=self.load_positions).pack(side=tk.LEFT, padx=5)

        # 상태 표시
        status_frame = ttk.Frame(self.root, padding="5")
        status_frame.pack(side=tk.TOP, fill=tk.X)

        self.status_label = ttk.Label(status_frame,
                                     text="Click Start point first",
                                     font=("Arial", 10, "bold"),
                                     foreground="blue")
        self.status_label.pack(side=tk.LEFT, padx=10)

        self.position_info_label = ttk.Label(status_frame, text="",
                                            foreground="green")
        self.position_info_label.pack(side=tk.LEFT, padx=20)

        self.label_info_label = ttk.Label(status_frame, text="",
                                          foreground="purple")
        self.label_info_label.pack(side=tk.LEFT, padx=20)

        # 그래프 영역
        self.fig = Figure(figsize=(14, 7), dpi=100)
        self.ax = self.fig.add_subplot(111)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # 툴바
        toolbar_frame = ttk.Frame(self.root)
        toolbar_frame.pack(side=tk.TOP, fill=tk.X)
        toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        toolbar.update()

        # 마우스 이벤트
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.canvas.mpl_connect('button_press_event', self.on_mouse_click)

        # 하단 도움말
        help_frame = ttk.Frame(self.root, padding="10")
        help_frame.pack(side=tk.BOTTOM, fill=tk.X)

        help_text = ("Usage: 1) Open CSV  2) (Optional) Load Label Excel  3) Click Start (Epidermis)  "
                    "4) Click P1 (Dermis Start)  5) Click P2 (Fascia Start)  6) Save Positions")
        ttk.Label(help_frame, text=help_text, foreground="gray").pack()

    def load_label_file(self):
        """Load Excel label file"""
        label_path = filedialog.askopenfilename(
            title="Select Label File",
            filetypes=[("Excel files", "*.xlsx"), ("CSV files", "*.csv"), ("All files", "*.*")],
            initialdir=os.path.dirname(__file__) or "."
        )

        if not label_path:
            return

        df = None
        error_messages = []

        # Try to read the file
        if label_path.endswith('.csv'):
            # Read CSV manually without pandas first
            import io
            encodings_to_try = ['utf-8', 'utf-8-sig', 'cp949', 'euc-kr', 'latin1', 'iso-8859-1']

            for encoding in encodings_to_try:
                try:
                    # Read CSV manually
                    with io.open(label_path, 'r', encoding=encoding, errors='ignore') as f:
                        lines = f.readlines()

                    # Parse CSV manually
                    data = []
                    headers = None
                    for i, line in enumerate(lines):
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split(',')
                        if i == 0:
                            headers = parts
                        else:
                            # Pad or trim to match header length
                            while len(parts) < len(headers):
                                parts.append('')
                            parts = parts[:len(headers)]
                            data.append(parts)

                    # Create DataFrame
                    if headers and data:
                        df = pd.DataFrame(data, columns=headers)
                        print(f"Successfully loaded label CSV with encoding: {encoding}")
                        break
                except Exception as e:
                    error_msg = f"{encoding}: {str(e)[:50]}"
                    error_messages.append(error_msg)
                    print(f"Failed with {error_msg}")
                    continue
        else:
            # Try to read Excel file
            try:
                df = pd.read_excel(label_path)
                print(f"Successfully loaded Excel file")
            except Exception as e:
                error_messages.append(f"Excel: {str(e)}")
                print(f"Failed to load Excel: {e}")

        # Check if loading was successful
        if df is None:
            error_detail = "\n".join(error_messages[:3])  # Show first 3 errors
            messagebox.showerror("Error",
                f"Failed to load label file.\n\nTried encodings: utf-8, cp949, euc-kr, latin1\n\nErrors:\n{error_detail}")
            return

        # Successfully loaded
        try:
            self.label_file_path = label_path
            filename = os.path.basename(label_path)
            self.label_info_label.config(text=f"Label file: {filename}", foreground="purple")

            # If CSV file is already loaded, try to load its labels
            if self.file_path:
                self.load_current_file_labels(df)

            messagebox.showinfo("Success", f"Label file loaded successfully:\n{filename}\n{len(df)} rows")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to process label file:\n{str(e)}")

    def load_current_file_labels(self, df=None):
        """Load manual labels for current CSV file"""
        if not self.file_path:
            print("DEBUG: No file_path set")
            return

        print(f"DEBUG: ===== load_current_file_labels called =====")
        print(f"DEBUG: Label file path: {self.label_file_path}")

        if df is None and self.label_file_path:
            try:
                if self.label_file_path.endswith('.csv'):
                    # Read CSV manually
                    import io
                    for encoding in ['utf-8', 'utf-8-sig', 'cp949', 'euc-kr', 'latin1', 'iso-8859-1']:
                        try:
                            with io.open(self.label_file_path, 'r', encoding=encoding, errors='ignore') as f:
                                lines = f.readlines()

                            # Parse CSV manually
                            data = []
                            headers = None
                            for i, line in enumerate(lines):
                                line = line.strip()
                                if not line:
                                    continue
                                parts = line.split(',')
                                if i == 0:
                                    headers = parts
                                else:
                                    # Pad or trim to match header length
                                    while len(parts) < len(headers):
                                        parts.append('')
                                    parts = parts[:len(headers)]
                                    data.append(parts)

                            if headers and data:
                                df = pd.DataFrame(data, columns=headers)
                                print(f"Loaded current file labels with encoding: {encoding}")
                                break
                        except:
                            continue
                else:
                    df = pd.read_excel(self.label_file_path)
            except:
                return

        if df is None:
            print("DEBUG: DataFrame is None, cannot load labels")
            return

        # Extract patient_id and position from filename
        # e.g., "bhjung-5M-1.csv" -> patient_id="bhjung", position=1
        filename = os.path.basename(self.file_path)
        base_name = os.path.splitext(filename)[0]
        print(f"DEBUG: Looking for labels for file: {filename}, base_name: {base_name}")

        # Find matching row
        print(f"DEBUG: DataFrame columns: {list(df.columns)}")
        print(f"DEBUG: DataFrame shape: {df.shape}")
        print(f"DEBUG: First few rows:\n{df.head()}")

        for col in df.columns:
            if 'file' in col.lower() or 'name' in col.lower():
                print(f"DEBUG: Checking column '{col}' for matching filename")
                print(f"DEBUG: Column values: {df[col].tolist()}")
                matching_rows = df[df[col].astype(str).str.contains(base_name, na=False)]
                print(f"DEBUG: Found {len(matching_rows)} matching rows")
                if not matching_rows.empty:
                    row = matching_rows.iloc[0]
                    print(f"DEBUG: Matched row: {row.to_dict()}")
                    labels = {}

                    # Find thickness1 and thickness2
                    for col_name in df.columns:
                        if 'thickness1' in col_name.lower():
                            val = row[col_name]
                            print(f"DEBUG: thickness1 raw value: '{val}' (type: {type(val)})")
                            if pd.notna(val) and str(val).strip():
                                try:
                                    labels['thickness1'] = float(val)
                                    print(f"Found thickness1: {labels['thickness1']} m (depth in meters)")
                                except (ValueError, TypeError) as e:
                                    print(f"Failed to convert thickness1 '{val}': {e}")
                                    labels['thickness1'] = None
                            else:
                                labels['thickness1'] = None
                        elif 'thickness2' in col_name.lower():
                            val = row[col_name]
                            print(f"DEBUG: thickness2 raw value: '{val}' (type: {type(val)})")
                            if pd.notna(val) and str(val).strip():
                                try:
                                    labels['thickness2'] = float(val)
                                    print(f"Found thickness2: {labels['thickness2']} m (depth in meters)")
                                except (ValueError, TypeError) as e:
                                    print(f"Failed to convert thickness2 '{val}': {e}")
                                    labels['thickness2'] = None
                            else:
                                labels['thickness2'] = None

                    if labels and (labels.get('thickness1') is not None or labels.get('thickness2') is not None):
                        self.current_manual_labels = labels
                        # Convert depth (in meters) to time (in μs)
                        # Formula: time = (depth * 2 / speed_of_sound) * 1e6
                        speed_of_sound = 1540  # m/s
                        if labels.get('thickness1') is not None:
                            depth1_m = labels['thickness1']
                            time1_us = (depth1_m * 2 / speed_of_sound) * 1e6
                            labels['time1_us'] = time1_us
                            print(f"Thickness1: {depth1_m*1000:.3f}mm -> Time: {time1_us:.2f}μs")
                        if labels.get('thickness2') is not None:
                            depth2_m = labels['thickness2']
                            time2_us = (depth2_m * 2 / speed_of_sound) * 1e6
                            labels['time2_us'] = time2_us
                            print(f"Thickness2: {depth2_m*1000:.3f}mm -> Time: {time2_us:.2f}μs")

                        self.plot_signal()
                        return

        # Try index-based matching
        # Check if this is the detailed Excel format with columns B, C, D, E (time in seconds)
        if len(df.columns) >= 5:  # Detailed format: B(1)=signal_start, C(2)=T0, D(3)=T1, E(4)=T2
            print("DEBUG: Using detailed Excel format (columns B, C, D, E - time values in seconds)")
            for idx, row in df.iterrows():
                if base_name in str(row.iloc[0]):
                    labels = {}

                    # Column B (index 1) = Signal start time from sensor
                    valB = row.iloc[1]
                    if pd.notna(valB) and str(valB).strip():
                        try:
                            labels['signal_start'] = float(valB)
                            print(f"[Index] Found Signal Start (col B): RAW='{valB}', FLOAT={labels['signal_start']}, SCIENTIFIC={labels['signal_start']:.6e}")
                        except (ValueError, TypeError) as e:
                            print(f"[Index] Failed to convert Signal Start '{valB}': {e}")
                            labels['signal_start'] = None
                    else:
                        labels['signal_start'] = None

                    # Column C (index 2) = T0 time (manual skin start position - Epidermis) in SECONDS
                    valC = row.iloc[2]
                    if pd.notna(valC) and str(valC).strip():
                        try:
                            labels['t0_start'] = float(valC)
                            print(f"[Index] Found T0 (col C): RAW='{valC}', FLOAT={labels['t0_start']}, SCIENTIFIC={labels['t0_start']:.6e}")
                        except (ValueError, TypeError) as e:
                            print(f"[Index] Failed to convert T0 '{valC}': {e}")
                            labels['t0_start'] = None
                    else:
                        labels['t0_start'] = None

                    # Column D (index 3) = T1 time (manual dermis start position) in SECONDS
                    valD = row.iloc[3]
                    if pd.notna(valD) and str(valD).strip():
                        try:
                            labels['thickness1'] = float(valD)
                            print(f"[Index] Found T1 (col D): RAW='{valD}', FLOAT={labels['thickness1']}, SCIENTIFIC={labels['thickness1']:.6e}")
                        except (ValueError, TypeError) as e:
                            print(f"[Index] Failed to convert T1 '{valD}': {e}")
                            labels['thickness1'] = None
                    else:
                        labels['thickness1'] = None

                    # Column E (index 4) = T2 time (manual fascia start position) in SECONDS
                    valE = row.iloc[4]
                    if pd.notna(valE) and str(valE).strip():
                        try:
                            labels['thickness2'] = float(valE)
                            print(f"[Index] Found T2 (col E): RAW='{valE}', FLOAT={labels['thickness2']}, SCIENTIFIC={labels['thickness2']:.6e}")
                        except (ValueError, TypeError) as e:
                            print(f"[Index] Failed to convert T2 '{valE}': {e}")
                            labels['thickness2'] = None
                    else:
                        labels['thickness2'] = None
        elif len(df.columns) >= 3:  # Simple CSV format: column 1=thickness1, column 2=thickness2
            print("DEBUG: Using simple CSV format (columns 1, 2)")
            for idx, row in df.iterrows():
                if base_name in str(row.iloc[0]):
                    labels = {}

                    # Convert thickness1
                    val1 = row.iloc[1]
                    if pd.notna(val1) and str(val1).strip():
                        try:
                            labels['thickness1'] = float(val1)
                            print(f"[Index] Found thickness1: {labels['thickness1']} m (depth in meters)")
                        except (ValueError, TypeError) as e:
                            print(f"[Index] Failed to convert thickness1 '{val1}': {e}")
                            labels['thickness1'] = None
                    else:
                        labels['thickness1'] = None

                    # Convert thickness2
                    val2 = row.iloc[2]
                    if pd.notna(val2) and str(val2).strip():
                        try:
                            labels['thickness2'] = float(val2)
                            print(f"[Index] Found thickness2: {labels['thickness2']} m (depth in meters)")
                        except (ValueError, TypeError) as e:
                            print(f"[Index] Failed to convert thickness2 '{val2}': {e}")
                            labels['thickness2'] = None
                    else:
                        labels['thickness2'] = None
        else:
            print("DEBUG: DataFrame does not have enough columns")
            return

        # Process the loaded labels
        if labels.get('t0_start') is not None or labels.get('thickness1') is not None or labels.get('thickness2') is not None:
            self.current_manual_labels = labels
            speed_of_sound = 1540  # m/s

            # Excel columns interpretation:
            # B = Signal start time from sensor (센서로부터 신호 시작 시간)
            # C = User-marked skin start time T0 (사용자가 표시한 피부 시작점)
            # D = User-marked dermis start time T1 (사용자가 표시한 진피 시작점)
            # E = User-marked fascia start time T2 (사용자가 표시한 근막 시작점)
            #
            # Formulas (from Excel):
            # F = ((C-B) × 1540 × 1000) / 2  (T0 depth in mm)
            # G = (((D-B) × 1540 × 1000) / 2) - F  (T1 thickness from start in mm)
            # H = (((E-B) × 1540 × 1000) / 2) - F  (T2 thickness from start in mm)
            #
            # Arrow positions on graph:
            # T0 position = C (absolute time)
            # T1 position = D (absolute time)
            # T2 position = E (absolute time)
            #
            # Arrow labels:
            # T0 label = F (absolute depth)
            # T1 label = G (thickness from skin start)
            # T2 label = H (thickness from skin start)

            signal_start_s = labels.get('signal_start', 0)

            # T0 - Manual skin start position (Epidermis)
            if labels.get('t0_start') is not None:
                # C column: absolute skin start time in seconds
                time0_s = labels['t0_start']
                time0_us = time0_s * 1e6  # Convert to microseconds
                # F = ((C-B) × 1540 × 1000) / 2
                depth0_mm = ((time0_s - signal_start_s) * speed_of_sound * 1000) / 2
                depth0_m = depth0_mm / 1000

                labels['time0_us'] = time0_us
                labels['t0_depth_m'] = depth0_m
                print(f"[Index] T0: Time={time0_us:.2f}μs, Depth F=((C-B)×1540×1000)/2={depth0_mm:.4f}mm")

            # T1 - Manual dermis position
            if labels.get('thickness1') is not None and labels.get('t0_start') is not None:
                # D column: absolute dermis time in seconds
                time1_s = labels['thickness1']
                time1_us = time1_s * 1e6
                # First calculate absolute depth: ((D-B) × 1540 × 1000) / 2
                absolute_depth1_mm = ((time1_s - signal_start_s) * speed_of_sound * 1000) / 2
                # G = absolute_depth1 - F (thickness from skin start)
                depth0_mm = ((labels['t0_start'] - signal_start_s) * speed_of_sound * 1000) / 2
                thickness1_mm = absolute_depth1_mm - depth0_mm
                thickness1_m = thickness1_mm / 1000

                labels['time1_us'] = time1_us
                labels['thickness1_depth_m'] = thickness1_m
                print(f"[Index] T1: Time={time1_us:.2f}μs, Thickness G=(((D-B)×1540×1000)/2)-F={thickness1_mm:.4f}mm")

            # T2 - Manual fascia position
            if labels.get('thickness2') is not None and labels.get('t0_start') is not None:
                # E column: absolute fascia time in seconds
                time2_s = labels['thickness2']
                time2_us = time2_s * 1e6
                # First calculate absolute depth: ((E-B) × 1540 × 1000) / 2
                absolute_depth2_mm = ((time2_s - signal_start_s) * speed_of_sound * 1000) / 2
                # H = absolute_depth2 - F (thickness from skin start)
                depth0_mm = ((labels['t0_start'] - signal_start_s) * speed_of_sound * 1000) / 2
                thickness2_mm = absolute_depth2_mm - depth0_mm
                thickness2_m = thickness2_mm / 1000

                labels['time2_us'] = time2_us
                labels['thickness2_depth_m'] = thickness2_m
                print(f"[Index] T2: Time={time2_us:.2f}μs, Thickness H=(((E-B)×1540×1000)/2)-F={thickness2_mm:.4f}mm")

            self.plot_signal()
            return

    def auto_find_label_file(self, csv_file_path):
        """Automatically find and load label file"""
        # Search paths for label file
        search_paths = [
            r"D:\util\masker\data",  # Windows path (for compiled exe)
            os.path.dirname(csv_file_path),  # Same directory as CSV
            os.path.dirname(__file__),  # Same directory as script (utils/)
            os.path.join(os.path.dirname(__file__), ".."),  # Project root
            os.path.join(os.path.dirname(__file__), "..", "data"),  # Project data directory
        ]

        print(f"DEBUG: Auto-finding label file for CSV: {csv_file_path}")
        print(f"DEBUG: Search paths: {search_paths}")

        # Possible label file names
        label_filenames = ['label.xlsx', 'label.csv', 'labels.xlsx', 'labels.csv']

        for search_dir in search_paths:
            if not os.path.exists(search_dir):
                print(f"DEBUG: Search path does not exist: {search_dir}")
                continue
            print(f"DEBUG: Searching in: {search_dir}")
            for label_name in label_filenames:
                label_path = os.path.join(search_dir, label_name)
                print(f"DEBUG: Checking: {label_path}")
                if os.path.exists(label_path):
                    print(f"DEBUG: Found label file: {label_path}")
                    try:
                        # Try to load the label file
                        df = None
                        if label_path.endswith('.csv'):
                            # Try different encodings - use manual CSV parsing
                            import io
                            for encoding in ['utf-8', 'utf-8-sig', 'cp949', 'euc-kr', 'latin1', 'iso-8859-1']:
                                try:
                                    with io.open(label_path, 'r', encoding=encoding, errors='ignore') as f:
                                        lines = f.readlines()

                                    # Parse CSV manually
                                    data = []
                                    headers = None
                                    for i, line in enumerate(lines):
                                        line = line.strip()
                                        if not line:
                                            continue
                                        parts = line.split(',')
                                        if i == 0:
                                            headers = parts
                                        else:
                                            # Pad or trim to match header length
                                            while len(parts) < len(headers):
                                                parts.append('')
                                            parts = parts[:len(headers)]
                                            data.append(parts)

                                    # Create DataFrame
                                    if headers and data:
                                        df = pd.DataFrame(data, columns=headers)
                                        print(f"Auto-loaded {label_path} with encoding: {encoding}")
                                        break
                                except Exception as e:
                                    print(f"Failed {label_path} with {encoding}: {e}")
                                    continue
                            if df is None:
                                print(f"Could not load {label_path} with any encoding")
                                continue
                        else:
                            df = pd.read_excel(label_path)

                        self.label_file_path = label_path
                        self.label_info_label.config(
                            text=f"Auto-loaded: {os.path.basename(label_path)}",
                            foreground="green"
                        )
                        return df
                    except Exception as e:
                        print(f"Failed to load {label_path}: {e}")
                        continue
        return None

    def load_file(self):
        """Load CSV file"""
        file_path = filedialog.askopenfilename(
            title="Select CSV File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialdir=os.path.join(os.path.dirname(__file__), "..", "data")
        )

        if not file_path:
            return

        times = []
        voltages = []
        error_messages = []

        # Try different encodings
        encodings_to_try = ['utf-8', 'utf-8-sig', 'cp949', 'euc-kr', 'latin1', 'iso-8859-1']

        for encoding in encodings_to_try:
            try:
                # Python 2/3 compatible way to open file with encoding
                import io
                with io.open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                    lines = f.readlines()[2:]
                    for line in lines:
                        parts = line.strip().split(',')
                        if len(parts) >= 2 and parts[1]:
                            try:
                                times.append(float(parts[0]))
                                voltages.append(float(parts[1]))
                            except ValueError:
                                continue

                print(f"Successfully loaded CSV with encoding: {encoding}")
                break  # Success, exit the loop

            except Exception as e:
                error_msg = f"{encoding}: {str(e)[:50]}"
                error_messages.append(error_msg)
                print(f"Failed to load CSV with {error_msg}")
                times = []  # Reset for next attempt
                voltages = []
                continue

        # Check if loading was successful
        if not times:
            error_detail = "\n".join(error_messages[:3])
            messagebox.showerror("Error",
                f"Failed to load CSV file.\n\nTried encodings: utf-8, cp949, euc-kr, latin1\n\nErrors:\n{error_detail}")
            return

        try:
            self.time_data = np.array(times) * 1e6  # μs
            self.voltage_data = np.array(voltages)
            self.file_path = file_path

            filename = os.path.basename(file_path)
            self.file_label.config(text=f"File: {filename}", foreground="black")

            self.reset_all()

            # Auto-find and load label file if not already loaded
            if not self.label_file_path:
                df = self.auto_find_label_file(file_path)
                if df is not None:
                    self.load_current_file_labels(df)
            else:
                # Load labels for this file if label file is already loaded
                self.load_current_file_labels()

            self.plot_signal()

            messagebox.showinfo("Success", f"File loaded successfully.\n{len(times)} samples")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to process file:\n{str(e)}")

    def plot_signal(self):
        """Plot signal with positions and manual labels"""
        if self.time_data is None:
            return

        self.ax.clear()

        # Raw signal
        self.ax.plot(self.time_data, self.voltage_data, 'b-',
                    linewidth=0.5, alpha=0.7, label='Raw Signal')

        # Start point (Epidermis)
        if self.start_point is not None:
            self.ax.axvline(x=self.start_point, color='red',
                          linestyle='--', linewidth=2,
                          label=f'Start (Epidermis: {self.start_point:.2f}μs)')

        # Manual labels from Excel
        if self.current_manual_labels is not None:
            print(f"DEBUG: Drawing manual labels - {self.current_manual_labels}")
            print(f"DEBUG: Current file: {self.file_path}")

            # T0 - Manual start position (Red arrow - independent of program's start_point)
            if 'time0_us' in self.current_manual_labels and self.current_manual_labels['time0_us'] is not None:
                time0 = self.current_manual_labels['time0_us']
                idx = np.argmin(np.abs(self.time_data - time0))
                voltage0 = self.voltage_data[idx]
                depth0_mm = self.current_manual_labels['t0_depth_m'] * 1000

                print(f"DEBUG: T0 at time={time0:.2f}μs, voltage={voltage0:.4f}V, depth={depth0_mm:.3f}mm")

                # Draw arrow pointing down
                y_range = self.ax.get_ylim()
                arrow_start_y = y_range[1] - (y_range[1] - y_range[0]) * 0.05
                self.ax.annotate('', xy=(time0, voltage0), xytext=(time0, arrow_start_y),
                               arrowprops=dict(arrowstyle='->', color='red', lw=3),
                               zorder=10)
                self.ax.text(time0, arrow_start_y, f'T0 (Manual Start)\n{depth0_mm:.2f}mm',
                           ha='center', va='bottom', color='red', fontsize=10, fontweight='bold')
                self.ax.axvline(x=time0, color='red', linestyle=':', linewidth=1.5, alpha=0.5)

            # T1 - Manual Dermis position (Blue arrow)
            if 'time1_us' in self.current_manual_labels and self.current_manual_labels['time1_us'] is not None:
                time1 = self.current_manual_labels['time1_us']
                idx = np.argmin(np.abs(self.time_data - time1))
                voltage1 = self.voltage_data[idx]
                depth1_mm = self.current_manual_labels['thickness1_depth_m'] * 1000

                print(f"DEBUG: T1 at time={time1:.2f}μs, voltage={voltage1:.4f}V, depth={depth1_mm:.3f}mm")

                # Draw arrow pointing down
                y_range = self.ax.get_ylim()
                arrow_start_y = y_range[1] - (y_range[1] - y_range[0]) * 0.15
                self.ax.annotate('', xy=(time1, voltage1), xytext=(time1, arrow_start_y),
                               arrowprops=dict(arrowstyle='->', color='blue', lw=3),
                               zorder=10)
                self.ax.text(time1, arrow_start_y, f'T1 (Dermis)\n{depth1_mm:.2f}mm',
                           ha='center', va='bottom', color='blue', fontsize=10, fontweight='bold')
                self.ax.axvline(x=time1, color='blue', linestyle=':', linewidth=1.5, alpha=0.5)

            # T2 - Manual Fascia position (Cyan arrow)
            if 'time2_us' in self.current_manual_labels and self.current_manual_labels['time2_us'] is not None:
                time2 = self.current_manual_labels['time2_us']
                idx = np.argmin(np.abs(self.time_data - time2))
                voltage2 = self.voltage_data[idx]
                depth2_mm = self.current_manual_labels['thickness2_depth_m'] * 1000

                print(f"DEBUG: T2 at time={time2:.2f}μs, voltage={voltage2:.4f}V, depth={depth2_mm:.3f}mm")

                # Draw arrow pointing down
                y_range = self.ax.get_ylim()
                arrow_start_y = y_range[1] - (y_range[1] - y_range[0]) * 0.25
                self.ax.annotate('', xy=(time2, voltage2), xytext=(time2, arrow_start_y),
                               arrowprops=dict(arrowstyle='->', color='cyan', lw=3),
                               zorder=10)
                self.ax.text(time2, arrow_start_y, f'T2 (Fascia)\n{depth2_mm:.2f}mm',
                           ha='center', va='bottom', color='cyan', fontsize=10, fontweight='bold')
                self.ax.axvline(x=time2, color='cyan', linestyle=':', linewidth=1.5, alpha=0.5)
        else:
            if self.current_manual_labels is None:
                print("DEBUG: No manual labels loaded")
            if self.start_point is None:
                print("DEBUG: Start point not set")

        # Position markers
        colors = ['orange', 'green']
        for i, position in enumerate(self.positions):
            if position is not None:
                position_name = self.position_names[i]
                self.ax.axvline(x=position, color=colors[i],
                              linestyle='--', linewidth=2,
                              label=f'P{i+1} ({position_name}: {position:.2f}μs)')

                # Calculate and display distance
                if self.start_point is not None:
                    distance = self.calculate_distance(self.start_point, position)
                    y_pos = self.ax.get_ylim()[1] * (0.9 - i * 0.15)
                    self.ax.text(position, y_pos,
                               f'{distance:.3f}mm',
                               bbox=dict(boxstyle='round,pad=0.5',
                                       facecolor=colors[i], alpha=0.7),
                               fontsize=10, weight='bold')

        self.ax.set_xlabel('Time (μs)', fontsize=12)
        self.ax.set_ylabel('Voltage (V)', fontsize=12)
        self.ax.set_title('Ultrasound Signal - Skin Layer Position Marking (Dermis/Fascia)', fontsize=14, weight='bold')
        self.ax.grid(True, alpha=0.3)

        # Set X-axis range to show only the signal data range
        # Find the range where voltage is significantly above noise level
        min_time = np.min(self.time_data)
        max_time = np.max(self.time_data)

        # Find where signal starts (first significant voltage change)
        voltage_threshold = np.max(np.abs(self.voltage_data)) * 0.01  # 1% of max voltage
        signal_indices = np.where(np.abs(self.voltage_data) > voltage_threshold)[0]

        if len(signal_indices) > 0:
            # Add some margin (10%) before and after the signal
            signal_start = self.time_data[signal_indices[0]]
            signal_end = self.time_data[signal_indices[-1]]
            margin = (signal_end - signal_start) * 0.1

            x_min = max(min_time, signal_start - margin)
            x_max = min(max_time, signal_end + margin)

            self.ax.set_xlim(x_min, x_max)

        # Add manual label info box in upper right
        if self.current_manual_labels is not None:
            info_lines = []
            if 't0_depth_m' in self.current_manual_labels and self.current_manual_labels['t0_depth_m'] is not None:
                depth0_mm = self.current_manual_labels['t0_depth_m'] * 1000
                info_lines.append(f'T0 Depth: {depth0_mm:.2f}mm')
            if 'thickness1_depth_m' in self.current_manual_labels and self.current_manual_labels['thickness1_depth_m'] is not None:
                thickness1_mm = self.current_manual_labels['thickness1_depth_m'] * 1000
                info_lines.append(f'T1 Thickness: {thickness1_mm:.2f}mm')
            if 'thickness2_depth_m' in self.current_manual_labels and self.current_manual_labels['thickness2_depth_m'] is not None:
                thickness2_mm = self.current_manual_labels['thickness2_depth_m'] * 1000
                info_lines.append(f'T2 Thickness: {thickness2_mm:.2f}mm')

            if info_lines:
                info_text = '\n'.join(info_lines)
                self.ax.text(0.98, 0.98, info_text,
                           transform=self.ax.transAxes,
                           verticalalignment='top',
                           horizontalalignment='right',
                           bbox=dict(boxstyle='round,pad=0.8',
                                   facecolor='lightblue',
                                   edgecolor='black',
                                   alpha=0.8),
                           fontsize=11,
                           fontweight='bold')

        self.ax.legend(loc='upper left')

        self.canvas.draw()

    def on_mouse_move(self, event):
        """Mouse movement event"""
        if event.inaxes != self.ax or self.time_data is None:
            return

        # Remove previous elements
        if self.crosshair_v is not None:
            try:
                self.crosshair_v.remove()
            except:
                pass
            self.crosshair_v = None

        if self.crosshair_h is not None:
            try:
                self.crosshair_h.remove()
            except:
                pass
            self.crosshair_h = None

        if self.distance_text is not None:
            try:
                self.distance_text.remove()
            except:
                pass
            self.distance_text = None

        # Draw crosshair
        self.crosshair_v = self.ax.axvline(x=event.xdata, color='gray',
                                          linestyle=':', linewidth=1, alpha=0.7)
        self.crosshair_h = self.ax.axhline(y=event.ydata, color='gray',
                                          linestyle=':', linewidth=1, alpha=0.7)

        # Distance display
        if self.start_point is not None:
            distance = self.calculate_distance(self.start_point, event.xdata)
            time_from_start = abs(event.xdata - self.start_point)
            text_str = f'Distance: {distance:.4f}mm\nTime from start: {time_from_start:.2f}μs\nCursor: {event.xdata:.2f}μs'

            # Position text above cursor
            y_offset = (self.ax.get_ylim()[1] - self.ax.get_ylim()[0]) * 0.05
            self.distance_text = self.ax.text(event.xdata, event.ydata + y_offset, text_str,
                                            bbox=dict(boxstyle='round,pad=0.7',
                                                    facecolor='yellow',
                                                    edgecolor='black',
                                                    alpha=0.9),
                                            fontsize=10,
                                            verticalalignment='bottom',
                                            horizontalalignment='center')

        self.canvas.draw_idle()

    def on_mouse_click(self, event):
        """Mouse click event"""
        if event.inaxes != self.ax or self.time_data is None:
            return

        if event.button == 1:  # Left click
            # Set start point
            if self.start_point is None:
                self.start_point = event.xdata
                self.status_label.config(
                    text=f"Start set: {self.start_point:.2f}μs. Click P1 (Dermis Start)",
                    foreground="blue")
                self.plot_signal()

            # Set position
            elif len(self.positions) < self.max_positions:
                self.positions.append(event.xdata)
                position_num = len(self.positions)
                distance = self.calculate_distance(self.start_point, event.xdata)

                if position_num < self.max_positions:
                    next_name = self.position_names[position_num]
                    self.status_label.config(
                        text=f"P{position_num} set ({distance:.3f}mm). Click P{position_num+1} ({next_name})",
                        foreground="blue")
                else:
                    self.status_label.config(
                        text="All positions set. Click 'Save Positions'",
                        foreground="green")

                self.update_position_info()
                self.plot_signal()
            else:
                messagebox.showwarning("Warning",
                    "Maximum 2 positions allowed.\nClick 'Reset All' to restart.")

    def calculate_distance(self, time1, time2):
        """거리 계산 (mm)"""
        speed_of_sound = 1540  # m/s
        time_diff = abs(time2 - time1) * 1e-6  # μs → s
        distance = (time_diff * speed_of_sound / 2) * 1000  # mm
        return distance

    def update_position_info(self):
        """Update position information"""
        if not self.positions:
            self.position_info_label.config(text="")
            return

        info_text = f"Positions set: {len(self.positions)}"
        if self.start_point is not None:
            distances = [self.calculate_distance(self.start_point, p)
                        for p in self.positions]
            info_text += f" | Distance: " + ", ".join([f"{d:.3f}mm" for d in distances])

        self.position_info_label.config(text=info_text)

    def reset_start(self):
        """Reset start point"""
        if self.start_point is None:
            return

        if messagebox.askyesno("Confirm", "Reset start point?"):
            self.start_point = None
            self.positions = []
            self.status_label.config(
                text="Click Start point first",
                foreground="blue")
            self.update_position_info()
            self.plot_signal()

    def reset_all(self):
        """Reset all markers"""
        self.start_point = None
        self.positions = []
        self.current_manual_labels = None
        self.status_label.config(
            text="Click Start point first",
            foreground="blue")
        self.update_position_info()
        if self.time_data is not None:
            self.plot_signal()

    def save_positions(self):
        """Save position data"""
        if self.file_path is None:
            messagebox.showwarning("Warning", "Open CSV file first")
            return

        if self.start_point is None:
            messagebox.showwarning("Warning", "Start point not set")
            return

        if len(self.positions) < 2:
            messagebox.showwarning("Warning", "Set both positions (P1 and P2)")
            return

        # Prepare data to save
        speed_of_sound = 1540

        layers = []
        prev_time = self.start_point

        for i, position_time in enumerate(self.positions):
            time_diff = (position_time - prev_time) * 1e-6
            thickness = (time_diff * speed_of_sound / 2) * 1000
            depth_start = self.calculate_distance(self.start_point, prev_time)
            depth_end = self.calculate_distance(self.start_point, position_time)

            layers.append({
                'position_number': i + 1,
                'position_name': self.position_names[i],
                'time_us': float(position_time),
                'thickness_mm': float(thickness),
                'depth_start_mm': float(depth_start),
                'depth_end_mm': float(depth_end)
            })

            prev_time = position_time

        data = {
            'source_file': self.file_path,
            'start_point_us': float(self.start_point),
            'num_positions': len(self.positions),
            'speed_of_sound': speed_of_sound,
            'positions': layers
        }

        # Save file path
        base_name = os.path.splitext(os.path.basename(self.file_path))[0]
        default_filename = f"{base_name}_positions.json"
        save_dir = os.path.join(os.path.dirname(self.file_path), "..", "manual_boundaries")
        os.makedirs(save_dir, exist_ok=True)

        save_path = filedialog.asksaveasfilename(
            title="Save Position Data",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir=save_dir,
            initialfile=default_filename
        )

        if save_path:
            try:
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

                messagebox.showinfo("Success",
                    f"Position data saved:\n{save_path}\n\n"
                    f"Start: {self.start_point:.2f}μs\n"
                    f"Dermis: {self.positions[0]:.2f}μs\n"
                    f"Fascia: {self.positions[1]:.2f}μs")

            except Exception as e:
                messagebox.showerror("Error", f"Save failed:\n{str(e)}")

    def load_positions(self):
        """Load saved positions"""
        if self.file_path is None:
            messagebox.showwarning("Warning", "Open CSV file first")
            return

        load_path = filedialog.askopenfilename(
            title="Load Position Data",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir=os.path.join(os.path.dirname(self.file_path), "..", "manual_boundaries")
        )

        if not load_path:
            return

        try:
            with open(load_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.start_point = data['start_point_us']
            self.positions = [pos['time_us'] for pos in data['positions']]

            self.status_label.config(
                text=f"Position data loaded ({len(self.positions)} positions)",
                foreground="green")

            self.update_position_info()
            self.plot_signal()

            messagebox.showinfo("Success",
                f"Position data loaded:\n"
                f"Start: {self.start_point:.2f}μs\n"
                f"Dermis: {self.positions[0]:.2f}μs\n"
                f"Fascia: {self.positions[1]:.2f}μs")

        except Exception as e:
            messagebox.showerror("Error", f"Load failed:\n{str(e)}")


def main():
    """메인 함수"""
    root = tk.Tk()
    app = UltrasoundPositionMarker(root)
    root.mainloop()


if __name__ == "__main__":
    main()
