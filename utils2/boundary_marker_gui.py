"""
Ultrasound Boundary Marker GUI
- CSV 로드: utils3 캡처 (ADC 값) + 구형 오실로스코프 (시간,전압) 자동 감지
- JSON 로드: 기존 마킹 파일 열기 → 대응 CSV 자동 탐색 → 확인/수정
- 시작점(Start), 진피(Dermis), 근막(Fascia) 경계 수동 마킹
- 시작점 이후 마우스 이동 시 거리(μs, mm) 실시간 표시
- JSON 저장 (manual_boundaries 형식 호환)
"""

import customtkinter as ctk
from tkinter import filedialog, messagebox
import os
import json

import numpy as np
from scipy.signal import hilbert
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator
import matplotlib.font_manager as fm
import platform

# Korean font setup for matplotlib (OS별 자동 선택)
def _setup_korean_font():
    candidates = {
        'Windows': ['Malgun Gothic', 'Microsoft YaHei', 'Arial Unicode MS'],
        'Darwin':  ['AppleGothic', 'Arial Unicode MS'],
        'Linux':   ['NanumGothic', 'NanumBarunGothic', 'UnDotum', 'DejaVu Sans'],
    }
    os_name = platform.system()
    for name in candidates.get(os_name, []):
        if any(name.lower() in f.name.lower() for f in fm.fontManager.ttflist):
            matplotlib.rcParams['font.family'] = name
            matplotlib.rcParams['axes.unicode_minus'] = False
            return
    # fallback: search all fonts for CJK support
    for f in fm.fontManager.ttflist:
        if any(k in f.name for k in ['Gothic', 'Gulim', 'Batang', 'Nanum']):
            matplotlib.rcParams['font.family'] = f.name
            matplotlib.rcParams['axes.unicode_minus'] = False
            return

_setup_korean_font()

from marker_config import MarkerConfig


# Constants
SAMPLE_INTERVAL_NS = 10
SPEED_OF_SOUND = 1540.0  # m/s

# Data trimming (must match utils3 capture settings)
TRIM_START = 1200       # Discard first 1200 samples (12.00 μs)
TRIM_COUNT = 1250       # Keep next 1250 samples (~10mm at 1540 m/s)
DISPLAY_OFFSET_US = 12.00  # Time offset for display (μs)


ctk.set_default_color_theme("blue")


class BoundaryMarkerGUI(ctk.CTk):
    """Ultrasound boundary marker GUI"""

    def __init__(self):
        super().__init__()

        self.title("VB5K Boundary Marker")

        # Center window
        w, h = 1400, 950
        x = (self.winfo_screenwidth() - w) // 2
        y = (self.winfo_screenheight() - h) // 2
        self.geometry(f"{w}x{h}+{x}+{y}")
        self.minsize(1100, 800)

        # Config
        self.config = MarkerConfig()
        ctk.set_appearance_mode(self.config.theme)

        # Data
        self.filepath = None
        self.adc_values = np.array([])
        self.time_us = np.array([])

        # File navigation (CSV files in same directory)
        self.file_list = []
        self.file_index = -1

        # Previous file data for compare overlay
        self.prev_adc_values = np.array([])
        self.prev_time_us = np.array([])
        self.prev_filepath = None
        self.show_compare = False

        # Compare-mode shift: PREVIOUS signal is shifted for alignment
        self.display_shift_us = 0.0   # visual shift only, original data unchanged
        self._drag_start_xdata = None
        self._drag_start_shift = 0.0
        self._is_shift_drag = False
        # Cached line ref for fast shift update (previous signal only)
        self._prev_plot_line = None

        # Markers (time in μs, None = not set)
        self.start_us = None
        self.dermis_us = None
        self.fascia_us = None
        self.json_path = None  # Track loaded/saved JSON path

        # Marker mode: "start" → "dermis" → "fascia"
        self.marker_mode = "start"

        # Envelope toggle (default: ON)
        self.show_envelope = True

        # Cursor tracking line and annotation
        self.cursor_line = None
        self.cursor_annot = None

        self._create_ui()

    def _create_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        # ===== Row 0: Title =====
        ctk.CTkLabel(self, text="VB5K Boundary Marker",
                     font=ctk.CTkFont(size=28, weight="bold")).grid(
            row=0, column=0, padx=15, pady=(10, 2))

        # ===== Row 1: File controls =====
        file_frame = ctk.CTkFrame(self, corner_radius=10)
        file_frame.grid(row=1, column=0, padx=15, pady=(5, 5), sticky="ew")

        ctk.CTkButton(file_frame, text="Open CSV", width=100,
                      fg_color="#007bff", hover_color="#0056b3",
                      command=self._open_csv).grid(
            row=0, column=0, padx=(15, 5), pady=10)

        ctk.CTkButton(file_frame, text="Open JSON", width=100,
                      fg_color="#e67e22", hover_color="#d35400",
                      command=self._open_json).grid(
            row=0, column=1, padx=5, pady=10)

        self.file_label = ctk.CTkLabel(file_frame, text="No file loaded",
                                       font=ctk.CTkFont(size=13))
        self.file_label.grid(row=0, column=2, padx=10, pady=10, sticky="w")

        file_frame.grid_columnconfigure(2, weight=1)

        ctk.CTkButton(file_frame, text="Set Folder", width=90,
                      fg_color="#17a2b8", hover_color="#138496",
                      command=self._set_save_folder).grid(
            row=0, column=3, padx=5, pady=10)

        self.folder_label = ctk.CTkLabel(file_frame, text="Save: (CSV folder)",
                                         font=ctk.CTkFont(size=11), text_color="gray")
        self.folder_label.grid(row=0, column=4, padx=5, pady=10, sticky="w")
        if self.config.save_directory:
            self.folder_label.configure(
                text=f"Save: {self.config.save_directory}",
                text_color="#17a2b8")

        file_frame.grid_columnconfigure(4, weight=0)

        self.save_btn = ctk.CTkButton(file_frame, text="Save JSON", width=100,
                                      fg_color="#28a745", hover_color="#218838",
                                      command=self._save_json, state="disabled")
        self.save_btn.grid(row=0, column=5, padx=10, pady=10)

        self.clear_btn = ctk.CTkButton(file_frame, text="Clear Markers", width=100,
                                       fg_color="#6c757d", hover_color="#545b62",
                                       command=self._clear_markers, state="disabled")
        self.clear_btn.grid(row=0, column=6, padx=(5, 15), pady=10)

        # ===== Row 2: Graph (with prev/next nav buttons on sides) =====
        graph_frame = ctk.CTkFrame(self, corner_radius=10)
        graph_frame.grid(row=2, column=0, padx=8, pady=4, sticky="nsew")
        graph_frame.grid_rowconfigure(0, weight=1)
        graph_frame.grid_columnconfigure(1, weight=1)

        # Prev file button (left side)
        self.prev_btn = ctk.CTkButton(
            graph_frame, text="◀\nPrev", width=48, font=ctk.CTkFont(size=13),
            fg_color="#495057", hover_color="#343a40",
            command=self._load_prev_file, state="disabled")
        self.prev_btn.grid(row=0, column=0, padx=(6, 2), pady=6, sticky="ns")

        # Graph canvas (center)
        canvas_inner = ctk.CTkFrame(graph_frame, fg_color="transparent")
        canvas_inner.grid(row=0, column=1, sticky="nsew", padx=2, pady=5)
        canvas_inner.grid_rowconfigure(0, weight=1)
        canvas_inner.grid_columnconfigure(0, weight=1)

        self.fig = Figure(figsize=(14, 7), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.fig.subplots_adjust(left=0.05, right=0.99, top=0.95, bottom=0.08)

        self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_inner)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        toolbar_frame = ctk.CTkFrame(canvas_inner, fg_color="transparent")
        toolbar_frame.grid(row=1, column=0, sticky="ew")
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()

        # Next file button (right side)
        self.next_btn = ctk.CTkButton(
            graph_frame, text="▶\nNext", width=48, font=ctk.CTkFont(size=13),
            fg_color="#495057", hover_color="#343a40",
            command=self._load_next_file, state="disabled")
        self.next_btn.grid(row=0, column=2, padx=(2, 6), pady=6, sticky="ns")

        # Connect mouse events
        # press: record position, release: place marker if no drag (zoom/pan compatible)
        self._press_pos = None
        self.canvas.mpl_connect('button_press_event', self._on_press)
        self.canvas.mpl_connect('button_release_event', self._on_release)
        self.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)

        # ===== Row 3: Marker controls + info =====
        ctrl_frame = ctk.CTkFrame(self, corner_radius=10)
        ctrl_frame.grid(row=3, column=0, padx=15, pady=5, sticky="ew")

        # Left: marker mode selection
        mode_frame = ctk.CTkFrame(ctrl_frame, fg_color="transparent")
        mode_frame.grid(row=0, column=0, padx=15, pady=10, sticky="w")

        ctk.CTkLabel(mode_frame, text="Marker:",
                     font=ctk.CTkFont(size=14, weight="bold")).grid(
            row=0, column=0, padx=(0, 10))

        self.mode_var = ctk.StringVar(value="start")

        self.start_radio = ctk.CTkRadioButton(
            mode_frame, text="Start", variable=self.mode_var,
            value="start", command=self._on_mode_change,
            fg_color="#ffaa00", hover_color="#cc8800")
        self.start_radio.grid(row=0, column=1, padx=8)

        self.dermis_radio = ctk.CTkRadioButton(
            mode_frame, text="Dermis", variable=self.mode_var,
            value="dermis", command=self._on_mode_change,
            fg_color="#ff4444", hover_color="#cc3333")
        self.dermis_radio.grid(row=0, column=2, padx=8)

        self.fascia_radio = ctk.CTkRadioButton(
            mode_frame, text="Fascia", variable=self.mode_var,
            value="fascia", command=self._on_mode_change,
            fg_color="#4488ff", hover_color="#3366cc")
        self.fascia_radio.grid(row=0, column=3, padx=8)

        # Separator
        sep = ctk.CTkFrame(ctrl_frame, width=2, fg_color="gray")
        sep.grid(row=0, column=1, padx=10, pady=8, sticky="ns")

        # Center: envelope toggle + theme
        toggle_frame = ctk.CTkFrame(ctrl_frame, fg_color="transparent")
        toggle_frame.grid(row=0, column=2, padx=10, pady=10)

        ctk.CTkLabel(toggle_frame, text="Envelope:").grid(row=0, column=0, padx=5)
        self.envelope_switch = ctk.CTkSwitch(toggle_frame, text="",
                                             command=self._toggle_envelope,
                                             width=40)
        self.envelope_switch.grid(row=0, column=1, padx=5)
        self.envelope_switch.select()  # default ON

        ctk.CTkLabel(toggle_frame, text="Dark:").grid(row=0, column=2, padx=(15, 5))
        self.theme_switch = ctk.CTkSwitch(toggle_frame, text="",
                                          command=self._toggle_theme,
                                          width=40)
        self.theme_switch.grid(row=0, column=3, padx=5)
        if self.config.theme == "dark":
            self.theme_switch.select()

        # Separator
        sep2 = ctk.CTkFrame(ctrl_frame, width=2, fg_color="gray")
        sep2.grid(row=0, column=3, padx=10, pady=8, sticky="ns")

        # Compare button
        self.compare_btn = ctk.CTkButton(
            ctrl_frame, text="Compare", width=90,
            fg_color="#6c757d", hover_color="#545b62",
            command=self._toggle_compare, state="disabled")
        self.compare_btn.grid(row=0, column=4, padx=(5, 2), pady=10)

        # Reset Shift button (only useful when compare+shift active)
        self.reset_shift_btn = ctk.CTkButton(
            ctrl_frame, text="⟳ Align", width=75,
            fg_color="#343a40", hover_color="#212529",
            command=self._reset_shift, state="disabled")
        self.reset_shift_btn.grid(row=0, column=5, padx=(2, 8), pady=10)

        # Separator
        sep3 = ctk.CTkFrame(ctrl_frame, width=2, fg_color="gray")
        sep3.grid(row=0, column=6, padx=5, pady=8, sticky="ns")

        # Right: marker info
        info_frame = ctk.CTkFrame(ctrl_frame, fg_color="transparent")
        info_frame.grid(row=0, column=7, padx=15, pady=10, sticky="e")
        ctrl_frame.grid_columnconfigure(7, weight=1)

        self.start_info = ctk.CTkLabel(
            info_frame, text="Start: --",
            font=ctk.CTkFont(size=13), text_color="#ffaa00")
        self.start_info.grid(row=0, column=0, padx=10, sticky="e")

        self.dermis_info = ctk.CTkLabel(
            info_frame, text="Dermis: --",
            font=ctk.CTkFont(size=13), text_color="#ff6666")
        self.dermis_info.grid(row=0, column=1, padx=10, sticky="e")

        self.fascia_info = ctk.CTkLabel(
            info_frame, text="Fascia: --",
            font=ctk.CTkFont(size=13), text_color="#6699ff")
        self.fascia_info.grid(row=0, column=2, padx=10, sticky="e")

        self.gap_info = ctk.CTkLabel(
            info_frame, text="Gap: --",
            font=ctk.CTkFont(size=13), text_color="#aaaaaa")
        self.gap_info.grid(row=0, column=3, padx=10, sticky="e")

        # ===== Row 4: Status bar (cursor distance + status) =====
        status_frame = ctk.CTkFrame(self, corner_radius=10, height=30)
        status_frame.grid(row=4, column=0, padx=15, pady=(5, 15), sticky="ew")

        self.cursor_info = ctk.CTkLabel(
            status_frame, text="",
            font=ctk.CTkFont(size=13, weight="bold"), text_color="#00ddaa")
        self.cursor_info.grid(row=0, column=0, padx=15, pady=6, sticky="w")

        self.status_label = ctk.CTkLabel(status_frame, text="Ready - Open a CSV file to begin",
                                         font=ctk.CTkFont(size=12))
        self.status_label.grid(row=0, column=1, padx=15, pady=6)

        self.data_info = ctk.CTkLabel(status_frame, text="",
                                      font=ctk.CTkFont(size=12), text_color="gray")
        self.data_info.grid(row=0, column=2, padx=15, pady=6, sticky="e")
        status_frame.grid_columnconfigure(2, weight=1)

        # Initial graph
        self._update_graph()

    # ---- File I/O ----

    def _set_save_folder(self):
        """Set save directory for JSON files"""
        initial_dir = self.config.save_directory or self.config.last_open_dir or os.getcwd()
        folder = filedialog.askdirectory(initialdir=initial_dir,
                                         title="Select Save Folder for JSON")
        if folder:
            self.config.update(save_directory=folder)
            self.folder_label.configure(text=f"Save: {folder}", text_color="#17a2b8")
            self.status_label.configure(text=f"Save folder: {folder}")

    def _open_csv(self):
        initial_dir = self.config.last_open_dir or os.getcwd()
        filepath = filedialog.askopenfilename(
            initialdir=initial_dir,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Open Capture CSV"
        )
        if not filepath:
            return

        try:
            adc_arr, time_us = self._load_csv(filepath)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV:\n{e}")
            return

        # Save current data as previous for compare
        if len(self.adc_values) > 0:
            self.prev_adc_values = self.adc_values.copy()
            self.prev_time_us = self.time_us.copy()
            self.prev_filepath = self.filepath
            self.compare_btn.configure(state="normal")

        self.filepath = filepath
        self.adc_values = adc_arr
        self.time_us = time_us

        # Reset display shift on new file load
        self.display_shift_us = 0.0
        self.reset_shift_btn.configure(state="disabled")

        # Clear markers
        self.start_us = None
        self.dermis_us = None
        self.fascia_us = None
        self.mode_var.set("start")
        self.marker_mode = "start"

        # Check for existing JSON
        self.json_path = self._get_json_path(filepath)
        if os.path.exists(self.json_path):
            self._load_existing_json(self.json_path)
        else:
            self.json_path = None

        # Update config
        self.config.update(last_open_dir=os.path.dirname(filepath))

        self._update_display_after_load()

    def _load_csv(self, filepath):
        """Load CSV with auto-detection (old oscilloscope or new utils3 format)

        Returns:
            (adc_values: np.ndarray, time_us: np.ndarray)
        """
        with open(filepath, 'r') as f:
            lines = f.readlines()

        # Detect old oscilloscope format (header: "x-axis,1" / "second,Volt")
        is_old = any('x-axis' in l or 'second,Volt' in l
                     for l in lines[:5])

        if is_old:
            return self._load_csv_old(lines)
        else:
            return self._load_csv_new(lines)

    def _load_csv_old(self, lines):
        """Old oscilloscope format: 2 header lines + time(s),voltage(V)"""
        VREF = 1.25
        ADC_MAX = 255
        adc_values = []
        time_us_list = []

        for line in lines:
            line = line.strip()
            if not line or 'x-axis' in line or 'second' in line:
                continue
            parts = line.split(',')
            if len(parts) >= 2:
                try:
                    t_us = float(parts[0]) * 1e6  # seconds -> μs
                    v = float(parts[1])
                    adc = int(np.clip(v / VREF * ADC_MAX, 0, 255))
                    time_us_list.append(t_us)
                    adc_values.append(adc)
                except ValueError:
                    continue

        if not adc_values:
            raise ValueError("No valid data found in CSV (old format)")

        return np.array(adc_values, dtype=np.int32), np.array(time_us_list)

    def _load_csv_new(self, lines):
        """New utils3 format: one ADC value (0-255) per line, 10ns interval"""
        adc_values = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                val = int(line)
                if 0 <= val <= 255:
                    adc_values.append(val)
            except ValueError:
                continue

        if not adc_values:
            raise ValueError("No valid ADC values found in file")

        adc_arr = np.array(adc_values, dtype=np.int32)

        # Trim long captures (4200+ samples) to relevant window
        if len(adc_arr) > TRIM_COUNT:
            end = TRIM_START + TRIM_COUNT
            if len(adc_arr) >= end:
                adc_arr = adc_arr[TRIM_START:end]
            elif len(adc_arr) > TRIM_START:
                adc_arr = adc_arr[TRIM_START:]

        time_us = np.arange(len(adc_arr)) * SAMPLE_INTERVAL_NS / 1000.0 + DISPLAY_OFFSET_US
        return adc_arr, time_us

    def _get_json_path(self, csv_path):
        """Get JSON output path (save_directory if set, else same dir as CSV)"""
        csv_name = os.path.basename(csv_path)
        base_name = os.path.splitext(csv_name)[0] + "_positions.json"

        if self.config.save_directory:
            return os.path.join(self.config.save_directory, base_name)
        else:
            return os.path.join(os.path.dirname(csv_path), base_name)

    def _load_existing_json(self, json_path):
        """Load previously saved markers from JSON"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.start_us = data.get('start_point_us')

            for pos in data.get('positions', []):
                name = pos.get('position_name', '')
                time_val = pos.get('time_us')
                if time_val is not None:
                    if name == 'Dermis':
                        self.dermis_us = time_val
                    elif name == 'Fascia':
                        self.fascia_us = time_val

            # Set mode to next unset marker
            if self.start_us is None:
                self.mode_var.set("start")
                self.marker_mode = "start"
            elif self.dermis_us is None:
                self.mode_var.set("dermis")
                self.marker_mode = "dermis"
            elif self.fascia_us is None:
                self.mode_var.set("fascia")
                self.marker_mode = "fascia"
            else:
                self.mode_var.set("fascia")
                self.marker_mode = "fascia"

            self.status_label.configure(
                text=f"Loaded existing markers from {os.path.basename(json_path)}")
        except (json.JSONDecodeError, IOError):
            pass

    def _open_json(self):
        """Open existing JSON marker file, find and load corresponding CSV"""
        initial_dir = self.config.last_open_dir or os.getcwd()
        json_filepath = filedialog.askopenfilename(
            initialdir=initial_dir,
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Open Marker JSON"
        )
        if not json_filepath:
            return

        try:
            with open(json_filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            messagebox.showerror("Error", f"Failed to load JSON:\n{e}")
            return

        # Find corresponding CSV
        source_file = data.get('source_file', '')
        if not source_file:
            messagebox.showerror("Error", "JSON has no 'source_file' field")
            return

        csv_path = self._find_csv(json_filepath, source_file)
        if not csv_path:
            return

        # Load CSV
        try:
            adc_arr, time_us = self._load_csv(csv_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV:\n{e}")
            return

        self.filepath = csv_path
        self.adc_values = adc_arr
        self.time_us = time_us
        self.json_path = json_filepath

        # Load markers from JSON
        self.start_us = data.get('start_point_us')
        self.dermis_us = None
        self.fascia_us = None
        for pos in data.get('positions', []):
            name = pos.get('position_name', '')
            time_val = pos.get('time_us')
            if time_val is not None:
                if name == 'Dermis':
                    self.dermis_us = time_val
                elif name == 'Fascia':
                    self.fascia_us = time_val

        # Set mode to next unset marker
        if self.start_us is None:
            self.mode_var.set("start")
            self.marker_mode = "start"
        elif self.dermis_us is None:
            self.mode_var.set("dermis")
            self.marker_mode = "dermis"
        elif self.fascia_us is None:
            self.mode_var.set("fascia")
            self.marker_mode = "fascia"
        else:
            self.mode_var.set("fascia")
            self.marker_mode = "fascia"

        # Update config
        self.config.update(last_open_dir=os.path.dirname(json_filepath))

        self._update_display_after_load(
            status=f"Loaded: {os.path.basename(json_filepath)} "
                   f"(CSV: {os.path.basename(csv_path)})")

    def _find_csv(self, json_path, source_file):
        """Find CSV file referenced by JSON source_file field"""
        json_dir = os.path.dirname(os.path.abspath(json_path))
        basename = os.path.basename(source_file)
        parent_dir = os.path.dirname(json_dir)

        candidates = [
            os.path.join(json_dir, source_file),           # relative to JSON dir
            os.path.join(json_dir, basename),               # basename in JSON dir
            os.path.join(parent_dir, source_file),          # relative to parent (project root)
            os.path.join(parent_dir, basename),             # basename in parent
            os.path.join(parent_dir, 'data', basename),     # data/ subfolder of parent
        ]

        for c in candidates:
            norm = os.path.normpath(c)
            if os.path.exists(norm):
                return norm

        # Not found - ask user
        messagebox.showinfo(
            "CSV Not Found",
            f"Cannot find: {source_file}\n"
            f"Please locate the CSV file manually.")
        filepath = filedialog.askopenfilename(
            initialdir=parent_dir,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title=f"Locate CSV: {basename}"
        )
        return filepath if filepath else None

    def _update_display_after_load(self, status=None):
        """Common UI update after loading CSV (from _open_csv or _open_json)"""
        filename = os.path.basename(self.filepath)
        self.file_label.configure(text=filename)
        self.save_btn.configure(state="normal")
        self.clear_btn.configure(state="normal")

        start_us = self.time_us[0] if len(self.time_us) > 0 else 0
        end_us = self.time_us[-1] if len(self.time_us) > 0 else 0
        self.data_info.configure(
            text=f"Samples: {len(self.adc_values)} | "
                 f"{start_us:.1f}~{end_us:.1f} us | "
                 f"Min: {np.min(self.adc_values)} | Max: {np.max(self.adc_values)}")

        if status:
            self.status_label.configure(text=status)
        else:
            self.status_label.configure(
                text=f"Loaded: {filename} - Click to mark Start point")

        self._build_file_list()
        self._update_nav_buttons()
        self._update_graph()
        self._update_marker_info()

    # ---- File navigation ----

    def _build_file_list(self):
        """Scan directory of current file and build sorted CSV file list"""
        if not self.filepath:
            return
        csv_dir = os.path.dirname(os.path.abspath(self.filepath))
        files = sorted([
            os.path.join(csv_dir, f)
            for f in os.listdir(csv_dir)
            if f.lower().endswith('.csv')
        ])
        self.file_list = files
        abs_path = os.path.abspath(self.filepath)
        self.file_index = files.index(abs_path) if abs_path in files else -1

    def _update_nav_buttons(self):
        """Enable/disable prev/next buttons based on current file position"""
        if not self.file_list or self.file_index < 0:
            self.prev_btn.configure(state="disabled")
            self.next_btn.configure(state="disabled")
            return
        total = len(self.file_list)
        self.prev_btn.configure(
            state="normal" if self.file_index > 0 else "disabled",
            text=f"◀\nPrev\n[{self.file_index}/{total}]" if self.file_index > 0
                 else "◀\nPrev")
        self.next_btn.configure(
            state="normal" if self.file_index < total - 1 else "disabled",
            text=f"▶\nNext\n[{self.file_index + 2}/{total}]" if self.file_index < total - 1
                 else "▶\nNext")

    def _load_prev_file(self):
        """Load previous CSV file in directory"""
        if self.file_index <= 0:
            return
        self._load_file_at_index(self.file_index - 1)

    def _load_next_file(self):
        """Load next CSV file in directory"""
        if self.file_index >= len(self.file_list) - 1:
            return
        self._load_file_at_index(self.file_index + 1)

    def _load_file_at_index(self, idx):
        """Load CSV at given index, saving current data as 'previous' for compare"""
        filepath = self.file_list[idx]

        try:
            adc_arr, time_us = self._load_csv(filepath)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV:\n{e}")
            return

        # Save current data as previous for compare
        if len(self.adc_values) > 0:
            self.prev_adc_values = self.adc_values.copy()
            self.prev_time_us = self.time_us.copy()
            self.prev_filepath = self.filepath
            self.compare_btn.configure(state="normal")

        self.filepath = filepath
        self.file_index = idx
        self.adc_values = adc_arr
        self.time_us = time_us

        # Reset display shift on new file load
        self.display_shift_us = 0.0
        self.reset_shift_btn.configure(state="disabled")

        # Clear markers
        self.start_us = None
        self.dermis_us = None
        self.fascia_us = None
        self.mode_var.set("start")
        self.marker_mode = "start"

        # Check for existing JSON
        self.json_path = self._get_json_path(filepath)
        if os.path.exists(self.json_path):
            self._load_existing_json(self.json_path)
        else:
            self.json_path = None

        self.config.update(last_open_dir=os.path.dirname(filepath))

        filename = os.path.basename(filepath)
        self.file_label.configure(text=filename)
        self.save_btn.configure(state="normal")
        self.clear_btn.configure(state="normal")

        start_us = self.time_us[0] if len(self.time_us) > 0 else 0
        end_us = self.time_us[-1] if len(self.time_us) > 0 else 0
        self.data_info.configure(
            text=f"Samples: {len(self.adc_values)} | "
                 f"{start_us:.1f}~{end_us:.1f} us | "
                 f"Min: {np.min(self.adc_values)} | Max: {np.max(self.adc_values)}")

        total = len(self.file_list)
        self.status_label.configure(
            text=f"[{idx + 1}/{total}] {filename} - Click to mark Start point")
        self._update_nav_buttons()
        self._update_graph()
        self._update_marker_info()

    def _toggle_compare(self):
        """Toggle compare overlay (prev data in red)"""
        if len(self.prev_adc_values) == 0:
            return
        self.show_compare = not self.show_compare
        if self.show_compare:
            self.compare_btn.configure(fg_color="#dc3545", hover_color="#c82333",
                                       text="Compare ●")
        else:
            # Reset shift when turning off compare
            self.display_shift_us = 0.0
            self.reset_shift_btn.configure(state="disabled")
            self.compare_btn.configure(fg_color="#6c757d", hover_color="#545b62",
                                       text="Compare")
        self._update_graph()

    def _reset_shift(self):
        """Reset display shift to zero (re-align signals)"""
        self.display_shift_us = 0.0
        self.reset_shift_btn.configure(state="disabled")
        self._update_graph()
        self.status_label.configure(text="Shift reset - signals aligned")

    def _save_json(self):
        if self.filepath is None:
            return

        if self.start_us is None or self.dermis_us is None or self.fascia_us is None:
            missing = []
            if self.start_us is None:
                missing.append("Start")
            if self.dermis_us is None:
                missing.append("Dermis")
            if self.fascia_us is None:
                missing.append("Fascia")
            messagebox.showwarning("Warning",
                                   f"Missing markers: {', '.join(missing)}\n"
                                   f"All 3 markers must be set before saving.")
            return

        speed = self.config.speed_of_sound

        # thickness = distance from start point
        dermis_mm = (self.dermis_us - self.start_us) * speed / 2000.0
        fascia_mm = (self.fascia_us - self.start_us) * speed / 2000.0

        data = {
            "source_file": os.path.basename(self.filepath),
            "start_point_us": round(self.start_us, 4),
            "num_positions": 2,
            "speed_of_sound": speed,
            "sample_interval_ns": SAMPLE_INTERVAL_NS,
            "num_samples": len(self.adc_values),
            "positions": [
                {
                    "position_number": 1,
                    "position_name": "Dermis",
                    "time_us": round(self.dermis_us, 4),
                    "thickness_mm": round(dermis_mm, 4),
                    "depth_start_mm": 0.0,
                    "depth_end_mm": round(dermis_mm, 4)
                },
                {
                    "position_number": 2,
                    "position_name": "Fascia",
                    "time_us": round(self.fascia_us, 4),
                    "thickness_mm": round(fascia_mm, 4),
                    "depth_start_mm": round(dermis_mm, 4),
                    "depth_end_mm": round(fascia_mm, 4)
                }
            ]
        }

        json_path = self.json_path or self._get_json_path(self.filepath)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        self.json_path = json_path
        self.status_label.configure(text=f"Saved: {os.path.basename(json_path)}")

    # ---- Graph ----

    def _update_graph(self):
        self.ax.clear()

        is_dark = ctk.get_appearance_mode() == "Dark"
        bg_color = '#2b2b2b' if is_dark else '#ffffff'
        text_color = 'white' if is_dark else 'black'
        grid_color = 'gray' if is_dark else 'lightgray'

        self.ax.set_facecolor(bg_color)
        self.fig.set_facecolor(bg_color)
        self.ax.grid(True, alpha=0.3, color=grid_color)
        self.ax.tick_params(colors=text_color)
        for spine in self.ax.spines.values():
            spine.set_color(grid_color)

        # Reset cached line ref
        self._prev_plot_line = None

        # Effective shift (only applied to PREVIOUS signal when compare is active)
        shift = self.display_shift_us if self.show_compare else 0.0

        if len(self.adc_values) > 0:
            # Compare overlay: PREVIOUS file in red, shifted for alignment
            if self.show_compare and len(self.prev_adc_values) > 0:
                prev_name = os.path.basename(self.prev_filepath) if self.prev_filepath else "Prev"
                prev_xdata = self.prev_time_us + shift
                if self.show_envelope:
                    prev_env = self._compute_envelope(self.prev_adc_values)
                    self._prev_plot_line = self.ax.plot(
                        prev_xdata, prev_env,
                        color='#ff4444', linewidth=0.8, alpha=0.6,
                        label=f"Prev: {prev_name}", linestyle='--')[0]
                else:
                    self._prev_plot_line = self.ax.plot(
                        prev_xdata, self.prev_adc_values,
                        color='#ff4444', linewidth=0.8, alpha=0.6,
                        label=f"Prev: {prev_name}", linestyle='--')[0]

            # Current file: FIXED (no shift) — markers placed here directly
            if self.show_envelope:
                envelope = self._compute_envelope(self.adc_values)
                line_color = '#00ff88' if is_dark else '#28a745'
                curr_name = os.path.basename(self.filepath) if self.filepath else "Current"
                self.ax.plot(self.time_us, envelope, color=line_color, linewidth=0.8,
                            label=f"Curr: {curr_name}" if self.show_compare else None)
                ylabel = "Envelope"
                title = "Ultrasound Envelope Signal"
            else:
                line_color = '#00bfff' if is_dark else '#007bff'
                curr_name = os.path.basename(self.filepath) if self.filepath else "Current"
                self.ax.plot(self.time_us, self.adc_values, color=line_color, linewidth=0.8,
                            label=f"Curr: {curr_name}" if self.show_compare else None)
                ylabel = "ADC Value"
                title = "Ultrasound RF Signal"

            # Show shift amount in title
            if self.show_compare and abs(shift) > 0.001:
                mm = shift * self.config.speed_of_sound / 2000.0
                title += f"  [prev shift {shift:+.3f} μs / {mm:+.3f} mm]"
            self.ax.set_ylabel(ylabel, color=text_color)
            self.ax.set_title(title, color=text_color)

            # Draw markers on CURRENT signal (no shift needed)
            has_legend = self.show_compare and len(self.prev_adc_values) > 0
            if self.start_us is not None:
                self.ax.axvline(x=self.start_us, color='#ffaa00', linewidth=2,
                               linestyle='-', label='Start', alpha=0.9)
                has_legend = True
            if self.dermis_us is not None:
                self.ax.axvline(x=self.dermis_us, color='#ff4444', linewidth=2,
                               linestyle='--', label='Dermis', alpha=0.9)
                has_legend = True
            if self.fascia_us is not None:
                self.ax.axvline(x=self.fascia_us, color='#4488ff', linewidth=2,
                               linestyle='--', label='Fascia', alpha=0.9)
                has_legend = True

            if has_legend:
                self.ax.legend(loc='upper right', fontsize=8,
                              facecolor=bg_color, edgecolor=grid_color,
                              labelcolor=text_color)

            # Fine x-axis ticks
            self.ax.xaxis.set_major_locator(MultipleLocator(1.0))
            self.ax.xaxis.set_minor_locator(MultipleLocator(0.5))
            self.ax.tick_params(axis='x', which='minor', length=3)
        else:
            self.ax.set_xlim(DISPLAY_OFFSET_US,
                            DISPLAY_OFFSET_US + (TRIM_COUNT - 1) * 0.01)
            self.ax.set_ylim(0, 256)
            self.ax.set_ylabel("ADC Value", color=text_color)
            self.ax.set_title("Ultrasound Signal (No Data)", color=text_color)

        self.ax.set_xlabel("Time (us)", color=text_color)
        self.cursor_line = None
        self.cursor_annot = None
        self.canvas.draw()

    def _compute_envelope(self, signal):
        signal_centered = signal - np.mean(signal)
        analytic_signal = hilbert(signal_centered)
        return np.abs(analytic_signal)

    def _fast_shift_redraw(self):
        """Shift only the previous (red) signal line — current signal and markers stay fixed."""
        # Lock axes to prevent auto-rescale (current signal stays visually fixed)
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        shift = self.display_shift_us
        if self._prev_plot_line is not None:
            self._prev_plot_line.set_xdata(self.prev_time_us + shift)

        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)

        # Update title shift info
        is_dark = ctk.get_appearance_mode() == "Dark"
        text_color = 'white' if is_dark else 'black'
        base = "Ultrasound Envelope Signal" if self.show_envelope else "Ultrasound RF Signal"
        if abs(shift) > 0.001:
            mm = shift * self.config.speed_of_sound / 2000.0
            self.ax.set_title(f"{base}  [prev shift {shift:+.3f} μs / {mm:+.3f} mm]",
                              color=text_color)
        else:
            self.ax.set_title(base, color=text_color)
        self.canvas.draw_idle()

    # ---- Mouse handling ----

    def _on_press(self, event):
        """Record press position to distinguish click vs drag"""
        if event.inaxes == self.ax and event.button == 1:
            self._press_pos = (event.x, event.y)
            self._is_shift_drag = False
            # Prepare for compare-mode shift drag
            if self.show_compare and event.xdata is not None:
                self._drag_start_xdata = event.xdata
                self._drag_start_shift = self.display_shift_us
            else:
                self._drag_start_xdata = None
        else:
            self._press_pos = None
            self._drag_start_xdata = None
            self._is_shift_drag = False

    def _on_release(self, event):
        """Place marker on click (not drag). Works even when zoomed."""
        was_shift_drag = self._is_shift_drag
        self._is_shift_drag = False
        self._drag_start_xdata = None

        if event.inaxes != self.ax or event.button != 1:
            return
        if len(self.adc_values) == 0:
            return
        if self._press_pos is None:
            return

        # Check if mouse moved (drag = zoom/pan/shift, not a marker click)
        dx = abs(event.x - self._press_pos[0])
        dy = abs(event.y - self._press_pos[1])
        self._press_pos = None
        if dx > 5 or dy > 5 or was_shift_drag:
            return  # was a drag, skip marking

        time_us = event.xdata
        if time_us is None:
            return

        # Clamp to data range (current signal is always at original time_us, no shift)
        time_us = max(self.time_us[0], min(time_us, self.time_us[-1]))

        if self.marker_mode == "start":
            self.start_us = time_us
            self.mode_var.set("dermis")
            self.marker_mode = "dermis"
            self.status_label.configure(text="Start set - Click to mark Dermis")
        elif self.marker_mode == "dermis":
            self.dermis_us = time_us
            self.mode_var.set("fascia")
            self.marker_mode = "fascia"
            self.status_label.configure(text="Dermis set - Click to mark Fascia")
        elif self.marker_mode == "fascia":
            self.fascia_us = time_us
            self.status_label.configure(text="Fascia set - All markers placed. Save when ready.")

        self._update_graph()
        self._update_marker_info()

    def _on_mouse_move(self, event):
        """Show distance from start point as cursor moves (status bar + on-graph annotation)"""
        # ---- Compare-mode shift drag ----
        if (self.show_compare and
                self._press_pos is not None and
                self._drag_start_xdata is not None and
                event.inaxes == self.ax and
                event.xdata is not None):
            dx_px = abs(event.x - self._press_pos[0])
            if dx_px > 3:
                self._is_shift_drag = True
                new_shift = self._drag_start_shift + (event.xdata - self._drag_start_xdata)
                self.display_shift_us = new_shift
                self._fast_shift_redraw()
                mm = new_shift * self.config.speed_of_sound / 2000.0
                self.status_label.configure(
                    text=f"Shift: {new_shift:+.3f} μs  ({mm:+.3f} mm)  "
                         f"│  ⟳ Align 버튼으로 초기화")
                self.reset_shift_btn.configure(state="normal")
                return  # skip cursor annotation during drag

        if event.inaxes != self.ax:
            self.cursor_info.configure(text="")
            # Remove cursor line and annotation
            removed = False
            if self.cursor_line is not None:
                self.cursor_line.remove()
                self.cursor_line = None
                removed = True
            if self.cursor_annot is not None:
                self.cursor_annot.remove()
                self.cursor_annot = None
                removed = True
            if removed:
                self.canvas.draw_idle()
            return

        if len(self.adc_values) == 0:
            return

        cursor_us = event.xdata
        if cursor_us is None:
            return

        is_dark = ctk.get_appearance_mode() == "Dark"
        annot_text = ""

        if self.start_us is not None:
            delta_us = cursor_us - self.start_us
            delta_mm = delta_us * self.config.speed_of_sound / 2000.0
            self.cursor_info.configure(
                text=f"Cursor: {cursor_us:.2f} us | "
                     f"From Start: {delta_us:+.2f} us ({delta_mm:+.2f} mm)")
            annot_text = f"{delta_us:+.2f} us\n{delta_mm:+.2f} mm"
        else:
            self.cursor_info.configure(text=f"Cursor: {cursor_us:.2f} us")
            annot_text = f"{cursor_us:.2f} us"

        # Update cursor vertical line
        if self.cursor_line is not None:
            self.cursor_line.set_xdata([cursor_us, cursor_us])
        else:
            cursor_color = '#ffffff55' if is_dark else '#00000033'
            self.cursor_line = self.ax.axvline(
                x=cursor_us, color=cursor_color, linewidth=1, linestyle=':')

        # Update annotation text near cursor
        annot_color = '#00ffaa' if is_dark else '#006644'
        bbox_color = '#333333cc' if is_dark else '#ffffffcc'
        if self.cursor_annot is not None:
            self.cursor_annot.remove()
        self.cursor_annot = self.ax.annotate(
            annot_text,
            xy=(cursor_us, event.ydata),
            xytext=(12, -25), textcoords='offset points',
            fontsize=10, fontweight='bold', color=annot_color,
            bbox=dict(boxstyle='round,pad=0.3', facecolor=bbox_color,
                      edgecolor=annot_color, alpha=0.9))

        self.canvas.draw_idle()

    # ---- UI callbacks ----

    def _on_mode_change(self):
        self.marker_mode = self.mode_var.get()

    def _toggle_envelope(self):
        self.show_envelope = self.envelope_switch.get()
        self._update_graph()

    def _toggle_theme(self):
        if self.theme_switch.get():
            ctk.set_appearance_mode("dark")
            self.config.update(theme="dark")
        else:
            ctk.set_appearance_mode("light")
            self.config.update(theme="light")
        self._update_graph()

    def _clear_markers(self):
        """Undo last marker: Fascia → Dermis → Start (reverse order)"""
        if self.fascia_us is not None:
            self.fascia_us = None
            self.mode_var.set("fascia")
            self.marker_mode = "fascia"
            self.status_label.configure(text="Fascia cleared - Click to mark Fascia")
        elif self.dermis_us is not None:
            self.dermis_us = None
            self.mode_var.set("dermis")
            self.marker_mode = "dermis"
            self.status_label.configure(text="Dermis cleared - Click to mark Dermis")
        elif self.start_us is not None:
            self.start_us = None
            self.mode_var.set("start")
            self.marker_mode = "start"
            self.cursor_info.configure(text="")
            self.status_label.configure(text="Start cleared - Click to mark Start point")
        self._update_graph()
        self._update_marker_info()

    def _update_marker_info(self):
        speed = self.config.speed_of_sound

        if self.start_us is not None:
            self.start_info.configure(text=f"Start: {self.start_us:.2f} us")
        else:
            self.start_info.configure(text="Start: --")

        if self.start_us is not None and self.dermis_us is not None:
            d_us = self.dermis_us - self.start_us
            dmm = d_us * speed / 2000.0
            self.dermis_info.configure(
                text=f"Dermis: {d_us:.2f} us ({dmm:.2f} mm)")
        else:
            self.dermis_info.configure(text="Dermis: --")

        if self.start_us is not None and self.fascia_us is not None:
            f_us = self.fascia_us - self.start_us
            fmm = f_us * speed / 2000.0
            self.fascia_info.configure(
                text=f"Fascia: {f_us:.2f} us ({fmm:.2f} mm)")
        else:
            self.fascia_info.configure(text="Fascia: --")

        if (self.start_us is not None and
                self.dermis_us is not None and self.fascia_us is not None):
            gap_us = self.fascia_us - self.dermis_us
            gap_mm = gap_us * speed / 2000.0
            self.gap_info.configure(text=f"Gap: {gap_mm:.2f} mm")
        else:
            self.gap_info.configure(text="Gap: --")


def main():
    app = BoundaryMarkerGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
