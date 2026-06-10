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
import re

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
# New format: drop=1850, keep=2050 (samples 1850~3900)
# Old format: drop=1200, keep=1250 (samples 1200~2450)
TRIM_START_NEW = 1850
TRIM_COUNT_NEW = 2050
DISPLAY_OFFSET_NEW = 18.50
TRIM_START_OLD = 1200
TRIM_COUNT_OLD = 1250
DISPLAY_OFFSET_OLD = 12.00
# defaults (new format)
TRIM_START = TRIM_START_NEW
TRIM_COUNT = TRIM_COUNT_NEW
DISPLAY_OFFSET_US = DISPLAY_OFFSET_NEW

# Filename pattern: 이름_날짜_시간_(번호)부위_성별.csv
FILENAME_RE = re.compile(
    r'^(?P<name>.+)_(?P<date>\d{8})_(?P<time>\d{6})_'
    r'\((?P<pos_num>\d+)\)(?P<position>.+?)_(?P<gender>[MF])\.csv$'
)

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

        # Same-position navigation (cross-patient, same pos_num)
        self.same_pos_list = []
        self.same_pos_index = -1

        # Previous file data for compare overlay (이전 파일, 파일 이동 시 자동 갱신)
        self.prev_adc_values = np.array([])
        self.prev_time_us = np.array([])
        self.prev_filepath = None
        self.show_compare = False

        # Fixed reference file for compare overlay (수동 지정, 파일 이동해도 유지)
        self.ref_adc_values = np.array([])
        self.ref_time_us = np.array([])
        self.ref_filepath = None
        self.show_ref_compare = False

        # Compare-mode shift: PREVIOUS signal is shifted for alignment
        self.display_shift_us = 0.0   # visual shift only, original data unchanged
        self._drag_start_xdata = None
        self._drag_start_shift = 0.0
        self._is_shift_drag = False
        # Cached line ref for fast shift update (previous signal only)
        self._prev_plot_line = None

        # Ref-mode shift: REF signal shift (persists across file navigation)
        self.ref_shift_us = 0.0
        self._ref_plot_line = None
        self._drag_target = None      # "prev" | "ref" | None
        self.ref_dir = None           # ref 디렉토리 (설정 시 인덱스 자동 매칭)
        self.ref_default_shift_mm = 0.439  # 기본 시프트(mm), ref 로드 시 자동 적용

        # Markers (time in μs, None = not set)
        self.start_us = None
        self.dermis_us = None
        self.fascia_us = None
        self.bone_us = None    # Optional bone echo marker
        self.json_path = None  # Track loaded/saved JSON path
        self.auto_epi_mm = None  # 자동 검출 표피두께 (analyze_epidermis.py와 동일 로직)

        # Marker mode: "start" → "dermis" → "fascia" → "bone"(optional)
        self.marker_mode = "start"

        # Envelope toggle (default: ON)
        self.show_envelope = True
        self.envelope_window = 8        # 이동평균 반폭 (±샘플)
        self.envelope_mode  = "mavg"    # "mavg" | "hilbert"

        # X-axis zoom (1.0 = full view, 0.5 = half width)
        self.xzoom = 1.0
        self.xzoom_end_us = None   # right edge of zoomed view (None = default)
        self.show_full_time = False  # 전체 타임 표시 (0μs~끝)
        self._saved_adc = None     # 전체보기 복원용 트림된 데이터
        self._saved_time = None
        self._triangle_drag = False

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

        ctk.CTkButton(file_frame, text="Data Root", width=90,
                      fg_color="#6f42c1", hover_color="#59359a",
                      command=self._set_data_root).grid(
            row=0, column=3, padx=5, pady=10)

        self.data_root_label = ctk.CTkLabel(file_frame, text="Root: (미설정)",
                                             font=ctk.CTkFont(size=11), text_color="gray")
        self.data_root_label.grid(row=0, column=4, padx=5, pady=10, sticky="w")
        if self.config.data_root:
            self.data_root_label.configure(
                text=f"Root: {self.config.data_root}",
                text_color="#b388ff")

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

        # Same-position navigation (cross-patient, same pos_num)
        self.same_pos_prev_btn = ctk.CTkButton(
            graph_frame, text="◀◀\n동부위", width=48, font=ctk.CTkFont(size=11),
            fg_color="#2d6a4f", hover_color="#1b4332",
            command=self._load_same_pos_prev, state="disabled")
        self.same_pos_prev_btn.grid(row=1, column=0, padx=(6, 2), pady=(2, 6), sticky="ew")

        self.same_pos_next_btn = ctk.CTkButton(
            graph_frame, text="동부위\n▶▶", width=48, font=ctk.CTkFont(size=11),
            fg_color="#2d6a4f", hover_color="#1b4332",
            command=self._load_same_pos_next, state="disabled")
        self.same_pos_next_btn.grid(row=1, column=2, padx=(2, 6), pady=(2, 6), sticky="ew")

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
            mode_frame, text="피하지방시작", variable=self.mode_var,
            value="dermis", command=self._on_mode_change,
            fg_color="#ff4444", hover_color="#cc3333")
        self.dermis_radio.grid(row=0, column=2, padx=8)

        self.fascia_radio = ctk.CTkRadioButton(
            mode_frame, text="Fascia", variable=self.mode_var,
            value="fascia", command=self._on_mode_change,
            fg_color="#4488ff", hover_color="#3366cc")
        self.fascia_radio.grid(row=0, column=3, padx=8)

        self.bone_radio = ctk.CTkRadioButton(
            mode_frame, text="뼈(선택)", variable=self.mode_var,
            value="bone", command=self._on_mode_change,
            fg_color="#ff8c00", hover_color="#cc7000")
        self.bone_radio.grid(row=0, column=4, padx=8)

        # T0 재설정 버튼 — 자동 검출 실패 시 수동으로 다시 찍기
        self.reset_t0_btn = ctk.CTkButton(
            mode_frame, text="↺T0", width=62, height=28,
            fg_color="#7a5500", hover_color="#553b00",
            command=self._reset_t0, state="disabled")
        self.reset_t0_btn.grid(row=0, column=5, padx=(14, 4))

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

        # 엔벨로프 방식 선택: MA / Hilbert 세그먼트 버튼
        self.env_mode_var = ctk.StringVar(value="mavg")
        self.btn_mavg = ctk.CTkSegmentedButton(
            toggle_frame,
            values=["MA", "Hilbert"],
            variable=self.env_mode_var,
            command=self._on_env_mode_changed,
            width=110, height=26,
        )
        self.btn_mavg.grid(row=0, column=2, padx=(10, 4))

        # MA 전용: -/+ 버튼 + 숫자 입력
        self.mavg_frame = ctk.CTkFrame(toggle_frame, fg_color="transparent")
        self.mavg_frame.grid(row=0, column=3, padx=(0, 0))
        self.btn_minus = ctk.CTkButton(self.mavg_frame, text="-", width=26, height=26,
                                       command=lambda: self._step_window(-1))
        self.btn_minus.grid(row=0, column=0, padx=(2, 0))
        self.window_entry = ctk.CTkEntry(self.mavg_frame, width=40, justify="center")
        self.window_entry.insert(0, str(self.envelope_window))
        self.window_entry.grid(row=0, column=1, padx=1)
        self.window_entry.bind("<Return>",   self._on_window_changed)
        self.window_entry.bind("<FocusOut>", self._on_window_changed)
        self.btn_plus = ctk.CTkButton(self.mavg_frame, text="+", width=26, height=26,
                                      command=lambda: self._step_window(+1))
        self.btn_plus.grid(row=0, column=2, padx=(0, 2))

        ctk.CTkLabel(toggle_frame, text="Dark:").grid(row=0, column=4, padx=(15, 5))
        self.theme_switch = ctk.CTkSwitch(toggle_frame, text="",
                                          command=self._toggle_theme,
                                          width=40)
        self.theme_switch.grid(row=0, column=5, padx=5)
        if self.config.theme == "dark":
            self.theme_switch.select()

        # Separator
        sep2 = ctk.CTkFrame(ctrl_frame, width=2, fg_color="gray")
        sep2.grid(row=0, column=3, padx=10, pady=8, sticky="ns")

        # Compare button (이전 파일과 비교)
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

        # Set Ref button (고정 참조 파일 선택)
        self.set_ref_btn = ctk.CTkButton(
            ctrl_frame, text="Set Ref", width=75,
            fg_color="#5a4a8a", hover_color="#3d3060",
            command=self._set_ref_file)
        self.set_ref_btn.grid(row=0, column=6, padx=(8, 2), pady=10)

        # Ref Compare toggle button (선택한 파일과 비교)
        self.ref_compare_btn = ctk.CTkButton(
            ctrl_frame, text="Ref", width=70,
            fg_color="#6c757d", hover_color="#545b62",
            command=self._toggle_ref_compare, state="disabled")
        self.ref_compare_btn.grid(row=0, column=7, padx=(2, 2), pady=10)

        # Ref shift mm entry (기본 시프트 값, ref 로드 시 자동 적용 + 드래그로 실시간 변경)
        ref_shift_frame = ctk.CTkFrame(ctrl_frame, fg_color="transparent")
        ref_shift_frame.grid(row=0, column=8, padx=(4, 6), pady=10)
        ctk.CTkLabel(ref_shift_frame, text="Ref Δ",
                     font=ctk.CTkFont(size=12)).grid(row=0, column=0, padx=(0, 2))
        self.ref_shift_entry = ctk.CTkEntry(ref_shift_frame, width=66, justify="center",
                                            font=ctk.CTkFont(size=12))
        self.ref_shift_entry.insert(0, f"{self.ref_default_shift_mm:+.3f}")
        self.ref_shift_entry.grid(row=0, column=1, padx=1)
        ctk.CTkLabel(ref_shift_frame, text="mm",
                     font=ctk.CTkFont(size=12)).grid(row=0, column=2, padx=(2, 0))
        self.ref_shift_entry.bind("<Return>",   self._on_ref_shift_entry_changed)
        self.ref_shift_entry.bind("<FocusOut>", self._on_ref_shift_entry_changed)

        # Separator
        sep3 = ctk.CTkFrame(ctrl_frame, width=2, fg_color="gray")
        sep3.grid(row=0, column=9, padx=5, pady=8, sticky="ns")

        # X-Zoom slider
        zoom_frame = ctk.CTkFrame(ctrl_frame, fg_color="transparent")
        zoom_frame.grid(row=0, column=10, padx=(8, 4), pady=10)

        ctk.CTkLabel(zoom_frame, text="X-Zoom:",
                     font=ctk.CTkFont(size=12)).grid(row=0, column=0, padx=(0, 4))

        self.xzoom_slider = ctk.CTkSlider(
            zoom_frame, from_=50, to=100, number_of_steps=50,
            width=130, command=self._on_xzoom_changed)
        self.xzoom_slider.set(100)  # 우측 = 전체보기 (기본값)
        self.xzoom_slider.grid(row=0, column=1, padx=4)

        self.xzoom_pct_label = ctk.CTkLabel(
            zoom_frame, text="100%", width=38,
            font=ctk.CTkFont(size=11))
        self.xzoom_pct_label.grid(row=0, column=2, padx=(0, 2))

        ctk.CTkButton(
            zoom_frame, text="1:1", width=36, height=26,
            fg_color="#495057", hover_color="#343a40",
            command=self._reset_xzoom).grid(row=0, column=3, padx=(0, 4))

        self.full_time_btn = ctk.CTkButton(
            zoom_frame, text="전체를 표시", height=26,
            fg_color="gray40", hover_color="#555555",
            command=self._toggle_full_time)
        self.full_time_btn.grid(row=1, column=0, columnspan=4,
                                padx=4, pady=(4, 0), sticky="ew")

        ctrl_frame.grid_columnconfigure(9, weight=1)

        # Row 1: marker info (always visible, spans full width)
        info_frame = ctk.CTkFrame(ctrl_frame, fg_color="transparent")
        info_frame.grid(row=1, column=0, columnspan=11, padx=10, pady=(0, 6), sticky="ew")

        self.start_info = ctk.CTkLabel(
            info_frame, text="Start: --",
            font=ctk.CTkFont(size=13), text_color="#ffaa00")
        self.start_info.grid(row=0, column=0, padx=10, sticky="w")

        self.dermis_info = ctk.CTkLabel(
            info_frame, text="피하지방시작: --",
            font=ctk.CTkFont(size=13), text_color="#ff6666")
        self.dermis_info.grid(row=0, column=1, padx=10, sticky="w")

        self.fascia_info = ctk.CTkLabel(
            info_frame, text="Fascia: --",
            font=ctk.CTkFont(size=13), text_color="#6699ff")
        self.fascia_info.grid(row=0, column=2, padx=10, sticky="w")

        self.bone_info = ctk.CTkLabel(
            info_frame, text="뼈: --",
            font=ctk.CTkFont(size=13), text_color="#ff8c00")
        self.bone_info.grid(row=0, column=3, padx=10, sticky="w")

        self.gap_info = ctk.CTkLabel(
            info_frame, text="Gap: --",
            font=ctk.CTkFont(size=13), text_color="#aaaaaa")
        self.gap_info.grid(row=0, column=4, padx=10, sticky="w")

        # 표피두께: 자동 검출 (analyze_epidermis.py 동일 로직)
        self.epi_info = ctk.CTkLabel(
            info_frame, text="표피두께: --",
            font=ctk.CTkFont(size=15, weight="bold"), text_color="#00e676")
        self.epi_info.grid(row=0, column=5, padx=20, sticky="w")

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

    def _set_data_root(self):
        """Set root data directory for cross-patient navigation"""
        initial_dir = self.config.data_root or self.config.last_open_dir or os.getcwd()
        folder = filedialog.askdirectory(initialdir=initial_dir,
                                         title="최상위 데이터 폴더 선택 (예: usdata/)")
        if folder:
            self.config.update(data_root=folder)
            self.data_root_label.configure(text=f"Root: {folder}", text_color="#b388ff")
            self.status_label.configure(text=f"Data root: {folder}")
            # Rebuild same-pos list with new root
            if self.filepath:
                self._build_same_pos_list()

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
        self.xzoom_end_us = None  # 줌 위치 초기화
        self.show_full_time = False
        self._saved_adc = None
        self._saved_time = None
        self.full_time_btn.configure(fg_color="gray40", hover_color="#555555")

        # Clear markers
        self.start_us = None
        self.dermis_us = None
        self.fascia_us = None
        self.bone_us = None
        self.mode_var.set("start")
        self.marker_mode = "start"

        # 자동 검출: T0(표피 시작)
        auto_t0, auto_epi_mm = self._auto_detect_t0(adc_arr, time_us)
        self.auto_epi_mm = auto_epi_mm

        # Check for existing JSON
        self.json_path = self._get_json_path(filepath)
        if os.path.exists(self.json_path):
            self._load_existing_json(self.json_path)
        else:
            self.json_path = None
            # JSON 없을 때만 자동 T0를 초기 시작점으로 설정
            if auto_t0 is not None:
                self.start_us = auto_t0
                self.mode_var.set("dermis")
                self.marker_mode = "dermis"

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
        VREF = 3.0   # signals up to ~2.9V; use 3.0V to avoid clipping
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

        # 파일 크기에 따라 포맷 자동 감지
        n = len(adc_arr)
        if n > TRIM_COUNT_NEW:
            # 원시(raw) 파일: 새 포맷으로 트림 (drop=1850, keep=2050)
            end = TRIM_START_NEW + TRIM_COUNT_NEW
            adc_arr = adc_arr[TRIM_START_NEW:end] if n >= end else adc_arr[TRIM_START_NEW:]
            offset_us = DISPLAY_OFFSET_NEW
        elif n > TRIM_COUNT_OLD:
            # 새 포맷 사전트림 파일 (~2050 샘플, drop=1850)
            offset_us = DISPLAY_OFFSET_NEW
        else:
            # 구 포맷 사전트림 파일 (~1250 샘플, drop=1200)
            offset_us = DISPLAY_OFFSET_OLD

        time_us = np.arange(len(adc_arr)) * SAMPLE_INTERVAL_NS / 1000.0 + offset_us
        return adc_arr, time_us

    def _load_csv_raw(self, filepath):
        """Load CSV without trimming — returns (adc_arr, time_us) starting from sample 0 (0μs).

        Used only by 전체를 표시 to show the full ~3900-sample raw capture including
        the pre-signal region (0~18.5μs).  Old oscilloscope format is returned as-is
        because its time values are already absolute.
        """
        with open(filepath, 'r') as f:
            lines = f.readlines()

        is_old = any('x-axis' in l or 'second,Volt' in l for l in lines[:5])
        if is_old:
            return self._load_csv_old(lines)

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
            raise ValueError("No valid ADC values found")

        adc_arr = np.array(adc_values, dtype=np.int32)
        time_us = np.arange(len(adc_arr)) * SAMPLE_INTERVAL_NS / 1000.0  # 0μs~
        return adc_arr, time_us

    def _get_json_path(self, csv_path):
        """Get JSON output path — always same directory as CSV"""
        csv_name = os.path.basename(csv_path)
        base_name = os.path.splitext(csv_name)[0] + "_positions.json"
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
                    if name in ('Dermis', '피하지방시작'):
                        self.dermis_us = time_val
                    elif name == 'Fascia':
                        self.fascia_us = time_val

            # Load bone echo (optional — auto or manual)
            bone_data = data.get('bone_echo')
            if bone_data and bone_data.get('time_us') is not None:
                self.bone_us = bone_data['time_us']

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
                self.mode_var.set("bone")
                self.marker_mode = "bone"

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
        self.show_full_time = False
        self._saved_adc = None
        self._saved_time = None
        self.full_time_btn.configure(fg_color="gray40", hover_color="#555555")

        # Load markers from JSON
        self.start_us = data.get('start_point_us')
        self.dermis_us = None
        self.fascia_us = None
        self.bone_us = None
        for pos in data.get('positions', []):
            name = pos.get('position_name', '')
            time_val = pos.get('time_us')
            if time_val is not None:
                if name in ('Dermis', '피하지방시작'):
                    self.dermis_us = time_val
                elif name == 'Fascia':
                    self.fascia_us = time_val

        # Load bone echo (optional — auto or manual)
        bone_data = data.get('bone_echo')
        if bone_data and bone_data.get('time_us') is not None:
            self.bone_us = bone_data['time_us']

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
            self.mode_var.set("bone")
            self.marker_mode = "bone"

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
        elif self.marker_mode == "dermis":
            self.status_label.configure(
                text=f"Loaded: {filename} - T0 자동 설정됨, 클릭하여 피하지방시작 마킹")
        else:
            self.status_label.configure(
                text=f"Loaded: {filename} - Click to mark Start point")

        self._build_file_list()
        self._build_same_pos_list()
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

    def _build_same_pos_list(self):
        """Build list of files with same position number (cross-patient navigation)"""
        self.same_pos_list = []
        self.same_pos_index = -1
        if not self.filepath:
            self._update_same_pos_nav_buttons()
            return

        basename = os.path.basename(self.filepath)
        m = FILENAME_RE.match(basename)
        if not m:
            self._update_same_pos_nav_buttons()
            return

        pos_num = int(m.group('pos_num'))

        # Use configured data_root; fall back to 3-levels-up heuristic
        if self.config.data_root and os.path.isdir(self.config.data_root):
            search_root = os.path.abspath(self.config.data_root)
        else:
            search_root = os.path.abspath(os.path.dirname(self.filepath))
            for _ in range(3):
                parent = os.path.dirname(search_root)
                if parent == search_root:
                    break
                search_root = parent

        # Collect all CSVs with the same position number
        same_pos = []
        for dirpath, _, filenames in os.walk(search_root):
            for filename in filenames:
                if not filename.lower().endswith('.csv'):
                    continue
                m2 = FILENAME_RE.match(filename)
                if m2 and int(m2.group('pos_num')) == pos_num:
                    same_pos.append(os.path.join(dirpath, filename))

        same_pos.sort()
        self.same_pos_list = same_pos
        abs_path = os.path.abspath(self.filepath)
        try:
            self.same_pos_index = same_pos.index(abs_path)
        except ValueError:
            self.same_pos_index = -1

        self._update_same_pos_nav_buttons()

    def _update_same_pos_nav_buttons(self):
        """Enable/disable same-position navigation buttons"""
        if not hasattr(self, 'same_pos_prev_btn'):
            return
        if not self.same_pos_list or self.same_pos_index < 0:
            self.same_pos_prev_btn.configure(state="disabled", text="◀◀\n동부위")
            self.same_pos_next_btn.configure(state="disabled", text="동부위\n▶▶")
            return
        total = len(self.same_pos_list)
        idx = self.same_pos_index
        self.same_pos_prev_btn.configure(
            state="normal" if idx > 0 else "disabled",
            text=f"◀◀\n동부위\n[{idx}/{total}]" if idx > 0 else "◀◀\n동부위")
        self.same_pos_next_btn.configure(
            state="normal" if idx < total - 1 else "disabled",
            text=f"동부위\n▶▶\n[{idx + 2}/{total}]" if idx < total - 1 else "동부위\n▶▶")

    def _load_same_pos_prev(self):
        """Load previous patient's file with same position number"""
        if self.same_pos_index <= 0:
            return
        self._load_same_pos_at_index(self.same_pos_index - 1)

    def _load_same_pos_next(self):
        """Load next patient's file with same position number"""
        if self.same_pos_index >= len(self.same_pos_list) - 1:
            return
        self._load_same_pos_at_index(self.same_pos_index + 1)

    def _load_same_pos_at_index(self, idx):
        """Load a same-position file at given index"""
        filepath = self.same_pos_list[idx]
        try:
            adc_arr, time_us = self._load_csv(filepath)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV:\n{e}")
            return

        if len(self.adc_values) > 0:
            self.prev_adc_values = self.adc_values.copy()
            self.prev_time_us = self.time_us.copy()
            self.prev_filepath = self.filepath
            self.compare_btn.configure(state="normal")

        self.filepath = filepath
        self.adc_values = adc_arr
        self.time_us = time_us
        self.display_shift_us = 0.0
        self.reset_shift_btn.configure(state="disabled")
        self.show_full_time = False
        self._saved_adc = None
        self._saved_time = None
        self.full_time_btn.configure(fg_color="gray40", hover_color="#555555")

        self.start_us = None
        self.dermis_us = None
        self.fascia_us = None
        self.bone_us = None
        self.mode_var.set("start")
        self.marker_mode = "start"

        auto_t0, auto_epi_mm = self._auto_detect_t0(adc_arr, time_us)
        self.auto_epi_mm = auto_epi_mm

        self.json_path = self._get_json_path(filepath)
        if os.path.exists(self.json_path):
            self._load_existing_json(self.json_path)
        else:
            self.json_path = None
            if auto_t0 is not None:
                self.start_us = auto_t0
                self.mode_var.set("dermis")
                self.marker_mode = "dermis"

        self.same_pos_index = idx
        self.config.update(last_open_dir=os.path.dirname(filepath))

        filename = os.path.basename(filepath)
        self.file_label.configure(text=filename)
        self.save_btn.configure(state="normal")
        self.clear_btn.configure(state="normal")

        s_us = self.time_us[0] if len(self.time_us) > 0 else 0
        e_us = self.time_us[-1] if len(self.time_us) > 0 else 0
        self.data_info.configure(
            text=f"Samples: {len(self.adc_values)} | "
                 f"{s_us:.1f}~{e_us:.1f} us | "
                 f"Min: {np.min(self.adc_values)} | Max: {np.max(self.adc_values)}")

        total = len(self.same_pos_list)
        m = FILENAME_RE.match(filename)
        pos_info = f" #pos{m.group('pos_num')}" if m else ""
        mode_hint = "클릭하여 피하지방시작 마킹" if self.marker_mode == "dermis" else "Click to mark Start"
        self.status_label.configure(
            text=f"[동부위 {idx + 1}/{total}]{pos_info} {filename} - {mode_hint}")

        self._build_file_list()
        self._update_nav_buttons()
        self._update_same_pos_nav_buttons()
        self._update_graph()
        self._update_marker_info()
        if self.ref_dir:
            self._auto_load_ref()

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
        self.show_full_time = False
        self._saved_adc = None
        self._saved_time = None
        self.full_time_btn.configure(fg_color="gray40", hover_color="#555555")

        # Clear markers
        self.start_us = None
        self.dermis_us = None
        self.fascia_us = None
        self.bone_us = None
        self.mode_var.set("start")
        self.marker_mode = "start"

        # 자동 검출: T0(표피 시작)
        auto_t0, auto_epi_mm = self._auto_detect_t0(adc_arr, time_us)
        self.auto_epi_mm = auto_epi_mm

        # Check for existing JSON
        self.json_path = self._get_json_path(filepath)
        if os.path.exists(self.json_path):
            self._load_existing_json(self.json_path)
        else:
            self.json_path = None
            # JSON 없을 때만 자동 T0를 초기 시작점으로 설정
            if auto_t0 is not None:
                self.start_us = auto_t0
                self.mode_var.set("dermis")
                self.marker_mode = "dermis"

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
        mode_hint = "클릭하여 피하지방시작 마킹" if self.marker_mode == "dermis" else "Click to mark Start point"
        self.status_label.configure(
            text=f"[{idx + 1}/{total}] {filename} - {mode_hint}")
        self._build_same_pos_list()
        self._update_nav_buttons()
        self._update_graph()
        self._update_marker_info()
        # ref_dir 설정 시 인덱스에 맞는 ref 자동 로드
        if self.ref_dir:
            self._auto_load_ref()

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

    def _set_ref_file(self):
        """참조 디렉토리 선택 — 현재 파일 인덱스에 맞는 ref CSV 자동 로드"""
        init_dir = os.path.dirname(self.filepath) if self.filepath else os.getcwd()
        dir_path = filedialog.askdirectory(
            title="참조 파일 디렉토리 선택 (파일명의 (숫자) 인덱스로 자동 매칭)",
            initialdir=init_dir
        )
        if not dir_path:
            return

        self.ref_dir = dir_path
        self.set_ref_btn.configure(fg_color="#7a3fc0", hover_color="#5a2ea0",
                                   text="Ref Dir ✓")
        self._auto_load_ref()

    def _find_ref_in_dir(self, pos_num):
        """ref_dir에서 인덱스 pos_num 에 해당하는 CSV 파일 탐색.
        파일명 내 첫 번째 (숫자) 를 정수로 비교하여 선택.
        """
        if not self.ref_dir or not os.path.isdir(self.ref_dir):
            return None
        candidates = []
        for f in os.listdir(self.ref_dir):
            if not f.lower().endswith('.csv'):
                continue
            m = re.search(r'\((\d+)\)', f)
            if m and int(m.group(1)) == pos_num:
                candidates.append(os.path.join(self.ref_dir, f))
        if not candidates:
            return None
        candidates.sort()
        return candidates[0]

    def _auto_load_ref(self):
        """현재 파일 인덱스에 맞는 ref 파일을 ref_dir에서 찾아 로드"""
        if not self.ref_dir or not self.filepath:
            return

        basename = os.path.basename(self.filepath)
        m = re.search(r'\((\d+)\)', basename)
        if not m:
            self.status_label.configure(
                text=f"인덱스 (숫자) 없음: {basename} — ref 자동 로드 불가")
            return

        pos_num = int(m.group(1))
        ref_path = self._find_ref_in_dir(pos_num)

        if not ref_path:
            # 매칭 파일 없음 → ref 클리어
            self.ref_adc_values = np.array([])
            self.ref_time_us = np.array([])
            self.ref_filepath = None
            self.show_ref_compare = False
            self.ref_compare_btn.configure(state="disabled",
                                           fg_color="#6c757d", text="Ref")
            self.status_label.configure(
                text=f"Ref Dir에서 인덱스 ({pos_num}) 파일 없음")
            self._update_graph()
            return

        try:
            adc, time_us = self._load_csv(ref_path)
        except Exception as e:
            messagebox.showerror("Error", f"Ref 파일 로드 실패:\n{e}")
            return

        self.ref_adc_values = adc
        self.ref_time_us = time_us
        self.ref_filepath = ref_path
        # 기본 시프트 적용
        self.ref_shift_us = self.ref_default_shift_mm * 2000.0 / self.config.speed_of_sound
        self._update_ref_shift_entry()
        # 사용자가 수동으로 Ref를 OFF 했다면 그 상태 유지, 아니면 ON으로 표시
        if self.show_ref_compare:
            self.ref_compare_btn.configure(
                state="normal", fg_color="#9b59b6", hover_color="#7d3c98",
                text="Ref ●")
        else:
            # 파일은 로드됐지만 표시는 OFF — 버튼은 활성화만
            self.ref_compare_btn.configure(
                state="normal", fg_color="#6c757d", hover_color="#545b62",
                text="Ref")

        fname = os.path.basename(ref_path)
        curr_start = self.time_us[0] if len(self.time_us) > 0 else 0
        hint = ""
        if len(time_us) > 0 and time_us[0] < curr_start - 1.0:
            hint = " (일부 구간 숨김 — '전체를 표시' 로 전체 비교)"
        self.status_label.configure(text=f"Ref ({pos_num}): {fname}{hint}")
        self._update_graph()

    def _update_ref_shift_entry(self):
        """ref_shift_us 값을 mm 엔트리 필드에 반영"""
        mm = self.ref_shift_us * self.config.speed_of_sound / 2000.0
        self.ref_shift_entry.delete(0, "end")
        self.ref_shift_entry.insert(0, f"{mm:+.3f}")

    def _on_ref_shift_entry_changed(self, event=None):
        """mm 엔트리 변경 → ref_shift_us 업데이트 및 그래프 재그리기"""
        try:
            mm = float(self.ref_shift_entry.get())
        except ValueError:
            self._update_ref_shift_entry()
            return
        self.ref_default_shift_mm = mm
        self.ref_shift_us = mm * 2000.0 / self.config.speed_of_sound
        self.ref_shift_entry.delete(0, "end")
        self.ref_shift_entry.insert(0, f"{mm:+.3f}")
        if len(self.ref_adc_values) > 0:
            self._update_graph()

    def _toggle_ref_compare(self):
        """고정 참조 파일 오버레이 토글"""
        if len(self.ref_adc_values) == 0:
            return
        self.show_ref_compare = not self.show_ref_compare
        if self.show_ref_compare:
            self.ref_compare_btn.configure(fg_color="#9b59b6", hover_color="#7d3c98",
                                           text="Ref ●")
        else:
            self.ref_compare_btn.configure(fg_color="#6c757d", hover_color="#545b62",
                                           text="Ref")
        self._update_graph()

    def _reset_shift(self):
        """Reset display shift to zero (re-align signals)"""
        self.display_shift_us = 0.0
        self.reset_shift_btn.configure(state="disabled")
        self._update_graph()
        self.status_label.configure(text="Shift reset - signals aligned")

    def _fast_ref_shift_redraw(self):
        """Ref(보라색) 신호 라인만 시프트 업데이트 — 마커/현재 신호는 고정."""
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        if self._ref_plot_line is not None:
            self._ref_plot_line.set_xdata(self.ref_time_us + self.ref_shift_us)

        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)

        is_dark = ctk.get_appearance_mode() == "Dark"
        text_color = 'white' if is_dark else 'black'
        base = "Ultrasound Envelope Signal" if self.show_envelope else "Ultrasound RF Signal"
        title = base
        if self.show_compare and abs(self.display_shift_us) > 0.001:
            mm = self.display_shift_us * self.config.speed_of_sound / 2000.0
            title += f"  [prev {self.display_shift_us:+.3f} μs / {mm:+.3f} mm]"
        if abs(self.ref_shift_us) > 0.001:
            mm = self.ref_shift_us * self.config.speed_of_sound / 2000.0
            title += f"  [ref {self.ref_shift_us:+.3f} μs / {mm:+.3f} mm]"
        self.ax.set_title(title, color=text_color)
        self._update_ref_shift_entry()
        self.canvas.draw_idle()

    def _reset_t0(self):
        """표피시작(T0) 초기화 — 수동으로 다시 마킹"""
        self.start_us = None
        self.mode_var.set("start")
        self.marker_mode = "start"
        self.cursor_info.configure(text="")
        self.status_label.configure(
            text="T0 초기화됨 — 그래프를 클릭하여 표피 시작(T0) 수동 마킹")
        self._update_graph()
        self._update_marker_info()

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

        # bone echo (optional)
        bone_echo = None
        if self.bone_us is not None:
            bone_rel_us = self.bone_us - self.start_us
            bone_mm = bone_rel_us * speed / 2000.0
            bone_echo = {
                "time_us":        round(self.bone_us, 4),
                "rel_us":         round(bone_rel_us, 4),
                "depth_mm":       round(bone_mm, 4),
                "manually_marked": True
            }

        data = {
            "source_file": os.path.basename(self.filepath),
            "start_point_us": round(self.start_us, 4),
            "num_positions": 2,
            "speed_of_sound": speed,
            "sample_interval_ns": SAMPLE_INTERVAL_NS,
            "num_samples": len(self.adc_values),
            "bone_echo": bone_echo,
            "positions": [
                {
                    "position_number": 1,
                    "position_name": "피하지방시작",
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

        # Reset cached line refs
        self._prev_plot_line = None
        self._ref_plot_line = None

        # Effective shift (only applied to PREVIOUS signal when compare is active)
        shift = self.display_shift_us if self.show_compare else 0.0

        if len(self.adc_values) > 0:
            # Compare overlay: PREVIOUS file in red, shifted for alignment
            if self.show_compare and len(self.prev_adc_values) > 0:
                prev_name = os.path.basename(self.prev_filepath) if self.prev_filepath else "Prev"
                prev_xdata = self.prev_time_us + shift
                if self.show_envelope:
                    prev_env = self._compute_envelope(self.prev_adc_values, self.envelope_window)
                    self._prev_plot_line = self.ax.plot(
                        prev_xdata, prev_env,
                        color='#ff4444', linewidth=0.8, alpha=0.6,
                        label=f"Prev: {prev_name}", linestyle='--')[0]
                else:
                    self._prev_plot_line = self.ax.plot(
                        prev_xdata, self.prev_adc_values,
                        color='#ff4444', linewidth=0.8, alpha=0.6,
                        label=f"Prev: {prev_name}", linestyle='--')[0]

            # Ref overlay: fixed reference file in purple (shifted by ref_shift_us)
            if self.show_ref_compare and len(self.ref_adc_values) > 0:
                ref_name = os.path.basename(self.ref_filepath) if self.ref_filepath else "Ref"
                ref_xdata = self.ref_time_us + self.ref_shift_us
                if self.show_envelope:
                    # ref 신호의 샘플 간격을 dev 장비(10ns)와 비교해 MA 창 크기 스케일
                    # → 시간 기준 동등한 스무딩 (e.g. 2.25ns 샘플 → window ×4.4배)
                    ref_window = self.envelope_window
                    if len(self.ref_time_us) > 1:
                        ref_dt_us = ((self.ref_time_us[-1] - self.ref_time_us[0])
                                     / (len(self.ref_time_us) - 1))
                        dev_dt_us = SAMPLE_INTERVAL_NS / 1000.0  # 0.01μs
                        if ref_dt_us > 0:
                            ref_window = max(self.envelope_window,
                                            int(self.envelope_window * dev_dt_us / ref_dt_us))
                    ref_env = self._compute_envelope(self.ref_adc_values,
                                                     ref_window,
                                                     auto_baseline=True)
                    self._ref_plot_line = self.ax.plot(
                        ref_xdata, ref_env,
                        color='#bb77ff', linewidth=0.8, alpha=0.65,
                        label=f"Ref: {ref_name}", linestyle='-.')[0]
                else:
                    self._ref_plot_line = self.ax.plot(
                        ref_xdata, self.ref_adc_values,
                        color='#bb77ff', linewidth=0.8, alpha=0.65,
                        label=f"Ref: {ref_name}", linestyle='-.')[0]

            # Current file: FIXED (no shift) — markers placed here directly
            if self.show_envelope:
                envelope = self._compute_envelope(self.adc_values, self.envelope_window)
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

            # Show shift amounts in title
            if self.show_compare and abs(shift) > 0.001:
                mm = shift * self.config.speed_of_sound / 2000.0
                title += f"  [prev {shift:+.3f} μs / {mm:+.3f} mm]"
            if self.show_ref_compare and abs(self.ref_shift_us) > 0.001:
                mm = self.ref_shift_us * self.config.speed_of_sound / 2000.0
                title += f"  [ref {self.ref_shift_us:+.3f} μs / {mm:+.3f} mm]"
            self.ax.set_ylabel(ylabel, color=text_color)
            self.ax.set_title(title, color=text_color)

            # Draw markers on CURRENT signal (no shift needed)
            has_legend = ((self.show_compare and len(self.prev_adc_values) > 0) or
                          (self.show_ref_compare and len(self.ref_adc_values) > 0))
            if self.start_us is not None:
                self.ax.axvline(x=self.start_us, color='#ffaa00', linewidth=2,
                               linestyle='-', label='표피시작(T0)', alpha=0.9)
                has_legend = True
            if self.dermis_us is not None:
                self.ax.axvline(x=self.dermis_us, color='#ff4444', linewidth=2,
                               linestyle='--', label='피하지방시작', alpha=0.9)
                has_legend = True
            if self.fascia_us is not None:
                self.ax.axvline(x=self.fascia_us, color='#4488ff', linewidth=2,
                               linestyle='--', label='근막', alpha=0.9)
                has_legend = True
            if self.bone_us is not None:
                self.ax.axvline(x=self.bone_us, color='#ff8c00', linewidth=2,
                               linestyle=':', label='뼈(Bone)', alpha=0.9)
                has_legend = True

            # Marker callout annotations (blended transform: x=data, y=axes fraction)
            _speed = self.config.speed_of_sound
            _trans = self.ax.get_xaxis_transform()
            if self.dermis_us is not None and self.start_us is not None:
                d_us = self.dermis_us - self.start_us
                dmm = d_us * _speed / 2000.0
                self.ax.annotate(
                    f"피하지방시작\n{d_us:.2f} μs\n{dmm:.2f} mm",
                    xy=(self.dermis_us, 0.28), xytext=(self.dermis_us, 0.68),
                    xycoords=_trans, textcoords=_trans,
                    arrowprops=dict(arrowstyle='->', color='#ff6666', lw=1.5),
                    ha='center', va='center', fontsize=9, color='white',
                    bbox=dict(boxstyle='round,pad=0.5',
                              facecolor='#881111', edgecolor='#ff4444', alpha=0.88))
            if self.fascia_us is not None and self.start_us is not None:
                f_us = self.fascia_us - self.start_us
                fmm = f_us * _speed / 2000.0
                self.ax.annotate(
                    f"Fascia\n{f_us:.2f} μs\n{fmm:.2f} mm",
                    xy=(self.fascia_us, 0.28), xytext=(self.fascia_us, 0.68),
                    xycoords=_trans, textcoords=_trans,
                    arrowprops=dict(arrowstyle='->', color='#6699ff', lw=1.5),
                    ha='center', va='center', fontsize=9, color='white',
                    bbox=dict(boxstyle='round,pad=0.5',
                              facecolor='#113388', edgecolor='#4488ff', alpha=0.88))
            if self.bone_us is not None and self.start_us is not None:
                b_us = self.bone_us - self.start_us
                bmm = b_us * _speed / 2000.0
                self.ax.annotate(
                    f"뼈\n{b_us:.2f} μs\n{bmm:.2f} mm",
                    xy=(self.bone_us, 0.10), xytext=(self.bone_us, 0.48),
                    xycoords=_trans, textcoords=_trans,
                    arrowprops=dict(arrowstyle='->', color='#ff8c00', lw=1.5),
                    ha='center', va='center', fontsize=9, color='white',
                    bbox=dict(boxstyle='round,pad=0.5',
                              facecolor='#7a3c00', edgecolor='#ff8c00', alpha=0.88))

            if has_legend:
                self.ax.legend(loc='upper right', fontsize=8,
                              facecolor=bg_color, edgecolor=grid_color,
                              labelcolor=text_color)

            # Fine x-axis ticks
            self.ax.xaxis.set_major_locator(MultipleLocator(1.0))
            self.ax.xaxis.set_minor_locator(MultipleLocator(0.5))
            self.ax.tick_params(axis='x', which='minor', length=3)

            # Apply X-axis zoom (Y-axis untouched)
            if self.show_full_time:
                self.ax.set_xlim(float(self.time_us[0]), float(self.time_us[-1]))
            else:
                xlims = self._get_xzoom_limits()
                if xlims:
                    self.ax.set_xlim(*xlims)
        else:
            self.ax.set_xlim(DISPLAY_OFFSET_NEW,
                            DISPLAY_OFFSET_NEW + (TRIM_COUNT_NEW - 1) * 0.01)
            self.ax.set_ylim(0, 256)
            self.ax.set_ylabel("ADC Value", color=text_color)
            self.ax.set_title("Ultrasound Signal (No Data)", color=text_color)

        self.ax.set_xlabel("Time (us)", color=text_color)
        self.cursor_line = None
        self.cursor_annot = None
        self.canvas.draw()

    # ---- X-axis zoom ----

    def _get_xzoom_limits(self):
        """현재 xzoom 설정에 따른 X축 표시 범위 반환. 데이터 없으면 None."""
        if len(self.adc_values) == 0:
            return None
        full_min = float(self.time_us[0])
        full_max = float(self.time_us[-1])
        if self.xzoom <= 1.0:
            return full_min, full_max
        # zoom out: xlim을 데이터 범위보다 넓게 설정 → 신호가 압축되어 보임
        full_w = full_max - full_min
        total_w = full_w * self.xzoom
        return full_min, full_min + total_w

    def _on_xzoom_changed(self, value):
        """슬라이더 콜백 — value: 50(좌/줌아웃 50%)~100(우/전체보기 100%)
        슬라이더 우측 = 1.0x 전체보기, 좌측 = xlim 2배 확장(신호 50% 압축)"""
        self.xzoom = 100.0 / value          # 100→1.0(전체), 50→2.0(50% 축소)
        self.xzoom_end_us = None
        self.xzoom_pct_label.configure(text=f"{int(value)}%")
        self.show_full_time = False
        self.full_time_btn.configure(fg_color="gray40", hover_color="#555555")
        self._update_graph()

    def _reset_xzoom(self):
        """X-zoom 100%(전체 보기)로 초기화"""
        self.xzoom = 1.0
        self.xzoom_end_us = None
        self.xzoom_slider.set(100)  # 우측 = 전체보기
        self.xzoom_pct_label.configure(text="100%")
        self.show_full_time = False
        self.full_time_btn.configure(fg_color="gray40", hover_color="#555555")
        self._update_graph()

    def _toggle_full_time(self):
        """전체 타임 표시 토글 — 활성: 원시 파일 재로드(0μs~끝), 비활성: 트림된 데이터 복원"""
        self.show_full_time = not self.show_full_time
        if self.show_full_time:
            self.full_time_btn.configure(fg_color="#1a5c38", hover_color="#145230")
            self.xzoom = 1.0
            self.xzoom_end_us = None
            self.xzoom_slider.set(100)
            self.xzoom_pct_label.configure(text="100%")
            # 원시 파일을 트리밍 없이 재로드 → 0~18.5μs 구간도 표시
            if self.filepath and self._saved_adc is None:
                try:
                    raw_adc, raw_time = self._load_csv_raw(self.filepath)
                    if raw_adc is not None and len(raw_adc) > len(self.adc_values):
                        self._saved_adc = self.adc_values
                        self._saved_time = self.time_us
                        self.adc_values = raw_adc
                        self.time_us = raw_time
                except Exception:
                    pass
        else:
            self.full_time_btn.configure(fg_color="gray40", hover_color="#555555")
            # 트림된 원본 데이터 복원
            if self._saved_adc is not None:
                self.adc_values = self._saved_adc
                self.time_us = self._saved_time
                self._saved_adc = None
                self._saved_time = None
        self._update_graph()

    def _auto_detect_t0(self, adc, time_us):
        """표피 시작(T0) 자동 검출.

        Returns:
            (t0_us, epi_mm) — 검출 실패 시 (None, None)
            analyze_epidermis.py와 동일 로직: 첫 클러스터 시작=T0, 끝=표피경계
        """
        THRESHOLD      = 100
        MIN_CLUSTER_W  = 10

        s   = adc.astype(float) - 128.0
        env = np.abs(hilbert(s))

        # 클러스터 검출 (envelope >= THRESHOLD 구간)
        clusters, in_c, cs = [], False, 0
        for i, v in enumerate(env >= THRESHOLD):
            if v and not in_c:
                in_c, cs = True, i
            elif not v and in_c:
                in_c = False
                if i - 1 - cs >= MIN_CLUSTER_W:
                    clusters.append((cs, i - 1))
        if in_c and len(env) - 1 - cs >= MIN_CLUSTER_W:
            clusters.append((cs, len(env) - 1))

        if not clusters:
            return None, None

        # T0 = 첫 번째 클러스터 시작, 표피끝 = 첫 번째 클러스터 끝
        t0_us_val  = float(time_us[clusters[0][0]])
        epi_end_us = float(time_us[clusters[0][1]])
        epi_mm     = (epi_end_us - t0_us_val) * self.config.speed_of_sound / 2000.0
        return t0_us_val, epi_mm

    def _compute_envelope(self, signal, window: int = 3, auto_baseline: bool = False):
        """엔벨로프 계산
        MA     : |ADC-baseline| → 이동평균(±window)
                 auto_baseline=True: 신호 평균을 기준으로 자동 설정 (단극성 펄서 신호용)
                 auto_baseline=False: 고정 기준 128 (개발 장비 양극성 신호 기본)
        Hilbert: |hilbert(ADC-mean)| (항상 자동 기준)
        """
        if self.envelope_mode == "hilbert":
            s = signal.astype(float) - np.mean(signal)
            return np.abs(hilbert(s))
        # MA
        baseline = float(np.mean(signal)) if auto_baseline else 128.0
        rectified = np.abs(signal.astype(float) - baseline)
        kernel = np.ones(2 * window + 1) / (2 * window + 1)
        return np.convolve(rectified, kernel, mode='same')

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
        title = base
        if abs(shift) > 0.001:
            mm = shift * self.config.speed_of_sound / 2000.0
            title += f"  [prev {shift:+.3f} μs / {mm:+.3f} mm]"
        if self.show_ref_compare and abs(self.ref_shift_us) > 0.001:
            mm = self.ref_shift_us * self.config.speed_of_sound / 2000.0
            title += f"  [ref {self.ref_shift_us:+.3f} μs / {mm:+.3f} mm]"
        self.ax.set_title(title, color=text_color)
        self.canvas.draw_idle()

    # ---- Mouse handling ----

    def _on_press(self, event):
        """Record press position to distinguish click vs drag"""
        self._triangle_drag = False

        # X축 끝 삼각형 드래그 (zoom out 모드에서는 사용 안 함)
        if False and event.button == 1 and self.xzoom < 1.0 and len(self.adc_values) > 0:
            xlims = self._get_xzoom_limits()
            if xlims:
                try:
                    x_end_disp, _ = self.ax.transData.transform((xlims[1], 0))
                    ax_bottom = self.ax.bbox.y0
                    # 삼각형 위치: x축 끝 ±20px, 축 하단 ±40px
                    if abs(event.x - x_end_disp) < 20 and ax_bottom - 50 < event.y < ax_bottom + 10:
                        self._triangle_drag = True
                        self._press_pos = None
                        self._is_shift_drag = False
                        self._drag_start_xdata = None
                        return
                except Exception:
                    pass

        if event.inaxes == self.ax and event.button == 1:
            self._press_pos = (event.x, event.y)
            self._is_shift_drag = False
            # Prepare for shift drag: prefer compare(prev) if both active, else ref
            if (self.show_compare or self.show_ref_compare) and event.xdata is not None:
                self._drag_start_xdata = event.xdata
                if self.show_compare:
                    self._drag_target = "prev"
                    self._drag_start_shift = self.display_shift_us
                else:
                    self._drag_target = "ref"
                    self._drag_start_shift = self.ref_shift_us
            else:
                self._drag_start_xdata = None
                self._drag_target = None
        else:
            self._press_pos = None
            self._drag_start_xdata = None
            self._drag_target = None
            self._is_shift_drag = False

    def _on_release(self, event):
        """Place marker on click (not drag). Works even when zoomed."""
        # 삼각형 드래그 종료
        was_triangle_drag = self._triangle_drag
        self._triangle_drag = False
        if was_triangle_drag:
            return

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
            self.status_label.configure(text="Start set - 클릭하여 피하지방시작 마킹")
        elif self.marker_mode == "dermis":
            self.dermis_us = time_us
            self.mode_var.set("fascia")
            self.marker_mode = "fascia"
            self.status_label.configure(text="피하지방시작 set - Click to mark Fascia")
        elif self.marker_mode == "fascia":
            self.fascia_us = time_us
            self.mode_var.set("bone")
            self.marker_mode = "bone"
            self.status_label.configure(
                text="Fascia set - 뼈가 보이면 클릭하여 마킹, 없으면 Save JSON")
        elif self.marker_mode == "bone":
            self.bone_us = time_us
            self.status_label.configure(
                text="뼈 마커 설정됨 - Save JSON 으로 저장")

        self._update_graph()
        self._update_marker_info()

    def _on_mouse_move(self, event):
        """Show distance from start point as cursor moves (status bar + on-graph annotation)"""
        # ---- X축 끝 삼각형 드래그 (줌 윈도우 이동) ----
        if self._triangle_drag and len(self.adc_values) > 0:
            # event.xdata가 없으면 display coords에서 변환
            xdata = event.xdata
            if xdata is None:
                try:
                    xdata, _ = self.ax.transData.inverted().transform((event.x, 0))
                except Exception:
                    return
            full_min = float(self.time_us[0])
            full_max = float(self.time_us[-1])
            zoomed_w = (full_max - full_min) * self.xzoom
            self.xzoom_end_us = max(full_min + zoomed_w, min(xdata, full_max))
            self._update_graph()
            return

        # ---- Shift drag (prev compare or ref) ----
        if (self._drag_target is not None and
                self._press_pos is not None and
                self._drag_start_xdata is not None and
                event.inaxes == self.ax and
                event.xdata is not None):
            dx_px = abs(event.x - self._press_pos[0])
            if dx_px > 3:
                self._is_shift_drag = True
                new_shift = self._drag_start_shift + (event.xdata - self._drag_start_xdata)
                mm = new_shift * self.config.speed_of_sound / 2000.0
                if self._drag_target == "ref":
                    self.ref_shift_us = new_shift
                    self.ref_default_shift_mm = mm  # 드래그 값을 기본값으로 갱신
                    self._fast_ref_shift_redraw()
                    self.status_label.configure(
                        text=f"Ref Shift: {new_shift:+.3f} μs  ({mm:+.3f} mm)")
                else:
                    self.display_shift_us = new_shift
                    self._fast_shift_redraw()
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

    def _step_window(self, delta: int):
        self.envelope_window = max(1, self.envelope_window + delta)
        self.window_entry.delete(0, "end")
        self.window_entry.insert(0, str(self.envelope_window))
        self._update_graph()

    def _on_env_mode_changed(self, value):
        self.envelope_mode = "hilbert" if value == "Hilbert" else "mavg"
        state = "normal" if self.envelope_mode == "mavg" else "disabled"
        self.btn_minus.configure(state=state)
        self.btn_plus.configure(state=state)
        self.window_entry.configure(state=state)
        self._update_graph()

    def _on_window_changed(self, event=None):
        try:
            val = int(self.window_entry.get())
            if val < 1:
                val = 1
            self.envelope_window = val
        except ValueError:
            pass
        self.window_entry.delete(0, "end")
        self.window_entry.insert(0, str(self.envelope_window))
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
        """Undo last marker: Bone → Fascia → Dermis → Start (reverse order)"""
        if self.bone_us is not None:
            self.bone_us = None
            self.mode_var.set("bone")
            self.marker_mode = "bone"
            self.status_label.configure(text="뼈 마커 삭제됨 - 다시 클릭하거나 Save JSON")
        elif self.fascia_us is not None:
            self.fascia_us = None
            self.mode_var.set("fascia")
            self.marker_mode = "fascia"
            self.status_label.configure(text="Fascia cleared - Click to mark Fascia")
        elif self.dermis_us is not None:
            self.dermis_us = None
            self.mode_var.set("dermis")
            self.marker_mode = "dermis"
            self.status_label.configure(text="피하지방시작 cleared - 클릭하여 피하지방시작 마킹")
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

        self.reset_t0_btn.configure(
            state="normal" if self.start_us is not None else "disabled")

        if self.start_us is not None:
            self.start_info.configure(text=f"Start: {self.start_us:.2f} us")
        else:
            self.start_info.configure(text="Start: --")

        if self.start_us is not None and self.dermis_us is not None:
            d_us = self.dermis_us - self.start_us
            dmm = d_us * speed / 2000.0
            self.dermis_info.configure(
                text=f"피하지방시작: {d_us:.2f} us ({dmm:.2f} mm)")
        else:
            self.dermis_info.configure(text="피하지방시작: --")

        # 표피두께: 수동 마커가 아닌 자동 검출 값 (analyze_epidermis.py와 동일)
        if self.auto_epi_mm is not None:
            self.epi_info.configure(text=f"표피두께: {self.auto_epi_mm:.3f} mm")
        else:
            self.epi_info.configure(text="표피두께: --")

        if self.start_us is not None and self.fascia_us is not None:
            f_us = self.fascia_us - self.start_us
            fmm = f_us * speed / 2000.0
            self.fascia_info.configure(
                text=f"Fascia: {f_us:.2f} us ({fmm:.2f} mm)")
        else:
            self.fascia_info.configure(text="Fascia: --")

        if self.bone_us is not None and self.start_us is not None:
            b_us = self.bone_us - self.start_us
            bmm = b_us * speed / 2000.0
            self.bone_info.configure(text=f"뼈: {b_us:.2f} us ({bmm:.2f} mm)")
        else:
            self.bone_info.configure(text="뼈: --")

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
