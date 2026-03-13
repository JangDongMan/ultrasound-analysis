"""
Ultrasound ADC Capture GUI Application
Windows 11 Style (CustomTkinter)
"""

import customtkinter as ctk
from tkinter import filedialog, messagebox, simpledialog
import threading
import queue
import time
import os
import sys
import io
from datetime import datetime

import numpy as np
from scipy.signal import hilbert
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
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
    for f in fm.fontManager.ttflist:
        if any(k in f.name for k in ['Gothic', 'Gulim', 'Batang', 'Nanum']):
            matplotlib.rcParams['font.family'] = f.name
            matplotlib.rcParams['axes.unicode_minus'] = False
            return

_setup_korean_font()

from serial_comm import UltrasoundSerial, save_to_csv, load_from_csv
from config import ConfigManager


# Data trimming
DEFAULT_DROP = 1200     # Default drop count (12.00 μs at 10ns interval)
TRIM_COUNT = 1250       # Keep samples (~10mm at 1540 m/s)

# CustomTkinter settings
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class PositionSetupDialog(ctk.CTkToplevel):
    """Dialog for setting up measurement positions"""

    def __init__(self, parent, existing_positions=None):
        super().__init__(parent)

        self.title("Measurement Position Setup")
        self.geometry("500x700")
        self.resizable(False, False)

        self.positions = existing_positions or []
        self.result = None

        # Make modal
        self.transient(parent)
        self.grab_set()

        self._create_ui()

        # Center on screen
        self.update_idletasks()
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = (screen_width - 500) // 2
        y = (screen_height - 700) // 2
        self.geometry(f"500x700+{x}+{y}")

    def _create_ui(self):
        # Title
        ctk.CTkLabel(self, text="Enter Measurement Positions",
                     font=ctk.CTkFont(size=16, weight="bold")).pack(pady=15)

        ctk.CTkLabel(self, text="(Minimum 4, Maximum 30 positions)").pack(pady=5)

        # Text area for positions
        self.text_frame = ctk.CTkFrame(self)
        self.text_frame.pack(fill="both", expand=True, padx=20, pady=10)

        self.textbox = ctk.CTkTextbox(self.text_frame, width=400, height=400)
        self.textbox.pack(fill="both", expand=True, padx=10, pady=10)

        # Pre-fill with existing or default positions
        if self.positions:
            self.textbox.insert("1.0", "\n".join(self.positions))
        else:
            default = [
                "Forehead-1", "Forehead-2",
                "Cheek-L", "Cheek-R",
                "Chin", "Nose",
                "Neck-L", "Neck-R"
            ]
            self.textbox.insert("1.0", "\n".join(default))

        # Info label
        ctk.CTkLabel(self, text="Enter one position per line",
                     text_color="gray").pack(pady=5)

        # Buttons
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill="x", padx=20, pady=15)

        ctk.CTkButton(btn_frame, text="Save Config", width=120,
                      fg_color="#28a745", hover_color="#218838",
                      command=self._on_ok).pack(side="right", padx=5)
        ctk.CTkButton(btn_frame, text="Cancel", width=100,
                      fg_color="gray", command=self._on_cancel).pack(side="right", padx=5)

    def _on_ok(self):
        text = self.textbox.get("1.0", "end-1c")
        positions = [p.strip() for p in text.split("\n") if p.strip()]

        if len(positions) < 4:
            messagebox.showerror("Error", "Minimum 4 positions required")
            return
        if len(positions) > 30:
            messagebox.showerror("Error", "Maximum 30 positions allowed")
            return

        self.result = positions
        self.destroy()

    def _on_cancel(self):
        self.result = None
        self.destroy()


class ADCCaptureGUI(ctk.CTk):
    """Ultrasound ADC Capture GUI"""

    def __init__(self):
        super().__init__()

        # Window settings
        self.title("VB5K Capture")

        # Center window on screen
        window_width = 1200
        window_height = 800
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.geometry(f"{window_width}x{window_height}+{x}+{y}")
        self.minsize(1000, 700)

        # Serial object
        self.serial = UltrasoundSerial()
        self.connected = False

        # Data storage
        self.time_ns = np.array([])
        self.adc_values = np.array([])
        self.drop_samples = DEFAULT_DROP  # Current drop count for display offset

        # Previous data for comparison
        self.prev_time_ns = np.array([])
        self.prev_adc_values = np.array([])

        # Thread queue
        self.data_queue = queue.Queue()

        # Raw data (before trimming) for debug
        self.raw_adc_values = np.array([])
        self.show_raw = False
        self.show_drop = False

        # Config manager
        self.config = ConfigManager()

        # Position selection
        self.current_position_index = 0

        # Load config and setup if needed
        self._check_config()

        # Create UI
        self._create_ui()

        # Refresh ports
        self._refresh_ports()

        # Queue check
        self._check_queue()

    def _check_config(self):
        """Check if config exists, show setup dialog if not"""
        if not self.config.config_exists():
            # Show setup dialog after mainloop starts
            self.after(100, self._show_position_setup)
        else:
            self.current_position_index = self.config.last_position_index

    def _show_position_setup(self):
        """Show position setup dialog"""
        dialog = PositionSetupDialog(self, self.config.positions)
        self.wait_window(dialog)

        if dialog.result:
            self.config.set_positions(dialog.result)
            self._update_position_display()
        else:
            # If cancelled and no config, exit
            if not self.config.positions:
                messagebox.showerror("Error", "Position configuration required")
                self.destroy()

    def _create_ui(self):
        """Create UI components"""

        # Grid setup
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(5, weight=1)

        # ===== Row 0: Title =====
        ctk.CTkLabel(self, text="VB5K Capture",
                     font=ctk.CTkFont(size=28, weight="bold")).grid(
            row=0, column=0, padx=15, pady=(10, 2))

        # ===== Row 1: Serial Connection =====
        conn_frame = ctk.CTkFrame(self, corner_radius=10)
        conn_frame.grid(row=1, column=0, padx=15, pady=(5, 5), sticky="ew")

        ctk.CTkLabel(conn_frame, text="Serial",
                     font=ctk.CTkFont(size=14, weight="bold")).grid(
            row=0, column=0, padx=15, pady=10, sticky="w")

        ctk.CTkLabel(conn_frame, text="Port:").grid(row=0, column=1, padx=5, pady=10)
        self.port_combo = ctk.CTkComboBox(conn_frame, width=120, state="readonly")
        self.port_combo.grid(row=0, column=2, padx=5, pady=10)

        self.refresh_btn = ctk.CTkButton(conn_frame, text="Refresh", width=70,
                                          command=self._refresh_ports)
        self.refresh_btn.grid(row=0, column=3, padx=5, pady=10)

        ctk.CTkLabel(conn_frame, text="Baud:").grid(row=0, column=4, padx=5, pady=10)
        self.baud_combo = ctk.CTkComboBox(conn_frame, width=90, state="readonly",
                                           values=["115200", "230400", "460800", "921600"])
        self.baud_combo.set("115200")
        self.baud_combo.grid(row=0, column=5, padx=5, pady=10)

        self.connect_btn = ctk.CTkButton(conn_frame, text="Connect", width=90,
                                          fg_color="#28a745", hover_color="#218838",
                                          command=self._toggle_connection)
        self.connect_btn.grid(row=0, column=6, padx=15, pady=10)

        self.status_indicator = ctk.CTkLabel(conn_frame, text="●", text_color="red",
                                              font=ctk.CTkFont(size=16))
        self.status_indicator.grid(row=0, column=7, padx=10, pady=10)

        # ===== Row 2: Patient Info & Position =====
        info_frame = ctk.CTkFrame(self, corner_radius=10)
        info_frame.grid(row=2, column=0, padx=15, pady=5, sticky="ew")

        # Patient name
        ctk.CTkLabel(info_frame, text="Patient:",
                     font=ctk.CTkFont(size=14, weight="bold")).grid(
            row=0, column=0, padx=15, pady=10, sticky="w")

        self.patient_entry = ctk.CTkEntry(info_frame, width=150,
                                           placeholder_text="Enter name")
        self.patient_entry.grid(row=0, column=1, padx=5, pady=10)
        if self.config.last_patient:
            self.patient_entry.insert(0, self.config.last_patient)

        # Gender
        ctk.CTkLabel(info_frame, text="Gender:").grid(row=0, column=2, padx=(20, 5), pady=10)

        self.gender_var = ctk.StringVar(value=self.config.last_gender)
        self.gender_m = ctk.CTkRadioButton(info_frame, text="M", variable=self.gender_var, value="M")
        self.gender_m.grid(row=0, column=3, padx=5, pady=10)
        self.gender_f = ctk.CTkRadioButton(info_frame, text="F", variable=self.gender_var, value="F")
        self.gender_f.grid(row=0, column=4, padx=5, pady=10)

        # Position selection (wheel scroll + right click)
        ctk.CTkLabel(info_frame, text="Position:",
                     font=ctk.CTkFont(size=14, weight="bold")).grid(
            row=0, column=5, padx=(30, 5), pady=10)

        self.position_frame = ctk.CTkFrame(info_frame, fg_color="#1a1a2e", corner_radius=8)
        self.position_frame.grid(row=0, column=6, padx=5, pady=10)

        self.position_label = ctk.CTkLabel(self.position_frame, text="[ Loading... ]",
                                            font=ctk.CTkFont(size=16, weight="bold"),
                                            width=180)
        self.position_label.pack(padx=15, pady=8)

        # Bind mouse events for position selection
        self.position_frame.bind("<MouseWheel>", self._on_position_wheel)
        self.position_label.bind("<MouseWheel>", self._on_position_wheel)
        self.position_frame.bind("<Button-3>", self._on_position_select)
        self.position_label.bind("<Button-3>", self._on_position_select)

        # Position hint
        ctk.CTkLabel(info_frame, text="(Wheel: change, Right-click: select)",
                     text_color="gray", font=ctk.CTkFont(size=10)).grid(
            row=0, column=7, padx=5, pady=10)

        # Edit positions button
        self.edit_pos_btn = ctk.CTkButton(info_frame, text="Edit", width=50,
                                           fg_color="gray", command=self._edit_positions)
        self.edit_pos_btn.grid(row=0, column=8, padx=15, pady=10)

        # ===== Row 3: Capture Settings =====
        cap_frame = ctk.CTkFrame(self, corner_radius=10)
        cap_frame.grid(row=3, column=0, padx=15, pady=5, sticky="ew")

        ctk.CTkLabel(cap_frame, text="Capture",
                     font=ctk.CTkFont(size=14, weight="bold")).grid(
            row=0, column=0, padx=15, pady=10, sticky="w")

        ctk.CTkLabel(cap_frame, text="Command:").grid(row=0, column=1, padx=5, pady=10)
        self.cmd_entry = ctk.CTkComboBox(
            cap_frame, width=160,
            values=["pwm start 5 1", "pwm start ulso"],
            state="normal")
        self.cmd_entry.set("pwm start 5 1")
        self.cmd_entry.grid(row=0, column=2, padx=5, pady=10)

        ctk.CTkLabel(cap_frame, text="Drop:").grid(row=0, column=3, padx=5, pady=10)
        self.drop_entry = ctk.CTkEntry(cap_frame, width=70)
        self.drop_entry.insert(0, str(DEFAULT_DROP))
        self.drop_entry.grid(row=0, column=4, padx=5, pady=10)

        ctk.CTkLabel(cap_frame, text="Timeout:").grid(row=0, column=5, padx=5, pady=10)
        self.timeout_entry = ctk.CTkEntry(cap_frame, width=50)
        self.timeout_entry.insert(0, "10.0")
        self.timeout_entry.grid(row=0, column=6, padx=5, pady=10)

        self.capture_btn = ctk.CTkButton(cap_frame, text="Capture", width=100,
                                          fg_color="#007bff", hover_color="#0056b3",
                                          command=self._start_capture, state="disabled")
        self.capture_btn.grid(row=0, column=7, padx=15, pady=10)

        self.send_btn = ctk.CTkButton(cap_frame, text="Send Cmd", width=80,
                                       fg_color="#6c757d", hover_color="#545b62",
                                       command=self._send_command_only, state="disabled")
        self.send_btn.grid(row=0, column=8, padx=5, pady=10)

        # Progress
        self.progress_bar = ctk.CTkProgressBar(cap_frame, width=150)
        self.progress_bar.grid(row=0, column=9, padx=10, pady=10)
        self.progress_bar.set(0)

        self.progress_label = ctk.CTkLabel(cap_frame, text="Ready", width=100)
        self.progress_label.grid(row=0, column=10, padx=5, pady=10)

        self.debug_btn = ctk.CTkButton(cap_frame, text="Debug ▼", width=80,
                                        fg_color="#6c757d", hover_color="#545b62",
                                        command=self._toggle_debug)
        self.debug_btn.grid(row=0, column=11, padx=(10, 15), pady=10)

        # ===== Row 4: Debug Log (collapsible) =====
        self.debug_visible = False
        self.debug_frame = ctk.CTkFrame(self, corner_radius=10)
        # Not gridded initially (hidden)

        log_header = ctk.CTkFrame(self.debug_frame, fg_color="transparent")
        log_header.pack(fill="x", padx=10, pady=(5, 0))
        ctk.CTkLabel(log_header, text="Serial Log",
                     font=ctk.CTkFont(size=12, weight="bold")).pack(side="left", padx=5)

        self.log_clear_btn = ctk.CTkButton(log_header, text="Clear", width=50,
                                            fg_color="gray", command=self._clear_log)
        self.log_clear_btn.pack(side="right", padx=5)

        self.raw_save_btn = ctk.CTkButton(log_header, text="Save Raw", width=80,
                                           fg_color="#6c757d", hover_color="#545b62",
                                           command=self._save_raw_data, state="disabled")
        self.raw_save_btn.pack(side="right", padx=5)

        self.raw_toggle_btn = ctk.CTkButton(log_header, text="Show Raw", width=80,
                                             fg_color="#17a2b8", hover_color="#138496",
                                             command=self._toggle_raw)
        self.raw_toggle_btn.pack(side="right", padx=5)

        self.drop_toggle_btn = ctk.CTkButton(log_header, text="Show Drop", width=80,
                                             fg_color="#e67e22", hover_color="#ca6f1e",
                                             command=self._toggle_drop)
        self.drop_toggle_btn.pack(side="right", padx=5)

        self.log_text = ctk.CTkTextbox(self.debug_frame, height=100,
                                        font=ctk.CTkFont(family="Courier", size=11))
        self.log_text.pack(fill="both", expand=True, padx=10, pady=(2, 8))
        self.log_text.configure(state="disabled")

        # ===== Row 5: Graph =====
        graph_frame = ctk.CTkFrame(self, corner_radius=10)
        graph_frame.grid(row=5, column=0, padx=15, pady=5, sticky="nsew")
        graph_frame.grid_columnconfigure(0, weight=1)
        graph_frame.grid_rowconfigure(0, weight=1)

        # matplotlib Figure (row 5)
        plt.style.use('dark_background')
        self.fig = Figure(figsize=(10, 4), dpi=100, facecolor='#2b2b2b')
        self.ax = self.fig.add_subplot(111)
        self._setup_graph_style()

        canvas_frame = ctk.CTkFrame(graph_frame, fg_color="#2b2b2b", corner_radius=5)
        canvas_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        canvas_frame.grid_columnconfigure(0, weight=1)
        canvas_frame.grid_rowconfigure(0, weight=1)

        self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        toolbar_frame = ctk.CTkFrame(canvas_frame, fg_color="#2b2b2b")
        toolbar_frame.grid(row=1, column=0, sticky="ew")
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.config(background='#2b2b2b')
        self.toolbar.update()

        # ===== Row 6: Bottom buttons =====
        bottom_frame = ctk.CTkFrame(self, corner_radius=10)
        bottom_frame.grid(row=6, column=0, padx=15, pady=(5, 15), sticky="ew")
        bottom_frame.grid_columnconfigure(8, weight=1)

        self.save_btn = ctk.CTkButton(bottom_frame, text="Quick Save", width=100,
                                       fg_color="#28a745", hover_color="#218838",
                                       command=self._quick_save, state="disabled")
        self.save_btn.grid(row=0, column=0, padx=(15, 5), pady=15)

        self.folder_btn = ctk.CTkButton(bottom_frame, text="Set Folder", width=80,
                                         fg_color="#6c757d", hover_color="#545b62",
                                         command=self._set_save_folder)
        self.folder_btn.grid(row=0, column=1, padx=5, pady=15)

        self.load_btn = ctk.CTkButton(bottom_frame, text="Load CSV", width=80,
                                       fg_color="#6c757d", hover_color="#545b62",
                                       command=self._load_data)
        self.load_btn.grid(row=0, column=2, padx=5, pady=15)

        self.compare_btn = ctk.CTkButton(bottom_frame, text="Compare", width=80,
                                         fg_color="#17a2b8", hover_color="#138496",
                                         command=self._toggle_compare, state="disabled")
        self.compare_btn.grid(row=0, column=3, padx=5, pady=15)
        self.show_compare = False

        self.clear_btn = ctk.CTkButton(bottom_frame, text="Clear", width=60,
                                        fg_color="#dc3545", hover_color="#c82333",
                                        command=self._clear_graph)
        self.clear_btn.grid(row=0, column=4, padx=5, pady=15)

        self.envelope_switch = ctk.CTkSwitch(bottom_frame, text="Envelope",
                                              command=self._update_graph)
        self.envelope_switch.select()
        self.envelope_switch.grid(row=0, column=5, padx=15, pady=15)

        self.theme_switch = ctk.CTkSwitch(bottom_frame, text="Dark",
                                           command=self._toggle_theme)
        self.theme_switch.select()
        self.theme_switch.grid(row=0, column=6, padx=10, pady=15)

        # Filename preview
        self.filename_label = ctk.CTkLabel(bottom_frame, text="",
                                            font=ctk.CTkFont(size=11),
                                            text_color="#00bfff")
        self.filename_label.grid(row=0, column=7, padx=10, pady=15)

        self.data_info = ctk.CTkLabel(bottom_frame, text="No Data",
                                       font=ctk.CTkFont(size=12))
        self.data_info.grid(row=0, column=8, padx=15, pady=15, sticky="e")

        # Update position display
        self._update_position_display()
        self._update_filename_preview()

        # Initialize graph with Start marker
        self._update_graph()

    def _setup_graph_style(self):
        """Setup graph style"""
        self.ax.set_facecolor('#2b2b2b')
        self.ax.set_xlabel("Time (us)", color='white')
        self.ax.set_ylabel("Envelope", color='white')
        self.ax.set_title("Ultrasound Signal", color='white')
        self.ax.grid(True, alpha=0.3, color='gray')
        self.ax.tick_params(colors='white')
        for spine in self.ax.spines.values():
            spine.set_color('gray')

    def _on_position_wheel(self, event):
        """Handle mouse wheel for position selection"""
        if not self.config.positions:
            return

        if event.delta > 0:
            self.current_position_index = (self.current_position_index - 1) % len(self.config.positions)
        else:
            self.current_position_index = (self.current_position_index + 1) % len(self.config.positions)

        self._update_position_display()
        self._update_filename_preview()
        self.config.update_last_used(position_index=self.current_position_index)

    def _on_position_select(self, event):
        """Handle right click for position confirmation"""
        pos = self.config.get_position(self.current_position_index)
        self.position_frame.configure(fg_color="#28a745")
        self.after(200, lambda: self.position_frame.configure(fg_color="#1a1a2e"))
        self.config.update_last_used(position_index=self.current_position_index)

    def _update_position_display(self):
        """Update position display"""
        if self.config.positions:
            pos = self.config.get_position(self.current_position_index)
            total = self.config.get_position_count()
            self.position_label.configure(
                text=f"[{self.current_position_index + 1}/{total}] {pos}")
        else:
            self.position_label.configure(text="[ No positions ]")

    def _edit_positions(self):
        """Edit measurement positions"""
        dialog = PositionSetupDialog(self, self.config.positions)
        self.wait_window(dialog)

        if dialog.result:
            self.config.set_positions(dialog.result)
            self.current_position_index = 0
            self._update_position_display()
            self._update_filename_preview()
            self.progress_label.configure(
                text=f"Config saved: {len(dialog.result)} positions")

    def _update_filename_preview(self):
        """Update filename preview"""
        patient = self.patient_entry.get().strip() or "unknown"
        gender = self.gender_var.get()
        position = self.config.get_position(self.current_position_index) or "pos"

        # Show folder and filename pattern
        if self.config.save_directory:
            folder = os.path.basename(self.config.save_directory)
            self.filename_label.configure(text=f"[{folder}] {patient}_{position}_{gender}.csv")
        else:
            self.filename_label.configure(text=f"[No Folder] {patient}_{position}_{gender}.csv")

    def _refresh_ports(self):
        """Refresh serial port list and restore last used port"""
        ports = UltrasoundSerial.list_ports()
        port_names = [p[0] for p in ports]
        self.port_combo.configure(values=port_names)
        if port_names:
            # 마지막 사용 포트가 있으면 선택
            if self.config.last_port and self.config.last_port in port_names:
                self.port_combo.set(self.config.last_port)
            else:
                self.port_combo.set(port_names[0])
        else:
            self.port_combo.set("")

    def _toggle_connection(self):
        """Toggle connection"""
        if self.connected:
            self._disconnect()
        else:
            self._connect()

    def _connect(self):
        """Connect to serial port"""
        port = self.port_combo.get()
        if not port:
            messagebox.showerror("Error", "Select a port")
            return

        baudrate = int(self.baud_combo.get())
        self.serial.port = port
        self.serial.baudrate = baudrate

        if self.serial.connect():
            # 버퍼 비우기
            self.serial.flush_buffer()

            # 포트 저장
            self.config.update_last_used(port=port)

            self.connected = True
            self.connect_btn.configure(text="Disconnect",
                                        fg_color="#dc3545", hover_color="#c82333")
            self.status_indicator.configure(text_color="#28a745")
            self.capture_btn.configure(state="normal")
            self.send_btn.configure(state="normal")
            self.port_combo.configure(state="disabled")
            self.baud_combo.configure(state="disabled")
            self.progress_label.configure(text=f"Connected: {port}")
        else:
            messagebox.showerror("Error", f"Connection failed: {port}")

    def _disconnect(self):
        """Disconnect from serial port"""
        self.serial.disconnect()
        self.connected = False
        self.connect_btn.configure(text="Connect",
                                    fg_color="#28a745", hover_color="#218838")
        self.status_indicator.configure(text_color="red")
        self.capture_btn.configure(state="disabled")
        self.send_btn.configure(state="disabled")
        self.port_combo.configure(state="readonly")
        self.baud_combo.configure(state="readonly")
        self.progress_label.configure(text="Disconnected")

    def _send_command_only(self):
        """Send command only"""
        cmd = self.cmd_entry.get()
        if self.serial.send_command(cmd):
            self.progress_label.configure(text=f"Sent: {cmd}")
        else:
            self.progress_label.configure(text="Send failed")

    def _start_capture(self):
        """Start capture"""
        self.capture_btn.configure(state="disabled")
        self.send_btn.configure(state="disabled")
        self.progress_label.configure(text="Capturing...")
        self.progress_bar.set(0)

        cmd = self.cmd_entry.get()
        try:
            self.drop_samples = int(self.drop_entry.get())
            timeout = float(self.timeout_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid drop or timeout value")
            self.capture_btn.configure(state="normal")
            self.send_btn.configure(state="normal")
            return

        # Capture: drop + keep
        capture_samples = self.drop_samples + TRIM_COUNT

        thread = threading.Thread(target=self._capture_thread,
                                   args=(cmd, capture_samples, timeout), daemon=True)
        thread.start()

    def _capture_thread(self, cmd, num_samples, capture_timeout):
        """Capture thread — stdout redirected to log panel"""
        def progress_callback(count, value):
            self.data_queue.put(('progress', count, num_samples))

        # Redirect serial_comm print() to log panel
        log_buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = log_buf

        try:
            time_ns, adc_values = self.serial.capture(cmd, num_samples, capture_timeout, progress_callback)
        finally:
            sys.stdout = old_stdout

        log_text = log_buf.getvalue()
        self.data_queue.put(('log', log_text, cmd, num_samples, adc_values))
        self.data_queue.put(('done', time_ns, adc_values))

    def _check_queue(self):
        """Check queue and update UI"""
        try:
            while True:
                msg = self.data_queue.get_nowait()

                if msg[0] == 'progress':
                    count, total = msg[1], msg[2]
                    progress = count / total if total > 0 else 0
                    self.progress_bar.set(progress)
                    self.progress_label.configure(text=f"Capturing... {count}/{total}")

                elif msg[0] == 'log':
                    log_text, cmd, num_samples, adc_values = msg[1], msg[2], msg[3], msg[4]
                    self._append_log(f"[CMD] {cmd}  (requested={num_samples})\n")
                    self._append_log(log_text)
                    # Show first/last values summary
                    if len(adc_values) > 0:
                        n = len(adc_values)
                        head = ' '.join(str(v) for v in adc_values[:10])
                        tail = ' '.join(str(v) for v in adc_values[-10:])
                        self._append_log(f"[RAW] total={n}  first10=[{head}]\n")
                        self._append_log(f"[RAW]              last10=[{tail}]\n")
                    self.raw_adc_values = adc_values.copy() if len(adc_values) > 0 else np.array([])
                    if len(adc_values) > 0:
                        self.raw_save_btn.configure(state="normal")

                elif msg[0] == 'done':
                    time_ns, adc_values = msg[1], msg[2]
                    self._on_capture_done(time_ns, adc_values)

        except queue.Empty:
            pass

        self.after(100, self._check_queue)

    def _trim_data(self, time_ns, adc_values, drop=None):
        """Trim data: discard first 'drop' samples, keep TRIM_COUNT, discard rest.
        Returns new (time_ns, adc_values) starting from index 0."""
        if drop is None:
            drop = self.drop_samples
        end = drop + TRIM_COUNT
        if len(adc_values) >= end:
            trimmed = adc_values[drop:end]
        elif len(adc_values) > drop:
            trimmed = adc_values[drop:]
        else:
            return time_ns, adc_values  # not enough data, return as-is
        new_time_ns = np.arange(len(trimmed)) * 10
        return new_time_ns, trimmed

    def _on_capture_done(self, time_ns, adc_values):
        """Capture done handler"""
        # 현재 데이터가 있으면 직전 버퍼로 이동
        if len(self.adc_values) > 0:
            self.prev_time_ns = self.time_ns.copy()
            self.prev_adc_values = self.adc_values.copy()
            self.compare_btn.configure(state="normal")

        # Trim: discard first drop_samples, keep TRIM_COUNT, discard rest
        raw_count = len(adc_values)
        time_ns, adc_values = self._trim_data(time_ns, adc_values)
        self.time_ns = time_ns
        self.adc_values = adc_values

        if len(adc_values) > 0:
            self.progress_bar.set(1)
            self.progress_label.configure(
                text=f"Done: {raw_count} captured, {len(adc_values)} kept")
            self.save_btn.configure(state="normal")
            self._update_graph()
            self._update_data_info()
        else:
            self.progress_bar.set(0)
            self.progress_label.configure(text="Capture failed")

        self.capture_btn.configure(state="normal")
        self.send_btn.configure(state="normal")

    def _compute_envelope(self, signal):
        """Hilbert transform envelope detection"""
        signal_centered = signal - np.mean(signal)
        analytic_signal = hilbert(signal_centered)
        envelope = np.abs(analytic_signal)
        return envelope

    def _toggle_compare(self):
        """Toggle compare overlay"""
        self.show_compare = not self.show_compare
        if self.show_compare:
            self.compare_btn.configure(fg_color="#0d8da6")
        else:
            self.compare_btn.configure(fg_color="#17a2b8")
        self._update_graph()

    def _update_graph(self):
        """Update graph"""
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

        show_envelope = self.envelope_switch.get()

        # Display offset from drop count
        offset_us = self.drop_samples * 0.01

        # 직전 데이터 (빨간색, 반투명) - add display offset
        if self.show_compare and len(self.prev_adc_values) > 0:
            prev_time_us = self.prev_time_ns / 1000.0 + offset_us
            prev_color = '#ff4444' if is_dark else '#cc3333'
            if show_envelope:
                prev_env = self._compute_envelope(self.prev_adc_values)
                self.ax.plot(prev_time_us, prev_env, color=prev_color,
                           linewidth=0.6, alpha=0.5, label="Previous")
            else:
                self.ax.plot(prev_time_us, self.prev_adc_values, color=prev_color,
                           linewidth=0.6, alpha=0.5, label="Previous")

        if self.show_raw and len(self.raw_adc_values) > 0:
            # Raw mode: show full untrimed data with drop region highlighted
            raw_time_us = np.arange(len(self.raw_adc_values)) * 0.01  # 10ns/sample → μs
            drop_n = self.drop_samples
            keep_end = drop_n + TRIM_COUNT

            if show_envelope:
                raw_env = self._compute_envelope(self.raw_adc_values)
                # Drop region (red/dim)
                if drop_n > 0:
                    self.ax.plot(raw_time_us[:drop_n], raw_env[:drop_n],
                                 color='#ff4444', linewidth=0.7, alpha=0.6, label=f"Drop({drop_n})")
                # Keep region (bright)
                end = min(keep_end, len(self.raw_adc_values))
                self.ax.plot(raw_time_us[drop_n:end], raw_env[drop_n:end],
                             color='#00ff88', linewidth=0.8, label=f"Keep({end - drop_n})")
                # Extra region beyond keep
                if len(self.raw_adc_values) > end:
                    self.ax.plot(raw_time_us[end:], raw_env[end:],
                                 color='#aaaaaa', linewidth=0.6, alpha=0.5, label="Tail")
                self.ax.set_ylabel("Envelope", color=text_color)
            else:
                if drop_n > 0:
                    self.ax.plot(raw_time_us[:drop_n], self.raw_adc_values[:drop_n],
                                 color='#ff4444', linewidth=0.7, alpha=0.6, label=f"Drop({drop_n})")
                end = min(keep_end, len(self.raw_adc_values))
                self.ax.plot(raw_time_us[drop_n:end], self.raw_adc_values[drop_n:end],
                             color='#00bfff', linewidth=0.8, label=f"Keep({end - drop_n})")
                if len(self.raw_adc_values) > end:
                    self.ax.plot(raw_time_us[end:], self.raw_adc_values[end:],
                                 color='#aaaaaa', linewidth=0.6, alpha=0.5, label="Tail")
                self.ax.set_ylabel("ADC Value", color=text_color)

            self.ax.set_title(f"RAW DATA — total {len(self.raw_adc_values)} samples", color='#ffaa00')
            self.ax.axvline(x=raw_time_us[drop_n] if drop_n < len(raw_time_us) else 0,
                            color='yellow', linewidth=1, linestyle='--', alpha=0.7)
            self.ax.legend(loc='upper right', fontsize=9,
                           facecolor=bg_color, edgecolor=grid_color, labelcolor=text_color)
            self.ax.xaxis.set_major_locator(MultipleLocator(1.0))
            self.ax.xaxis.set_minor_locator(MultipleLocator(0.5))

        elif len(self.adc_values) > 0:
            time_us = self.time_ns / 1000.0 + offset_us

            # Show Drop: raw_adc_values[:drop_n]을 kept 데이터 앞에 붙여 표시
            drop_n = self.drop_samples
            has_drop = (self.show_drop and len(self.raw_adc_values) >= drop_n > 0)
            if has_drop:
                drop_time_us = (np.arange(drop_n) - drop_n) * 0.01 + time_us[0]
                drop_data = self.raw_adc_values[:drop_n]
                if show_envelope:
                    drop_env = self._compute_envelope(drop_data)
                    self.ax.plot(drop_time_us, drop_env, color='#ff6644',
                                 linewidth=0.7, alpha=0.7, label=f"Drop({drop_n})")
                else:
                    self.ax.plot(drop_time_us, drop_data, color='#ff6644',
                                 linewidth=0.7, alpha=0.7, label=f"Drop({drop_n})")
                self.ax.axvline(x=time_us[0], color='yellow', linewidth=1,
                                linestyle='--', alpha=0.7)

            if show_envelope:
                envelope = self._compute_envelope(self.adc_values)
                line_color = '#00ff88' if is_dark else '#28a745'
                self.ax.plot(time_us, envelope, color=line_color, linewidth=0.8,
                           label="Current")
                self.ax.set_ylabel("Envelope", color=text_color)
                self.ax.set_title("Ultrasound Envelope Signal", color=text_color)
            else:
                line_color = '#00bfff' if is_dark else '#007bff'
                self.ax.plot(time_us, self.adc_values, color=line_color, linewidth=0.8,
                           label="Current")
                self.ax.set_ylabel("ADC Value", color=text_color)
                self.ax.set_title("Ultrasound RF Signal", color=text_color)

            # legend: Compare 또는 Drop 활성 시
            if (self.show_compare and len(self.prev_adc_values) > 0) or has_drop:
                self.ax.legend(loc='upper right', fontsize=9,
                             facecolor=bg_color, edgecolor=grid_color,
                             labelcolor=text_color)

            # Fine x-axis ticks for better resolution
            self.ax.xaxis.set_major_locator(MultipleLocator(1.0))   # 1μs major
            self.ax.xaxis.set_minor_locator(MultipleLocator(0.5))   # 0.5μs minor
            self.ax.tick_params(axis='x', which='minor', length=3)

        else:
            # Default view when no data
            self.ax.set_xlim(offset_us,
                            offset_us + (TRIM_COUNT - 1) * 0.01)
            self.ax.set_ylim(0, 256)
            if show_envelope:
                self.ax.set_ylabel("Envelope", color=text_color)
            else:
                self.ax.set_ylabel("ADC Value", color=text_color)
            self.ax.set_title("Ultrasound Signal (No Data)", color=text_color)

        # Set x-axis label
        self.ax.set_xlabel("Time (us)", color=text_color)

        self.canvas.draw()

    def _update_data_info(self):
        """Update data info"""
        if len(self.adc_values) > 0:
            offset_us = self.drop_samples * 0.01
            start_us = offset_us
            end_us = self.time_ns[-1] / 1000.0 + offset_us
            max_val = np.max(self.adc_values)
            min_val = np.min(self.adc_values)
            info = f"Samples: {len(self.adc_values)} | {start_us:.1f}~{end_us:.1f}us | Min: {min_val} | Max: {max_val}"
            self.data_info.configure(text=info)
        else:
            self.data_info.configure(text="No Data")

    def _toggle_debug(self):
        """Show/hide debug log panel"""
        self.debug_visible = not self.debug_visible
        if self.debug_visible:
            self.debug_frame.grid(row=4, column=0, padx=15, pady=(0, 5), sticky="ew")
            self.debug_btn.configure(text="Debug ▲")
        else:
            self.debug_frame.grid_forget()
            self.debug_btn.configure(text="Debug ▼")
            # 패널 닫을 때 drop 오버레이 끄기
            if self.show_drop:
                self.show_drop = False
                self.drop_toggle_btn.configure(text="Show Drop", fg_color="#e67e22")
                self._update_graph()

    def _append_log(self, text: str):
        """Append text to log panel"""
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _clear_log(self):
        """Clear log panel"""
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    def _toggle_raw(self):
        """Toggle raw (untrimmed) data view in graph"""
        self.show_raw = not self.show_raw
        if self.show_raw:
            self.raw_toggle_btn.configure(text="Show Trim", fg_color="#0d8da6")
        else:
            self.raw_toggle_btn.configure(text="Show Raw", fg_color="#17a2b8")
        self._update_graph()

    def _toggle_drop(self):
        """Toggle drop region overlay on normal graph"""
        self.show_drop = not self.show_drop
        if self.show_drop:
            self.drop_toggle_btn.configure(text="Hide Drop", fg_color="#c0392b")
        else:
            self.drop_toggle_btn.configure(text="Show Drop", fg_color="#e67e22")
        self._update_graph()

    def _save_raw_data(self):
        """Save raw (untrimmed) data to CSV"""
        if len(self.raw_adc_values) == 0:
            messagebox.showwarning("Warning", "No raw data")
            return
        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialfile="raw_debug.csv"
        )
        if filepath:
            raw_time = np.arange(len(self.raw_adc_values)) * 10
            save_to_csv(filepath, raw_time, self.raw_adc_values)
            self.progress_label.configure(text=f"Raw saved: {os.path.basename(filepath)}")


    def _clear_graph(self):
        """Clear graph and data"""
        self.time_ns = np.array([])
        self.adc_values = np.array([])
        self.save_btn.configure(state="disabled")
        self.progress_bar.set(0)
        self._update_graph()
        self._update_data_info()
        self.progress_label.configure(text="Ready")

    def _set_save_folder(self):
        """Set default save folder"""
        initial_dir = self.config.save_directory if self.config.save_directory else os.getcwd()
        folder = filedialog.askdirectory(initialdir=initial_dir, title="Select Save Folder")
        if folder:
            self.config.update_last_used(save_directory=folder)
            self._update_filename_preview()
            self.progress_label.configure(text=f"Folder: {folder}")

    def _quick_save(self):
        """Quick save - save directly without dialog"""
        if len(self.adc_values) == 0:
            messagebox.showwarning("Warning", "No data to save")
            return

        # Check if save folder is set
        if not self.config.save_directory:
            messagebox.showinfo("Info", "Please set save folder first")
            self._set_save_folder()
            if not self.config.save_directory:
                return

        # Generate filename
        patient = self.patient_entry.get().strip() or "unknown"
        gender = self.gender_var.get()
        position = self.config.get_position(self.current_position_index) or "pos"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        filename = f"{patient}_{timestamp}_{position}_{gender}.csv"
        filepath = os.path.join(self.config.save_directory, filename)

        # Update last used settings
        self.config.update_last_used(
            patient=patient,
            gender=gender,
            position_index=self.current_position_index
        )

        metadata = {
            "patient": patient,
            "gender": gender,
            "position": position,
            "capture_time": datetime.now().isoformat(),
            "sample_count": len(self.adc_values),
            "sample_interval_ns": 10
        }
        save_to_csv(filepath, self.time_ns, self.adc_values, metadata)
        self.progress_label.configure(text=f"Saved: {filename}")

    def _load_data(self):
        """Load CSV file"""
        filepath = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if filepath:
            try:
                time_ns, adc_values, metadata = load_from_csv(filepath)
                # Update drop from UI
                try:
                    self.drop_samples = int(self.drop_entry.get())
                except ValueError:
                    self.drop_samples = DEFAULT_DROP
                # Trim if loaded file has more than TRIM_COUNT samples
                if len(adc_values) > TRIM_COUNT:
                    time_ns, adc_values = self._trim_data(time_ns, adc_values)
                self.time_ns = time_ns
                self.adc_values = adc_values
                self._update_graph()
                self._update_data_info()
                self.save_btn.configure(state="normal")
                self.progress_label.configure(text=f"Loaded: {os.path.basename(filepath)}")
            except Exception as e:
                messagebox.showerror("Error", f"Load failed: {e}")

    def _toggle_theme(self):
        """Toggle theme"""
        if self.theme_switch.get():
            ctk.set_appearance_mode("dark")
        else:
            ctk.set_appearance_mode("light")
        self._update_graph()


def main():
    app = ADCCaptureGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
