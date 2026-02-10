"""
Configuration management for ultrasound ADC capture
"""

import json
import os
import sys
from typing import List

CONFIG_FILE = "us_capture.config"


def _get_app_dir() -> str:
    """Get the directory where the executable/script is located"""
    if getattr(sys, 'frozen', False):
        # PyInstaller EXE
        return os.path.dirname(sys.executable)
    else:
        return os.path.dirname(os.path.abspath(__file__))


class ConfigManager:
    """Configuration manager for measurement positions and settings"""

    def __init__(self, config_dir: str = None):
        if config_dir is None:
            config_dir = _get_app_dir()

        self.config_path = os.path.join(config_dir, CONFIG_FILE)

        self.positions: List[str] = []
        self.last_patient: str = ""
        self.last_gender: str = "M"
        self.last_position_index: int = 0
        self.save_directory: str = ""
        self.last_port: str = ""

    def load(self) -> bool:
        """Load all config from us_capture.config

        Returns:
            True if config exists and positions loaded, False otherwise
        """
        if not os.path.exists(self.config_path):
            return False

        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.positions = data.get('positions', [])
            self.last_patient = data.get('last_patient', "")
            self.last_gender = data.get('last_gender', "M")
            self.last_position_index = data.get('last_position_index', 0)
            self.save_directory = data.get('save_directory', "")
            self.last_port = data.get('last_port', "")

            if len(self.positions) < 4:
                return False

            return True
        except (json.JSONDecodeError, IOError):
            return False

    def save(self):
        """Save all config to us_capture.config"""
        data = {
            'positions': self.positions,
            'last_patient': self.last_patient,
            'last_gender': self.last_gender,
            'last_position_index': self.last_position_index,
            'save_directory': self.save_directory,
            'last_port': self.last_port
        }

        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def set_positions(self, positions: List[str]):
        """Set measurement positions (min 4, max 22)"""
        if len(positions) < 4:
            raise ValueError("Minimum 4 positions required")
        if len(positions) > 22:
            raise ValueError("Maximum 22 positions allowed")

        self.positions = positions
        self.save()

    def get_positions(self) -> List[str]:
        """Get measurement positions"""
        return self.positions

    def get_position(self, index: int) -> str:
        """Get position by index"""
        if 0 <= index < len(self.positions):
            return self.positions[index]
        return ""

    def get_position_count(self) -> int:
        """Get number of positions"""
        return len(self.positions)

    def update_last_used(self, patient: str = None, gender: str = None,
                         position_index: int = None, save_directory: str = None,
                         port: str = None):
        """Update last used settings and save"""
        if patient is not None:
            self.last_patient = patient
        if gender is not None:
            self.last_gender = gender
        if position_index is not None:
            self.last_position_index = position_index
        if save_directory is not None:
            self.save_directory = save_directory
        if port is not None:
            self.last_port = port
        self.save()

    def config_exists(self) -> bool:
        """Check if config exists and has valid positions"""
        return os.path.exists(self.config_path) and self.load()
