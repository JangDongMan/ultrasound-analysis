"""
Configuration management for boundary marker GUI
"""

import json
import os
import sys


CONFIG_FILE = "us_marker.config"


def _get_app_dir() -> str:
    """Get the directory where the executable/script is located"""
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    else:
        return os.path.dirname(os.path.abspath(__file__))


class MarkerConfig:
    """Configuration manager for boundary marker settings"""

    def __init__(self, config_dir: str = None):
        if config_dir is None:
            config_dir = _get_app_dir()

        self.config_path = os.path.join(config_dir, CONFIG_FILE)

        self.last_open_dir: str = ""
        self.save_directory: str = ""
        self.speed_of_sound: float = 1540.0
        self.theme: str = "dark"

        self.load()

    def load(self) -> bool:
        if not os.path.exists(self.config_path):
            return False

        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.last_open_dir = data.get('last_open_dir', "")
            self.save_directory = data.get('save_directory', "")
            self.speed_of_sound = data.get('speed_of_sound', 1540.0)
            self.theme = data.get('theme', "dark")
            return True
        except (json.JSONDecodeError, IOError):
            return False

    def save(self):
        data = {
            'last_open_dir': self.last_open_dir,
            'save_directory': self.save_directory,
            'speed_of_sound': self.speed_of_sound,
            'theme': self.theme,
        }

        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.save()
