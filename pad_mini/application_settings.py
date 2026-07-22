"""Kullaniciya ozel uygulama ayarlarini kalici olarak saklar."""

import json
import os
from pathlib import Path


class ApplicationSettings:
    def __init__(self, settings_path=None):
        self.settings_path = settings_path or self._default_settings_path()

    def get_droidcam_address(self):
        settings = self._read()
        value = settings.get("droidcam_address", "")
        if isinstance(value, str):
            return value
        return ""

    def save_droidcam_address(self, address):
        settings = self._read()
        settings["version"] = 1
        settings["droidcam_address"] = address.strip()
        try:
            self._write(settings)
        except OSError:
            return False
        return True

    def _read(self):
        try:
            with self.settings_path.open("r", encoding="utf-8") as file:
                data = json.load(file)
        except (OSError, json.JSONDecodeError):
            return {}

        if isinstance(data, dict):
            return data
        return {}

    def _write(self, settings):
        self.settings_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.settings_path.with_suffix(".tmp")

        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(settings, file, ensure_ascii=False, indent=2)
            file.write("\n")

        temporary_path.replace(self.settings_path)

    def _default_settings_path(self):
        config_directory = os.environ.get("XDG_CONFIG_HOME")
        if config_directory:
            base_directory = Path(config_directory)
        else:
            base_directory = Path.home() / ".config"

        return base_directory / "arvis-face-quality" / "settings.json"
