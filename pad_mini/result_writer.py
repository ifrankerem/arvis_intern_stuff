from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from data_models import FaceAnalysisResult, LivenessResult


class JsonResultWriter:
    """Canlılık sonucunu okunabilir bir JSON dosyasına kaydeder."""

    def __init__(self, output_directory: Path) -> None:
        self.output_directory = output_directory
        self.output_directory.mkdir(parents=True, exist_ok=True)

    def write(self, result: FaceAnalysisResult | LivenessResult) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_path = self.output_directory / f"face_analysis_{timestamp}.json"

        with output_path.open("w", encoding="utf-8") as json_file:
            json.dump(
                result.to_dict(),
                json_file,
                ensure_ascii=False,
                indent=4,
            )

        return output_path
