import json
from datetime import datetime


class JsonResultWriter:
    """Yuz analiz sonucunu JSON dosyasina yazar."""

    def __init__(self, output_directory):
        self.output_directory = output_directory
        self.output_directory.mkdir(parents=True, exist_ok=True)

    def write(self, analysis_result):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        file_name = "face_analysis_" + timestamp + ".json"
        output_path = self.output_directory / file_name

        json_data = analysis_result.to_dictionary()

        with output_path.open("w", encoding="utf-8") as json_file:
            json.dump(
                json_data,
                json_file,
                ensure_ascii=False,
                indent=4,
            )

        return output_path
