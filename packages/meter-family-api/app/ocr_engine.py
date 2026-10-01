from pathlib import Path
from time import perf_counter
import threading

import numpy as np
from paddleocr import PaddleOCR


class OcrEngine:
    """Loads the finalized OCR detector and recognizer once and reuses them."""

    def __init__(
        self,
        detector_model_path: Path,
        recognizer_model_path: Path,
        character_dictionary_path: Path,
    ) -> None:
        self._inference_lock = threading.Lock()

        self.ocr = PaddleOCR(
            use_angle_cls=False,
            use_gpu=False,
            show_log=False,
            lang="en",
            det_model_dir=str(detector_model_path),
            rec_model_dir=str(recognizer_model_path),
            rec_char_dict_path=str(character_dictionary_path),
            drop_score=0.3,
        )

    def predict(self, image_bgr: np.ndarray):
        inference_started = perf_counter()

        with self._inference_lock:
            raw_result = self.ocr.ocr(
                image_bgr,
                cls=False,
            )

        inference_time_ms = (
            perf_counter() - inference_started
        ) * 1000

        if not raw_result:
            return [], inference_time_ms

        if len(raw_result) == 1 and isinstance(raw_result[0], list):
            result_lines = raw_result[0]
        else:
            result_lines = raw_result

        text_occurrences = []

        for result_line in result_lines or []:
            if not result_line or len(result_line) < 2:
                continue

            raw_polygon = result_line[0]
            recognition = result_line[1]

            if not recognition or len(recognition) < 2:
                continue

            polygon = [
                [
                    int(round(float(point[0]))),
                    int(round(float(point[1]))),
                ]
                for point in raw_polygon
            ]

            text_occurrences.append(
                {
                    "value": str(recognition[0]),
                    "confidence": float(recognition[1]),
                    "polygon": polygon,
                }
            )

        return text_occurrences, inference_time_ms
