from __future__ import annotations

import base64
import json
import os
import subprocess
import threading
from collections import deque
from pathlib import Path
from time import perf_counter

import cv2


class V6OcrEngine:
    def __init__(self, python_executable: Path, paddleocr_root: Path, detector_model_path: Path, recognizer_model_path: Path, character_dictionary_path: Path, worker_program: Path):
        self.lock = threading.Lock()
        self.non_json_lines = deque(maxlen=30)
        environment = os.environ.copy()
        environment["PADDLEOCR_ROOT"] = str(paddleocr_root)
        self.process = subprocess.Popen(
            [str(python_executable), "-u", str(worker_program), "--det-model", str(detector_model_path), "--rec-model", str(recognizer_model_path), "--dictionary", str(character_dictionary_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=environment,
        )
        ready = self._read_message({"ready", "error"})
        if ready.get("status") != "ready":
            raise RuntimeError(f"PP-OCRv6 worker startup failed: {ready}")

    def _stderr_after_exit(self):
        if self.process.poll() is None or self.process.stderr is None:
            return ""
        try:
            return self.process.stderr.read().strip()
        except Exception:
            return ""

    def _read_message(self, statuses):
        assert self.process.stdout is not None
        while True:
            line = self.process.stdout.readline()
            if line == "":
                raise RuntimeError(
                    "PP-OCRv6 worker closed stdout. "
                    f"exit_code={self.process.poll()}; "
                    f"recent_stdout={list(self.non_json_lines)!r}; "
                    f"stderr={self._stderr_after_exit()!r}"
                )
            line = line.strip().lstrip("\ufeff")
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                self.non_json_lines.append(line)
                continue
            if isinstance(message, dict) and message.get("status") in statuses:
                return message
            self.non_json_lines.append(line)

    def predict(self, image_bgr):
        started = perf_counter()
        ok, encoded = cv2.imencode(".png", image_bgr)
        if not ok:
            raise RuntimeError("Could not encode image for PP-OCRv6 worker")
        payload = {"image_base64": base64.b64encode(encoded.tobytes()).decode("ascii")}
        with self.lock:
            if self.process.poll() is not None:
                raise RuntimeError(f"PP-OCRv6 worker is not running: {self._stderr_after_exit()}")
            assert self.process.stdin is not None
            self.process.stdin.write(json.dumps(payload) + "\n")
            self.process.stdin.flush()
            response = self._read_message({"success", "error"})
        if response.get("status") != "success":
            detail = response.get("error", "PP-OCRv6 inference failed")
            if response.get("traceback"):
                detail += "\n" + response["traceback"]
            raise RuntimeError(detail)
        return response["text_occurrences"], (perf_counter() - started) * 1000

    def close(self):
        if self.process.poll() is None:
            try:
                assert self.process.stdin is not None
                self.process.stdin.write('{"command":"shutdown"}\n')
                self.process.stdin.flush()
                self.process.wait(timeout=10)
            except Exception:
                self.process.kill()
