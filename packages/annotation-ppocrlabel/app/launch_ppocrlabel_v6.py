from __future__ import annotations

import base64
import json
import os
import runpy
import subprocess
import sys
import threading
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import paddleocr

TOOL_ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = TOOL_ROOT.parent
REPO_ROOT = PACKAGE_ROOT.parent.parent
API_ROOT = REPO_ROOT / "packages" / "meter-family-api"
V6_PYTHON = API_ROOT / ".venv-v6" / "Scripts" / "python.exe"
PADDLEOCR_ROOT = API_ROOT / "runtime" / "PaddleOCR-3.7"
MODEL_ROOT = API_ROOT / "models" / "experimental_v6"
WORKER = TOOL_ROOT / "ppocrv6_worker.py"


class V6PaddleOCR:
    def __init__(self, *args, **kwargs):
        environment = os.environ.copy()
        environment["PADDLEOCR_ROOT"] = str(PADDLEOCR_ROOT)
        self._lock = threading.Lock()
        self._noise = deque(maxlen=30)
        self._process = subprocess.Popen(
            [
                str(V6_PYTHON), "-u", str(WORKER),
                "--det-model", str(MODEL_ROOT / "models" / "detector" / "inference"),
                "--rec-model", str(MODEL_ROOT / "models" / "recognizer" / "inference"),
                "--dictionary", str(MODEL_ROOT / "models" / "recognizer" / "character_dict.txt"),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=environment,
        )
        response = self._read_message({"ready", "error"})
        if response.get("status") != "ready":
            raise RuntimeError(f"PP-OCRv6 startup failed: {response}")

    def _read_message(self, statuses):
        while True:
            line = self._process.stdout.readline()
            if line == "":
                raise RuntimeError(
                    f"PP-OCRv6 worker stopped with exit code {self._process.poll()}; "
                    f"recent output: {list(self._noise)}"
                )
            line = line.strip().lstrip("\ufeff")
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                self._noise.append(line)
                continue
            if isinstance(message, dict) and message.get("status") in statuses:
                return message
            self._noise.append(line)

    def ocr(self, img, det=True, rec=True, cls=True, **kwargs):
        image = img if isinstance(img, np.ndarray) else cv2.imread(str(img))
        if image is None:
            raise ValueError("Could not read image")
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            raise RuntimeError("Could not encode image")
        request = {"image_base64": base64.b64encode(encoded.tobytes()).decode("ascii")}
        with self._lock:
            self._process.stdin.write(json.dumps(request) + "\n")
            self._process.stdin.flush()
            response = self._read_message({"success", "error"})
        if response.get("status") != "success":
            raise RuntimeError(response.get("error", "PP-OCRv6 inference failed"))
        rows = []
        for item in response.get("results", []):
            polygon = [[float(x), float(y)] for x, y in item["polygon"]]
            rows.append([polygon, (item["text"], float(item["confidence"]))])
        return [rows]

    def close(self):
        if self._process.poll() is None:
            try:
                self._process.stdin.write('{"command":"shutdown"}\n')
                self._process.stdin.flush()
                self._process.wait(timeout=10)
            except Exception:
                self._process.kill()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# PPOCRLabel imports PaddleOCR from the paddleocr module.
# Replace it before executing PPOCRLabel, without editing PPOCRLabel.py.
paddleocr.PaddleOCR = V6PaddleOCR
os.chdir(TOOL_ROOT)
sys.argv = [str(TOOL_ROOT / "PPOCRLabel.py"), "--lang", "en"]
runpy.run_path(str(TOOL_ROOT / "PPOCRLabel.py"), run_name="__main__")
