from __future__ import annotations
import base64, json, os, subprocess, threading
from collections import deque
import cv2
import numpy as np

class PPORCv6Adapter:
    def __init__(self, python_executable, paddleocr_root, detector_model_dir, recognizer_model_dir, character_dict_path, worker_script, **kwargs):
        self.lock = threading.Lock()
        self.noise = deque(maxlen=40)
        env = os.environ.copy()
        env["PADDLEOCR_ROOT"] = str(paddleocr_root)
        self.process = subprocess.Popen([str(python_executable), "-u", str(worker_script), "--det-model", str(detector_model_dir), "--rec-model", str(recognizer_model_dir), "--dictionary", str(character_dict_path)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace", bufsize=1, env=env)
        ready = self._read({"ready", "error"})
        if ready.get("status") != "ready": raise RuntimeError(f"PP-OCRv6 worker startup failed: {ready}")
    def _read(self, statuses):
        while True:
            line = self.process.stdout.readline()
            if line == "": raise RuntimeError(f"PP-OCRv6 worker stopped: {self.process.poll()}")
            line = line.strip().lstrip("\ufeff")
            if not line: continue
            try: message = json.loads(line)
            except json.JSONDecodeError:
                self.noise.append(line); continue
            if isinstance(message, dict) and message.get("status") in statuses: return message
    def ocr(self, img, det=True, rec=True, cls=True, **kwargs):
        if not det or not rec: raise NotImplementedError("Combined detection and recognition only")
        image = img if isinstance(img, np.ndarray) else cv2.imread(str(img))
        if image is None: raise ValueError("Could not read image")
        ok, encoded = cv2.imencode(".png", image)
        if not ok: raise RuntimeError("Could not encode image")
        with self.lock:
            self.process.stdin.write(json.dumps({"image_base64": base64.b64encode(encoded.tobytes()).decode("ascii")}) + "\n")
            self.process.stdin.flush()
            response = self._read({"success", "error"})
        if response.get("status") != "success": raise RuntimeError(response.get("error", "PP-OCRv6 failed"))
        rows = []
        for item in response.get("results", []):
            rows.append([[[float(x), float(y)] for x, y in item["polygon"]], (item["text"], float(item["confidence"]))])
        return [rows]
    def close(self):
        if self.process.poll() is None:
            try:
                self.process.stdin.write('{"command":"shutdown"}\n'); self.process.stdin.flush(); self.process.wait(timeout=10)
            except Exception: self.process.kill()
    def __del__(self):
        try: self.close()
        except Exception: pass
