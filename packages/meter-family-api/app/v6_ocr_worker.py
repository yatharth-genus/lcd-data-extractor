import argparse
import base64
import json
import os
import sys
import traceback
from pathlib import Path
import cv2
import numpy as np

def send(value):
    sys.stdout.write(json.dumps(value, ensure_ascii=False) + "\n")
    sys.stdout.flush()

def create_system(args):
    root = Path(os.environ["PADDLEOCR_ROOT"])
    sys.path.insert(0, str(root))
    from tools.infer.predict_system import TextSystem
    from tools.infer.utility import parse_args
    saved = sys.argv[:]
    try:
        sys.argv = ["predict_system.py", "--det_model_dir", str(args.det_model), "--rec_model_dir", str(args.rec_model), "--rec_char_dict_path", str(args.dictionary), "--use_gpu", "false", "--use_angle_cls", "false", "--use_space_char", "false", "--det_algorithm", "DB", "--rec_algorithm", "SVTR_LCNet", "--det_limit_type", "max", "--det_limit_side_len", "1280", "--det_db_thresh", "0.20", "--det_db_box_thresh", "0.45", "--det_db_unclip_ratio", "1.40", "--drop_score", "0.0", "--show_log", "false"]
        parsed = parse_args()
    finally:
        sys.argv = saved
    return TextSystem(parsed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--det-model", type=Path, required=True)
    parser.add_argument("--rec-model", type=Path, required=True)
    parser.add_argument("--dictionary", type=Path, required=True)
    args = parser.parse_args()
    system = create_system(args)
    send({"status": "ready"})
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get("command") == "shutdown":
                send({"status": "stopped"})
                break
            raw = base64.b64decode(request["image_base64"])
            image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            boxes, recs, _ = system(image)
            values = []
            for box, rec in zip(boxes or [], recs or []):
                values.append({"value": str(rec[0]), "confidence": float(rec[1]), "polygon": [[int(round(float(x))), int(round(float(y)))] for x, y in box]})
            send({"status": "success", "text_occurrences": values})
        except Exception as exc:
            send({"status": "error", "error": str(exc), "traceback": traceback.format_exc()})

if __name__ == "__main__":
    main()
