from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import cv2
import numpy as np

SCRIPT_FILE = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_FILE.parent
MODEL4 = SCRIPT_DIR.parent
PACKAGES_DIR = MODEL4.parent
REPO_ROOT = PACKAGES_DIR.parent
ICON_PACKAGE = PACKAGES_DIR / "icon-detector-openvino"
ICON_MODULE = ICON_PACKAGE / "runtime" / "icon_detector.py"
PADDLE_REPO = Path(os.environ.get("PADDLEOCR_ROOT", str(REPO_ROOT.parent / "PaddleOCR-3.7"))).expanduser().resolve()
PADDLE_PYTHON = Path(os.environ.get("PPOCRV6_PYTHON", sys.executable)).expanduser().resolve()
DET_MODEL = MODEL4 / "models" / "detector" / "inference"
REC_MODEL = MODEL4 / "models" / "recognizer" / "inference"
REC_DICT = MODEL4 / "models" / "recognizer" / "character_dict.txt"
DEFAULT_OUTPUT = MODEL4 / "results" / "full_v6_hybrid_validation"
SUPPORTED = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

def load_icon_class():
    spec = importlib.util.spec_from_file_location("lcd_icon_detector", ICON_MODULE)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {ICON_MODULE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.IconDetector


def files_by_suffix(root: Path, suffixes: set[str]):
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in suffixes)


def pick_unique(candidates: list[Path], words: tuple[str, ...], label: str) -> Path:
    ranked = [p for p in candidates if any(word in str(p).lower() for word in words)]
    pool = ranked or candidates
    if len(pool) == 1:
        return pool[0]
    if not pool:
        raise FileNotFoundError(f"No candidate found for {label}")
    # Prefer files in model/icon directories, then shortest path.
    pool.sort(key=lambda p: ("icon" not in str(p).lower(), "model" not in str(p).lower(), len(str(p))))
    best = pool[0]
    print(f"Auto-selected {label}: {best}")
    return best


def build_icon_detector():
    """Load the reviewed OpenVINO icon detector with explicit package assets."""
    IconDetector = load_icon_class()

    icon_package = CLEAN / "packages" / "icon-detector-openvino"
    model_path = icon_package / "model" / "saved_model.xml"
    labels_path = icon_package / "dm.json"
    transforms_path = icon_package / "model_meta" / "transforms.yaml"
    train_config_path = icon_package / "model_meta" / "train.yaml"

    required_assets = {
        "model_path": model_path,
        "model_weights": model_path.with_suffix(".bin"),
        "labels_path": labels_path,
        "transforms_path": transforms_path,
        "train_config_path": train_config_path,
    }

    missing = [
        f"{name}: {path}"
        for name, path in required_assets.items()
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Required icon-detector assets are missing:\n" + "\n".join(missing)
        )

    print("IconDetector constructor:", inspect.signature(IconDetector))
    print("Using reviewed icon-detector package:")
    for name, path in required_assets.items():
        print(f"  {name}: {path}")
    print("  threshold: 0.38")
    print("  device: CPU")

    return IconDetector(
        model_path=model_path,
        labels_path=labels_path,
        transforms_path=transforms_path,
        train_config_path=train_config_path,
        threshold=0.38,
        device="CPU",
    )

def first_value(item: dict, names: tuple[str, ...], default=None):
    for name in names:
        if name in item:
            return item[name]
    return default


def normalize_icon(item: Any, index: int) -> dict:
    if hasattr(item, "model_dump"):
        item = item.model_dump()
    elif hasattr(item, "dict"):
        item = item.dict()
    elif hasattr(item, "__dict__") and not isinstance(item, dict):
        item = vars(item)

    if isinstance(item, dict):
        box = first_value(item, ("box", "bbox", "xyxy", "coordinates", "rect"))
        if isinstance(box, dict):
            box = [box.get("x1", box.get("left")), box.get("y1", box.get("top")),
                   box.get("x2", box.get("right")), box.get("y2", box.get("bottom"))]
        cls = first_value(item, ("class_name", "label", "name", "class", "class_id"), "icon")
        score = first_value(item, ("confidence", "score", "probability", "conf"), 0.0)
    elif isinstance(item, (list, tuple)) and len(item) >= 6:
        box, score, cls = item[:4], item[4], item[5]
    else:
        raise ValueError(f"Unsupported icon detection item: {item!r}")

    if box is None or len(box) != 4:
        raise ValueError(f"Invalid icon box in item: {item!r}")
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    return {
        "id": f"icon_{index + 1}",
        "class_name": str(cls),
        "confidence": float(score),
        "box": [x1, y1, x2, y2],
    }


def _collect_detection_dicts(value: Any) -> list[dict]:
    """Extract detections from list, tuple, dict, or tuple-with-metadata output."""
    detections: list[dict] = []

    if value is None:
        return detections

    if hasattr(value, "model_dump"):
        value = value.model_dump()
    elif hasattr(value, "dict") and not isinstance(value, dict):
        value = value.dict()

    if isinstance(value, dict):
        has_box = any(key in value for key in ("box", "bbox", "xyxy", "coordinates", "rect"))
        has_class = any(key in value for key in ("class_id", "class_name", "label", "name", "class"))
        if has_box and has_class:
            return [value]
        for key in ("detections", "icons", "results", "predictions", "items", "output"):
            if key in value:
                detections.extend(_collect_detection_dicts(value[key]))
        return detections

    if isinstance(value, (list, tuple)):
        if len(value) >= 6 and all(
            isinstance(item, (int, float, str, np.integer, np.floating))
            for item in value[:6]
        ):
            return [{
                "box": list(value[:4]),
                "confidence": value[4],
                "class_name": value[5],
            }]
        for item in value:
            detections.extend(_collect_detection_dicts(item))
        return detections

    if hasattr(value, "__dict__"):
        return _collect_detection_dicts(vars(value))

    return detections


def normalize_predictions(raw: Any) -> list[dict]:
    items = _collect_detection_dicts(raw)
    return [normalize_icon(item, index) for index, item in enumerate(items)]

def mask_icons(image: np.ndarray, icons: list[dict], padding: int = 2) -> np.ndarray:
    masked = image.copy()
    h, w = masked.shape[:2]
    # Use the median display color around each icon instead of pure black.
    for icon in icons:
        x1, y1, x2, y2 = icon["box"]
        x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
        x2, y2 = min(w - 1, x2 + padding), min(h - 1, y2 + padding)
        rings = []
        if y1 > 0: rings.append(masked[max(0, y1-3):y1, x1:x2+1])
        if y2 + 1 < h: rings.append(masked[y2+1:min(h, y2+4), x1:x2+1])
        if x1 > 0: rings.append(masked[y1:y2+1, max(0, x1-3):x1])
        if x2 + 1 < w: rings.append(masked[y1:y2+1, x2+1:min(w, x2+4)])
        pixels = [r.reshape(-1, 3) for r in rings if r.size]
        color = np.median(np.concatenate(pixels), axis=0).astype(np.uint8) if pixels else np.array([0, 0, 0], np.uint8)
        masked[y1:y2+1, x1:x2+1] = color
    return masked


def draw_icons(image: np.ndarray, icons: list[dict]) -> np.ndarray:
    rendered = image.copy()
    for icon in icons:
        x1, y1, x2, y2 = icon["box"]
        label = f"{icon['class_name']} {icon['confidence']:.2f}"
        cv2.rectangle(rendered, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.putText(rendered, label, (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 0, 255), 1, cv2.LINE_AA)
    return rendered


def find_ocr_visual(ocr_dir: Path, name: str) -> Path | None:
    exact = ocr_dir / name
    if exact.is_file():
        return exact
    stem = Path(name).stem
    matches = [p for p in ocr_dir.rglob("*") if p.is_file() and p.stem == stem]
    return matches[0] if matches else None


def main():
    parser = argparse.ArgumentParser(description="Run icon masking plus PP-OCRv6 OCR on all validation images.")
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--icon-threshold", type=float, default=0.38)
    args = parser.parse_args()

    for path in (ICON_MODULE, PADDLE_PYTHON, DET_MODEL, REC_MODEL, REC_DICT, args.images):
        if not path.exists():
            raise FileNotFoundError(path)

    images = sorted(p for p in args.images.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED)
    if not images:
        raise FileNotFoundError(f"No images in {args.images}")
    print(f"Validation images found: {len(images)}")

    output = args.output.resolve()
    if output.exists():
        shutil.rmtree(output)
    originals = output / "originals"
    masked_dir = output / "masked"
    ocr_dir = output / "ocr_rendered"
    rendered_dir = output / "rendered"
    json_dir = output / "json"
    for folder in (originals, masked_dir, ocr_dir, rendered_dir, json_dir):
        folder.mkdir(parents=True, exist_ok=True)

    detector = build_icon_detector()
    predictions, failures = [], []
    stage_times = {"icon_ms": [], "mask_ms": []}

    for index, image_path in enumerate(images, 1):
        try:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("OpenCV could not read image")
            shutil.copy2(image_path, originals / image_path.name)
            start = time.perf_counter()
            raw = detector.predict(image)
            icon_ms = (time.perf_counter() - start) * 1000
            icons = [i for i in normalize_predictions(raw) if i["confidence"] >= args.icon_threshold]
            start = time.perf_counter()
            masked = mask_icons(image, icons)
            mask_ms = (time.perf_counter() - start) * 1000
            cv2.imwrite(str(masked_dir / image_path.name), masked)
            record = {
                "image": image_path.name,
                "status": "icon_stage_complete",
                "icon_occurrences": icons,
                "timing_ms": {"icon_detection": icon_ms, "icon_masking": mask_ms},
            }
            predictions.append(record)
            (json_dir / f"{image_path.stem}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
            stage_times["icon_ms"].append(icon_ms)
            stage_times["mask_ms"].append(mask_ms)
            print(f"[{index}/{len(images)}] {image_path.name}: {len(icons)} icons")
        except Exception as error:
            failures.append({"image": image_path.name, "stage": "icon_or_mask", "error": str(error), "traceback": traceback.format_exc()})
            print(f"[{index}/{len(images)}] FAILED {image_path.name}: {error}")

    command = [
        str(PADDLE_PYTHON), str(PADDLE_REPO / "tools" / "infer" / "predict_system.py"),
        "--image_dir", str(masked_dir),
        "--det_model_dir", str(DET_MODEL),
        "--rec_model_dir", str(REC_MODEL),
        "--rec_char_dict_path", str(REC_DICT),
        "--draw_img_save_dir", str(ocr_dir),
        "--use_gpu", "false", "--use_angle_cls", "false", "--use_space_char", "false",
        "--det_algorithm", "DB", "--rec_algorithm", "SVTR_LCNet",
        "--det_limit_type", "max", "--det_limit_side_len", "1280",
        "--det_db_thresh", "0.20", "--det_db_box_thresh", "0.45",
        "--det_db_unclip_ratio", "1.40", "--drop_score", "0.0", "--show_log", "true",
    ]
    ocr_start = time.perf_counter()
    process = subprocess.run(command, cwd=str(PADDLE_REPO), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, encoding="utf-8", errors="replace")
    ocr_total_ms = (time.perf_counter() - ocr_start) * 1000
    (output / "pipeline_console.txt").write_text(process.stdout, encoding="utf-8")
    print(process.stdout)
    if process.returncode != 0:
        raise RuntimeError(f"OCR stage failed with exit code {process.returncode}")

    record_by_name = {r["image"]: r for r in predictions}
    for image_path in images:
        if image_path.name not in record_by_name:
            continue
        visual = find_ocr_visual(ocr_dir, image_path.name)
        if visual is None:
            failures.append({"image": image_path.name, "stage": "render", "error": "OCR visual not found"})
            continue
        ocr_image = cv2.imread(str(visual))
        final = draw_icons(ocr_image, record_by_name[image_path.name]["icon_occurrences"])
        cv2.imwrite(str(rendered_dir / image_path.name), final)
        record_by_name[image_path.name]["status"] = "success"
        record_by_name[image_path.name]["rendered"] = str(rendered_dir / image_path.name)

    successful = sum(r.get("status") == "success" for r in predictions)
    summary = {
        "total_validation_images": len(images),
        "icon_stage_completed": len(predictions),
        "successful_rendered": successful,
        "failed": len(failures),
        "total_icons": sum(len(r["icon_occurrences"]) for r in predictions),
        "average_icon_detection_ms": sum(stage_times["icon_ms"]) / len(stage_times["icon_ms"]) if stage_times["icon_ms"] else None,
        "average_icon_masking_ms": sum(stage_times["mask_ms"]) / len(stage_times["mask_ms"]) if stage_times["mask_ms"] else None,
        "ocr_total_ms": ocr_total_ms,
        "detector_model": str(DET_MODEL),
        "recognizer_model": str(REC_MODEL),
        "icon_threshold": args.icon_threshold,
    }
    (output / "predictions.json").write_text(json.dumps(predictions, indent=2), encoding="utf-8")
    (output / "failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    subprocess.Popen(["explorer.exe", str(rendered_dir)])


if __name__ == "__main__":
    main()
