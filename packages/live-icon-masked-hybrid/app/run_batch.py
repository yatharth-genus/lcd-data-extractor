from __future__ import annotations
import argparse
import json
from pathlib import Path
from time import perf_counter
import cv2
from app.main import (
    IconDetector, OcrEngine,
    ICON_MODEL_PATH, ICON_LABELS_PATH, ICON_TRANSFORMS_PATH, ICON_TRAIN_CONFIG_PATH,
    OCR_DETECTOR_MODEL_PATH, OCR_RECOGNIZER_MODEL_PATH, OCR_CHARACTER_DICTIONARY_PATH,
    mask_detected_icons, render_combined_result,
)
EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}

def main():
    parser = argparse.ArgumentParser(description="Run live icon detection, masking, and OCR.")
    parser.add_argument("images", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rendered = args.output / "rendered"
    rendered.mkdir(exist_ok=True)
    icon = IconDetector(ICON_MODEL_PATH, ICON_LABELS_PATH, ICON_TRANSFORMS_PATH, ICON_TRAIN_CONFIG_PATH, 0.38, "CPU")
    ocr = OcrEngine(OCR_DETECTOR_MODEL_PATH, OCR_RECOGNIZER_MODEL_PATH, OCR_CHARACTER_DICTIONARY_PATH)
    images = [args.images] if args.images.is_file() else sorted(
        item for item in args.images.iterdir()
        if item.is_file() and item.suffix.lower() in EXTENSIONS
    )
    records = []
    for number, path in enumerate(images, start=1):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            records.append({"image": path.name, "status": "decode_failed"})
            continue
        started = perf_counter()
        raw_icons, icon_ms = icon.predict(image)
        icons = [{**item, "id": f"icon_{i:03d}", "detection_order": i, "masked_before_ocr": True} for i, item in enumerate(raw_icons, 1)]
        masked, masking_ms = mask_detected_icons(image, icons)
        raw_text, ocr_ms = ocr.predict(masked)
        texts = [{**item, "id": f"text_{i:03d}", "reading_order": i} for i, item in enumerate(raw_text, 1)]
        visual, rendering_ms = render_combined_result(image, icons, texts)
        visual_path = rendered / f"{path.stem}_result.png"
        cv2.imwrite(str(visual_path), visual)
        records.append({
            "image": path.name,
            "status": "success",
            "text_occurrences": texts,
            "icon_occurrences": icons,
            "timing_ms": {
                "icon_detection": icon_ms,
                "icon_masking": masking_ms,
                "ocr_detection_and_recognition": ocr_ms,
                "rendering": rendering_ms,
                "total": (perf_counter() - started) * 1000,
            },
            "rendered_result": str(visual_path),
        })
        print(f"[{number}/{len(images)}] {path.name}: {len(icons)} icons, {len(texts)} text occurrences")
    (args.output / "predictions.json").write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Completed {len(records)} records. Output: {args.output}")

if __name__ == "__main__":
    main()
