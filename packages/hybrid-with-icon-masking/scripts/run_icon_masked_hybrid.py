from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def normalize_name(value: str) -> str:
    return Path(value.replace("\\", "/")).name


def load_icon_results(path: Path) -> dict[str, dict]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, list):
        raise ValueError("Icon detections JSON must contain a list of image records.")
    return {normalize_name(item["image"]): item for item in data}


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(value, high))


def mask_icons(
    source_dir: Path,
    icon_results: dict[str, dict],
    output_dir: Path,
    padding: int,
    fill_value: int,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    total_masks = 0

    image_paths = sorted(
        item for item in source_dir.iterdir()
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise RuntimeError(f"No supported images found in: {source_dir}")

    for index, image_path in enumerate(image_paths, start=1):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            records.append({"image": image_path.name, "status": "decode_failed", "mask_count": 0})
            continue

        height, width = image.shape[:2]
        icon_record = icon_results.get(image_path.name, {"detections": []})
        detections = icon_record.get("detections", [])
        applied = []

        for detection in detections:
            box = detection.get("box")
            if not box or len(box) != 4:
                continue

            x1 = clamp(int(round(float(box[0]))) - padding, 0, width - 1)
            y1 = clamp(int(round(float(box[1]))) - padding, 0, height - 1)
            x2 = clamp(int(round(float(box[2]))) + padding, 0, width - 1)
            y2 = clamp(int(round(float(box[3]))) + padding, 0, height - 1)

            if x2 <= x1 or y2 <= y1:
                continue

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (fill_value, fill_value, fill_value),
                thickness=-1,
            )

            applied.append({
                "class_id": detection.get("class_id"),
                "class_name": detection.get("class_name"),
                "confidence": detection.get("confidence"),
                "original_box": box,
                "masked_box": [x1, y1, x2, y2],
            })

        destination = output_dir / image_path.name
        if not cv2.imwrite(str(destination), image):
            raise RuntimeError(f"Failed to write masked image: {destination}")

        total_masks += len(applied)
        records.append({
            "image": image_path.name,
            "status": "ok",
            "width": width,
            "height": height,
            "mask_count": len(applied),
            "masks": applied,
            "masked_image": str(destination),
        })
        print(f"[{index}/{len(image_paths)}] {image_path.name}: masked {len(applied)} icon regions")

    return {
        "images_requested": len(image_paths),
        "images_completed": sum(record["status"] == "ok" for record in records),
        "total_icon_masks": total_masks,
        "padding_pixels": padding,
        "fill_value": fill_value,
        "records": records,
    }


def run_ocr(args: argparse.Namespace, masked_dir: Path, rendered_dir: Path, log_path: Path) -> int:
    rendered_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        str(args.ocr_python),
        str(args.paddle_root / "tools" / "infer" / "predict_system.py"),
        "--image_dir", str(masked_dir),
        "--det_model_dir", str(args.det_model),
        "--rec_model_dir", str(args.rec_model),
        "--rec_char_dict_path", str(args.dictionary),
        "--use_angle_cls", "false",
        "--use_gpu", "false",
        "--drop_score", str(args.drop_score),
        "--vis_font_path", str(args.font),
        "--draw_img_save_dir", str(rendered_dir),
        "--save_log_path", str(rendered_dir / "system_results.txt"),
    ]

    print()
    print("Running hybrid OCR on icon-masked images...")
    print(" ".join(f'"{part}"' if " " in part else part for part in command))
    print()

    with log_path.open("w", encoding="utf-8", newline="\n") as log_file:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        return process.wait()


def validate_paths(args: argparse.Namespace) -> None:
    required_files = [
        args.icon_results,
        args.ocr_python,
        args.paddle_root / "tools" / "infer" / "predict_system.py",
        args.dictionary,
        args.font,
    ]
    required_dirs = [
        args.images,
        args.det_model,
        args.rec_model,
    ]

    for path in required_files:
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")
    for path in required_dirs:
        if not path.is_dir():
            raise FileNotFoundError(f"Required folder not found: {path}")


def main() -> None:
    clean_root = Path(r"D:\Actual Project\Indali_Lcd-Data-Extractor")
    hybrid_package = clean_root / "packages" / "hybrid-model-2-detector-model-3-recognizer"

    parser = argparse.ArgumentParser(
        description="Mask every verified icon detection, then run the Model 2 detector plus Model 3 recognizer and save visual OCR results."
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=hybrid_package / "data" / "validation" / "images",
    )
    parser.add_argument(
        "--icon-results",
        type=Path,
        default=clean_root / "shared" / "icon-detector-evaluation" / "reports" / "icon_detections.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=clean_root / "shared" / "hybrid-with-icon-masking",
    )
    parser.add_argument(
        "--ocr-python",
        type=Path,
        default=Path(r"D:\Actual Project\lcd-data-extractor\paddle_train_env\Scripts\python.exe"),
    )
    parser.add_argument(
        "--paddle-root",
        type=Path,
        default=Path(r"D:\Actual Project\PaddleOCR-2.8"),
    )
    parser.add_argument(
        "--det-model",
        type=Path,
        default=hybrid_package / "models" / "detector" / "inference",
    )
    parser.add_argument(
        "--rec-model",
        type=Path,
        default=hybrid_package / "models" / "recognizer" / "inference",
    )
    parser.add_argument(
        "--dictionary",
        type=Path,
        default=hybrid_package / "models" / "recognizer" / "character_dict.txt",
    )
    parser.add_argument(
        "--font",
        type=Path,
        default=Path(r"C:\Windows\Fonts\arial.ttf"),
    )
    parser.add_argument("--padding", type=int, default=2)
    parser.add_argument("--fill-value", type=int, default=0)
    parser.add_argument("--drop-score", type=float, default=0.3)
    parser.add_argument("--keep-existing", action="store_true")
    args = parser.parse_args()

    if not 0 <= args.fill_value <= 255:
        raise ValueError("fill-value must be between 0 and 255")
    if args.padding < 0:
        raise ValueError("padding must be zero or greater")

    validate_paths(args)

    masked_dir = args.output / "masked_images"
    rendered_dir = args.output / "ocr_visual_results"
    reports_dir = args.output / "reports"

    if args.output.exists() and not args.keep_existing:
        shutil.rmtree(args.output)

    masked_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    icon_results = load_icon_results(args.icon_results)
    mask_summary = mask_icons(
        source_dir=args.images,
        icon_results=icon_results,
        output_dir=masked_dir,
        padding=args.padding,
        fill_value=args.fill_value,
    )

    (reports_dir / "masking_summary.json").write_text(
        json.dumps(mask_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    exit_code = run_ocr(
        args=args,
        masked_dir=masked_dir,
        rendered_dir=rendered_dir,
        log_path=reports_dir / "ocr_console.txt",
    )

    rendered_images = [
        item for item in rendered_dir.iterdir()
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
    ] if rendered_dir.exists() else []

    final_summary = {
        "mode": "icon_regions_masked_before_ocr",
        "source_images": mask_summary["images_requested"],
        "masked_images": mask_summary["images_completed"],
        "icon_regions_hidden": mask_summary["total_icon_masks"],
        "mask_padding_pixels": args.padding,
        "mask_fill_value": args.fill_value,
        "ocr_exit_code": exit_code,
        "ocr_visual_result_images": len(rendered_images),
        "masked_images_folder": str(masked_dir),
        "ocr_visual_results_folder": str(rendered_dir),
        "ocr_results_file": str(rendered_dir / "system_results.txt"),
        "ocr_console_log": str(reports_dir / "ocr_console.txt"),
    }
    (reports_dir / "summary.json").write_text(
        json.dumps(final_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print(json.dumps(final_summary, indent=2, ensure_ascii=False))

    if exit_code != 0:
        raise SystemExit(f"OCR inference failed with exit code {exit_code}. Review {reports_dir / 'ocr_console.txt'}")

    print()
    print("Combined run completed successfully.")
    print(f"Masked source images: {masked_dir}")
    print(f"OCR visual results: {rendered_dir}")
    print(f"Summary: {reports_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
