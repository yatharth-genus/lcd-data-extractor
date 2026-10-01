from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import openvino as ov
import yaml


ROOT = Path(__file__).resolve().parent

DEFAULT_MODEL = ROOT / "model" / "saved_model.xml"
DEFAULT_LABELS = ROOT / "dm.json"
DEFAULT_TRANSFORMS = ROOT / "model_meta" / "transforms.yaml"
DEFAULT_TRAIN_CONFIG = ROOT / "model_meta" / "train.yaml"

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}

COLORS = {
    "earth": (230, 50, 230),
    "magnet": (0, 165, 255),
    "comm": (255, 220, 0),
    "switchON": (255, 100, 0),
    "switchOFF": (0, 0, 255),
    "battery": (0, 128, 255),
    "reverseCN": (50, 50, 220),
    "ForwardCN": (0, 0, 180),
    "neutral": (0, 230, 100),
    "powerON": (0, 215, 255),
    "coverOpen": (100, 255, 100),
}


def load_labels(path: Path) -> dict[int, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        int(class_id): class_data["name"]
        for class_id, class_data in data.items()
    }


def load_preprocessing(
    transforms_path: Path,
    train_config_path: Path,
) -> dict[str, Any]:
    transforms = yaml.safe_load(
        transforms_path.read_text(encoding="utf-8")
    )

    train_config = yaml.safe_load(
        train_config_path.read_text(encoding="utf-8")
    )

    valid_transforms = transforms["valid"]

    resize_config = next(
        item["RescaleWithPadding"]
        for item in valid_transforms
        if "RescaleWithPadding" in item
    )

    normalize_config = next(
        item["NormalizeMeanStd"]
        for item in valid_transforms
        if "NormalizeMeanStd" in item
    )

    return {
        "width": int(resize_config["width"]),
        "height": int(resize_config["height"]),
        "mean": np.asarray(
            normalize_config["mean"],
            dtype=np.float32,
        ),
        "std": np.asarray(
            normalize_config["std"],
            dtype=np.float32,
        ),
        "threshold": float(
            train_config["model"].get(
                "score_threshold",
                0.38,
            )
        ),
    }


def collect_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(
                f"Unsupported image extension: {input_path}"
            )

        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(
            f"Input image or folder not found: {input_path}"
        )

    images = sorted(
        item
        for item in input_path.iterdir()
        if item.is_file()
        and item.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not images:
        raise RuntimeError(
            f"No supported images found in: {input_path}"
        )

    return images


def prepare_image(
    image_bgr: np.ndarray,
    config: dict[str, Any],
    pad_position: str,
) -> tuple[np.ndarray, dict[str, float | int]]:
    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB,
    )

    source_height, source_width = image_rgb.shape[:2]

    target_width = config["width"]
    target_height = config["height"]

    scale = min(
        target_width / source_width,
        target_height / source_height,
    )

    resized_width = max(
        1,
        int(round(source_width * scale)),
    )

    resized_height = max(
        1,
        int(round(source_height * scale)),
    )

    resized = cv2.resize(
        image_rgb,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )

    padded = np.zeros(
        (target_height, target_width, 3),
        dtype=np.uint8,
    )

    if pad_position == "top_left":
        top = 0
        left = 0
    else:
        top = (target_height - resized_height) // 2
        left = (target_width - resized_width) // 2

    padded[
        top : top + resized_height,
        left : left + resized_width,
    ] = resized

    normalized = (
        padded.astype(np.float32) - config["mean"]
    ) / config["std"]

    tensor = np.transpose(
        normalized,
        (2, 0, 1),
    )[None, ...].astype(
        np.float32,
        copy=False,
    )

    geometry = {
        "source_width": source_width,
        "source_height": source_height,
        "scale": float(scale),
        "pad_left": left,
        "pad_top": top,
        "resized_width": resized_width,
        "resized_height": resized_height,
    }

    return tensor, geometry


def restore_box(
    box: np.ndarray,
    geometry: dict[str, float | int],
) -> list[int]:
    scale = float(geometry["scale"])
    left = int(geometry["pad_left"])
    top = int(geometry["pad_top"])

    source_width = int(geometry["source_width"])
    source_height = int(geometry["source_height"])

    xmin = int(round((float(box[0]) - left) / scale))
    ymin = int(round((float(box[1]) - top) / scale))
    xmax = int(round((float(box[2]) - left) / scale))
    ymax = int(round((float(box[3]) - top) / scale))

    xmin = max(0, min(xmin, source_width - 1))
    ymin = max(0, min(ymin, source_height - 1))
    xmax = max(0, min(xmax, source_width - 1))
    ymax = max(0, min(ymax, source_height - 1))

    if xmax < xmin:
        xmin, xmax = xmax, xmin

    if ymax < ymin:
        ymin, ymax = ymax, ymin

    return [xmin, ymin, xmax, ymax]


def draw_detections(
    image_bgr: np.ndarray,
    detections: list[dict[str, Any]],
) -> np.ndarray:
    output = image_bgr.copy()

    for detection in detections:
        xmin, ymin, xmax, ymax = detection["box"]

        class_name = detection["class_name"]
        confidence = detection["confidence"]

        color = COLORS.get(
            class_name,
            (128, 128, 128),
        )

        cv2.rectangle(
            output,
            (xmin, ymin),
            (xmax, ymax),
            color,
            2,
        )

        label = f"{class_name} {confidence:.3f}"

        label_y = max(15, ymin - 5)

        cv2.putText(
            output,
            label,
            (xmin, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )

        cv2.putText(
            output,
            label,
            (xmin, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )

    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the OpenVINO LCD icon detector on an image "
            "or a folder of images."
        )
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Input image or folder.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "icon_detector_results",
        help="Output directory.",
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="OpenVINO XML model.",
    )

    parser.add_argument(
        "--labels",
        type=Path,
        default=DEFAULT_LABELS,
        help="Class mapping JSON.",
    )

    parser.add_argument(
        "--transforms",
        type=Path,
        default=DEFAULT_TRANSFORMS,
        help="Preprocessing YAML.",
    )

    parser.add_argument(
        "--train-config",
        type=Path,
        default=DEFAULT_TRAIN_CONFIG,
        help="Training metadata YAML.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Confidence threshold override.",
    )

    parser.add_argument(
        "--device",
        default="CPU",
        help="OpenVINO device, normally CPU.",
    )

    parser.add_argument(
        "--pad-position",
        choices=("center", "top_left"),
        default="center",
        help="Letterbox padding position.",
    )

    parser.add_argument(
        "--include-ok",
        action="store_true",
        help="Include class 0, ok.",
    )

    args = parser.parse_args()

    for required_path in (
        args.model,
        args.labels,
        args.transforms,
        args.train_config,
    ):
        if not required_path.exists():
            raise FileNotFoundError(required_path)

    images = collect_images(args.input)

    labels = load_labels(args.labels)

    config = load_preprocessing(
        args.transforms,
        args.train_config,
    )

    threshold = (
        config["threshold"]
        if args.threshold is None
        else float(args.threshold)
    )

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    rendered_folder = args.output / "rendered"
    reports_folder = args.output / "reports"

    rendered_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    reports_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    core = ov.Core()

    load_started = time.perf_counter()

    model = core.read_model(args.model)

    compiled_model = core.compile_model(
        model,
        args.device,
    )

    load_seconds = time.perf_counter() - load_started

    input_port = compiled_model.input(0)
    input_name = input_port.get_any_name()

    output_names = [
        output.get_any_name()
        for output in compiled_model.outputs
    ]

    if "dets" not in output_names:
        raise RuntimeError(
            f"Expected output 'dets'. Found: {output_names}"
        )

    if "labels" not in output_names:
        raise RuntimeError(
            f"Expected output 'labels'. Found: {output_names}"
        )

    warmup = np.zeros(
        (1, 3, config["height"], config["width"]),
        dtype=np.float32,
    )

    compiled_model({input_name: warmup})

    all_records: list[dict[str, Any]] = []
    class_counts: Counter[str] = Counter()

    total_inference_seconds = 0.0

    for image_index, image_path in enumerate(images, start=1):
        image_bgr = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR,
        )

        if image_bgr is None:
            all_records.append(
                {
                    "image": image_path.name,
                    "status": "decode_failed",
                    "detections": [],
                }
            )

            continue

        tensor, geometry = prepare_image(
            image_bgr,
            config,
            args.pad_position,
        )

        inference_started = time.perf_counter()

        raw_outputs = compiled_model(
            {input_name: tensor}
        )

        inference_seconds = (
            time.perf_counter() - inference_started
        )

        total_inference_seconds += inference_seconds

        output_map = {
            port.get_any_name(): np.asarray(value)
            for port, value in raw_outputs.items()
        }

        raw_detections = output_map["dets"][0]
        raw_labels = output_map["labels"][0]

        detections: list[dict[str, Any]] = []

        for raw_detection, raw_label in zip(
            raw_detections,
            raw_labels,
            strict=False,
        ):
            score = float(raw_detection[4])
            class_id = int(raw_label)

            if score < threshold:
                continue

            if class_id == 0 and not args.include_ok:
                continue

            class_name = labels.get(
                class_id,
                str(class_id),
            )

            box = restore_box(
                raw_detection[:4],
                geometry,
            )

            xmin, ymin, xmax, ymax = box

            if xmax <= xmin or ymax <= ymin:
                continue

            detection = {
                "class_id": class_id,
                "class_name": class_name,
                "confidence": round(score, 6),
                "box": box,
                "box_format": "xyxy",
            }

            detections.append(detection)
            class_counts[class_name] += 1

        detections.sort(
            key=lambda item: item["confidence"],
            reverse=True,
        )

        rendered = draw_detections(
            image_bgr,
            detections,
        )

        rendered_path = (
            rendered_folder
            / f"{image_path.stem}_icons.png"
        )

        cv2.imwrite(
            str(rendered_path),
            rendered,
        )

        record = {
            "image": image_path.name,
            "source_path": str(image_path.resolve()),
            "width": int(image_bgr.shape[1]),
            "height": int(image_bgr.shape[0]),
            "status": "ok",
            "inference_ms": round(
                inference_seconds * 1000,
                3,
            ),
            "threshold": threshold,
            "pad_position": args.pad_position,
            "detection_count": len(detections),
            "detections": detections,
            "rendered_file": rendered_path.name,
        }

        all_records.append(record)

        print(
            f"[{image_index}/{len(images)}] "
            f"{image_path.name}: "
            f"{len(detections)} detections, "
            f"{record['inference_ms']:.3f} ms"
        )

    combined_json = reports_folder / "icon_detections.json"

    combined_json.write_text(
        json.dumps(
            all_records,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    jsonl_path = reports_folder / "icon_detections.jsonl"

    with jsonl_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as jsonl_file:
        for record in all_records:
            jsonl_file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    completed_records = [
        record
        for record in all_records
        if record["status"] == "ok"
    ]

    total_detections = sum(
        record["detection_count"]
        for record in completed_records
    )

    average_inference_ms = (
        total_inference_seconds
        / len(completed_records)
        * 1000
        if completed_records
        else 0.0
    )

    summary = {
        "model": str(args.model.resolve()),
        "device": args.device,
        "input_name": input_name,
        "input_shape": [
            1,
            3,
            config["height"],
            config["width"],
        ],
        "output_names": output_names,
        "threshold": threshold,
        "pad_position": args.pad_position,
        "include_ok": args.include_ok,
        "model_load_seconds": round(
            load_seconds,
            4,
        ),
        "images_requested": len(images),
        "images_completed": len(completed_records),
        "total_detections": total_detections,
        "average_inference_ms": round(
            average_inference_ms,
            3,
        ),
        "class_counts": dict(
            sorted(class_counts.items())
        ),
    }

    summary_path = reports_folder / "summary.json"

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("Icon detector run completed.")
    print(f"Images completed: {len(completed_records)}")
    print(f"Total detections: {total_detections}")
    print(
        "Average inference time: "
        f"{average_inference_ms:.3f} ms"
    )
    print(f"Rendered images: {rendered_folder}")
    print(f"JSON report: {combined_json}")
    print(f"JSONL report: {jsonl_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
