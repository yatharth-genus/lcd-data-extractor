from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

DEFAULT_ZIP = Path.home() / "Downloads" / "ppocrv6_detector_best.zip"
MODEL4 = Path(r"D:\Actual Project\Indali_Lcd-Data-Extractor\packages\model-4-ppocrv6-small-baseline")
DESTINATION = MODEL4 / "models" / "detector"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def locate_result_root(extracted: Path) -> Path:
    candidates = [extracted / "ppocrv6_detector_best", extracted]
    candidates.extend(path for path in extracted.rglob("ppocrv6_detector_best") if path.is_dir())
    for path in candidates:
        if (path / "best_metrics.json").is_file() and (path / "inference").is_dir():
            return path
    raise FileNotFoundError("Could not locate ppocrv6_detector_best in the downloaded ZIP.")


def validate(root: Path) -> tuple[dict, Path, list[Path]]:
    metrics_path = root / "best_metrics.json"
    checkpoint = root / "checkpoint" / "best_accuracy.pdparams"
    inference = root / "inference"
    config_candidates = list(root.glob("PP-OCRv6_small_det_lcd_finetune.y*ml"))

    for required in (metrics_path, checkpoint, inference):
        if not required.exists():
            raise FileNotFoundError(required)

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    expected = {
        "precision": 0.7112068965517241,
        "recall": 0.8823529411764706,
        "hmean": 0.7875894988066825,
        "best_epoch": 40,
    }
    for key, value in expected.items():
        if key not in metrics:
            raise KeyError(f"Metric missing: {key}")
        if key == "best_epoch":
            if int(metrics[key]) != value:
                raise ValueError(f"Unexpected {key}: {metrics[key]}")
        elif abs(float(metrics[key]) - value) > 1e-9:
            raise ValueError(f"Unexpected {key}: {metrics[key]}")

    inference_files = [path for path in inference.rglob("*") if path.is_file()]
    if not inference_files:
        raise FileNotFoundError("The exported inference folder is empty.")
    if checkpoint.stat().st_size < 1_000_000:
        raise ValueError("The best detector checkpoint is unexpectedly small.")
    if any(path.stat().st_size == 0 for path in inference_files):
        raise ValueError("A zero-byte inference file was found.")

    return metrics, config_candidates[0] if config_candidates else None, inference_files


def backup_destination() -> Path | None:
    if not DESTINATION.exists() or not any(DESTINATION.rglob("*")):
        return None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = MODEL4 / "backups" / f"detector_before_colab_import_{timestamp}"
    shutil.copytree(DESTINATION, backup)
    return backup


def main() -> None:
    parser = argparse.ArgumentParser(description="Import PP-OCRv6 detector Colab result into Model 4.")
    parser.add_argument("--source", type=Path, default=DEFAULT_ZIP)
    args = parser.parse_args()
    source = args.source.resolve()

    if not source.is_file() or not zipfile.is_zipfile(source):
        raise FileNotFoundError(f"Valid result ZIP not found: {source}")

    with tempfile.TemporaryDirectory(prefix="ppocrv6_detector_import_") as temporary:
        extracted = Path(temporary)
        with zipfile.ZipFile(source, "r") as archive:
            archive.extractall(extracted)
        result_root = locate_result_root(extracted)
        metrics, config, inference_files = validate(result_root)

        backup = backup_destination()
        if DESTINATION.exists():
            shutil.rmtree(DESTINATION)
        (DESTINATION / "training").mkdir(parents=True)
        (DESTINATION / "inference").mkdir(parents=True)

        shutil.copytree(result_root / "checkpoint", DESTINATION / "training", dirs_exist_ok=True)
        shutil.copytree(result_root / "inference", DESTINATION / "inference", dirs_exist_ok=True)

        config_destination = None
        if config:
            config_destination = MODEL4 / "config" / "PP-OCRv6_small_det_lcd_finetune.yml"
            config_destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(config, config_destination)

        log_source = result_root / "detector_training_console.txt"
        log_destination = None
        if log_source.is_file():
            log_destination = MODEL4 / "logs" / "detector_training_console_colab.txt"
            log_destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(log_source, log_destination)

    imported_checkpoint = DESTINATION / "training" / "best_accuracy.pdparams"
    imported_inference = [path for path in (DESTINATION / "inference").rglob("*") if path.is_file()]
    report = {
        "imported_at": datetime.now().isoformat(timespec="seconds"),
        "source_zip": str(source),
        "metrics": metrics,
        "checkpoint": str(imported_checkpoint),
        "checkpoint_bytes": imported_checkpoint.stat().st_size,
        "checkpoint_sha256": sha256(imported_checkpoint),
        "inference_file_count": len(imported_inference),
        "inference_files": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in imported_inference
        ],
        "config": str(config_destination) if config_destination else None,
        "training_log": str(log_destination) if log_destination else None,
        "backup": str(backup) if backup else None,
    }
    report_path = MODEL4 / "docs" / "colab_detector_import_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Detector import completed successfully.")
    print(f"Precision: {metrics['precision']:.6f}")
    print(f"Recall: {metrics['recall']:.6f}")
    print(f"Hmean: {metrics['hmean']:.6f}")
    print(f"Best epoch: {metrics['best_epoch']}")
    print(f"Checkpoint: {imported_checkpoint}")
    print(f"Inference files: {len(imported_inference)}")
    print(f"Report: {report_path}")
    if backup:
        print(f"Previous detector backed up to: {backup}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nIMPORT FAILED: {error}", file=sys.stderr)
        raise
