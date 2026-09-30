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

DEFAULT_DOWNLOADS = Path.home() / "Downloads"
DEFAULT_MODEL4 = Path(
    r"D:\Actual Project\Indali_Lcd-Data-Extractor"
    r"\packages\model-4-ppocrv6-small-baseline"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def newest(paths: list[Path]) -> Path | None:
    valid = [path for path in paths if path.exists()]
    if not valid:
        return None
    return max(valid, key=lambda path: path.stat().st_mtime)


def discover_source(downloads: Path) -> Path:
    exact_candidates = [
        downloads / "ppocrv6_colab.zip",
        downloads / "ppocrv6_colab",
        downloads / "PP-OCRv6_LCD_Recognizer_Finetuning.zip",
    ]
    exact = newest(exact_candidates)
    if exact:
        return exact

    candidates: list[Path] = []
    for pattern in (
        "*ppocrv6*colab*.zip",
        "*PP-OCRv6*.zip",
        "*ppocrv6*colab*",
        "*PP-OCRv6*",
    ):
        candidates.extend(downloads.glob(pattern))

    candidates = [
        path
        for path in candidates
        if path.is_dir() or path.suffix.lower() == ".zip"
    ]
    source = newest(candidates)
    if source is None:
        raise FileNotFoundError(
            "No Colab result ZIP or folder was found in Downloads.\n"
            "Download the complete ppocrv6_colab folder from Google Drive "
            "as a ZIP, then run this script again."
        )
    return source


def extract_if_needed(source: Path, temporary_root: Path) -> Path:
    if source.is_dir():
        return source
    if not zipfile.is_zipfile(source):
        raise RuntimeError(f"Not a valid ZIP file: {source}")
    extract_root = temporary_root / "extracted"
    extract_root.mkdir(parents=True, exist_ok=True)
    print(f"Extracting: {source}")
    with zipfile.ZipFile(source, "r") as archive:
        archive.extractall(extract_root)
    return extract_root


def find_package_root(search_root: Path) -> Path:
    direct_candidates = [
        search_root,
        search_root / "ppocrv6_colab",
    ]
    for candidate in direct_candidates:
        if (candidate / "output").is_dir() or (candidate / "exported").is_dir():
            return candidate

    candidates = []
    for candidate in search_root.rglob("ppocrv6_colab"):
        if candidate.is_dir() and (
            (candidate / "output").is_dir()
            or (candidate / "exported").is_dir()
        ):
            candidates.append(candidate)

    if not candidates:
        # Some browser ZIP downloads omit the top-level folder name.
        for candidate in [search_root, *search_root.iterdir()]:
            if candidate.is_dir() and (
                (candidate / "output" / "recognizer").is_dir()
                or (candidate / "exported").is_dir()
            ):
                candidates.append(candidate)

    if not candidates:
        raise FileNotFoundError(
            "Could not find the Colab result structure. Expected folders such as:\n"
            "  ppocrv6_colab/output/recognizer\n"
            "  ppocrv6_colab/exported/PP-OCRv6_small_rec_lcd"
        )
    return newest(candidates) or candidates[0]


def first_existing(candidates: list[Path]) -> Path | None:
    for path in candidates:
        if path.exists():
            return path
    return None


def locate_assets(package_root: Path) -> dict[str, Path | None]:
    training = first_existing([
        package_root / "output" / "recognizer",
        package_root / "recognizer",
    ])

    inference = first_existing([
        package_root / "exported" / "PP-OCRv6_small_rec_lcd",
        package_root / "output" / "recognizer" / "inference",
        package_root / "exported",
    ])

    config = first_existing([
        package_root / "config" / "PP-OCRv6_small_rec_lcd_finetune.yml",
        package_root / "config" / "PP-OCRv6_small_rec_lcd_finetune.yaml",
    ])

    training_log = first_existing([
        package_root / "output" / "recognizer_training_console.txt",
        package_root / "recognizer_training_console.txt",
    ])

    preflight_log = first_existing([
        package_root / "output" / "preflight_evaluation_console.txt",
        package_root / "preflight_evaluation_console.txt",
    ])

    dictionary = first_existing([
        package_root / "dataset" / "character_dict.txt",
        inference / "character_dict.txt" if inference else Path("__missing__"),
    ])

    return {
        "training": training,
        "inference": inference,
        "config": config,
        "training_log": training_log,
        "preflight_log": preflight_log,
        "dictionary": dictionary,
    }


def validate_assets(assets: dict[str, Path | None]) -> None:
    required = ["training", "inference", "config"]
    missing = [name for name in required if assets[name] is None]
    if missing:
        raise FileNotFoundError(
            "Required Colab result items are missing: " + ", ".join(missing)
        )

    training = assets["training"]
    inference = assets["inference"]
    assert training is not None
    assert inference is not None

    best_params = list(training.rglob("best_accuracy.pdparams"))
    if not best_params:
        raise FileNotFoundError(
            f"best_accuracy.pdparams was not found under: {training}"
        )

    inference_files = [path for path in inference.rglob("*") if path.is_file()]
    if not inference_files:
        raise FileNotFoundError(f"No exported inference files found under: {inference}")

    suspicious = [path for path in best_params + inference_files if path.stat().st_size == 0]
    if suspicious:
        raise RuntimeError(
            "Zero-byte result files were found:\n" + "\n".join(map(str, suspicious))
        )


def backup_existing(model4: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_root = model4 / "backups" / f"before_colab_import_{timestamp}"
    mappings = {
        model4 / "models" / "recognizer" / "training": backup_root / "training",
        model4 / "models" / "recognizer" / "inference": backup_root / "inference",
        model4 / "logs" / "recognizer_training_console.txt": backup_root / "logs" / "recognizer_training_console.txt",
        model4 / "logs" / "preflight_evaluation_console.txt": backup_root / "logs" / "preflight_evaluation_console.txt",
        model4 / "config" / "PP-OCRv6_small_rec_lcd_finetune.yml": backup_root / "config" / "PP-OCRv6_small_rec_lcd_finetune.yml",
    }

    backed_up = False
    for source, destination in mappings.items():
        if not source.exists():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            if any(source.iterdir()):
                shutil.copytree(source, destination, dirs_exist_ok=True)
                backed_up = True
        else:
            shutil.copy2(source, destination)
            backed_up = True

    if not backed_up and backup_root.exists():
        shutil.rmtree(backup_root)
    return backup_root


def replace_directory(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)


def copy_optional(source: Path | None, destination: Path) -> bool:
    if source is None or not source.is_file():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return True


def import_results(package_root: Path, assets: dict[str, Path | None], model4: Path) -> dict:
    training_destination = model4 / "models" / "recognizer" / "training"
    inference_destination = model4 / "models" / "recognizer" / "inference"
    log_destination = model4 / "logs"
    config_destination = model4 / "config"

    backup_root = backup_existing(model4)

    training = assets["training"]
    inference = assets["inference"]
    config = assets["config"]
    assert training is not None
    assert inference is not None
    assert config is not None

    print("Copying Colab training outputs...")
    replace_directory(training, training_destination)

    print("Copying exported inference model...")
    replace_directory(inference, inference_destination)

    print("Copying fine-tuning configuration...")
    config_name = "PP-OCRv6_small_rec_lcd_finetune" + config.suffix.lower()
    shutil.copy2(config, config_destination / config_name)

    logs_copied = {
        "training_log": copy_optional(
            assets["training_log"],
            log_destination / "recognizer_training_console_colab.txt",
        ),
        "preflight_log": copy_optional(
            assets["preflight_log"],
            log_destination / "preflight_evaluation_console_colab.txt",
        ),
    }

    dictionary = assets["dictionary"]
    if dictionary is not None:
        shutil.copy2(
            dictionary,
            model4 / "models" / "recognizer" / "character_dict.txt",
        )

    best_params = list(training_destination.rglob("best_accuracy.pdparams"))
    inference_files = [
        path for path in inference_destination.rglob("*") if path.is_file()
    ]

    return {
        "source_package": str(package_root),
        "model4": str(model4),
        "backup": str(backup_root) if backup_root.exists() else None,
        "best_checkpoint": str(best_params[0]),
        "best_checkpoint_bytes": best_params[0].stat().st_size,
        "best_checkpoint_sha256": sha256(best_params[0]),
        "inference_file_count": len(inference_files),
        "inference_files": [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in inference_files
        ],
        "logs_copied": logs_copied,
        "config": str(config_destination / config_name),
        "imported_at": datetime.now().isoformat(timespec="seconds"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Safely import PP-OCRv6 Colab recognizer results into Model 4."
    )
    parser.add_argument(
        "--source",
        type=Path,
        help="Downloaded ppocrv6_colab folder or ZIP. If omitted, Downloads is searched automatically.",
    )
    parser.add_argument(
        "--model4",
        type=Path,
        default=DEFAULT_MODEL4,
        help="Local Model 4 package root.",
    )
    args = parser.parse_args()

    model4 = args.model4.resolve()
    if not model4.is_dir():
        raise FileNotFoundError(f"Model 4 package not found: {model4}")

    source = args.source.resolve() if args.source else discover_source(DEFAULT_DOWNLOADS)
    print(f"Detected source: {source}")
    print(f"Destination package: {model4}")

    with tempfile.TemporaryDirectory(prefix="ppocrv6_colab_import_") as temp:
        search_root = extract_if_needed(source, Path(temp))
        package_root = find_package_root(search_root)
        print(f"Detected package root: {package_root}")

        assets = locate_assets(package_root)
        print("Detected assets:")
        for name, path in assets.items():
            print(f"  {name}: {path}")

        validate_assets(assets)
        report = import_results(package_root, assets, model4)

    report_path = model4 / "docs" / "colab_recognizer_import_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nColab result import completed successfully.")
    print(f"Best checkpoint: {report['best_checkpoint']}")
    print(f"Inference files: {report['inference_file_count']}")
    print(f"Configuration: {report['config']}")
    print(f"Import report: {report_path}")
    if report["backup"]:
        print(f"Previous local outputs backed up to: {report['backup']}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nIMPORT FAILED: {error}", file=sys.stderr)
        raise
