from __future__ import annotations

import ast
import csv
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(r"D:\Actual Project")
CLEAN = ROOT / "Indali_Lcd-Data-Extractor"
DATASET = CLEAN / "packages" / "model-3-full-reviewed" / "data" / "corrected_dataset" / "recognizer" / "validation"
CROPS = DATASET / "crop_img"
LABELS = DATASET / "rec_gt.txt"

MODEL3 = CLEAN / "packages" / "model-3-full-reviewed"
MODEL4 = CLEAN / "packages" / "model-4-ppocrv6-small-baseline"

PPOCR_V3_ROOT = ROOT / "PaddleOCR-2.8"
PPOCR_V6_ROOT = ROOT / "PaddleOCR-3.7"
V3_PYTHON = ROOT / "lcd-data-extractor" / "paddle_train_env" / "Scripts" / "python.exe"
V6_PYTHON = ROOT / "ppocrv6_env" / "Scripts" / "python.exe"

V3_MODEL = MODEL3 / "models" / "recognizer" / "inference"
V3_DICT = MODEL3 / "models" / "recognizer" / "character_dict.txt"
V6_MODEL = MODEL4 / "models" / "recognizer" / "inference"
V6_DICT = MODEL4 / "models" / "recognizer" / "character_dict.txt"

OUTPUT = MODEL4 / "results" / "recognizer_comparison"

SAMPLES = [
    "additional_100__batch_01__231__crop_0.jpg",
    "additional_100__batch_01__231__crop_1.jpg",
    "additional_100__batch_01__231__crop_2.jpg",
    "additional_100__batch_01__38__crop_0.jpg",
    "additional_100__batch_01__38__crop_1.jpg",
]


@dataclass
class Prediction:
    text: str
    confidence: float | None
    raw_output: str


def run(command: list[str], cwd: Path | None = None) -> str:
    print("\nRUN:", subprocess.list2cmdline(command))
    process = subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    print(process.stdout)
    if process.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {process.returncode}")
    return process.stdout


def require(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")


def load_labels() -> dict[str, str]:
    result = {}
    for line_number, raw in enumerate(LABELS.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not raw.strip():
            continue
        parts = raw.split("\t", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid label line {line_number}: {raw!r}")
        result[Path(parts[0].replace("\\", "/")).name] = parts[1]
    return result


def levenshtein(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            current.append(min(
                current[-1] + 1,
                previous[j] + 1,
                previous[j - 1] + (char_a != char_b),
            ))
        previous = current
    return previous[-1]


def parse_predict_rec(output: str, filename: str) -> Prediction:
    lines = [line for line in output.splitlines() if filename in line]
    candidates = lines or output.splitlines()
    patterns = [
        re.compile(r"\[['\"](?P<text>.*?)['\"],\s*(?P<score>[0-9.]+)\]"),
        re.compile(r"\('(?P<text>.*?)',\s*(?P<score>[0-9.]+)\)"),
    ]
    for line in reversed(candidates):
        for pattern in patterns:
            match = pattern.search(line)
            if match:
                return Prediction(match.group("text"), float(match.group("score")), output)
    raise ValueError(f"Could not parse prediction for {filename}")


def parse_zero_shot(output: str) -> Prediction:
    matches = re.findall(r"'rec_text':\s*'([^']*)'.*?'rec_score':\s*([0-9.]+)", output, flags=re.S)
    if not matches:
        matches = re.findall(r'"rec_text"\s*:\s*"([^"]*)".*?"rec_score"\s*:\s*([0-9.]+)', output, flags=re.S)
    if not matches:
        raise ValueError("Could not parse zero-shot PP-OCRv6 output")
    text, score = matches[-1]
    return Prediction(text, float(score), output)


def predict_zero_shot(image: Path) -> Prediction:
    command = [
        str(V6_PYTHON), "-m", "paddleocr", "text_recognition",
        "-i", str(image),
        "--model_name", "PP-OCRv6_small_rec",
        "--device", "cpu",
        "--save_path", str(OUTPUT / "zero_shot" / image.stem),
    ]
    return parse_zero_shot(run(command, PPOCR_V6_ROOT))


def predict_inference(image: Path, python_exe: Path, repo: Path, model: Path, dictionary: Path) -> Prediction:
    command = [
        str(python_exe), str(repo / "tools" / "infer" / "predict_rec.py"),
        "--image_dir", str(image),
        "--rec_model_dir", str(model),
        "--rec_char_dict_path", str(dictionary),
        "--use_gpu", "false",
        "--use_space_char", "false",
    ]
    return parse_predict_rec(run(command, repo), image.name)


def main() -> None:
    for path, description in [
        (LABELS, "validation labels"),
        (CROPS, "validation crops"),
        (V3_PYTHON, "PP-OCRv3 Python"),
        (V6_PYTHON, "PP-OCRv6 Python"),
        (PPOCR_V3_ROOT / "tools" / "infer" / "predict_rec.py", "PP-OCRv3 inference script"),
        (PPOCR_V6_ROOT / "tools" / "infer" / "predict_rec.py", "PP-OCRv6 inference script"),
        (V3_MODEL, "Model 3 inference model"),
        (V3_DICT, "Model 3 dictionary"),
        (V6_MODEL, "Model 4 inference model"),
        (V6_DICT, "Model 4 dictionary"),
    ]:
        require(path, description)

    labels = load_labels()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    raw_dir = OUTPUT / "raw_logs"
    raw_dir.mkdir(parents=True, exist_ok=True)

    for filename in SAMPLES:
        image = CROPS / filename
        require(image, "comparison crop")
        if filename not in labels:
            raise KeyError(f"Ground truth not found for {filename}")
        ground_truth = labels[filename]
        print("\n" + "=" * 80)
        print(filename, "GROUND TRUTH:", ground_truth)

        predictions = {
            "ppocrv6_zero_shot": predict_zero_shot(image),
            "ppocrv6_finetuned": predict_inference(image, V6_PYTHON, PPOCR_V6_ROOT, V6_MODEL, V6_DICT),
            "model3_ppocrv3": predict_inference(image, V3_PYTHON, PPOCR_V3_ROOT, V3_MODEL, V3_DICT),
        }

        for model_name, prediction in predictions.items():
            distance = levenshtein(ground_truth, prediction.text)
            rows.append({
                "image": filename,
                "ground_truth": ground_truth,
                "model": model_name,
                "prediction": prediction.text,
                "confidence": prediction.confidence,
                "exact_match": prediction.text == ground_truth,
                "edit_distance": distance,
            })
            (raw_dir / f"{image.stem}__{model_name}.txt").write_text(prediction.raw_output, encoding="utf-8")

    csv_path = OUTPUT / "five_crop_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {}
    for model_name in sorted({row["model"] for row in rows}):
        model_rows = [row for row in rows if row["model"] == model_name]
        summary[model_name] = {
            "samples": len(model_rows),
            "exact_matches": sum(bool(row["exact_match"]) for row in model_rows),
            "exact_accuracy": sum(bool(row["exact_match"]) for row in model_rows) / len(model_rows),
            "total_edit_distance": sum(int(row["edit_distance"]) for row in model_rows),
            "mean_edit_distance": sum(int(row["edit_distance"]) for row in model_rows) / len(model_rows),
        }

    report = {
        "full_validation_metrics": {
            "model3_ppocrv3": {
                "accuracy": 0.9402,
                "normalized_edit_similarity": 0.9728,
            },
            "ppocrv6_finetuned": {
                "accuracy": 0.9171122749435221,
                "normalized_edit_similarity": 0.9653043044672901,
                "best_epoch": 30,
                "fps": 642.1894288673552,
            },
        },
        "five_crop_summary": summary,
        "predictions": rows,
    }
    json_path = OUTPUT / "comparison_report.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nFINAL FIVE-CROP SUMMARY")
    for model_name, values in summary.items():
        print(
            f"{model_name}: {values['exact_matches']}/{values['samples']} exact, "
            f"mean edit distance {values['mean_edit_distance']:.3f}"
        )
    print(f"\nCSV: {csv_path}")
    print(f"JSON: {json_path}")
    print(f"Raw logs: {raw_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nCOMPARISON FAILED: {error}", file=sys.stderr)
        raise
