# Full Icon-Masked PP-OCRv6 LCD Pipeline

This package contains the completed experimental LCD pipeline:

```text
LCD image
  -> OpenVINO icon detector
  -> icon masking
  -> fine-tuned PP-OCRv6 small detector
  -> fine-tuned PP-OCRv6 small recognizer
  -> rendered images and JSON reports
```

## Results

### Detector

- Precision: `0.7112068966`
- Recall: `0.8823529412`
- Hmean: `0.7875894988`
- Best epoch: `40`

### Recognizer

- Validation accuracy: `0.9171122749`
- Normalized edit similarity: `0.9653043045`
- Best epoch: `30`

The existing PP-OCRv3 Model 3 recognizer remains the stronger general fallback. This package is the experimental full PP-OCRv6 path and a base for future meter-family routing.

## Contents

```text
config/                         Training and reference configurations
models/detector/training/       Best detector checkpoint
models/detector/inference/      Exported detector model
models/recognizer/training/     Best recognizer checkpoint
models/recognizer/inference/    Exported recognizer model
models/recognizer/character_dict.txt
scripts/                        Pipeline, comparison, and import utilities
results/                        Compact metrics and reports
logs/                           Colab training logs
docs/                           Import and environment records
```

The reviewed icon detector is stored at:

```text
packages/icon-detector-openvino/
```

## Required local paths

```text
D:\Actual Project\lcd-fastapi-learning\icon_detector.py
D:\Actual Project\PaddleOCR-3.7
D:\Actual Project\ppocrv6_env
D:\Actual Project\Indali_Lcd-Data-Extractor
```

## Run the full validation pipeline

Activate the environment containing OpenVINO and the icon detector dependencies:

```powershell
& "D:\Actual Project\lcd_fastapi_env\Scripts\Activate.ps1"
```

If the environment is inside the FastAPI folder, use:

```powershell
& "D:\Actual Project\lcd-fastapi-learning\lcd_fastapi_env\Scripts\Activate.ps1"
```

Run all reviewed validation images:

```powershell
cd "D:\Actual Project\Indali_Lcd-Data-Extractor"
python ".\packages\model-4-ppocrv6-small-baseline\scriptsun_full_v6_hybrid_validation.py"
```

The expected validation count is 97. The OCR stage automatically invokes `D:\Actual Project\ppocrv6_env\Scripts\python.exe`.

## Run another image folder

```powershell
python ".\packages\model-4-ppocrv6-small-baseline\scriptsun_full_v6_hybrid_validation.py" `
  --images "D:\path	o\lcd_images" `
  --output "D:\path	o\output"
```

Set a different icon confidence threshold:

```powershell
python ".\packages\model-4-ppocrv6-small-baseline\scriptsun_full_v6_hybrid_validation.py" `
  --icon-threshold 0.45
```

## Outputs

```text
results/full_v6_hybrid_validation/
  originals/          Ignored by Git
  masked/             Ignored by Git
  ocr_rendered/       Ignored by Git
  rendered/           Ignored by Git
  json/               Ignored by Git
  predictions.json
  failures.json
  summary.json
  pipeline_console.txt
```

## Compare recognizers

```powershell
python ".\packages\model-4-ppocrv6-small-baseline\scripts\compare_lcd_recognizers.py"
```

## Git LFS

Model binaries use Git LFS. After cloning or pulling:

```powershell
git lfs install
git lfs pull
```

Do not commit copied datasets, Colab archives, virtual environments, backups, or bulk generated renderings.

## Temporary API routing plan

```text
general_v3
  -> Model 2 detector
  -> Model 3 recognizer

experimental_v6
  -> icon detector and masking
  -> fine-tuned PP-OCRv6 small detector
  -> fine-tuned PP-OCRv6 small recognizer
```
