# Full Icon-Masked PP-OCRv6 LCD Pipeline

This package is portable. It does not depend on the original developer's drive letter, username, virtual-environment path, or validation-data path.

## Pipeline

```text
LCD image -> OpenVINO icon detector -> icon masking -> PP-OCRv6 detector -> PP-OCRv6 recognizer -> rendered output and JSON
```

## Metrics

Detector: precision `0.7112068966`, recall `0.8823529412`, Hmean `0.7875894988`, best epoch `40`.

Recognizer: accuracy `0.9171122749`, normalized edit similarity `0.9653043045`, best epoch `30`.

## Clone and fetch model files

```powershell
git clone <COMPANY_REPOSITORY_URL>
cd Indali_Lcd-Data-Extractor
git switch feature/ppocrv6-full-pipeline
git lfs install
git lfs pull
```

## Create the icon runtime

```powershell
py -3.10 -m venv .venv-icon
& ".\.venv-icon\Scripts\Activate.ps1"
python -m pip install --upgrade pip
python -m pip install -r ".\packages\model-4-ppocrv6-small-baseline\requirements\requirements-inference.txt"
```

## Prepare PaddleOCR 3.7

Clone PaddleOCR next to this repository. It can also be stored elsewhere.

```powershell
cd ..
git clone --depth 1 --branch v3.7.0 https://github.com/PaddlePaddle/PaddleOCR.git PaddleOCR-3.7
cd PaddleOCR-3.7
py -3.10 -m venv .venv-ppocrv6
& ".\.venv-ppocrv6\Scripts\Activate.ps1"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

Install the appropriate PaddlePaddle CPU or GPU build in `.venv-ppocrv6`.

## Configure runtime locations

From the company repository root:

```powershell
$env:PADDLEOCR_ROOT = (Resolve-Path "..\PaddleOCR-3.7").Path
$env:PPOCRV6_PYTHON = (Resolve-Path "..\PaddleOCR-3.7\.venv-ppocrv6\Scripts\python.exe").Path
```

These environment variables make the setup independent of installation location.

## Run

```powershell
& ".\.venv-icon\Scripts\Activate.ps1"
python ".\packages\model-4-ppocrv6-small-baseline\scripts\run_full_v6_hybrid_validation.py" `
  --images "C:\path\to\lcd_images" `
  --output ".\pipeline-output"
```

Optional threshold:

```powershell
python ".\packages\model-4-ppocrv6-small-baseline\scripts\run_full_v6_hybrid_validation.py" `
  --images "C:\path\to\lcd_images" `
  --output ".\pipeline-output" `
  --icon-threshold 0.45
```

## Outputs

```text
pipeline-output/originals
pipeline-output/masked
pipeline-output/ocr_rendered
pipeline-output/rendered
pipeline-output/json
pipeline-output/predictions.json
pipeline-output/failures.json
pipeline-output/summary.json
pipeline-output/pipeline_console.txt
```

The validation dataset is not committed. Supply any image directory with `--images`.
