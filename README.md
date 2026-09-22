# LCD OCR Model v1: Pilot 60

This branch packages the first remotely testable LCD OCR model. It contains the 60-image detector training split, 10-image validation split, selected detector and recognizer checkpoints, exported inference models when available, evaluation results, and reproducibility scripts.

## Final metrics

### Recognizer

- Run: `rec_lcd_v2_finetuned`
- Training source: 60 pilot full images and their recognition crops
- Validation source: 10 pilot full images and their recognition crops
- Exact validation accuracy: 87.50%
- Normalized edit similarity: 94.18%
- Best epoch: 45
- Selected checkpoint: `best_accuracy`

### Detector

- Run: `det_lcd_v1_60`
- Training images: 60
- Validation images: 10
- Precision: 55.56%
- Recall: 39.47%
- Hmean: 46.15%
- Best epoch: 33
- Selected checkpoint: `best_accuracy`

## Remote quick start

Install Git LFS before cloning or pulling model files.

```powershell
git clone REPOSITORY_URL
cd lcd-data-extractor
git lfs install
git switch model/v1-pilot-60
git lfs pull
```

Create the environment:

```powershell
py -3.10 -m venv paddle_inference_env
& ".\paddle_inference_env\Scripts\Activate.ps1"
python -m pip install --upgrade pip
python -m pip install -r ".\model_package\requirements\requirements-inference.txt"
```

A PaddleOCR release/2.8 source checkout is required. Set its path when running the scripts if it is not located at `D:\Actual Project\PaddleOCR-2.8`.

Verify the package:

```powershell
powershell -ExecutionPolicy Bypass -File ".\model_package\scripts\verify_package.ps1"
```

Run detector-only inference:

```powershell
powershell -ExecutionPolicy Bypass -File ".\model_package\scripts\run_detector.ps1" -ImagePath "PATH_TO_FULL_LCD_IMAGE_OR_FOLDER" -PaddleRoot "PATH_TO_PaddleOCR-2.8"
```

Run recognizer-only inference:

```powershell
powershell -ExecutionPolicy Bypass -File ".\model_package\scripts\run_recognizer.ps1" -CropPath "PATH_TO_TEXT_CROP" -PaddleRoot "PATH_TO_PaddleOCR-2.8"
```

Run end-to-end inference after confirming both inference folders contain exported models:

```powershell
powershell -ExecutionPolicy Bypass -File ".\model_package\scripts\run_end_to_end.ps1" -ImagePath "PATH_TO_FULL_LCD_IMAGE_OR_FOLDER" -PaddleRoot "PATH_TO_PaddleOCR-2.8"
```

## Package layout

- `config`: detector and recognizer YAML files
- `data`: detector full images and recognizer text crops
- `models`: selected training checkpoints and exported inference models
- `scripts`: verification and inference scripts
- `results`: detector, recognizer, and end-to-end outputs
- `docs`: results, known issues, troubleshooting, and changelog
- `requirements`: package versions and environment requirements

## Important limitations

- The v1 detector is a baseline and has modest recall.
- The detector may identify icons, inactive LCD segments, or borders as text.
- The recognizer expects correctly cropped text regions.
- Full-image OCR requires both the detector and recognizer inference models.
- The original sealed test split was not used for model tuning.

See `model_package/docs/TROUBLESHOOTING.md` for all known issues encountered during development.