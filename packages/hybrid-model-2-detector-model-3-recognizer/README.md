# Hybrid OCR Pipeline

## Pipeline

This package combines:

- Model 2 detector: det_lcd_v2_expanded_240
- Model 3 recognizer: rec_lcd_v3_full_reviewed

The detector and recognizer were trained separately.

This pipeline is retrospective and was not jointly trained.

## Why this combination was selected

Model 2 has the strongest standalone detector:

- Precision: 65.03 percent
- Recall: 72.66 percent
- Hmean: 68.63 percent

Model 3 has the strongest standalone recognizer:

- Exact corrected validation accuracy: 94.02 percent
- Corrected normalized edit similarity: 97.28 percent

## Evaluation

The hybrid pipeline was evaluated on 97 corrected validation images.

Generated detector crops:

393

The rendered outputs and detected crops are included under:

results

No aggregate end-to-end accuracy is claimed because the outputs have not yet
been manually scored against the corrected full-image annotations.

## Package contents

- config: source detector and recognizer configurations
- data: corrected validation images and ground truth
- models: exported detector and recognizer inference models
- results: rendered outputs, generated crops, and execution logs
- scripts: package validation and end-to-end inference
- docs: evaluation provenance and native-versus-hybrid comparison

## Requirements

- Windows 10 or Windows 11
- Python 3.10
- PaddlePaddle 2.6.2
- PaddleOCR 2.8.1
- PaddleOCR release/2.8 source checkout
- Git LFS

## Validate the package

Run:

powershell -ExecutionPolicy Bypass -File ".\scripts\validate_package.ps1"

## Run the hybrid pipeline

Run on one image or a folder:

powershell -ExecutionPolicy Bypass -File ".\scripts\run_hybrid.ps1" -ImagePath "PATH_TO_IMAGE_OR_FOLDER" -PaddleRoot "D:\Actual Project\PaddleOCR-2.8"

New outputs are written under:

results/new_inference

## Important limitation

The icon model is not yet integrated.

Graphical icons and inactive LCD segments may still be detected as text.

The next integration phase will use icon detections to suppress OCR regions
that substantially overlap trusted non-text icon predictions.
