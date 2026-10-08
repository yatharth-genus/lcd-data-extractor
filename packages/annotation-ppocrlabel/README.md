# PPOCRLabel Annotation Package with Model 5 PP-OCRv6

Portable Windows annotation environment for utility-meter LCD text detection and recognition.

This package combines the patched PPOCRLabel v2 interface with the selected Model 5 PP-OCRv6 detector and recognizer.

## Package location

packages/annotation-ppocrlabel

## OCR configuration

The annotation interface uses the Model 5 artifacts packaged under:

packages/meter-family-api/models/experimental_v6

Runtime configuration:

- Detector: Model 5 fine-tuned PP-OCRv6 small detector
- Recognizer: Model 5 fine-tuned PP-OCRv6 small recognizer
- Detector algorithm: DB
- Recognizer algorithm: SVTR_LCNet
- Recognition shape: 3,48,320
- Space-character support: enabled
- Dictionary: exact Model 5 character dictionary

The package does not maintain a second copy of the OCR models. It references the existing meter-family API Model 5 package.

## Package structure

    packages/annotation-ppocrlabel/
    ├── app/
    │   ├── PPOCRLabel.py
    │   ├── launch_ppocrlabel_v6.py
    │   ├── ppocrv6_worker.py
    │   ├── libs/
    │   └── resources/
    ├── audit/
    │   ├── critical_versions.txt
    │   ├── model5-runtime-identity.txt
    │   └── PPOCRLabel-local-changes.patch
    ├── requirements/
    │   └── requirements-annotation.txt
    ├── setup_annotation.ps1
    ├── validate_package.ps1
    ├── run_annotation.ps1
    └── README.md

## System requirements

- Windows 10 or Windows 11
- 64-bit Python 3.10
- Git
- Git LFS
- Internet access during initial setup
- The repository cloned with Model 5 Git LFS files retrieved

## Clone and retrieve Git LFS files

After cloning the company branch:

    git lfs install
    git lfs pull
    git lfs checkout

Confirm that the detector and recognizer `.pdiparams` files are real binary files rather than small Git LFS pointer files.

## One-time setup

Open PowerShell in:

    packages/annotation-ppocrlabel

Run:

    powershell -NoProfile -ExecutionPolicy Bypass -File .\setup_annotation.ps1

The setup process creates the local PPOCRLabel annotation environment.

The Model 5 worker reuses the PP-OCRv6 environment and PaddleOCR runtime from:

    packages/meter-family-api/.venv-v6
    packages/meter-family-api/runtime/PaddleOCR-3.7

If the shared meter-family runtime has not been installed, the setup process invokes the meter-family API setup.

## Validate the package

Run:

    powershell -NoProfile -ExecutionPolicy Bypass -File .\validate_package.ps1

Validation checks include:

- Required PPOCRLabel application files
- Required setup and runner scripts
- Model 5 detector artifacts
- Model 5 recognizer artifacts
- Exact Model 5 character dictionary
- Package-relative paths
- Python syntax
- PowerShell syntax
- DB detector configuration
- SVTR_LCNet recognizer configuration
- Recognition shape 3,48,320
- Enabled space-character support

## Start the annotation interface

Run:

    powershell -NoProfile -ExecutionPolicy Bypass -File .\run_annotation.ps1

The launcher starts the patched PPOCRLabel interface and routes automatic labeling through the Model 5 PP-OCRv6 worker.

## Annotation workflow

1. Open an image directory in PPOCRLabel.
2. Run automatic detection and recognition.
3. Review every generated polygon.
4. Review every recognized string.
5. Correct labels before saving.
6. Keep graphical icons outside OCR annotations.
7. Save annotations only after manual review.

Automatic predictions are draft annotations and must not be treated as final ground truth.

## Annotation conventions

- Use one polygon for each complete, visually continuous text region.
- Keep complete readings together, including leading zeros.
- Preserve decimal points, colons, spaces, and capitalization.
- Preserve the exact display form of units such as kW, kWh, kVA, kVAh, Hz, V, and A.
- Use tight four-point polygons.
- Include punctuation within the polygon.
- Keep coordinates inside the image.
- Do not annotate battery, lock, bell, bulb, grounding, arrow, relay, or other graphical icons as OCR text.
- Do not guess ambiguous characters.

Pay particular attention to:

- Zero and letter O
- One and letter I
- Five and letter S
- Unit capitalization
- Spaces within units and labels
- Missing or extra zeros
- Decimal points and colons

## Model files

The launcher resolves the following repository-relative paths:

    packages/meter-family-api/models/experimental_v6/models/detector/inference
    packages/meter-family-api/models/experimental_v6/models/recognizer/inference
    packages/meter-family-api/models/experimental_v6/models/recognizer/character_dict.txt

Changing the dictionary or recognizer preprocessing can produce incorrect text even when confidence is high.

## Local files excluded from Git

The following must remain local:

- `.venv-annotation`
- `.venv-v6`
- PaddleOCR runtime clones
- `__pycache__`
- Python bytecode
- Logs
- Backup files
- `fileState.txt`
- `Cache.cach`
- Annotation images
- `Label.txt`
- Generated recognition crops
- Unapproved datasets

Do not use `git add .` for annotation work. Stage the package paths explicitly.

## Troubleshooting

### Annotation environment is missing

Run:

    powershell -NoProfile -ExecutionPolicy Bypass -File .\setup_annotation.ps1

### Model 5 files are missing or unexpectedly small

Run from the repository root:

    git lfs pull
    git lfs checkout

### PP-OCRv6 worker does not start

Verify:

    packages/meter-family-api/.venv-v6/Scripts/python.exe
    packages/meter-family-api/runtime/PaddleOCR-3.7
    packages/meter-family-api/models/experimental_v6

Then run the package validator again.

### Recognition produces incorrect high-confidence text

Verify all of the following:

- The exact Model 5 dictionary is used.
- The recognizer algorithm is SVTR_LCNet.
- The recognition shape is 3,48,320.
- Space-character support is enabled.
- The annotation launcher points to experimental_v6.

## Acceptance checklist

Before handover:

- Git LFS files are present.
- One-time setup succeeds.
- Package validation succeeds.
- PPOCRLabel opens.
- An existing image directory loads.
- Automatic detection produces polygons.
- Automatic recognition produces readable strings.
- Saved labels can be reopened.
- No machine-specific paths are required.
- No environment, cache, log, image, or annotation-output files are staged.

## Branch

    feature/model5-ppocrlabel-annotation

## Related package

    packages/meter-family-api

## Current status

The annotation package is configured to use the selected Model 5 PP-OCRv6 detector and recognizer for automatic labeling.