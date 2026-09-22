# Model 2: Expanded PP-OCRv3 Detector with Reused Recognizer

## Overview

This package contains the second LCD OCR detector experiment.

Model 2 improved text detection by expanding the detector dataset and
initializing the detector from the fully trained PP-OCRv3 student detector.

The recognizer was not retrained for Model 2. Model 2 reuses the same
successful recognizer packaged with Model 1.

## Component identity

### Detector

Run name:

det_lcd_v2_expanded_240

Initialization:

Fully trained PP-OCRv3 student detector

Training images:

240

Validation images:

28

Epochs completed:

50

Best epoch:

15

Precision:

65.03 percent

Recall:

72.66 percent

Hmean:

68.63 percent

Validation FPS:

0.6063

Selected checkpoint:

best_accuracy

### Recognizer

Run name:

rec_lcd_v2_finetuned

Initialization:

Official English PP-OCRv3 recognition checkpoint

Training source:

281 text crops produced from the original 60 pilot training images

Validation source:

44 text crops produced from the original 10 pilot validation images

Best epoch:

45

Exact validation accuracy:

87.50 percent

Normalized edit similarity:

94.18 percent

Recognizer status:

Reused unchanged from Model 1

The recognizer checkpoint and exported inference files were verified against
Model 1 using SHA256 hashes.

## Dataset composition

### Detector training set

240 full LCD images

Annotation file:

data/detector/train/Label.txt

Image folder:

data/detector/train/images

### Detector validation set

28 full LCD images

Annotation file:

data/detector/validation/Label.txt

Image folder:

data/detector/validation/images

### Recognizer training set

281 manually reviewed text crops

Ground-truth file:

data/recognizer/train/rec_gt.txt

Crop folder:

data/recognizer/train/crop_img

### Recognizer validation set

44 manually reviewed text crops

Ground-truth file:

data/recognizer/validation/rec_gt.txt

Crop folder:

data/recognizer/validation/crop_img

## Package structure

config

Contains the historical and portable detector and recognizer configurations.

data

Contains detector full-image data and recognition crop data.

models

Contains the selected training checkpoints, exported inference models, and
the recognition character dictionary.

results

Contains standalone metrics, retrospective detector results, retrospective
recognizer logs, and retrospective end-to-end outputs.

scripts

Contains package validation, environment setup, detector inference, and
recognizer inference scripts.

requirements

Contains Python package requirements and development version records.

docs

Contains checkpoint provenance, inference provenance, package validation,
and retrospective evaluation documentation.

## Requirements

Windows 10 or Windows 11

Python 3.10

PaddlePaddle 2.6.2

PaddleOCR 2.8.1

PaddleOCR release 2.8 source checkout

CPU inference is supported.

## Environment setup

Open PowerShell inside this package and run:

powershell -ExecutionPolicy Bypass -File ".\scripts\setup_environment.ps1"

Activate the environment using the command printed by the setup script.

## PaddleOCR source checkout

The inference scripts require a separate PaddleOCR release 2.8 source
checkout.

Example location:

D:\Actual Project\PaddleOCR-2.8

Required PaddleOCR scripts:

tools\infer\predict_det.py

tools\infer\predict_rec.py

tools\infer\predict_system.py

## Validate the package

Run:

powershell -ExecutionPolicy Bypass -File ".\scripts\validate_package.ps1"

Expected detector counts:

240 training images

240 training annotation rows

28 validation images

28 validation annotation rows

Expected recognizer counts:

281 training crops

281 training ground-truth rows

44 validation crops

44 validation ground-truth rows

## Run standalone detector inference

Run on one image:

powershell -ExecutionPolicy Bypass -File ".\scripts\run_detector.ps1" -ImagePath ".\data\detector\validation\images\IMAGE_NAME.png" -PaddleRoot "D:\Actual Project\PaddleOCR-2.8"

Run on the full validation folder:

powershell -ExecutionPolicy Bypass -File ".\scripts\run_detector.ps1" -ImagePath ".\data\detector\validation\images" -PaddleRoot "D:\Actual Project\PaddleOCR-2.8"

Default detector results are written to:

results/detector/inference_results

## Run standalone recognizer inference

Choose a crop from:

data/recognizer/validation/crop_img

Run:

powershell -ExecutionPolicy Bypass -File ".\scripts\run_recognizer.ps1" -CropPath ".\data\recognizer\validation\crop_img\CROP_NAME.jpg" -PaddleRoot "D:\Actual Project\PaddleOCR-2.8"

The console displays the recognized text and confidence.

## Historical end-to-end status

No verified end-to-end evaluation was performed during the original Model 2
detector experiment.

The original detector and recognizer metrics are standalone component
metrics.

## Retrospective evaluation

A retrospective evaluation was subsequently performed using:

Detector:

det_lcd_v2_expanded_240

Recognizer:

rec_lcd_v2_finetuned

Full-image inputs:

All 28 Model 2 detector validation images

Standalone recognizer inputs:

All 44 recognition validation crops

Retrospective outputs are stored under:

results/retrospective_evaluation

The retrospective evaluation is intended for qualitative inspection and
remote demonstration.

The retrospective results must not be presented as part of the original
training-time experiment.

No aggregate end-to-end accuracy is claimed until the generated predictions
are manually scored against the reviewed annotations.

## Model 1 comparison

Model 1 detector:

Precision: 55.56 percent

Recall: 39.47 percent

Hmean: 46.15 percent

Model 2 detector:

Precision: 65.03 percent

Recall: 72.66 percent

Hmean: 68.63 percent

Model 2 improved detector precision by approximately 9.47 percentage points.

Model 2 improved detector recall by approximately 33.19 percentage points.

Model 2 improved detector Hmean by approximately 22.48 percentage points.

The recognizer remained unchanged between Model 1 and Model 2.

## Known limitations

The detector may still detect graphical LCD icons as text.

The detector may detect inactive LCD segments.

Nearby text and icon regions may be merged.

Some text regions may be fragmented.

Low-contrast text may remain undetected.

Recognizer quality on detector-generated crops may be lower than the
standalone recognizer accuracy measured on manually reviewed crops.

The icon model has not yet been integrated into these retrospective outputs.

## Configuration files

Historical detector configuration:

config/detector_original.yml

Portable detector configuration:

config/detector_portable.yml

Historical recognizer configuration:

config/recognizer_original.yml

Portable recognizer configuration:

config/recognizer_portable.yml

The original files preserve historical experiment settings.

The portable files point to the clean Model 2 package.

## Selected checkpoints

Detector:

models/detector/training/best_accuracy.pdparams

Recognizer:

models/recognizer/training/best_accuracy.pdparams

For final inference, use the exported models under:

models/detector/inference

models/recognizer/inference

## Metrics

Detector metrics:

results/detector/standalone_metrics.json

Recognizer metrics:

results/recognizer/standalone_metrics.json

## Provenance

Checkpoint provenance:

docs/CHECKPOINT_PROVENANCE.txt

Inference provenance:

docs/INFERENCE_PROVENANCE.txt

Retrospective evaluation log:

docs/RETROSPECTIVE_EVALUATION_LOG.md

Package validation record:

docs/PACKAGE_VALIDATION.txt
