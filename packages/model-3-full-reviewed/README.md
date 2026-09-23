# Model 3: Full Reviewed Dataset Experiment

## Overview

Model 3 trained a new PP-OCRv3 detector and recognizer using the complete
compiled dataset.

The detector and recognizer were trained using the pre-correction
`dataset_final` snapshot.

Additional ground-truth corrections were discovered and applied to the
authoritative annotations on September 22, 2026. The corrected dataset was
compiled separately and is included in this package.

The active training dataset was not modified while training was running.

## Dataset versions

### Training snapshot

The Model 3 checkpoints were trained using:

D:\Actual Project\lcd-data-extractor\dataset_final

The training-snapshot metadata is stored under:

data/training_snapshot

### Corrected submission dataset

The corrected authoritative dataset is stored under:

data/corrected_dataset

The corrected dataset is intended for:

- Dataset submission
- Corrected evaluation
- Future production retraining

The packaged Model 3 checkpoints must not be described as trained on the
corrected dataset.

## Detector

Run:

det_lcd_v3_full_reviewed

Initialization:

Fully trained PP-OCRv3 student detector

Scheduled epochs:

25

Selected checkpoint:

best_accuracy

Best epoch:

11

Precision:

42.91 percent

Recall:

68.46 percent

Hmean:

52.75 percent

Corrected-evaluation FPS:

0.6936

## Detector interpretation

The Model 3 detector did not improve over the Model 2 detector.

Model 2 detector Hmean:

68.63 percent

Model 3 detector Hmean:

52.75 percent

Model 2 remains the strongest standalone detector.

## Recognizer

Run:

rec_lcd_v3_full_reviewed

Initialization:

Official English PP-OCRv3 recognition checkpoint

Completed epochs:

10

Selected checkpoint:

best_accuracy

Best epoch:

10

Corrected validation accuracy:

94.02 percent

Corrected normalized edit similarity:

97.28 percent

Corrected-evaluation FPS:

4.2249

## Recognizer interpretation

The Model 3 recognizer improved over the recognizer used by Models 1 and 2.

Earlier recognizer accuracy:

87.50 percent

Model 3 recognizer accuracy:

94.02 percent

Model 3 therefore contains the strongest standalone recognizer.

## Corrected validation result

The quality metrics from the corrected validation data were identical to the
training-snapshot validation metrics.

Detector corrected Hmean:

0.5275181723779855

Recognizer corrected accuracy:

0.940217365754963

The ground-truth corrections did not change the aggregate validation-quality
metrics for the selected checkpoints.

## Package contents

### config

Contains the training-snapshot configurations and corrected portable
configurations.

### data

Contains training-snapshot metadata and the corrected compiled dataset.

### models

Contains selected detector and recognizer checkpoints, exported inference
models, and the recognition dictionary.

### results

Contains training-snapshot metrics, corrected evaluation metrics, training
logs, and corrected retrospective end-to-end outputs.

### scripts

Contains dataset compilation and inference utilities.

### docs

Contains ground-truth correction records, checkpoint provenance, inference
provenance, dataset comparison, validation comparison, and training results.

## Model 3 end-to-end evaluation

A retrospective native Model 3 pipeline evaluation was performed using:

- Detector: det_lcd_v3_full_reviewed
- Recognizer: rec_lcd_v3_full_reviewed
- Inputs: 97 corrected validation images

The outputs are stored under:

results/corrected_retrospective_evaluation

No aggregate end-to-end accuracy is claimed until the predictions are
manually scored against the corrected full-image annotations.

## Recommended hybrid candidate

The leading candidate combines:

- Model 2 detector: det_lcd_v2_expanded_240
- Model 3 recognizer: rec_lcd_v3_full_reviewed

The hybrid system is stored separately under the repository `shared`
directory because the two components came from different model experiments.

The hybrid pipeline is retrospective and was not jointly trained.

## Requirements

- Windows 10 or Windows 11
- Python 3.10
- PaddlePaddle 2.6.2
- PaddleOCR 2.8.1
- PaddleOCR release/2.8 source checkout
- Git LFS for remote model and dataset retrieval

## Validate the package

Run:

powershell -ExecutionPolicy Bypass -File ".\scripts\validate_package.ps1"

## Run detector inference

Run:

powershell -ExecutionPolicy Bypass -File ".\scripts\run_detector.ps1" -ImagePath ".\data\corrected_dataset\detector\validation\images" -PaddleRoot "D:\Actual Project\PaddleOCR-2.8"

## Run recognizer inference

Choose a crop from:

data/corrected_dataset/recognizer/validation/crop_img

Run:

powershell -ExecutionPolicy Bypass -File ".\scripts\run_recognizer.ps1" -CropPath "PATH_TO_CROP" -PaddleRoot "D:\Actual Project\PaddleOCR-2.8"

## Known limitations

- The Model 3 detector has lower precision and Hmean than Model 2.
- Icons and inactive LCD segments may still be detected as text.
- Nearby text and graphical regions may be merged.
- Some valid text regions may be missed.
- The checkpoints were trained on the pre-correction dataset snapshot.
- End-to-end predictions have not been manually scored.
- The icon model has not yet been integrated into these results.

## Important provenance documents

- docs/CHECKPOINT_PROVENANCE.txt
- docs/INFERENCE_PROVENANCE.txt
- docs/GROUND_TRUTH_STATUS.md
- docs/GROUND_TRUTH_CORRECTIONS_2026-09-22.md
- docs/DATASET_COMPARISON.txt
- docs/VALIDATION_COMPARISON.md
- docs/TRAINING_RESULTS.md
