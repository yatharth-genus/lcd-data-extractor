# Indali LCD Data Extractor

This repository preserves the development history, datasets, selected
checkpoints, exported inference models, results, and reproducibility
instructions for utility-meter LCD OCR.

## Branches

### model/model-1-pilot-mobilenet

First detector baseline:

- Detector: det_lcd_v1_60
- Detector initialization: ImageNet-pretrained MobileNetV3 backbone
- Detector training images: 60
- Detector validation images: 10
- Recognizer: rec_lcd_v2_finetuned
- Recognizer validation accuracy: 87.50 percent

### model/model-2-expanded-ppocr

Expanded detector:

- Detector: det_lcd_v2_expanded_240
- Detector initialization: fully trained PP-OCRv3 student
- Detector training images: 240
- Detector validation images: 28
- Recognizer: rec_lcd_v2_finetuned, reused unchanged from Model 1
- Detector Hmean: 68.63 percent

### model/model-3-full-reviewed

Full reviewed experiment:

- Detector: det_lcd_v3_full_reviewed
- Recognizer: rec_lcd_v3_full_reviewed
- Training snapshot: pre-correction compiled dataset
- Corrected submission dataset: annotations corrected on September 22, 2026
- Training status: in progress or pending final packaging

## Git LFS

This repository uses Git LFS for images, model checkpoints, exported models,
Word files, and Excel files.

After cloning a model branch, run:

git lfs install

git lfs pull

git lfs checkout

## Historical and retrospective results

Historical metrics are preserved separately from retrospective end-to-end
tests.

Model 1 and Model 2 did not originally have verified end-to-end results.
Their packaged end-to-end outputs are labeled retrospective.

## Model instructions

Each model branch contains its own README, validation scripts, environment
requirements, checkpoint provenance, inference provenance, metrics, and
testing instructions.
