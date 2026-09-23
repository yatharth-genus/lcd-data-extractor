# Hybrid OCR Pipeline Evaluation

## Evaluation identity

Evaluation date: 2026-09-23 10:07:07

Evaluation type: Retrospective hybrid evaluation

## Pipeline components

Detector package:

model-2-expanded-ppocr

Detector run:

det_lcd_v2_expanded_240

Detector checkpoint:

best_accuracy

Detector standalone Hmean:

0.6863468634686346

Recognizer package:

model-3-full-reviewed

Recognizer run:

rec_lcd_v3_full_reviewed

Recognizer checkpoint:

best_accuracy

Recognizer corrected validation accuracy:

0.940217365754963

Recognizer corrected normalized edit similarity:

0.9728260876949433

## Evaluation inputs

Corrected Model 3 validation images:

97

## Generated outputs

Rendered image files:

0

All files in rendered-results folder:

98

Detector-generated recognition crops:

393

## Result locations

Rendered results:

rendered_results

Detector-generated crops:

detected_crops

Console log:

reports/end_to_end_console.txt

## Interpretation

This retrospective pipeline combines the strongest standalone detector with
the strongest standalone recognizer.

The detector and recognizer were trained in separate experiments and were not
jointly optimized.

The pipeline generated 393 recognition crops from 97 corrected validation
images.

No aggregate end-to-end accuracy is claimed until the detected regions and
recognized strings are compared against the corrected full-image ground
truth.

## Historical status

This hybrid pipeline did not exist during the original Model 2 or Model 3
training experiments.

All hybrid results are retrospective.
