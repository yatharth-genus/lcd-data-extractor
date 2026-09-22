# Model 2 Retrospective Evaluation Log

## Evaluation identity

Evaluation date: 2026-09-22 16:49:44

Package: model-2-expanded-ppocr

Evaluation type: Retrospective standalone and end-to-end evaluation

This evaluation was performed after the original Model 2 detector experiment.
It is not a historical training-time end-to-end result.

## Model components

Detector run: det_lcd_v2_expanded_240

Detector initialization: Fully trained PP-OCRv3 student detector

Detector checkpoint: best_accuracy

Detector best epoch: 15

Recognizer run: rec_lcd_v2_finetuned

Recognizer initialization: Official English PP-OCRv3 recognition checkpoint

Recognizer checkpoint: best_accuracy

Recognizer best epoch: 45

Recognizer reuse status:

The recognizer was reused unchanged from Model 1. It was not retrained for
Model 2.

## Historical standalone metrics

Detector precision: 0.6503496503496503

Detector recall: 0.7265625

Detector Hmean: 0.6863468634686346

Detector validation FPS: 0.6062845692150092

Recognizer exact validation accuracy: 0.8749997265625854

Recognizer normalized edit similarity: 0.9418402959526853

## Retrospective evaluation scope

Full validation images tested: 28

Standalone recognition crops tested: 44

Standalone detector rendered files: 29

End-to-end rendered files: 29

Detector-generated text crops: 130

## Inputs

Full-image inputs:

data/detector/validation/images

Standalone recognizer inputs:

data/recognizer/validation/crop_img

## Outputs

Standalone detector outputs:

results/retrospective_evaluation/detector

Standalone recognizer report:

results/retrospective_evaluation/recognizer/reports/recognizer_console.txt

End-to-end rendered results:

results/retrospective_evaluation/end_to_end/rendered_results

Detector-generated crops:

results/retrospective_evaluation/end_to_end/detected_crops

End-to-end console report:

results/retrospective_evaluation/end_to_end/reports/end_to_end_console.txt

## Interpretation

Model 2 improves the detector substantially over Model 1.

The expanded detector was paired retrospectively with the same successful
pilot recognizer used in Model 1.

The retrospective outputs are intended for qualitative pipeline inspection
and remote demonstration.

No aggregate end-to-end accuracy is claimed because the generated outputs
have not yet been manually scored against the reviewed full-image ground
truth.

## Known limitations

The detector may still identify LCD icons, inactive segments, or display
graphics as text.

Some nearby regions may be merged.

Some low-contrast text may remain undetected.

Recognizer performance on detector-generated crops may be lower than its
standalone manually cropped recognition validation result.

These retrospective results must not be described as part of the original
Model 2 detector experiment.
