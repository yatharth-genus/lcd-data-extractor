# Model 1 Retrospective Evaluation Log

## Evaluation identity

Evaluation date: 2026-09-22 16:02:12

Package: model-1-pilot-mobilenet

Evaluation type: Retrospective standalone and end-to-end evaluation

This evaluation was performed after the original Model 1 component
experiments. It is not a historical training-time end-to-end result.

## Packaged model components

Detector run: det_lcd_v1_60

Detector initialization: ImageNet-pretrained MobileNetV3 backbone

Detector checkpoint: best_accuracy

Detector best epoch: 33

Recognizer run: rec_lcd_v2_finetuned

Recognizer initialization: Official English PP-OCRv3 recognition checkpoint

Recognizer checkpoint: best_accuracy

Recognizer best epoch: 45

## Historical standalone metrics

Detector precision: 0.5555555555555556

Detector recall: 0.39473684210526316

Detector Hmean: 0.46153846153846156

Recognizer exact validation accuracy: 0.8749997265625854

Recognizer normalized edit similarity: 0.9418402959526853

## Retrospective evaluation scope

Full validation images tested: 10

Standalone recognition crops tested: 44

End-to-end rendered files produced: 11

Detector-generated text crops produced: 25

## Inputs

Full-image inputs:

data/detector/validation/images

Standalone recognition inputs:

data/recognizer/validation/crop_img

## Outputs

Standalone detector outputs:

results/retrospective_evaluation/detector

Standalone recognizer report:

results/retrospective_evaluation/recognizer/reports/recognizer_console.txt

End-to-end rendered outputs:

results/retrospective_evaluation/end_to_end/rendered_results

Detector-generated crops:

results/retrospective_evaluation/end_to_end/detected_crops

End-to-end console log:

results/retrospective_evaluation/end_to_end/reports/end_to_end_console.txt

## Interpretation

The original detector and recognizer were evaluated separately.

This retrospective evaluation combines the first detector baseline with the
first successful pretrained recognizer for qualitative pipeline inspection
and remote demonstration.

No aggregate end-to-end accuracy is claimed because the end-to-end
predictions have not yet been manually scored against the reviewed
full-image annotations.

## Known limitations

The Model 1 detector has limited recall.

The detector may miss valid text regions.

The detector may identify graphical icons, inactive LCD segments, or display
borders as text.

Recognizer performance on detector-generated crops may be lower than its
historical performance on manually reviewed recognition crops.

These retrospective results must not be described as part of the original
Model 1 experiment.
