# End-to-End Evaluation Status

## Historical status

No verified end-to-end evaluation was performed for this exact detector and
recognizer pair during the original Model 1 experiment.

The original detector and recognizer metrics are standalone validation
metrics.

## Retrospective status

A retrospective end-to-end evaluation was later performed using:

Detector: det_lcd_v1_60

Recognizer: rec_lcd_v2_finetuned

Inputs: all 10 original pilot validation images

The generated results are stored under:

results/retrospective_evaluation/end_to_end

The retrospective outputs are provided for qualitative inspection and remote
demonstration.

No aggregate end-to-end accuracy is claimed until each generated prediction
is manually compared with the reviewed full-image ground truth.
