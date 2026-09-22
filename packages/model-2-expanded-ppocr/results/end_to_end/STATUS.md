# End-to-End Evaluation Status

## Historical status

No verified end-to-end evaluation was performed during the original Model 2
detector experiment.

The detector and recognizer metrics are standalone validation metrics.

## Retrospective status

A retrospective end-to-end evaluation was later performed using:

Detector: det_lcd_v2_expanded_240

Recognizer: rec_lcd_v2_finetuned

The recognizer was reused unchanged from Model 1.

Inputs: all 28 Model 2 detector validation images

Outputs:

results/retrospective_evaluation/end_to_end

The retrospective outputs are provided for qualitative inspection and remote
demonstration.

No aggregate end-to-end accuracy is claimed until the predictions are
manually compared against the reviewed full-image ground truth.
