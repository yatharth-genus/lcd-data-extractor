# Model 3 Evaluation Status

## Training-snapshot evaluation

The selected Model 3 checkpoints were evaluated against the original
training-snapshot validation data.

These results reproduce the validation conditions used during training and
checkpoint selection.

## Corrected-dataset evaluation

The same checkpoints were separately evaluated against the corrected
validation data compiled after the September 22, 2026 ground-truth updates.

The corrected evaluation does not change the data on which the models were
trained.

## Retrospective end-to-end evaluation

Retrospective end-to-end OCR was run on the corrected validation images using:

Detector:

det_lcd_v3_full_reviewed

Recognizer:

rec_lcd_v3_full_reviewed

The results are stored under:

results/corrected_retrospective_evaluation

No aggregate end-to-end accuracy is claimed until the rendered predictions
are manually scored against the corrected full-image annotations.
