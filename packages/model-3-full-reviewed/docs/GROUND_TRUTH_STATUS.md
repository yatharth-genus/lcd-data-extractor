# Model 3 Ground-Truth Status

## Current training snapshot

The running Model 3 detector and recognizer use the dataset compiled before
additional ground-truth corrections were discovered on September 22, 2026.

The active dataset was not modified during training.

## Corrected dataset

The corrected authoritative annotations were compiled separately into:

data/corrected_dataset

The corrected dataset passed image, label, crop, and split consistency
validation.

## Reporting classification

The current Model 3 training runs are intermediate full-dataset experiments.

The corrected dataset is the authoritative submission dataset and should be
used for any future production retraining.

Completed Model 3 results must state that the models were trained on the
training snapshot, not the corrected submission dataset.
