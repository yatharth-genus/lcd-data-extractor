# Model 3 Validation Comparison

## Training source

Both Model 3 components were trained using the pre-correction `dataset_final`
snapshot.

Additional ground-truth corrections were applied to the authoritative source
annotations on September 22, 2026.

## Detector

### Training-snapshot validation

Precision: 0.42905405405405406

Recall: 0.6846361185983828

Hmean: 0.5275181723779855

### Corrected validation

Precision: 0.42905405405405406

Recall: 0.6846361185983828

Hmean: 0.5275181723779855

Evaluation FPS: 0.6936077281342832

### Detector conclusion

The corrected validation labels did not change detector precision, recall, or
Hmean.

Model 3 detector performance remains below Model 2 detector performance.

Model 2 detector Hmean: 0.6863468634686346

Model 3 detector Hmean: 0.5275181723779855

## Recognizer

### Training-snapshot validation

Exact accuracy: 0.940217365754963

Normalized edit similarity: 0.9728260876949433

### Corrected validation

Exact accuracy: 0.940217365754963

Normalized edit similarity: 0.9728260876949433

Evaluation FPS: 4.224947567669982

### Recognizer conclusion

The corrected validation labels did not change recognizer accuracy or
normalized edit similarity.

The Model 3 recognizer remains substantially better than the recognizer used
by Models 1 and 2.

Previous recognizer accuracy: 0.8749997265625854

Model 3 recognizer accuracy: 0.940217365754963

## Recommended deployment candidate

Detector:

Model 2 detector, det_lcd_v2_expanded_240

Recognizer:

Model 3 recognizer, rec_lcd_v3_full_reviewed

This hybrid combination requires a separate retrospective evaluation before
being selected as the recommended pipeline.
