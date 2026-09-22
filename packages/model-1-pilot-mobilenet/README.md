
## Retrospective end-to-end evaluation

A retrospective end-to-end evaluation was performed after the original
component experiments.

The evaluation used:

- Detector: det_lcd_v1_60
- Recognizer: rec_lcd_v2_finetuned
- Input set: all 10 original pilot validation images

The generated outputs are stored under:

results/retrospective_evaluation

This evaluation is intended for qualitative inspection and remote
demonstration. It does not alter the original standalone detector or
recognizer metrics.

No aggregate end-to-end accuracy is claimed because the generated
predictions have not yet been manually scored against the full-image ground
truth.

See:

docs/RETROSPECTIVE_EVALUATION_LOG.md
