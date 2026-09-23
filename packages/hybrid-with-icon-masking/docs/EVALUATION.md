Icon-masked hybrid evaluation record

Dataset:
97 corrected validation images

Objective:
Prevent trusted icon detections from being emitted as OCR text.

Visual review:
Icon-related OCR outputs were removed from the end-to-end visual results.

Unmasked detector metrics:
Precision 0.5115384615384615
Recall 0.7169811320754716
Hmean 0.5970819304152638
FPS 0.6987833929373587

Masked detector metrics:
Precision 0.4649910233393178
Recall 0.6981132075471698
Hmean 0.5581896551724138
FPS 0.6942090873756062

Interpretation:
Pre-OCR black masking achieved the application-level icon-removal objective but reduced standalone text-detection precision, recall, and Hmean on the controlled 97-image evaluation.

Decision:
Retain the masked pipeline as a transparent functional baseline. Continue improving OCR with more representative training data, hard negatives, and broader testing. Preserve the unmasked hybrid as the metric baseline and fallback.
