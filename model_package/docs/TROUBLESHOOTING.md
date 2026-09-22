# Troubleshooting and Known Issues

## Environment

- Use Python 3.10 with PaddlePaddle 2.6.2 and PaddleOCR release/2.8.
- Use `num_workers: 0` on Windows to avoid DataLoader process issues.
- Keep annotation and training environments separate.
- Install `openpyxl` for Excel report generation and `Pillow` for image validation.

## PPOCRLabel

- Older PPOCRLabel code may pass floating-point values to PyQt methods that require integers. Wrap affected coordinates and scrollbar values with `int()`.
- Automatic OCR output may contain an outer page-level list. Flatten the result before saving annotations.
- A no-detection result may appear as `[None]`; handle it before iterating.
- Editing `Label.txt` does not automatically refresh `rec_gt.txt` or `crop_img`. Regenerate recognition exports after label corrections.

## Dataset

- Read label files as UTF-8 with optional BOM handling.
- Use relative crop paths joined to split-specific `data_dir` values.
- Polygon coordinates must be less than image width and height. A point equal to the height must be clamped to `height - 1`.
- PPOCRLabel crop names may contain the original extension, such as `425.png_crop_0.jpg`.
- Ground truths require visual review. Known mistakes found during development included `Pu5h` and `KVA h`.
- Split at the full-image level. Never place crops from one image in both training and validation.

## Training

- Scratch recognition failed on the small pilot dataset. Use official pretrained recognition weights.
- The initial detector used an ImageNet backbone and had limited performance. Later experiments use the fully trained PP-OCRv3 student detector.
- Use `best_accuracy` for model selection. Use `latest` only to resume optimizer and epoch state.
- PowerShell may display `NativeCommandError` when progress bars write to stderr through `Tee-Object`; if training continues and checkpoints are saved, this is not necessarily a failure.
- CPU training is slow. Prevent Windows sleep and avoid running multiple heavy jobs if memory pressure is high.

## Inference

- The recognizer accepts text crops, not full LCD images.
- PaddleOCR 2.8 recognition logs each prediction as two lines: `infer_img:` followed by `result:`.
- End-to-end inference requires exported detector and recognizer inference models, plus the exact character dictionary.
- DB thresholds trade precision against recall and do not replace training.
- Icon false positives may later be suppressed using overlap with a trusted icon detector.