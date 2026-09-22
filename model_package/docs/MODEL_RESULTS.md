# Model Results: v1 Pilot 60

## Model identity

- Branch: `model/v1-pilot-60`
- Detector run: `det_lcd_v1_60`
- Recognizer run: `rec_lcd_v2_finetuned`
- Training device: CPU
- Framework: PaddleOCR release/2.8 with PaddlePaddle 2.6.2

## Detector

- Training images: 60
- Validation images: 10
- Initialization: MobileNetV3 ImageNet-pretrained backbone
- Epochs completed: 50
- Best epoch: 33
- Precision: 0.5555555555555556
- Recall: 0.39473684210526316
- Hmean: 0.46153846153846156
- Validation FPS: 0.24734973737458457

## Recognizer

- Training source: recognition crops from 60 full images
- Validation source: recognition crops from 10 full images
- Initialization: official English PP-OCRv3 recognition checkpoint
- Epochs completed: 50
- Best epoch: 45
- Exact validation accuracy: 0.8749997265625854
- Normalized edit similarity: 0.9418402959526853
- Validation FPS: 0.4514875830857429

## Interpretation

The recognizer learned successfully from pretrained weights. The detector learned a usable baseline but remained the pipeline bottleneck because recall was below 40%. The v1 branch is retained as the controlled pilot baseline for comparison with later expanded models.