# OpenVINO LCD Icon Detector

This package contains the independently verified LCD icon detector received in the handover package.

## Runtime

Primary runtime: OpenVINO

Primary files:

model/saved_model.xml
model/saved_model.bin

Alternate export:

model/saved_model.onnx

The XML and BIN files must remain together.

## Input and preprocessing

Runtime tensor: 1 x 3 x 640 x 640, float32

The runner decodes with OpenCV, converts BGR to RGB, resizes while preserving aspect ratio, center-pads to 640 x 640, normalizes using the packaged mean and standard deviation, transposes HWC to CHW, and adds the batch dimension.

## Default threshold

0.38

## Classes

0 ok
1 earth
2 magnet
3 comm
4 switchON
5 switchOFF
6 battery
7 reverseCN
8 ForwardCN
9 neutral
10 powerON
11 coverOpen

Class zero is excluded from normal output.

## Environment

Create a dedicated Python 3.10 environment and run:

python -m pip install -r requirements/requirements.txt

## Run one image

python scripts/run_icon_detector.py PATH_TO_IMAGE --output results/new_run --device CPU --threshold 0.38

## Run a folder

python scripts/run_icon_detector.py PATH_TO_IMAGE_FOLDER --output results/new_run --device CPU --threshold 0.38

## Verified evaluation

The detector completed standalone evaluation on 97 corrected validation images. Inference, class mapping, original-coordinate box restoration, rendering, and report generation were reviewed and verified.

## Integration status

OCR overlap filtering has not started. The standalone icon package is preserved before integration work begins.
