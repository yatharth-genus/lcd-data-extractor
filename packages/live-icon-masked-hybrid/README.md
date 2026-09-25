# Live Icon-Masked Hybrid

This is a self-contained live inference package for arbitrary LCD images.

Every image runs through all three stages:

1. OpenVINO icon detection
2. Black masking of freshly detected icon regions
3. Model 2 OCR detection plus Model 3 OCR recognition

No pre-generated `icon_detections.json` is required.

## Install

```powershell
python -m pip install -r .\requirements.txt
```

## Visual tester

```powershell
.\run_api.ps1
```

Open `http://127.0.0.1:8000/`.

## Batch inference

```powershell
.\run_batch.ps1 -Images "C:\path\to\images" -Output ".\results\new_run"
```

Duplicate text and icon occurrences are preserved. Development outputs include confidence values, positions, timings, JSON, and rendered images.
