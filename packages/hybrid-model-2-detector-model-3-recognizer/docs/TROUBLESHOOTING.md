# Troubleshooting

## OSError: cannot open resource

### Cause

PaddleOCR completed detection and recognition successfully but failed while
drawing recognized text because the configured TrueType font could not be
opened.

Typical traceback location:

PIL.ImageFont.truetype

### Solution

Supply a valid TrueType font using the FontPath argument.

Example:

powershell -ExecutionPolicy Bypass -File ".\scripts\run_hybrid.ps1" -ImagePath "PATH_TO_IMAGE" -PaddleRoot "D:\Actual Project\PaddleOCR-2.8" -FontPath "C:\Windows\Fonts\arial.ttf"

The font path must point to an existing TTF or OTF file.

### Important

This error affects rendered visualization only. Detection and recognition may
already have completed successfully before the rendering failure.

## PowerShell NativeCommandError

PaddleOCR may write progress or diagnostic output to stderr. The packaged
runner records the Python exit code and does not treat stderr output alone as
a model failure.

## Git LFS pointer files

If images or model files are approximately 130 bytes and cannot be opened,
run:

git lfs install

git lfs pull

git lfs checkout
