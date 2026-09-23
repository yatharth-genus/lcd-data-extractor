param(
    [Parameter(Mandatory = $true)]
    [string]$ImagePath,

    [string]$PaddleRoot = "D:\Actual Project\PaddleOCR-2.8",

    [string]$OutputFolder,

    [string]$FontPath = "C:\Windows\Fonts\arial.ttf"
)

$ErrorActionPreference = "Stop"

$PackageRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

if (-not (Test-Path -LiteralPath $ImagePath)) {
    throw "Input image or folder not found: $ImagePath"
}

if (-not (Test-Path -LiteralPath $PaddleRoot -PathType Container)) {
    throw "PaddleOCR folder not found: $PaddleRoot"
}

if (-not (Test-Path -LiteralPath $FontPath -PathType Leaf)) {
    throw "Visualization font not found: $FontPath"
}

if (-not $OutputFolder) {
    $OutputFolder = Join-Path $PackageRoot "results\new_inference"
}

$PredictSystem = Get-ChildItem `
    -LiteralPath (Join-Path $PaddleRoot "tools") `
    -Recurse `
    -File `
    -Filter "predict_system.py" |
Select-Object -First 1

if (-not $PredictSystem) {
    throw "predict_system.py was not found under $PaddleRoot\tools"
}

$DetectorModel = Join-Path $PackageRoot "models\detector\inference"

$RecognizerModel = Join-Path $PackageRoot "models\recognizer\inference"

$Dictionary = Join-Path $PackageRoot "models\recognizer\character_dict.txt"

if (-not (Test-Path -LiteralPath $DetectorModel -PathType Container)) {
    throw "Detector inference model folder not found: $DetectorModel"
}

if (-not (Test-Path -LiteralPath $RecognizerModel -PathType Container)) {
    throw "Recognizer inference model folder not found: $RecognizerModel"
}

if (-not (Test-Path -LiteralPath $Dictionary -PathType Leaf)) {
    throw "Recognizer character dictionary not found: $Dictionary"
}

$DetectorInferenceFiles = @(
    Get-ChildItem `
        -LiteralPath $DetectorModel `
        -File `
        -ErrorAction SilentlyContinue
)

$RecognizerInferenceFiles = @(
    Get-ChildItem `
        -LiteralPath $RecognizerModel `
        -File `
        -ErrorAction SilentlyContinue
)

if ($DetectorInferenceFiles.Count -lt 2) {
    throw "Detector inference model folder is incomplete: $DetectorModel"
}

if ($RecognizerInferenceFiles.Count -lt 2) {
    throw "Recognizer inference model folder is incomplete: $RecognizerModel"
}

$CropFolder = Join-Path $OutputFolder "detected_crops"

$RenderedFolder = Join-Path $OutputFolder "rendered_results"

$ReportsFolder = Join-Path $OutputFolder "reports"

$ConsoleLog = Join-Path $ReportsFolder "end_to_end_console.txt"

New-Item `
    -ItemType Directory `
    -Path $CropFolder `
    -Force |
Out-Null

New-Item `
    -ItemType Directory `
    -Path $RenderedFolder `
    -Force |
Out-Null

New-Item `
    -ItemType Directory `
    -Path $ReportsFolder `
    -Force |
Out-Null

Write-Host ""
Write-Host "Running hybrid LCD OCR"
Write-Host "Input:      $ImagePath"
Write-Host "Detector:   $DetectorModel"
Write-Host "Recognizer: $RecognizerModel"
Write-Host "Dictionary: $Dictionary"
Write-Host "Font:       $FontPath"
Write-Host "Output:     $OutputFolder"
Write-Host ""

$PreviousErrorActionPreference = $ErrorActionPreference

$ErrorActionPreference = "Continue"

python $PredictSystem.FullName `
    --image_dir $ImagePath `
    --det_model_dir $DetectorModel `
    --rec_model_dir $RecognizerModel `
    --rec_char_dict_path $Dictionary `
    --vis_font_path $FontPath `
    --use_angle_cls false `
    --use_gpu false `
    --drop_score 0.3 `
    --draw_img_save_dir $RenderedFolder `
    --save_crop_res true `
    --crop_res_save_dir $CropFolder 2>&1 |
Tee-Object -FilePath $ConsoleLog

$PythonExitCode = $LASTEXITCODE

$ErrorActionPreference = $PreviousErrorActionPreference

if ($PythonExitCode -ne 0) {
    throw "Hybrid OCR failed with exit code $PythonExitCode. Review: $ConsoleLog"
}

Write-Host ""
Write-Host "Hybrid OCR completed successfully." -ForegroundColor Green
Write-Host "Rendered results: $RenderedFolder"
Write-Host "Detected crops:   $CropFolder"
Write-Host "Console log:      $ConsoleLog"
