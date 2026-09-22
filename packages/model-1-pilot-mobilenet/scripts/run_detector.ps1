param(
    [Parameter(Mandatory = $true)]
    [string]$ImagePath,

    [string]$PaddleRoot = "D:\Actual Project\PaddleOCR-2.8",

    [string]$OutputFolder
)

$ErrorActionPreference = "Stop"

$PackageRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

if (-not (Test-Path -LiteralPath $ImagePath)) {
    throw "Input image or folder not found: $ImagePath"
}

if (-not $OutputFolder) {
    $OutputFolder = Join-Path $PackageRoot "results\detector\inference_results"
}

$PredictDetector = Join-Path $PaddleRoot "tools\infer\predict_det.py"
$DetectorModel = Join-Path $PackageRoot "models\detector\inference"

if (-not (Test-Path -LiteralPath $PredictDetector -PathType Leaf)) {
    throw "PaddleOCR detector script not found: $PredictDetector"
}

if (-not (Test-Path -LiteralPath $DetectorModel -PathType Container)) {
    throw "Detector inference model folder not found: $DetectorModel"
}

$InferenceFiles = @(
    Get-ChildItem -LiteralPath $DetectorModel -File
)

if ($InferenceFiles.Count -lt 2) {
    throw "Detector inference model folder is incomplete: $DetectorModel"
}

New-Item -ItemType Directory -Path $OutputFolder -Force | Out-Null

Write-Host ""
Write-Host "Running Model 1 detector inference"
Write-Host "Input:  $ImagePath"
Write-Host "Model:  $DetectorModel"
Write-Host "Output: $OutputFolder"
Write-Host ""

python $PredictDetector `
    --image_dir $ImagePath `
    --det_model_dir $DetectorModel `
    --use_gpu false `
    --draw_img_save_dir $OutputFolder

if ($LASTEXITCODE -ne 0) {
    throw "Detector inference failed with exit code $LASTEXITCODE"
}

Write-Host ""
Write-Host "Detector inference completed successfully." -ForegroundColor Green
Write-Host "Results: $OutputFolder"
