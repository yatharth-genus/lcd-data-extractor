param(
    [string]$PackageRoot
)

$ErrorActionPreference = "Stop"

if (-not $PackageRoot) {
    $PackageRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}

$RequiredFiles = @(
    "$PackageRoot\README.md"
    "$PackageRoot\PIPELINE_IDENTITY.yml"
    "$PackageRoot\config\detector_source.yml"
    "$PackageRoot\config\recognizer_source.yml"
    "$PackageRoot\data\validation\ground_truth\Label.txt"
    "$PackageRoot\models\recognizer\character_dict.txt"
    "$PackageRoot\scripts\run_hybrid.ps1"
)

$Problems = @()

foreach ($FilePath in $RequiredFiles) {
    if (-not (Test-Path -LiteralPath $FilePath -PathType Leaf)) {
        $Problems += "Missing required file: $FilePath"
    }
}

$ValidationImages = @(
    Get-ChildItem `
        -LiteralPath "$PackageRoot\data\validation\images" `
        -File
).Count

$ValidationLabels = @(
    Get-Content `
        -LiteralPath "$PackageRoot\data\validation\ground_truth\Label.txt" |
    Where-Object {
        $_.Trim().Length -gt 0
    }
).Count

$DetectorInferenceFiles = @(
    Get-ChildItem `
        -LiteralPath "$PackageRoot\models\detector\inference" `
        -File
).Count

$RecognizerInferenceFiles = @(
    Get-ChildItem `
        -LiteralPath "$PackageRoot\models\recognizer\inference" `
        -File
).Count

if ($ValidationImages -ne 97) {
    $Problems += "Expected 97 corrected validation images, found $ValidationImages"
}

if ($ValidationImages -ne $ValidationLabels) {
    $Problems += "Validation image and full-image label counts differ."
}

if ($DetectorInferenceFiles -lt 2) {
    $Problems += "Detector inference model is incomplete."
}

if ($RecognizerInferenceFiles -lt 2) {
    $Problems += "Recognizer inference model is incomplete."
}

if ($Problems.Count -gt 0) {
    Write-Host "Hybrid package validation problems:" -ForegroundColor Red

    foreach ($Problem in $Problems) {
        Write-Host "  $Problem"
    }

    throw "Hybrid package validation failed."
}

Write-Host "Hybrid package validation passed." -ForegroundColor Green
Write-Host ""
Write-Host "Validation images:          $ValidationImages"
Write-Host "Validation labels:          $ValidationLabels"
Write-Host "Detector inference files:   $DetectorInferenceFiles"
Write-Host "Recognizer inference files: $RecognizerInferenceFiles"
