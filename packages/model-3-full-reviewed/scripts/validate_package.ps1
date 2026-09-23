param(
    [string]$PackageRoot
)

$ErrorActionPreference = "Stop"

if (-not $PackageRoot) {
    $PackageRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}

Write-Host ""
Write-Host "Validating Model 3 package"
Write-Host "Package root: $PackageRoot"
Write-Host ""

$RequiredFiles = @(
    "$PackageRoot\README.md"
    "$PackageRoot\MODEL_IDENTITY.yml"
    "$PackageRoot\data\training_snapshot\split_manifest.csv"
    "$PackageRoot\data\training_snapshot\split_settings.json"
    "$PackageRoot\data\corrected_dataset\split_manifest.csv"
    "$PackageRoot\data\corrected_dataset\split_settings.json"
    "$PackageRoot\data\corrected_dataset\detector\train\Label.txt"
    "$PackageRoot\data\corrected_dataset\detector\validation\Label.txt"
    "$PackageRoot\data\corrected_dataset\recognizer\train\rec_gt.txt"
    "$PackageRoot\data\corrected_dataset\recognizer\validation\rec_gt.txt"
    "$PackageRoot\models\detector\training\best_accuracy.pdparams"
    "$PackageRoot\models\detector\training\best_accuracy.states"
    "$PackageRoot\models\recognizer\training\best_accuracy.pdparams"
    "$PackageRoot\models\recognizer\training\best_accuracy.states"
    "$PackageRoot\models\recognizer\character_dict.txt"
    "$PackageRoot\results\detector\training_snapshot_metrics.json"
    "$PackageRoot\results\detector\corrected_evaluation\metrics.json"
    "$PackageRoot\results\recognizer\training_snapshot_metrics.json"
    "$PackageRoot\results\recognizer\corrected_evaluation\metrics.json"
    "$PackageRoot\docs\CHECKPOINT_PROVENANCE.txt"
    "$PackageRoot\docs\INFERENCE_PROVENANCE.txt"
    "$PackageRoot\docs\VALIDATION_COMPARISON.md"
    "$PackageRoot\docs\GROUND_TRUTH_STATUS.md"
)

$MissingFiles = @()

foreach ($FilePath in $RequiredFiles) {
    if (-not (Test-Path -LiteralPath $FilePath -PathType Leaf)) {
        $MissingFiles += $FilePath
    }
}

$DetectorTrainImages = @(
    Get-ChildItem `
        -LiteralPath "$PackageRoot\data\corrected_dataset\detector\train\images" `
        -File
).Count

$DetectorValidationImages = @(
    Get-ChildItem `
        -LiteralPath "$PackageRoot\data\corrected_dataset\detector\validation\images" `
        -File
).Count

$DetectorTrainLabels = @(
    Get-Content `
        -LiteralPath "$PackageRoot\data\corrected_dataset\detector\train\Label.txt" |
    Where-Object {
        $_.Trim().Length -gt 0
    }
).Count

$DetectorValidationLabels = @(
    Get-Content `
        -LiteralPath "$PackageRoot\data\corrected_dataset\detector\validation\Label.txt" |
    Where-Object {
        $_.Trim().Length -gt 0
    }
).Count

$RecognizerTrainCrops = @(
    Get-ChildItem `
        -LiteralPath "$PackageRoot\data\corrected_dataset\recognizer\train\crop_img" `
        -File
).Count

$RecognizerValidationCrops = @(
    Get-ChildItem `
        -LiteralPath "$PackageRoot\data\corrected_dataset\recognizer\validation\crop_img" `
        -File
).Count

$RecognizerTrainLabels = @(
    Get-Content `
        -LiteralPath "$PackageRoot\data\corrected_dataset\recognizer\train\rec_gt.txt" |
    Where-Object {
        $_.Trim().Length -gt 0
    }
).Count

$RecognizerValidationLabels = @(
    Get-Content `
        -LiteralPath "$PackageRoot\data\corrected_dataset\recognizer\validation\rec_gt.txt" |
    Where-Object {
        $_.Trim().Length -gt 0
    }
).Count

$DetectorInferenceFiles = @(
    Get-ChildItem `
        -LiteralPath "$PackageRoot\models\detector\inference" `
        -File `
        -ErrorAction SilentlyContinue
).Count

$RecognizerInferenceFiles = @(
    Get-ChildItem `
        -LiteralPath "$PackageRoot\models\recognizer\inference" `
        -File `
        -ErrorAction SilentlyContinue
).Count

$Problems = @()

if ($MissingFiles.Count -gt 0) {
    foreach ($FilePath in $MissingFiles) {
        $Problems += "Missing required file: $FilePath"
    }
}

if ($DetectorTrainImages -ne $DetectorTrainLabels) {
    $Problems += "Detector training image and label counts differ."
}

if ($DetectorValidationImages -ne $DetectorValidationLabels) {
    $Problems += "Detector validation image and label counts differ."
}

if ($RecognizerTrainCrops -ne $RecognizerTrainLabels) {
    $Problems += "Recognizer training crop and label counts differ."
}

if ($RecognizerValidationCrops -ne $RecognizerValidationLabels) {
    $Problems += "Recognizer validation crop and label counts differ."
}

if (($DetectorTrainImages + $DetectorValidationImages) -ne 646) {
    $Problems += "Expected 646 corrected development images."
}

if ($DetectorInferenceFiles -lt 2) {
    $Problems += "Detector inference export is incomplete."
}

if ($RecognizerInferenceFiles -lt 2) {
    $Problems += "Recognizer inference export is incomplete."
}

if ($Problems.Count -gt 0) {
    Write-Host "Validation problems:" -ForegroundColor Red

    foreach ($Problem in $Problems) {
        Write-Host "  $Problem"
    }

    throw "Model 3 package validation failed."
}

Write-Host "Model 3 package validation passed." -ForegroundColor Green
Write-Host ""
Write-Host "Corrected detector training images:       $DetectorTrainImages"
Write-Host "Corrected detector training labels:       $DetectorTrainLabels"
Write-Host "Corrected detector validation images:     $DetectorValidationImages"
Write-Host "Corrected detector validation labels:     $DetectorValidationLabels"
Write-Host "Corrected recognizer training crops:      $RecognizerTrainCrops"
Write-Host "Corrected recognizer training labels:     $RecognizerTrainLabels"
Write-Host "Corrected recognizer validation crops:    $RecognizerValidationCrops"
Write-Host "Corrected recognizer validation labels:   $RecognizerValidationLabels"
Write-Host "Detector inference files:                 $DetectorInferenceFiles"
Write-Host "Recognizer inference files:               $RecognizerInferenceFiles"
