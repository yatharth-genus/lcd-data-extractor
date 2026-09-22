param(
    [string]$PackageRoot
)

$ErrorActionPreference = "Stop"

if (-not $PackageRoot) {
    $PackageRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}

Write-Host ""
Write-Host "Validating Model 2 package"
Write-Host "Package root: $PackageRoot"
Write-Host ""

$RequiredFiles = @(
    "$PackageRoot\MODEL_IDENTITY.yml",
    "$PackageRoot\config\detector_original.yml",
    "$PackageRoot\config\detector_portable.yml",
    "$PackageRoot\config\recognizer_original.yml",
    "$PackageRoot\config\recognizer_portable.yml",
    "$PackageRoot\data\detector\train\Label.txt",
    "$PackageRoot\data\detector\validation\Label.txt",
    "$PackageRoot\data\recognizer\train\rec_gt.txt",
    "$PackageRoot\data\recognizer\validation\rec_gt.txt",
    "$PackageRoot\models\detector\training\best_accuracy.pdparams",
    "$PackageRoot\models\recognizer\training\best_accuracy.pdparams",
    "$PackageRoot\models\recognizer\character_dict.txt",
    "$PackageRoot\results\detector\standalone_metrics.json",
    "$PackageRoot\results\recognizer\standalone_metrics.json",
    "$PackageRoot\results\end_to_end\NOT_EVALUATED.md"
)

$MissingFiles = @()

foreach ($File in $RequiredFiles) {
    if (-not (Test-Path -LiteralPath $File -PathType Leaf)) {
        $MissingFiles += $File
    }
}

if ($MissingFiles.Count -gt 0) {
    Write-Host "Missing required files:" -ForegroundColor Red

    foreach ($File in $MissingFiles) {
        Write-Host "  $File"
    }

    throw "Model 2 package validation failed because required files are missing."
}

$DetectorTrainImages = @(
    Get-ChildItem `
        -LiteralPath "$PackageRoot\data\detector\train\images" `
        -File
).Count

$DetectorValidationImages = @(
    Get-ChildItem `
        -LiteralPath "$PackageRoot\data\detector\validation\images" `
        -File
).Count

$DetectorTrainLabelRows = @(
    Get-Content `
        -LiteralPath "$PackageRoot\data\detector\train\Label.txt" |
    Where-Object {
        $_.Trim().Length -gt 0
    }
).Count

$DetectorValidationLabelRows = @(
    Get-Content `
        -LiteralPath "$PackageRoot\data\detector\validation\Label.txt" |
    Where-Object {
        $_.Trim().Length -gt 0
    }
).Count

$RecognizerTrainCrops = @(
    Get-ChildItem `
        -LiteralPath "$PackageRoot\data\recognizer\train\crop_img" `
        -File
).Count

$RecognizerValidationCrops = @(
    Get-ChildItem `
        -LiteralPath "$PackageRoot\data\recognizer\validation\crop_img" `
        -File
).Count

$RecognizerTrainLabelRows = @(
    Get-Content `
        -LiteralPath "$PackageRoot\data\recognizer\train\rec_gt.txt" |
    Where-Object {
        $_.Trim().Length -gt 0
    }
).Count

$RecognizerValidationLabelRows = @(
    Get-Content `
        -LiteralPath "$PackageRoot\data\recognizer\validation\rec_gt.txt" |
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

if ($DetectorTrainImages -ne 240) {
    $Problems += "Expected 240 detector training images, found $DetectorTrainImages"
}

if ($DetectorTrainLabelRows -ne 240) {
    $Problems += "Expected 240 detector training label rows, found $DetectorTrainLabelRows"
}

if ($DetectorValidationImages -ne 28) {
    $Problems += "Expected 28 detector validation images, found $DetectorValidationImages"
}

if ($DetectorValidationLabelRows -ne 28) {
    $Problems += "Expected 28 detector validation label rows, found $DetectorValidationLabelRows"
}

if ($RecognizerTrainCrops -ne 281) {
    $Problems += "Expected 281 recognizer training crops, found $RecognizerTrainCrops"
}

if ($RecognizerTrainLabelRows -ne 281) {
    $Problems += "Expected 281 recognizer training label rows, found $RecognizerTrainLabelRows"
}

if ($RecognizerValidationCrops -ne 44) {
    $Problems += "Expected 44 recognizer validation crops, found $RecognizerValidationCrops"
}

if ($RecognizerValidationLabelRows -ne 44) {
    $Problems += "Expected 44 recognizer validation label rows, found $RecognizerValidationLabelRows"
}

if ($RecognizerTrainCrops -ne $RecognizerTrainLabelRows) {
    $Problems += "Recognizer training crop and label counts do not match"
}

if ($RecognizerValidationCrops -ne $RecognizerValidationLabelRows) {
    $Problems += "Recognizer validation crop and label counts do not match"
}

if ($DetectorInferenceFiles -lt 2) {
    $Problems += "Detector inference model folder is incomplete"
}

if ($RecognizerInferenceFiles -lt 2) {
    $Problems += "Recognizer inference model folder is incomplete"
}

if ($Problems.Count -gt 0) {
    Write-Host "Validation problems:" -ForegroundColor Red

    foreach ($Problem in $Problems) {
        Write-Host "  $Problem"
    }

    throw "Model 2 package validation failed."
}

Write-Host "Model 2 package validation passed." -ForegroundColor Green
Write-Host ""
Write-Host "Detector training images:       $DetectorTrainImages"
Write-Host "Detector training label rows:   $DetectorTrainLabelRows"
Write-Host "Detector validation images:     $DetectorValidationImages"
Write-Host "Detector validation label rows: $DetectorValidationLabelRows"
Write-Host "Recognizer training crops:      $RecognizerTrainCrops"
Write-Host "Recognizer training labels:     $RecognizerTrainLabelRows"
Write-Host "Recognizer validation crops:    $RecognizerValidationCrops"
Write-Host "Recognizer validation labels:   $RecognizerValidationLabelRows"
Write-Host "Detector inference files:       $DetectorInferenceFiles"
Write-Host "Recognizer inference files:     $RecognizerInferenceFiles"
