param(
    [string]$PackageRoot = ""
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($PackageRoot)) {
    $PackageRoot = (
        Resolve-Path "$PSScriptRoot\.."
    ).Path
}

Write-Host "Validating Model 1 package"
Write-Host "Package root: $PackageRoot"
Write-Host ""

$RequiredFiles = @(
    "$PackageRoot\MODEL_IDENTITY.yml",
    "$PackageRoot\config\detector_original.yml",
    "$PackageRoot\config\detector_portable.yml",
    "$PackageRoot\config\recognizer_original.yml",
    "$PackageRoot\config\recognizer_portable.yml",
    "$PackageRoot\data\calibration\Label.txt",
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

    throw "Model 1 package validation failed."
}

$CalibrationImages = @(
    Get-ChildItem `
        -LiteralPath "$PackageRoot\data\calibration\images" `
        -File
).Count

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

$CalibrationLabelRows = @(
    Get-Content `
        -LiteralPath "$PackageRoot\data\calibration\Label.txt" |
    Where-Object {
        $_.Trim().Length -gt 0
    }
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

$ValidationErrors = @()

if ($CalibrationImages -ne 10) {
    $ValidationErrors += "Expected 10 calibration images, found $CalibrationImages"
}

if ($CalibrationLabelRows -ne 10) {
    $ValidationErrors += "Expected 10 calibration label rows, found $CalibrationLabelRows"
}

if ($DetectorTrainImages -ne 60) {
    $ValidationErrors += "Expected 60 detector training images, found $DetectorTrainImages"
}

if ($DetectorTrainLabelRows -ne 60) {
    $ValidationErrors += "Expected 60 detector training label rows, found $DetectorTrainLabelRows"
}

if ($DetectorValidationImages -ne 10) {
    $ValidationErrors += "Expected 10 detector validation images, found $DetectorValidationImages"
}

if ($DetectorValidationLabelRows -ne 10) {
    $ValidationErrors += "Expected 10 detector validation label rows, found $DetectorValidationLabelRows"
}

if ($RecognizerTrainCrops -ne $RecognizerTrainLabelRows) {
    $ValidationErrors += (
        "Recognizer training mismatch: " +
        "$RecognizerTrainCrops crops and " +
        "$RecognizerTrainLabelRows label rows"
    )
}

if ($RecognizerValidationCrops -ne $RecognizerValidationLabelRows) {
    $ValidationErrors += (
        "Recognizer validation mismatch: " +
        "$RecognizerValidationCrops crops and " +
        "$RecognizerValidationLabelRows label rows"
    )
}

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

if ($DetectorInferenceFiles -lt 2) {
    $ValidationErrors += "Detector inference model is incomplete."
}

if ($RecognizerInferenceFiles -lt 2) {
    $ValidationErrors += "Recognizer inference model is incomplete."
}

if ($ValidationErrors.Count -gt 0) {
    Write-Host "Validation problems:" -ForegroundColor Red

    foreach ($Problem in $ValidationErrors) {
        Write-Host "  $Problem"
    }

    throw "Model 1 package validation failed."
}

Write-Host "Model 1 package validation passed." -ForegroundColor Green
Write-Host ""
Write-Host "Calibration images: $CalibrationImages"
Write-Host "Detector training images: $DetectorTrainImages"
Write-Host "Detector validation images: $DetectorValidationImages"
Write-Host "Recognizer training crops: $RecognizerTrainCrops"
Write-Host "Recognizer validation crops: $RecognizerValidationCrops"
Write-Host "Detector inference files: $DetectorInferenceFiles"
Write-Host "Recognizer inference files: $RecognizerInferenceFiles"

