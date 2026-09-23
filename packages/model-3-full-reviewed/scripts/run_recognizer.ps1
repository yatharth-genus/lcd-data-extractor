param(
    [Parameter(Mandatory = $true)]
    [string]$CropPath,

    [string]$PaddleRoot = "D:\Actual Project\PaddleOCR-2.8"
)

$ErrorActionPreference = "Stop"

$PackageRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

if (-not (Test-Path -LiteralPath $CropPath)) {
    throw "Input crop or folder not found: $CropPath"
}

$PredictRecognizer = Join-Path $PaddleRoot "tools\infer\predict_rec.py"

if (-not (Test-Path -LiteralPath $PredictRecognizer -PathType Leaf)) {
    throw "PaddleOCR recognizer script not found: $PredictRecognizer"
}

$RecognizerModel = Join-Path $PackageRoot "models\recognizer\inference"

if (-not (Test-Path -LiteralPath $RecognizerModel -PathType Container)) {
    throw "Recognizer inference model folder not found: $RecognizerModel"
}

$CharacterDictionary = Join-Path $PackageRoot "models\recognizer\character_dict.txt"

if (-not (Test-Path -LiteralPath $CharacterDictionary -PathType Leaf)) {
    throw "Character dictionary not found: $CharacterDictionary"
}

$InferenceFiles = @(
    Get-ChildItem -LiteralPath $RecognizerModel -File
)

if ($InferenceFiles.Count -lt 2) {
    throw "Recognizer inference model folder is incomplete: $RecognizerModel"
}

Write-Host ""
Write-Host "Running Model 1 recognizer inference"
Write-Host "Input:      $CropPath"
Write-Host "Model:      $RecognizerModel"
Write-Host "Dictionary: $CharacterDictionary"
Write-Host ""

python $PredictRecognizer `
    --image_dir $CropPath `
    --rec_model_dir $RecognizerModel `
    --rec_char_dict_path $CharacterDictionary `
    --use_gpu false

if ($LASTEXITCODE -ne 0) {
    throw "Recognizer inference failed with exit code $LASTEXITCODE"
}

Write-Host ""
Write-Host "Recognizer inference completed successfully." -ForegroundColor Green
