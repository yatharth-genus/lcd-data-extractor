$ErrorActionPreference = "Stop"
$Root = Resolve-Path "$PSScriptRoot\.."
$Required = @(
    "$Root\config\detector.yml",
    "$Root\config\recognizer.yml",
    "$Root\data\detector\train\Label.txt",
    "$Root\data\detector\validation\Label.txt",
    "$Root\data\recognizer\train\rec_gt.txt",
    "$Root\data\recognizer\validation\rec_gt.txt",
    "$Root\models\detector\training\best_accuracy.pdparams",
    "$Root\models\recognizer\training\best_accuracy.pdparams",
    "$Root\models\recognizer\character_dict.txt"
)
$Missing = @($Required | Where-Object { -not (Test-Path -LiteralPath $_) })
if ($Missing.Count -gt 0) {
    Write-Host "Missing required files:" -ForegroundColor Red
    $Missing | ForEach-Object { Write-Host "  $_" }
    exit 1
}
$TrainImages = @(Get-ChildItem "$Root\data\detector\train\images" -File).Count
$ValImages = @(Get-ChildItem "$Root\data\detector\validation\images" -File).Count
$TrainCrops = @(Get-ChildItem "$Root\data\recognizer\train\crop_img" -File).Count
$ValCrops = @(Get-ChildItem "$Root\data\recognizer\validation\crop_img" -File).Count
Write-Host "Package verification passed." -ForegroundColor Green
Write-Host "Detector train images: $TrainImages"
Write-Host "Detector validation images: $ValImages"
Write-Host "Recognizer train crops: $TrainCrops"
Write-Host "Recognizer validation crops: $ValCrops"