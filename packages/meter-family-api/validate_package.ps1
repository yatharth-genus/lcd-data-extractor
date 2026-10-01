$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Required = @(
    "app\main.py",
    "app\pipeline_registry.py",
    "app\icon_detector.py",
    "app\ocr_engine.py",
    "app\v6_ocr_engine.py",
    "app\v6_ocr_worker.py",
    "models\icon_detector\model\saved_model.xml",
    "models\icon_detector\model\saved_model.bin",
    "models\icon_detector\dm.json",
    "models\icon_detector\model_meta\transforms.yaml",
    "models\icon_detector\model_meta\train.yaml",
    "models\general_v3\models\detector\inference",
    "models\general_v3\models\recognizer\inference",
    "models\general_v3\models\recognizer\character_dict.txt",
    "models\experimental_v6\models\detector\inference",
    "models\experimental_v6\models\recognizer\inference",
    "models\experimental_v6\models\recognizer\character_dict.txt",
    "requirements\requirements-api.txt",
    "requirements\requirements-v6.txt",
    "setup.ps1",
    "run_api.ps1",
    "README.md"
)

$Missing = @()
foreach ($Relative in $Required) {
    if (-not (Test-Path (Join-Path $Root $Relative))) {
        $Missing += $Relative
    }
}
if ($Missing.Count -gt 0) {
    $Missing | ForEach-Object { Write-Host "MISSING: $_" -ForegroundColor Red }
    throw "Package validation failed."
}
Write-Host "Package validation passed." -ForegroundColor Green
