param(
    [string]$PackageRoot
)

$ErrorActionPreference = "Stop"

if (-not $PackageRoot) {
    $PackageRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}

$RequiredFiles = @(
    "$PackageRoot\README.md"
    "$PackageRoot\PACKAGE_IDENTITY.yml"
    "$PackageRoot\model\saved_model.xml"
    "$PackageRoot\model\saved_model.bin"
    "$PackageRoot\model\saved_model.onnx"
    "$PackageRoot\model_meta\train.yaml"
    "$PackageRoot\model_meta\transforms.yaml"
    "$PackageRoot\dm.json"
    "$PackageRoot\meta.json"
    "$PackageRoot\scripts\run_icon_detector.py"
    "$PackageRoot\requirements\requirements.txt"
    "$PackageRoot\results\baseline_97_images\reports\summary.json"
    "$PackageRoot\results\baseline_97_images\reports\icon_detections.json"
)

$Problems = @()

foreach ($FilePath in $RequiredFiles) {
    if (-not (Test-Path -LiteralPath $FilePath -PathType Leaf)) {
        $Problems += "Missing required file: $FilePath"
    }
}

$RenderedCount = @(
    Get-ChildItem -LiteralPath "$PackageRoot\results\baseline_97_images\rendered" -File -ErrorAction SilentlyContinue
).Count

if ($RenderedCount -ne 97) {
    $Problems += "Expected 97 rendered baseline images, found $RenderedCount"
}

if ($Problems.Count -gt 0) {
    Write-Host "Icon package validation problems:" -ForegroundColor Red
    foreach ($Problem in $Problems) {
        Write-Host "  $Problem"
    }
    throw "Icon detector package validation failed."
}

Write-Host "Icon detector package validation passed." -ForegroundColor Green
Write-Host "Rendered baseline images: $RenderedCount"
