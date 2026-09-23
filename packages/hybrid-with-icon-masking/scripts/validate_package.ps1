param(
    [string]$PackageRoot
)

$ErrorActionPreference = "Stop"

if (-not $PackageRoot) {
    $PackageRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}

$Required = @(
    "$PackageRoot\README.md"
    "$PackageRoot\PIPELINE_IDENTITY.yml"
    "$PackageRoot\docs\EVALUATION.md"
    "$PackageRoot\scripts\run_icon_masked_hybrid.py"
    "$PackageRoot\results\reports\summary.json"
    "$PackageRoot\results\reports\masking_summary.json"
    "$PackageRoot\results\reports\ocr_console.txt"
    "$PackageRoot\results\ocr_visual_results\system_results.txt"
    "$PackageRoot\results\detector_evaluation\evaluation_console.txt"
    "$PackageRoot\results\unmasked_control_evaluation\evaluation_console.txt"
)

$Problems = @()
foreach ($Path in $Required) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        $Problems += "Missing required file: $Path"
    }
}

$MaskedCount = @(Get-ChildItem -LiteralPath "$PackageRoot\results\masked_images" -File -ErrorAction SilentlyContinue).Count
$RenderedCount = @(Get-ChildItem -LiteralPath "$PackageRoot\results\ocr_visual_results" -File -ErrorAction SilentlyContinue | Where-Object { $_.Extension.ToLowerInvariant() -in @(".png", ".jpg", ".jpeg", ".bmp", ".webp") }).Count

if ($MaskedCount -ne 97) {
    $Problems += "Expected 97 masked images, found $MaskedCount"
}
if ($RenderedCount -ne 97) {
    $Problems += "Expected 97 OCR visual result images, found $RenderedCount"
}

if ($Problems.Count -gt 0) {
    foreach ($Problem in $Problems) {
        Write-Host $Problem -ForegroundColor Red
    }
    throw "Package validation failed."
}

Write-Host "Icon-masked hybrid package validation passed." -ForegroundColor Green
Write-Host "Masked images: $MaskedCount"
Write-Host "OCR visual result images: $RenderedCount"
