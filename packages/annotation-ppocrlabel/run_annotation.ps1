#requires -Version 5.1

[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$PackageRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

$AnnotationPython = Join-Path `
    $PackageRoot `
    ".venv-annotation\Scripts\python.exe"

$Launcher = Join-Path `
    $PackageRoot `
    "app\launch_ppocrlabel_v6.py"

$Validator = Join-Path `
    $PackageRoot `
    "validate_package.ps1"

if (-not (Test-Path $AnnotationPython -PathType Leaf)) {
    throw (
        "The annotation environment is not installed. Run: " +
        "powershell -NoProfile -ExecutionPolicy Bypass " +
        "-File `"$PackageRoot\setup_annotation.ps1`""
    )
}

if (-not (Test-Path $Launcher -PathType Leaf)) {
    throw "PPOCRLabel launcher not found: $Launcher"
}

powershell `
    -NoProfile `
    -ExecutionPolicy Bypass `
    -File $Validator

if ($LASTEXITCODE -ne 0) {
    throw "Annotation package validation failed."
}

Set-Location (Split-Path -Parent $Launcher)

Write-Host `
    "Starting PPOCRLabel with Model 5 PP-OCRv6..." `
    -ForegroundColor Cyan

& $AnnotationPython $Launcher

if ($LASTEXITCODE -ne 0) {
    throw "PPOCRLabel exited with code $LASTEXITCODE."
}