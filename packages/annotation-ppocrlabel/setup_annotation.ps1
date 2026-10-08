#requires -Version 5.1

[CmdletBinding()]
param(
    [string]$PythonLauncher = "py",
    [string]$PythonVersion = "3.10"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$PackageRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

$RepoRoot = (
    Resolve-Path (
        Join-Path $PackageRoot "..\.."
    )
).Path

$AnnotationEnv = Join-Path `
    $PackageRoot `
    ".venv-annotation"

$AnnotationPython = Join-Path `
    $AnnotationEnv `
    "Scripts\python.exe"

$AnnotationRequirements = Join-Path `
    $PackageRoot `
    "requirements\requirements-annotation.txt"

$ApiRoot = Join-Path `
    $RepoRoot `
    "packages\meter-family-api"

$ApiSetup = Join-Path `
    $ApiRoot `
    "setup.ps1"

$ApiV6Python = Join-Path `
    $ApiRoot `
    ".venv-v6\Scripts\python.exe"

$ApiPaddleOCR = Join-Path `
    $ApiRoot `
    "runtime\PaddleOCR-3.7"

$ModelRoot = Join-Path `
    $ApiRoot `
    "models\experimental_v6"

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Program,

        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    & $Program @Arguments

    if ($LASTEXITCODE -ne 0) {
        throw (
            "Command failed with exit code " +
            "$LASTEXITCODE`: $Program " +
            "$($Arguments -join ' ')"
        )
    }
}

if (-not (Test-Path $AnnotationRequirements -PathType Leaf)) {
    throw "Missing annotation requirements: $AnnotationRequirements"
}

if (-not (Test-Path $ApiSetup -PathType Leaf)) {
    throw "Missing meter-family API setup script: $ApiSetup"
}

if (
    -not (Test-Path $ApiV6Python -PathType Leaf) -or
    -not (Test-Path $ApiPaddleOCR -PathType Container)
) {
    Write-Host `
        "Preparing the shared meter-family PP-OCRv6 runtime..." `
        -ForegroundColor Cyan

    powershell `
        -NoProfile `
        -ExecutionPolicy Bypass `
        -File $ApiSetup

    if ($LASTEXITCODE -ne 0) {
        throw "The meter-family API setup failed."
    }
}

if (-not (Test-Path $ApiV6Python -PathType Leaf)) {
    throw "Shared PP-OCRv6 Python was not created: $ApiV6Python"
}

if (-not (Test-Path $ApiPaddleOCR -PathType Container)) {
    throw "Shared PaddleOCR runtime was not created: $ApiPaddleOCR"
}

$RequiredModelFiles = @(
    (
        Join-Path `
            $ModelRoot `
            "models\detector\inference\inference.json"
    )
    (
        Join-Path `
            $ModelRoot `
            "models\detector\inference\inference.pdiparams"
    )
    (
        Join-Path `
            $ModelRoot `
            "models\recognizer\inference\inference.json"
    )
    (
        Join-Path `
            $ModelRoot `
            "models\recognizer\inference\inference.pdiparams"
    )
    (
        Join-Path `
            $ModelRoot `
            "models\recognizer\character_dict.txt"
    )
)

foreach ($File in $RequiredModelFiles) {
    if (-not (Test-Path $File -PathType Leaf)) {
        throw "Missing Model 5 file: $File"
    }
}

if (-not (Test-Path $AnnotationPython -PathType Leaf)) {
    if (
        -not (
            Get-Command `
                $PythonLauncher `
                -ErrorAction SilentlyContinue
        )
    ) {
        throw (
            "Python launcher '$PythonLauncher' was not found. " +
            "Install 64-bit Python 3.10."
        )
    }

    Write-Host `
        "Creating PPOCRLabel annotation environment..." `
        -ForegroundColor Cyan

    Invoke-Checked `
        -Program $PythonLauncher `
        -Arguments @(
            "-$PythonVersion"
            "-m"
            "venv"
            $AnnotationEnv
        )
}

Write-Host `
    "Installing the audited annotation dependencies..." `
    -ForegroundColor Cyan

Invoke-Checked `
    -Program $AnnotationPython `
    -Arguments @(
        "-m"
        "pip"
        "install"
        "--upgrade"
        "pip"
    )

Invoke-Checked `
    -Program $AnnotationPython `
    -Arguments @(
        "-m"
        "pip"
        "install"
        "-r"
        $AnnotationRequirements
    )

Write-Host "`nChecking annotation imports..." `
    -ForegroundColor Cyan

$ImportCheck = @"
import importlib

modules = [
    "PyQt5",
    "cv2",
    "numpy",
    "PIL",
    "paddle",
    "paddleocr",
    "shapely",
    "yaml",
]

for name in modules:
    importlib.import_module(name)
    print(f"PASS {name}")
"@

$ImportCheckFile = Join-Path `
    $env:TEMP `
    "validate_annotation_imports.py"

[System.IO.File]::WriteAllText(
    $ImportCheckFile,
    $ImportCheck,
    [System.Text.UTF8Encoding]::new($false)
)

try {
    & $AnnotationPython $ImportCheckFile

    if ($LASTEXITCODE -ne 0) {
        throw "One or more annotation imports failed."
    }
}
finally {
    Remove-Item `
        $ImportCheckFile `
        -Force `
        -ErrorAction SilentlyContinue
}

Write-Host "`nPASS: Annotation setup completed." `
    -ForegroundColor Green

Write-Host "Annotation Python: $AnnotationPython"
Write-Host "Shared v6 Python: $ApiV6Python"
Write-Host "Model: Model 5 PP-OCRv6"

Write-Host "`nNext command:"
Write-Host (
    "powershell -NoProfile -ExecutionPolicy Bypass " +
    "-File `"$PackageRoot\validate_package.ps1`""
)