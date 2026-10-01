param(
    [ValidateSet("general_v3", "experimental_v6")]
    [string]$MeterFamily = "experimental_v6",
    [string]$HostAddress = "127.0.0.1",
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"
$PackageRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ApiPython = Join-Path $PackageRoot ".venv-api\Scripts\python.exe"
$V6Python = Join-Path $PackageRoot ".venv-v6\Scripts\python.exe"
$PaddleRoot = Join-Path $PackageRoot "runtime\PaddleOCR-3.7"

if (-not (Test-Path $ApiPython)) {
    throw "API environment missing. Run .\setup.ps1 first."
}
if (-not (Test-Path $V6Python)) {
    throw "PP-OCRv6 environment missing. Run .\setup.ps1 first."
}
if (-not (Test-Path (Join-Path $PaddleRoot "tools\infer\predict_system.py"))) {
    throw "PaddleOCR runtime missing. Run .\setup.ps1 first."
}

$env:LCD_METER_FAMILY = $MeterFamily
$env:PPOCRV6_ROOT = $PaddleRoot
$env:PPOCRV6_PYTHON = $V6Python

New-Item -ItemType Directory -Path (Join-Path $PackageRoot "results\rendered") -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $PackageRoot "results\debug") -Force | Out-Null

Push-Location $PackageRoot
try {
    & $ApiPython -m uvicorn app.main:app --host $HostAddress --port $Port
}
finally {
    Pop-Location
}
