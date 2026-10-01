param(
    [string]$PythonCommand = "py",
    [string]$PythonVersion = "3.10"
)

$ErrorActionPreference = "Stop"
$PackageRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ApiEnv = Join-Path $PackageRoot ".venv-api"
$V6Env = Join-Path $PackageRoot ".venv-v6"
$RuntimeRoot = Join-Path $PackageRoot "runtime\PaddleOCR-3.7"

if (-not (Test-Path $ApiEnv)) {
    & $PythonCommand "-$PythonVersion" -m venv $ApiEnv
}
& (Join-Path $ApiEnv "Scripts\python.exe") -m pip install --upgrade pip setuptools wheel
& (Join-Path $ApiEnv "Scripts\python.exe") -m pip install -r (Join-Path $PackageRoot "requirements\requirements-api.txt")

if (-not (Test-Path $V6Env)) {
    & $PythonCommand "-$PythonVersion" -m venv $V6Env
}
& (Join-Path $V6Env "Scripts\python.exe") -m pip install --upgrade pip setuptools wheel
& (Join-Path $V6Env "Scripts\python.exe") -m pip install -r (Join-Path $PackageRoot "requirements\requirements-v6.txt")

if (-not (Test-Path $RuntimeRoot)) {
    New-Item -ItemType Directory -Path (Split-Path -Parent $RuntimeRoot) -Force | Out-Null
    git clone --depth 1 --branch v3.7.0 https://github.com/PaddlePaddle/PaddleOCR.git $RuntimeRoot
}

& (Join-Path $V6Env "Scripts\python.exe") -m pip install -r (Join-Path $RuntimeRoot "requirements.txt")
& (Join-Path $V6Env "Scripts\python.exe") -m pip install -e $RuntimeRoot

Write-Host "Setup completed."
Write-Host "Start with: .\run_api.ps1 -MeterFamily experimental_v6"
