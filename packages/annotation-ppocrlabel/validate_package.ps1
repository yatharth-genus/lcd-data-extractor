#requires -Version 5.1

[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$PackageRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

$RepoRoot = (
    Resolve-Path (
        Join-Path $PackageRoot "..\.."
    )
).Path

$AppRoot = Join-Path $PackageRoot "app"

$Launcher = Join-Path `
    $AppRoot `
    "launch_ppocrlabel_v6.py"

$Worker = Join-Path `
    $AppRoot `
    "ppocrv6_worker.py"

$Application = Join-Path `
    $AppRoot `
    "PPOCRLabel.py"

$AnnotationPython = Join-Path `
    $PackageRoot `
    ".venv-annotation\Scripts\python.exe"

$ApiRoot = Join-Path `
    $RepoRoot `
    "packages\meter-family-api"

$V6Python = Join-Path `
    $ApiRoot `
    ".venv-v6\Scripts\python.exe"

$PaddleOCRRoot = Join-Path `
    $ApiRoot `
    "runtime\PaddleOCR-3.7"

$ModelRoot = Join-Path `
    $ApiRoot `
    "models\experimental_v6"

$RequiredFiles = @(
    $Application
    $Launcher
    $Worker
    (Join-Path $PackageRoot "setup_annotation.ps1")
    (Join-Path $PackageRoot "run_annotation.ps1")
    (Join-Path $PackageRoot "README.md")
    (
        Join-Path `
            $PackageRoot `
            "requirements\requirements-annotation.txt"
    )
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

foreach ($File in $RequiredFiles) {
    if (-not (Test-Path $File -PathType Leaf)) {
        throw "Missing required file: $File"
    }

    if ((Get-Item $File).Length -eq 0) {
        throw "Required file is empty: $File"
    }
}

$PortableFiles = @(
    $Application
    $Launcher
    $Worker
    (Join-Path $PackageRoot "setup_annotation.ps1")
    (Join-Path $PackageRoot "run_annotation.ps1")
    (Join-Path $PackageRoot "README.md")
)

$AbsolutePaths = Select-String `
    -Path $PortableFiles `
    -Pattern "D:\\Actual Project|D:/Actual Project" `
    -CaseSensitive:$false

if ($AbsolutePaths) {
    $AbsolutePaths |
        Format-Table Path, LineNumber, Line -AutoSize

    throw "Machine-specific paths remain in portable files."
}

$LauncherText = [System.IO.File]::ReadAllText($Launcher)
$WorkerText = [System.IO.File]::ReadAllText($Worker)

$ExpectedLauncherSettings = @(
    'PACKAGE_ROOT = TOOL_ROOT.parent'
    'REPO_ROOT = PACKAGE_ROOT.parent.parent'
    'API_ROOT = REPO_ROOT / "packages" / "meter-family-api"'
    'V6_PYTHON = API_ROOT / ".venv-v6" / "Scripts" / "python.exe"'
    'PADDLEOCR_ROOT = API_ROOT / "runtime" / "PaddleOCR-3.7"'
    'MODEL_ROOT = API_ROOT / "models" / "experimental_v6"'
)

foreach ($Setting in $ExpectedLauncherSettings) {
    if (-not $LauncherText.Contains($Setting)) {
        throw "Missing portable launcher setting: $Setting"
    }
}

$ExpectedWorkerSettings = @(
    '"--det_algorithm","DB"'
    '"--rec_algorithm","SVTR_LCNet"'
    '"--rec_image_shape","3,48,320"'
    '"--use_space_char","true"'
)

foreach ($Setting in $ExpectedWorkerSettings) {
    if (-not $WorkerText.Contains($Setting)) {
        throw "Missing Model 5 worker setting: $Setting"
    }
}

if (-not (Test-Path $AnnotationPython -PathType Leaf)) {
    throw (
        "Annotation environment is not installed. " +
        "Run setup_annotation.ps1."
    )
}

if (-not (Test-Path $V6Python -PathType Leaf)) {
    throw (
        "Shared PP-OCRv6 environment is not installed. " +
        "Run setup_annotation.ps1."
    )
}

if (-not (Test-Path $PaddleOCRRoot -PathType Container)) {
    throw (
        "Shared PaddleOCR runtime is not installed. " +
        "Run setup_annotation.ps1."
    )
}

$AnnotationPythonFiles = @(
    Get-ChildItem `
        -Path $AppRoot `
        -File `
        -Recurse `
        -Filter "*.py" |
    Where-Object {
        $_.Name -ne "ppocrv6_worker.py"
    }
)

foreach ($File in $AnnotationPythonFiles) {
    & $AnnotationPython -m py_compile $File.FullName

    if ($LASTEXITCODE -ne 0) {
        throw "Python syntax failure: $($File.FullName)"
    }
}

& $V6Python -m py_compile $Worker

if ($LASTEXITCODE -ne 0) {
    throw "PP-OCRv6 worker syntax validation failed."
}

$PowerShellFiles = @(
    Get-ChildItem `
        -Path $PackageRoot `
        -File `
        -Filter "*.ps1"
)

foreach ($File in $PowerShellFiles) {
    $Tokens = $null
    $ParseErrors = $null

    [System.Management.Automation.Language.Parser]::ParseFile(
        $File.FullName,
        [ref]$Tokens,
        [ref]$ParseErrors
    ) | Out-Null

    if ($ParseErrors.Count -gt 0) {
        $ParseErrors | Format-List
        throw "PowerShell syntax failure: $($File.FullName)"
    }
}

Write-Host "`nPASS: Annotation package validation passed." `
    -ForegroundColor Green

Write-Host "Model: Model 5 PP-OCRv6"
Write-Host "Detector algorithm: DB"
Write-Host "Recognizer algorithm: SVTR_LCNet"
Write-Host "Recognition shape: 3,48,320"
Write-Host "Space-character support: enabled"