param([Parameter(Mandatory=$true)][string]$Images,[string]$Output=".\results\batch")
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
python -m app.run_batch "$Images" --output "$Output"
