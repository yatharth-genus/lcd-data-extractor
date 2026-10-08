param(
    [Parameter(Mandatory = $true)]
    [ValidateScript({ Test-Path $_ -PathType Leaf })]
    [string]$ImagePath,

    [ValidateSet("experimental_v6", "general_v3")]
    [string]$MeterFamily = "experimental_v6",

    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"

$PackageRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$RunApi = Join-Path $PackageRoot "run_api.ps1"
$ApiPython = Join-Path $PackageRoot ".venv-api\Scripts\python.exe"
$HealthUrl = "http://127.0.0.1:$Port/health"
$PredictUrl = "http://127.0.0.1:$Port/predict"

if (-not (Test-Path $RunApi -PathType Leaf)) {
    throw "Missing API runner: $RunApi"
}

if (-not (Test-Path $ApiPython -PathType Leaf)) {
    throw @"
The API environment is not installed.

Run this once:
powershell -ExecutionPolicy Bypass -File "$PackageRoot\setup.ps1"
"@
}

$ResolvedImage = (
    Resolve-Path $ImagePath
).Path

$StartedApi = $false
$ApiProcess = $null

try {
    $ApiReady = $false

    try {
        $Health = Invoke-RestMethod `
            -Uri $HealthUrl `
            -Method Get `
            -TimeoutSec 3

        $ApiReady = $true
        Write-Host "Using the API already running on port $Port." `
            -ForegroundColor Green
    }
    catch {
        Write-Host "Starting the LCD Understanding API..." `
            -ForegroundColor Cyan

        $ApiProcess = Start-Process `
            -FilePath "powershell.exe" `
            -ArgumentList @(
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                "`"$RunApi`"",
                "-MeterFamily",
                $MeterFamily
            ) `
            -WorkingDirectory $PackageRoot `
            -PassThru

        $StartedApi = $true

        for ($Attempt = 1; $Attempt -le 90; $Attempt++) {
            if ($ApiProcess.HasExited) {
                throw (
                    "The API process exited during startup " +
                    "with code $($ApiProcess.ExitCode)."
                )
            }

            try {
                $Health = Invoke-RestMethod `
                    -Uri $HealthUrl `
                    -Method Get `
                    -TimeoutSec 3

                $ApiReady = $true
                break
            }
            catch {
                Start-Sleep -Seconds 1
            }
        }
    }

    if (-not $ApiReady) {
        throw "The API did not become ready at $HealthUrl."
    }

    Write-Host "Running prediction..." -ForegroundColor Cyan
    Write-Host "Image: $ResolvedImage"
    Write-Host "Pipeline: $MeterFamily"

    $CurlArguments = @(
        "-sS",
        "-X", "POST",
        $PredictUrl,
        "-F", "meter_family=$MeterFamily",
        "-F", "lcd_image=@$ResolvedImage"
    )

    $ResponseText = & curl.exe @CurlArguments

    if ($LASTEXITCODE -ne 0) {
        throw "curl.exe failed with exit code $LASTEXITCODE."
    }

    try {
        $Response = $ResponseText | ConvertFrom-Json
    }
    catch {
        Write-Host $ResponseText
        throw "The API response was not valid JSON."
    }

    if ($Response.status -ne "success") {
        $Response | ConvertTo-Json -Depth 20
        throw "Prediction did not return success."
    }

    Write-Host "`nPREDICTION SUCCEEDED" -ForegroundColor Green
    Write-Host "Resolved pipeline: $($Response.resolved_pipeline)"
    Write-Host "Fallback used:     $($Response.fallback_used)"

    Write-Host "`nRecognized text" -ForegroundColor Cyan

    foreach ($Occurrence in $Response.text_occurrences) {
        Write-Host (
            "{0}  {1}  confidence={2:N4}" -f
            $Occurrence.id,
            $Occurrence.value,
            [double]$Occurrence.confidence
        )
    }

    Write-Host "`nDetected icons" -ForegroundColor Cyan

    foreach ($Occurrence in $Response.icon_occurrences) {
        Write-Host (
            "{0}  {1}  confidence={2:N4}" -f
            $Occurrence.id,
            $Occurrence.class_name,
            [double]$Occurrence.confidence
        )
    }

    Write-Host "`nSaved results" -ForegroundColor Cyan
    Write-Host "Masked:     $($Response.saved_results.masked_url)"
    Write-Host "Rendered:   $($Response.saved_results.rendered_url)"
    Write-Host "Prediction: $($Response.saved_results.prediction_url)"

    $Response |
        ConvertTo-Json -Depth 20
}
finally {
    if (
        $StartedApi -and
        $null -ne $ApiProcess -and
        -not $ApiProcess.HasExited
    ) {
        Write-Host "`nStopping the API started by this runner..." `
            -ForegroundColor DarkGray

        Stop-Process `
            -Id $ApiProcess.Id `
            -Force `
            -ErrorAction SilentlyContinue
    }
}
