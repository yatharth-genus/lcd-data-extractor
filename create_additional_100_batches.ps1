$RepoPath = "D:\Actual Project\lcd-data-extractor"
$SourceRoot = Join-Path $RepoPath "data"
$DatasetRoot = Join-Path $RepoPath "additional_100"
$CsvPath = Join-Path $DatasetRoot "annotations\PaddleOCR_Additional_100_Selected_Images.csv"
$BatchNames = @("batch_01","batch_02","batch_03","batch_04","batch_05")
New-Item -ItemType Directory -Path (Join-Path $DatasetRoot "annotations") -Force | Out-Null
foreach ($BatchName in $BatchNames) { New-Item -ItemType Directory -Path (Join-Path $DatasetRoot "$BatchName\images") -Force | Out-Null }
if (-not (Test-Path -LiteralPath $CsvPath)) { throw "CSV missing: $CsvPath" }
$Rows = @(Import-Csv -LiteralPath $CsvPath)
if ($Rows.Count -ne 100) { throw "Expected 100 rows; found $($Rows.Count)" }
$Index = @{}
Get-ChildItem -LiteralPath $SourceRoot -Recurse -File | ForEach-Object { $Key=$_.Name.ToLowerInvariant(); if (-not $Index.ContainsKey($Key)) { $Index[$Key]=$_.FullName } }
$Copied=0; $Missing=@()
for ($BatchIndex=0; $BatchIndex -lt 5; $BatchIndex++) {
  $Destination=Join-Path $DatasetRoot "$($BatchNames[$BatchIndex])\images"
  for ($Offset=0; $Offset -lt 20; $Offset++) {
    $RowIndex=($BatchIndex*20)+$Offset
    $Name=$Rows[$RowIndex].image.Trim(); $Key=$Name.ToLowerInvariant()
    if ($Index.ContainsKey($Key)) { Copy-Item -LiteralPath $Index[$Key] -Destination $Destination -Force; $Copied++ } else { $Missing += $Name }
  }
}
Write-Host "Copied: $Copied"; Write-Host "Missing: $($Missing.Count)"
foreach ($BatchName in $BatchNames) { $Count=@(Get-ChildItem (Join-Path $DatasetRoot "$BatchName\images") -File).Count; Write-Host "$BatchName : $Count" }
