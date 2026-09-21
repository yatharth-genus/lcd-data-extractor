$RepoPath = "D:\Actual Project\lcd-data-extractor"
$SourceRoot = Join-Path $RepoPath "data"
$DatasetRoot = Join-Path $RepoPath "next_100"
$CsvPath = Join-Path $DatasetRoot "annotations\PaddleOCR_Next_100_Selected_Images.csv"

$folders = @("batch_01", "batch_02", "batch_03", "batch_04", "batch_05")
foreach ($folder in $folders) {
    New-Item -ItemType Directory -Path (Join-Path $DatasetRoot "$folder\images") -Force | Out-Null
}
New-Item -ItemType Directory -Path (Join-Path $DatasetRoot "annotations") -Force | Out-Null

if (-not (Test-Path -LiteralPath $CsvPath)) { throw "CSV not found: $CsvPath" }
$index = @{}
Get-ChildItem -LiteralPath $SourceRoot -Recurse -File | ForEach-Object {
    $key = $_.Name.ToLowerInvariant()
    if (-not $index.ContainsKey($key)) { $index[$key] = $_.FullName }
}

$rows = Import-Csv -LiteralPath $CsvPath
$copied = 0
$missing = @()
for ($i = 0; $i -lt $rows.Count; $i++) {
    $row = $rows[$i]
    $batchNumber = [math]::Floor($i / 20) + 1
    $batch = "batch_{0:D2}" -f $batchNumber
    $destination = Join-Path $DatasetRoot "$batch\images"
    $key = $row.image.Trim().ToLowerInvariant()
    if ($index.ContainsKey($key)) {
        Copy-Item -LiteralPath $index[$key] -Destination $destination -Force
        $copied++
    } else {
        $missing += $row.image
    }
}
Write-Host "Copied: $copied"
Write-Host "Missing: $($missing.Count)"
if ($missing.Count -gt 0) { $missing | Set-Content (Join-Path $DatasetRoot "annotations\missing_images.txt") -Encoding UTF8 }
foreach ($folder in $folders) {
    $count = @(Get-ChildItem (Join-Path $DatasetRoot "$folder\images") -File).Count
    Write-Host "$folder : $count"
}
