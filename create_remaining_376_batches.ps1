$ErrorActionPreference = "Stop"

$RepoPath = "D:\Actual Project\lcd-data-extractor"
$SourceRoot = Join-Path $RepoPath "data"
$OutputRoot = Join-Path $RepoPath "remaining_376"

$PilotCsv = Join-Path `
    $RepoPath `
    "pilot_90\annotations\PaddleOCR_90_Selected_Images.csv"

$Next100Csv = Join-Path `
    $RepoPath `
    "next_100\annotations\PaddleOCR_Next_100_Selected_Images.csv"

$Additional100Csv = Join-Path `
    $RepoPath `
    "additional_100\annotations\PaddleOCR_Additional_100_Selected_Images.csv"

$AnnotationFolder = Join-Path $OutputRoot "annotations"
$ManifestCsv = Join-Path `
    $AnnotationFolder `
    "PaddleOCR_Remaining_376_Images.csv"

$ManifestTxt = Join-Path `
    $AnnotationFolder `
    "PaddleOCR_Remaining_376_Images.txt"

$MissingFile = Join-Path `
    $AnnotationFolder `
    "missing_images.txt"

$ImageExtensions = @(
    ".png",
    ".jpg",
    ".jpeg"
)

Write-Host ""
Write-Host "Checking required inputs..." -ForegroundColor Cyan

foreach ($RequiredPath in @(
    $SourceRoot,
    $PilotCsv,
    $Next100Csv,
    $Additional100Csv
)) {
    if (-not (Test-Path -LiteralPath $RequiredPath)) {
        throw "Required path not found: $RequiredPath"
    }

    Write-Host "Found: $RequiredPath"
}

New-Item `
    -ItemType Directory `
    -Path $AnnotationFolder `
    -Force |
    Out-Null

Write-Host ""
Write-Host "Reading the 290 previously selected images..." `
    -ForegroundColor Cyan

$SelectedNames = @(
    Import-Csv -LiteralPath $PilotCsv
    Import-Csv -LiteralPath $Next100Csv
    Import-Csv -LiteralPath $Additional100Csv
) |
ForEach-Object {
    if ($null -ne $_.image) {
        $_.image.Trim().ToLowerInvariant()
    }
} |
Where-Object {
    $_ -ne ""
} |
Sort-Object -Unique

Write-Host "Unique selected images: $($SelectedNames.Count)"

if ($SelectedNames.Count -ne 290) {
    throw (
        "Expected 290 unique selected images, but found " +
        "$($SelectedNames.Count). Check the three selection CSV files."
    )
}

Write-Host ""
Write-Host "Indexing the complete master dataset..." `
    -ForegroundColor Cyan

$MasterImages = @(
    Get-ChildItem `
        -LiteralPath $SourceRoot `
        -Recurse `
        -File |
    Where-Object {
        $_.Extension.ToLowerInvariant() -in $ImageExtensions
    }
)

Write-Host "Master images found: $($MasterImages.Count)"

if ($MasterImages.Count -ne 666) {
    throw (
        "Expected 666 master images, but found " +
        "$($MasterImages.Count). No batches were created."
    )
}

$DuplicateMasterNames = @(
    $MasterImages |
    Group-Object {
        $_.Name.ToLowerInvariant()
    } |
    Where-Object {
        $_.Count -gt 1
    }
)

if ($DuplicateMasterNames.Count -gt 0) {
    Write-Host ""
    Write-Host "Duplicate filenames exist in the master dataset:" `
        -ForegroundColor Red

    $DuplicateMasterNames |
        Select-Object Name, Count |
        Format-Table -AutoSize

    throw (
        "Duplicate master filenames make filename-based selection " +
        "ambiguous. Resolve them before continuing."
    )
}

$RemainingImages = @(
    $MasterImages |
    Where-Object {
        $_.Name.ToLowerInvariant() -notin $SelectedNames
    } |
    Sort-Object {
        $NumericName = 0

        if (
            [int]::TryParse(
                $_.BaseName,
                [ref]$NumericName
            )
        ) {
            $NumericName
        }
        else {
            [int]::MaxValue
        }
    }, Name
)

Write-Host "Remaining unselected images: $($RemainingImages.Count)"

if ($RemainingImages.Count -ne 376) {
    throw (
        "Expected 376 remaining images, but found " +
        "$($RemainingImages.Count). No batches were created."
    )
}

Write-Host ""
Write-Host "Preparing 13 batch folders..." `
    -ForegroundColor Cyan

$BatchNames = @(
    "batch_01",
    "batch_02",
    "batch_03",
    "batch_04",
    "batch_05",
    "batch_06",
    "batch_07",
    "batch_08",
    "batch_09",
    "batch_10",
    "batch_11",
    "batch_12",
    "batch_13"
)

foreach ($BatchName in $BatchNames) {
    $BatchImageFolder = Join-Path `
        $OutputRoot `
        "$BatchName\images"

    New-Item `
        -ItemType Directory `
        -Path $BatchImageFolder `
        -Force |
        Out-Null

    # Clear only source images from an earlier incomplete run.
    Get-ChildItem `
        -LiteralPath $BatchImageFolder `
        -File `
        -ErrorAction SilentlyContinue |
    Where-Object {
        $_.Extension.ToLowerInvariant() -in $ImageExtensions
    } |
    Remove-Item -Force
}

$ManifestRows = @()
$Copied = 0
$Missing = @()

Write-Host ""
Write-Host "Copying remaining images..." -ForegroundColor Cyan

for (
    $ImageIndex = 0;
    $ImageIndex -lt $RemainingImages.Count;
    $ImageIndex++
) {
    $SourceImage = $RemainingImages[$ImageIndex]

    $BatchIndex = [math]::Truncate($ImageIndex / 30
    )

    $BatchName = $BatchNames[$BatchIndex]

    $DestinationFolder = Join-Path `
        $OutputRoot `
        "$BatchName\images"

    $DestinationFile = Join-Path `
        $DestinationFolder `
        $SourceImage.Name

    try {
        Copy-Item `
            -LiteralPath $SourceImage.FullName `
            -Destination $DestinationFile `
            -Force

        $Copied++

        $ManifestRows += [PSCustomObject]@{
            sequence_id = $ImageIndex + 1
            batch = $BatchName
            image = $SourceImage.Name
            source_path = $SourceImage.FullName
            relative_destination = (
                "$BatchName/images/$($SourceImage.Name)"
            )
            annotation_status = "Not started"
            annotator = ""
            reviewer = ""
            notes = ""
        }
    }
    catch {
        $Missing += $SourceImage.FullName

        Write-Host `
            "Copy failed: $($SourceImage.FullName)" `
            -ForegroundColor Red
    }
}

$ManifestRows |
    Export-Csv `
        -LiteralPath $ManifestCsv `
        -NoTypeInformation `
        -Encoding UTF8

$ManifestRows.image |
    Set-Content `
        -LiteralPath $ManifestTxt `
        -Encoding UTF8

if ($Missing.Count -gt 0) {
    $Missing |
        Set-Content `
            -LiteralPath $MissingFile `
            -Encoding UTF8
}
elseif (Test-Path -LiteralPath $MissingFile) {
    Remove-Item -LiteralPath $MissingFile -Force
}

Write-Host ""
Write-Host "Verifying batch contents..." -ForegroundColor Cyan

$ActualImages = @()

foreach ($BatchName in $BatchNames) {
    $BatchFolder = Join-Path `
        $OutputRoot `
        "$BatchName\images"

    $BatchImages = @(
        Get-ChildItem `
            -LiteralPath $BatchFolder `
            -File |
        Where-Object {
            $_.Extension.ToLowerInvariant() -in $ImageExtensions
        }
    )

    $ActualImages += $BatchImages

    Write-Host (
        "{0} : {1}" -f
        $BatchName,
        $BatchImages.Count
    )
}

$DuplicateOutputNames = @(
    $ActualImages |
    Group-Object {
        $_.Name.ToLowerInvariant()
    } |
    Where-Object {
        $_.Count -gt 1
    }
)

$UnexpectedSelectedImages = @(
    $ActualImages |
    Where-Object {
        $_.Name.ToLowerInvariant() -in $SelectedNames
    }
)

Write-Host ""
Write-Host "Final summary" -ForegroundColor Cyan
Write-Host "Master images:              $($MasterImages.Count)"
Write-Host "Previously selected:        $($SelectedNames.Count)"
Write-Host "Expected remaining:         376"
Write-Host "Copied:                     $Copied"
Write-Host "Files found in batches:     $($ActualImages.Count)"
Write-Host "Copy failures:              $($Missing.Count)"
Write-Host "Duplicate output filenames: $($DuplicateOutputNames.Count)"
Write-Host "Previously selected copied: $($UnexpectedSelectedImages.Count)"

if (
    $Copied -ne 376 -or
    $ActualImages.Count -ne 376 -or
    $Missing.Count -ne 0 -or
    $DuplicateOutputNames.Count -ne 0 -or
    $UnexpectedSelectedImages.Count -ne 0
) {
    throw "Final verification failed. Review the summary above."
}

Write-Host ""
Write-Host (
    "Success: all 376 remaining images were copied " +
    "into 13 batches."
) -ForegroundColor Green

Write-Host "Manifest CSV: $ManifestCsv"
Write-Host "Manifest TXT: $ManifestTxt"
