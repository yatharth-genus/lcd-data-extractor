param(
    [Parameter(Mandatory=$true)][string]$CropPath,
    [string]$PaddleRoot = "D:\Actual Project\PaddleOCR-2.8"
)
$ErrorActionPreference = "Stop"
python "$PaddleRoot\tools\infer_rec.py" `
    -c "$PSScriptRoot\..\config\recognizer.yml" `
    -o "Global.checkpoints=$($PSScriptRoot.Replace('\','/'))/../models/recognizer/training/best_accuracy" `
       "Global.infer_img=$($CropPath.Replace('\','/'))"