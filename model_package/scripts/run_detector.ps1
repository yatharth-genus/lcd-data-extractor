param(
    [string]$ImagePath = "$PSScriptRoot\..\results\end_to_end\sample_images",
    [string]$PaddleRoot = "D:\Actual Project\PaddleOCR-2.8"
)
$ErrorActionPreference = "Stop"
python "$PaddleRoot\tools\infer_det.py" `
    -c "$PSScriptRoot\..\config\detector.yml" `
    -o "Global.checkpoints=$($PSScriptRoot.Replace('\','/'))/../models/detector/training/best_accuracy" `
       "Global.infer_img=$($ImagePath.Replace('\','/'))"