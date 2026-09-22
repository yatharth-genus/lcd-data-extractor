param(
    [string]$ImagePath = "$PSScriptRoot\..\results\end_to_end\sample_images",
    [string]$PaddleRoot = "D:\Actual Project\PaddleOCR-2.8"
)
$ErrorActionPreference = "Stop"
$PredictSystem = Get-ChildItem "$PaddleRoot\tools" -Recurse -File -Filter "predict_system.py" | Select-Object -First 1
if (-not $PredictSystem) { throw "predict_system.py not found below $PaddleRoot\tools" }
$DetInference = Resolve-Path "$PSScriptRoot\..\models\detector\inference"
$RecInference = Resolve-Path "$PSScriptRoot\..\models\recognizer\inference"
$Dictionary = Resolve-Path "$PSScriptRoot\..\models\recognizer\character_dict.txt"
$Output = "$PSScriptRoot\..\results\end_to_end\rendered_results"
New-Item -ItemType Directory -Path $Output -Force | Out-Null
python $PredictSystem.FullName `
    --image_dir $ImagePath `
    --det_model_dir $DetInference.Path `
    --rec_model_dir $RecInference.Path `
    --rec_char_dict_path $Dictionary.Path `
    --use_angle_cls false `
    --use_gpu false `
    --drop_score 0.3 `
    --draw_img_save_dir $Output