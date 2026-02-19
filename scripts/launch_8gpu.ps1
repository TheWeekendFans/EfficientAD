$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

if (-not (Test-Path .\logs)) {
    New-Item -ItemType Directory -Path .\logs | Out-Null
}

Write-Host "Launching parallel training jobs..."

$gridCmd = "$env:CUDA_VISIBLE_DEVICES='0'; python main.py --mode train --data_path F:\lhd\2 --class_name grid --output_dir output --batch_size 8 --epochs 100 --lr 1e-4 --num_workers 8 --device cuda --amp --save_every 10"
$transistorCmd = "$env:CUDA_VISIBLE_DEVICES='1'; python main.py --mode train --data_path F:\lhd\2 --class_name transistor --output_dir output --batch_size 8 --epochs 100 --lr 1e-4 --num_workers 8 --device cuda --amp --save_every 10"

Start-Process powershell -ArgumentList "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", "$gridCmd *> .\logs\train_grid.log"
Start-Process powershell -ArgumentList "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", "$transistorCmd *> .\logs\train_transistor.log"

Write-Host "Started:"
Write-Host "- Grid training on GPU0 -> logs/train_grid.log"
Write-Host "- Transistor training on GPU1 -> logs/train_transistor.log"
