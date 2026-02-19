# 半导体晶圆缺陷检测系统 - 8卡服务器部署指南

## 1. 目标与适用环境
本指南用于在 **Windows Server + 8x RTX 2080Ti (11GB)** 环境部署并运行本项目。

- 算法：EfficientAD（无监督，仅良品训练）
- 数据：MVTec AD 子类 `grid`、`transistor`
- 入口：`main.py`（`train` / `evaluate` / `analysis`）

## 2. 当前项目目录（与你工作区一致）

```text
2/
├─ main.py
├─ train.py
├─ evaluate.py
├─ production_analysis.py
├─ dataset.py
├─ model.py
├─ README.md
├─ server_deployment_guide.md
├─ runbook_8gpu.md
├─ requirements.txt
├─ scripts/
│  └─ launch_8gpu.ps1
├─ grid/
│  └─ grid/
│     ├─ train/good
│     ├─ test/{good,bent,broken,...}
│     └─ ground_truth/{bent,broken,...}
└─ transistor/
   └─ transistor/
      ├─ train/good
      ├─ test/{good,bent_lead,...}
      └─ ground_truth/{bent_lead,...}
```

> 代码已支持三种 `--data_path` 传法：
> - 项目根：`F:\lhd\2`
> - 类目录上一级：`F:\lhd\2\grid`
> - 类目录本身：`F:\lhd\2\grid\grid`

## 3. 环境安装（CUDA 11.8 推荐）

### 3.1 创建虚拟环境
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 3.2 安装 PyTorch + 依赖
> 2080Ti 推荐安装 CUDA 11.8 对应轮子。

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### 3.3 GPU 可见性检查
```powershell
python -c "import torch; print(torch.cuda.device_count()); print(torch.cuda.get_device_name(0))"
```

## 4. 关键参数（已在代码中支持）

- `--device`：`auto` / `cuda` / `cuda:0` / `cpu`
- `--num_workers`：DataLoader 并行读取线程
- `--amp`：开启混合精度训练（2080Ti 推荐开启）
- `--save_every`：每 N 轮保存一次 epoch checkpoint
- `--epoch`：评估/分析时选择 checkpoint 轮次
- `--adaptive_k`：自适应阈值公式 `mu + k*sigma` 中的 `k`

## 5. 训练（建议并行跑两个类别）

### 5.1 Grid（GPU0）
```powershell
$env:CUDA_VISIBLE_DEVICES="0"
python main.py --mode train --data_path F:\lhd\2 --class_name grid --output_dir output --batch_size 8 --epochs 100 --lr 1e-4 --num_workers 8 --device cuda --amp --save_every 10
```

### 5.2 Transistor（GPU1）
```powershell
$env:CUDA_VISIBLE_DEVICES="1"
python main.py --mode train --data_path F:\lhd\2 --class_name transistor --output_dir output --batch_size 8 --epochs 100 --lr 1e-4 --num_workers 8 --device cuda --amp --save_every 10
```

> 说明：当前项目是按“类别/任务并行”使用多卡，而不是单任务DDP。对你这个两类别任务，这是最直接稳定的方式。

## 6. 评估（AUROC）

### 6.1 Grid
```powershell
$env:CUDA_VISIBLE_DEVICES="0"
python main.py --mode evaluate --data_path F:\lhd\2 --class_name grid --output_dir output --epoch 100 --num_workers 4 --device cuda
```

### 6.2 Transistor
```powershell
$env:CUDA_VISIBLE_DEVICES="1"
python main.py --mode evaluate --data_path F:\lhd\2 --class_name transistor --output_dir output --epoch 100 --num_workers 4 --device cuda
```

## 7. 产线分析（Escape / Overkill / 自适应阈值）

### 7.1 Grid
```powershell
$env:CUDA_VISIBLE_DEVICES="0"
python main.py --mode analysis --data_path F:\lhd\2 --class_name grid --output_dir output --epoch 100 --adaptive_k 3.0 --num_workers 4 --device cuda
```

### 7.2 Transistor
```powershell
$env:CUDA_VISIBLE_DEVICES="1"
python main.py --mode analysis --data_path F:\lhd\2 --class_name transistor --output_dir output --epoch 100 --adaptive_k 3.0 --num_workers 4 --device cuda
```

输出位于 `output/`：
- `output/models/<class_name>/...`
- `output/vis/*.png`
- `output/tradeoff_curve.png`

## 8. 一键并行启动（可选）
使用 `scripts/launch_8gpu.ps1` 同时拉起两个训练进程（分别占 GPU0/1），并保存日志到 `logs/`。

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\launch_8gpu.ps1
```

## 9. 常见故障

1) `ModuleNotFoundError: No module named 'torch'`
- 重新执行第3节安装命令。

2) `FileNotFoundError`（数据路径）
- 确认目录存在 `train/test/ground_truth`。
- 使用 `--data_path F:\lhd\2` 最稳妥。

3) CUDA OOM
- 降低 `--batch_size`（先改为 4 或 2）。
- 维持 `--amp` 开启。

4) `teacher_initial.pth` 缺失
- 先完整跑一次训练（至少第1个epoch写入 teacher checkpoint）。
