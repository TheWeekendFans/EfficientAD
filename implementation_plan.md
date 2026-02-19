# 实施计划 - 基于EfficientAD的无监督半导体晶圆表面缺陷检测系统

## 目标描述
本项目旨在构建一个面向半导体产线的无监督晶圆表面缺陷检测系统。核心基于 **EfficientAD** 算法，仅使用良品数据进行训练。系统不仅追求高学术指标（AUROC > 98.5%, AUPRO > 92.0%），更注重产线实际应用，引入**漏检率 (Escape Rate)** 和 **过杀率 (Overkill Rate)** 指标，并在此基础上实现**零漏检**约束下的过杀率控制。此外，通过模拟光照和噪声扰动，提升算法在真实工况下的鲁棒性。

## 用户审查要求
> [!IMPORTANT]
> 请确认以下两点：
> 1. 数据集路径确认：系统将默认读取 `g:\半导体offer\2\grid\grid` 和 `g:\半导体offer\2\transistor\transistor` 作为数据源。
> 2. 算力需求：EfficientAD 训练涉及 Knowledge Distillation，建议使用 GPU 进行训练。

## 拟议变更

### 项目结构
将在 `g:\半导体offer\2\` 目录下创建以下模块：

#### 数据处理 (`dataset.py`)
- 实现 `MVTecDataset` 类，加载训练集（仅良品）和测试集。
- **鲁棒性增强**：在数据加载流程中加入 `RobustnessAugmentation` 模块，包含：
    - `RandomBrightnessContrast`: 模拟照明衰减。
    - `GaussianNoise`: 模拟传感器抖动。
    - 这些增强仅用于“自适应阈值拟合”阶段或评估阶段的鲁棒性测试，训练阶段保持标准预处理以维持特征纯净度（EfficientAD原文通常不做强增强，依靠特征提取器的泛化能力，但此处为了拟合自适应阈值，我们需要构造“困难良品”及其统计分布）。

#### 模型架构 (`model.py`) (基于 EfficientAD)
- **PDN (Patch Description Network)**: 
    - Teacher: 预训练并在ImageNet上蒸馏过的轻量级CNN。
    - Student: 结构相同，训练目标是匹配Teacher的输出。
- **Autoencoder**: 用于重建Teacher的特征，检测逻辑异常。
- loss函数：MSE loss (蒸馏损失 + 重建损失)。

#### 训练流程 (`train.py`)
- 按照 EfficientAD 论文流程：
    1. 训练 PDN Student 逼近 Teacher。
    2. 训练 Autoencoder 重建 Teacher 特征。
    3. 全流程端到端微调（可选，视收敛情况而定）。
- 保存最佳模型权重。

#### 推理与评估 (`evaluate.py`)
- 计算异常得分图 (Anomaly Map)。
- 计算标准指标：**Image-level AUROC**, **Pixel-level AU-PRO**。
- 生成可视化结果：原图 + 热力图 + 分割掩码叠加。

#### 产线指标分析 (`production_analysis.py`)
- **漏检率 (Escape Rate)**: $FN / (TP + FN)$ (针对缺陷样本)
- **过杀率 (Overkill Rate)**: $FP / (TN + FP)$ (针对良品样本)
- **Trade-off 曲线**: 遍历阈值，绘制 Escape Rate vs. Overkill Rate 曲线。
- **自适应阈值**:
    - 利用增强后的良品数据（模拟工况扰动）计算异常得分分布。
    - 设定阈值 $T = \mu + k \cdot \sigma$，其中 $k$ 由“零漏检”目标在验证集上反推或根据统计学置信度设定（如 3-sigma, 6-sigma）。
    - 对比传统固定阈值，展示过杀率降低效果。

#### 配置与工具 (`config.py`, `utils.py`)
- 集中管理超参数（学习率、Batch Size、图像尺寸=256x256等）。
- 通用工具函数。

#### 主程序 (`main.py`)
- 集成训练、评估、产线分析全流程。
- 提供命令行接口选择模式（train/evaluate/analysis）。

## 验证计划

### 自动化验证
1. **模型收敛性**: 检查训练 loss 是否下降。
2. **指标达标**: 
    - 运行 `python main.py --mode evaluate`
    - 验证 Grid/Transistor 类别上 AUROC >= 98.5%。
3. **产线指标**:
    - 运行 `python main.py --mode analysis`
    - 验证在 Escape Rate = 0 时，Overkill Rate 是否在 3-5% 区间内。

### 手动验证
- 查看生成的 `output/images` 目录下的热力图，确认异常区域定位准确。
- 检查 `tradeoff_curve.png` 确认曲线走势符合预期。
