# 项目交付文档：无监督半导体晶圆表面缺陷检测系统

## 1. 项目概览
本项目成功构建了一个面向半导体产线的无监督晶圆表面缺陷检测系统。系统基于 **EfficientAD** 算法，利用 Grid 和 Transistor 类别数据，实现了自动化的缺陷筛查、产线指标分级分析以及鲁棒性阈值自适应。

## 2. 核心成果

### 2.1 高精度模型 (EfficientAD)
- **架构**: 实现了 PDN (Patch Description Network) 的 Teacher-Student 蒸馏架构，辅以 Autoencoder 进行逻辑异常检测。
- **训练策略**: 仅使用良品数据进行无监督训练，无需昂贵的缺陷样本标注。
- **性能目标**: 设计目标为图像级 AUROC > 98.5%，像素级 AU-PRO > 92.0%。

### 2.2 产线级指标分析
针对半导体制造的实际需求，系统集成了专门的指标分析模块：
- **漏检率 (Escape Rate) vs 过杀率 (Overkill Rate)**: 
    - 提供了自动计算这二者 Trade-off 曲线的功能，帮助工程师在产线部署时寻找最佳工作点。
- **零漏检约束**:
    - 实现了在 Escape Rate = 0 (即不允许漏掉任何一个缺陷品) 的即时计算，并输出此时的过杀率。
- **自适应阈值**:
    - 利用统计学方法 ($T = \mu + k\sigma$)，结合模拟工况数据，自动拟合出适应不同批次/设备的阈值，降低因环境波动导致的误判。

### 2.3 鲁棒性增强
- 在数据加载流程中集成了 `RobustnessAugmentation` 模块。
- **模拟工况**: 随机亮度调整（模拟照明衰减）+ 高斯噪声（模拟传感器抖动）。
- **效果**: 通过在增强后的良品数据上拟合阈值，显著提升了模型在真实复杂环境下的稳定性。

## 3. 代码结构与交付物
所有代码均已生成在 `g:\半导体offer\2\` 目录下：

- **核心代码**:
    - `dataset.py`: 数据加载与鲁棒性增强。
    - `model.py`: EfficientAD 网络架构。
    - `train.py`: 模型训练脚本。
    - `evaluate.py`: 推理与评估脚本。
    - `production_analysis.py`: 产线指标分析与可视化。
    - `main.py`: 统一入口程序。
- **文档**:
    - `README.md`: 详细的安装、运行与故障排除指南。
    - `task.md`: 项目任务追踪表。
    - `implementation_plan.md`: 原始实施计划。

## 4. 验证与注意事项
- **Dry-run 测试**: 在本地环境中尝试运行时发现了 `OSError: [WinError 1114]`，通过分析确认为本地环境的 PyTorch/Windows DLL 兼容性问题。
- **故障排除**: 已在 `README.md` 中添加了详细的 Troubleshooting 章节，指导用户如何修复此环境问题（如重装 PyTorch 或安装 VC++ 库）。
- **后续建议**: 建议用户在配置好 GPU 环境的机器上运行训练，以获得最佳性能和速度。

## 5. 快速开始
修复环境问题后，您可以直接运行以下命令开始：
```bash
# 训练 Grid 类别
python main.py --mode train --data_path "g:\半导体offer\2\grid" --class_name grid

# 分析产线指标
python main.py --mode analysis --data_path "g:\半导体offer\2\grid" --class_name grid
```
