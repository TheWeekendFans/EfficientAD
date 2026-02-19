# VS Code 直接点运行指南（不走命令行）

这份指南改为“只改代码里的配置，然后点击右上角运行”。

> 重要：模型已升级为更完整的 EfficientAD 风格实现（预训练 Teacher + Student + AE + 固定特征统计）。
> 旧版权重与新结构不兼容，必须先重新训练，再评估/分析。

## 0) 先做一次设置（只需一次）

1. 打开命令面板：`Ctrl+Shift+P`
2. 执行：`Python: Select Interpreter`
3. 选择：`GF` 环境（`C:\Users\zpl\miniconda3\envs\GF\python.exe`）

> 之后所有任务都使用这个解释器。

---

## 1) 环境与GPU检查（直接点运行）

### 运行
- 在 VS Code 打开 `scripts/env_check.py`
- 点击右上角运行按钮（Run Python File）

### 预期结果
- 终端打印：`python ok`
- 打印 `torch` 版本
- `cuda=True`
- `gpus=8`

如果 `cuda=False` 或 `gpus<8`，先不要继续训练。

---

## 2) 训练 Grid（直接运行 main.py）

### 操作
1. 打开 `main.py`
2. 修改顶部这两行：
   - `EMBEDDED_MODE = 'train'`
   - `EMBEDDED_CLASS = 'grid'`
3. 右上角点击运行

### 预期结果
- 输出 `Starting Training for grid...`
- 出现 `Epoch x/100` 进度条
- 生成：
  - `output/models/grid/student_last.pth`
  - `output/models/grid/autoencoder_last.pth`
  - `output/models/grid/teacher_initial.pth`

---

## 3) 训练 Transistor（直接运行 main.py）

### 操作
1. 打开 `main.py`
2. 修改：`EMBEDDED_CLASS = 'transistor'`（`EMBEDDED_MODE` 保持 `train`）
3. 再次点击运行

### 预期结果
- 输出 `Starting Training for transistor...`
- 生成 `output/models/transistor/*.pth`

---

## 4) 评估 Grid AUROC（直接运行 main.py）

### 操作
1. `EMBEDDED_MODE = 'evaluate'`
2. `EMBEDDED_CLASS = 'grid'`
3. 运行 `main.py`

### 预期输出
- `Image AUROC: xx.xx%`
- `Pixel AUROC: xx.xx%`
- 可视化图在 `output/vis/`

---

## 5) 评估 Transistor AUROC（直接运行 main.py）

### 操作
1. `EMBEDDED_MODE = 'evaluate'`
2. `EMBEDDED_CLASS = 'transistor'`
3. 运行 `main.py`

### 预期输出
- `Image AUROC: xx.xx%`
- `Pixel AUROC: xx.xx%`

---

## 6) 分析 Grid（Escape/Overkill + Adaptive）

### 操作
1. `EMBEDDED_MODE = 'analysis'`
2. `EMBEDDED_CLASS = 'grid'`
3. 运行 `main.py`

### 预期输出
- `Zero Escape Threshold: ...`
- `At Zero Escape Rate -> Overkill Rate: ...%`
- `Adaptive Threshold (k-sigma): ...`
- 生成 `output/tradeoff_curve.png`

---

## 7) 分析 Transistor（Escape/Overkill + Adaptive）

### 操作
1. `EMBEDDED_MODE = 'analysis'`
2. `EMBEDDED_CLASS = 'transistor'`
3. 运行 `main.py`

### 预期输出
- 同上

---

## 8) 最终验收标准（对应你的项目目标）

- 图像级：`Image AUROC >= 98.5%`
- 像素级：目标关注 `AU-PRO >= 92.0%`（当前代码主输出 Pixel AUROC，若你需要我可以补上严格 AU-PRO 计算）
- 产线约束：`Escape Rate = 0` 时 `Overkill Rate` 尽量落在 `3~5%`
- 鲁棒性：比较固定阈值与自适应阈值，验证过杀下降

---

## 9) 常见问题

1. 右上角运行时提示 `No module named xxx`
- 说明解释器不是 GF，重新执行 `Python: Select Interpreter`。

2. CUDA OOM
- 把训练任务中的 `--batch_size 8` 改成 `4` 或 `2`。

3. 评估报找不到 epoch 权重
- 先确认训练已完成；或在命令中把 `--epoch` 改为你已有的轮次。

4. 数据路径问题,     
- 打开 `runtime_config.py`，把 `DEFAULTS['data_path']` 改成你的项目目录。
