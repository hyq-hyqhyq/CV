# 任务 1：Oxford-IIIT Pet 宠物分类

这个目录对应“微调 ImageNet 预训练卷积神经网络完成宠物分类”的作业内容。当前版本已经统一为以 `ResNet-18` 为主线。

目前支持的实验内容：

- 基线模型：`ResNet-18` 的 ImageNet 预训练微调
- 预训练消融：`ResNet-18` 从随机初始化开始训练
- 注意力机制：在 `ResNet-18` 基础上加入 `SE Block`，得到 `SE-ResNet18`
- 超参数实验：支持比较不同学习率、训练轮数等设置
- 训练可视化：自动保存训练集和验证集的 `loss`、`accuracy` 曲线
- 日志后端：支持 `wandb` 或 `swanlab`

## 1. 环境准备

建议使用一个干净的 Python 环境，避免和本机已有包冲突。

### 方式一：venv

```bash
cd task1
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 方式二：Conda

```bash
conda create -n cv-pet python=3.10 -y
conda activate cv-pet
pip install -r requirements.txt
```

## 2. 目录结构

```text
task1/
|-- configs/           # 实验配置文件
|-- outputs/           # 训练输出目录
|-- src/pet_cls/       # 核心训练代码
|-- tests/             # 轻量自检脚本
|-- train.py           # 单次训练入口
|-- evaluate.py        # 模型评测入口
|-- run_experiments.py # 批量实验入口
|-- requirements.txt   # 依赖列表
`-- README.md
```

## 3. 已准备好的实验

### 3.1 基线模型：ResNet-18 预训练微调

```bash
python train.py --config configs/resnet18_pretrained.yaml
```

### 3.2 预训练消融：ResNet-18 随机初始化

```bash
python train.py --config configs/resnet18_scratch.yaml
```

### 3.3 注意力机制：SE-ResNet18

```bash
python train.py --config configs/se_resnet18_pretrained.yaml
```

## 4. 超参数实验

已经准备了几组可以直接运行的配置：

```bash
python run_experiments.py --configs configs/resnet18_pretrained.yaml configs/resnet18_lr_low.yaml configs/resnet18_lr_high.yaml configs/resnet18_longer.yaml
```

如果只是临时改参数，也可以直接覆盖配置项：

```bash
python train.py --config configs/resnet18_pretrained.yaml --set training.epochs=30 training.backbone_lr=5e-5 training.head_lr=5e-4
```

## 5. 训练日志与曲线

默认情况下，程序会在本地自动保存训练日志和曲线图，不依赖外部平台。

每次训练会在 `outputs/<时间戳>_<实验名>/` 下生成：

- `best.pt`：验证集最优模型权重
- `last.pt`：最后一个 epoch 的模型权重
- `metrics.csv`：每个 epoch 的训练和验证指标
- `metrics.jsonl`：逐行 JSON 格式日志
- `summary.json`：实验结果汇总
- `loss_curve.png`：训练集和验证集的 loss 曲线
- `accuracy_curve.png`：训练集和验证集的 accuracy 曲线
- `training_curves.png`：整合后的总曲线图，适合直接放进实验报告
- `data_split.json`：训练/验证划分信息
- `config_snapshot.yaml`：本次实验配置快照

其中 `training_curves.png` 最适合直接用于交报告。

## 6. wandb / swanlab 可视化

如果你需要在线可视化截图，可以启用相应日志后端。

### 使用 wandb

```bash
wandb login
python train.py --config configs/resnet18_pretrained.yaml --set logging.backend=wandb logging.project=oxfordiiit-pet
```

### 使用 swanlab

```bash
python train.py --config configs/resnet18_pretrained.yaml --set logging.backend=swanlab logging.project=oxfordiiit-pet
```

## 7. 数据集与实验设置

本项目使用 `torchvision.datasets.OxfordIIITPet`。

- 数据集：Oxford-IIIT Pet Dataset
- 类别数：37 类
- 测试集：官方 `test` split
- 训练/验证集：官方 `trainval` split，再做分类别分层划分
- 默认验证集比例：`0.15`
- 输入尺寸：`224 x 224`
- 损失函数：`CrossEntropyLoss`
- 评价指标：`Accuracy`
- 优化器：默认 `AdamW`
- 学习率策略：对预训练微调分别设置 `backbone_lr` 和 `head_lr`

## 8. 模型评测

训练完成后，可以这样评测模型：

```bash
python evaluate.py --config configs/resnet18_pretrained.yaml --checkpoint outputs/<your_run>/best.pt --split test
```

## 9. 正确性自检

项目带了一个不依赖真实数据集的 smoke test，用随机张量检查以下内容是否正常：

- `resnet18`
- `se_resnet18`
- 优化器和学习率调度器
- 单轮训练与评估流程

运行方式：

```bash
python tests/smoke_test.py
```

## 10. 建议的实验顺序

如果你想尽快把第一题做完整，建议按下面顺序跑：

1. 先跑 `ResNet-18` 预训练基线
2. 再跑 `ResNet-18` 随机初始化，用于预训练消融对比
3. 再跑 `SE-ResNet18`，完成注意力机制对比
4. 最后跑几组不同学习率或训练轮数，补齐超参数分析

这样最后写实验报告时结构会比较自然：

- 基线结果
- 超参数分析
- 预训练消融
- 注意力机制对比

## 11. Hyperparameter Analysis

本节提供一套可复现的“三轮超参搜索”脚本，基于 `configs/resnet18_pretrained.yaml`（ImageNet 预训练 `ResNet-18`）对学习率、训练轮数与正则化进行系统对比，并尽可能找到优于 baseline 的配置。

### 11.1 第一轮：学习率搜索（7 张 GPU 并行）

从 `task1/` 根目录执行：

```bash
bash scripts/run_hparam_phase1_lr_7gpu.sh
```

脚本会同时启动 7 个单卡训练进程（GPU 0-6），并将日志保存到 `logs/hparam/<experiment_name>.log`。

### 11.2 收集结果（生成汇总 CSV + 打印 Top 10）

训练结束后执行：

```bash
python scripts/collect_hparam_results.py
```

会生成 `outputs/hparam_summary.csv`，并在终端打印 `best_val_acc` 排名前 10 的实验。

### 11.3 根据第一轮结果设置 BEST_BACKBONE_LR / BEST_HEAD_LR

打开 `outputs/hparam_summary.csv`，在第一轮（`experiment_name` 以 `hparam_p1_lr_` 开头）里选择 `best_val_acc` 最高的一组，
取其 `backbone_lr` 与 `head_lr` 作为第二轮/第三轮的默认学习率。

你可以用环境变量传入：

```bash
BEST_BACKBONE_LR=2e-4 BEST_HEAD_LR=2e-3 bash scripts/run_hparam_phase2_reg_epoch_7gpu.sh
```

### 11.4 第二轮：训练轮数与正则化（7 张 GPU 并行）

默认使用 `BEST_BACKBONE_LR=1e-4`、`BEST_HEAD_LR=1e-3`，也可由环境变量覆盖：

```bash
bash scripts/run_hparam_phase2_reg_epoch_7gpu.sh
```

### 11.5 第三轮：组合实验（可选，7 张 GPU 并行）

第三轮允许传入：

- `BEST_BACKBONE_LR`（默认 `1e-4`）
- `BEST_HEAD_LR`（默认 `1e-3`）
- `BEST_EPOCHS`（默认 `20`）
- `BEST_WEIGHT_DECAY`（默认 `1e-4`）

运行：

```bash
bash scripts/run_hparam_phase3_final_7gpu.sh
```

### 11.6 生成汇总 CSV 与可视化图

```bash
python scripts/collect_hparam_results.py
python scripts/plot_hparam_bar.py
```

输出：

- `outputs/hparam_summary.csv`：所有已完成实验的汇总表（按 `best_val_acc` 降序）
- `outputs/hparam_plots/`：
  - `top10_best_val_acc.png`
  - `phase1_lr_best_val_acc.png`
  - `phase2_reg_epoch_best_val_acc.png`

### 11.7 报告建议引用

建议在实验报告中：

- 用 `outputs/hparam_summary.csv` 的表格对比不同超参设置；
- 用 `outputs/hparam_plots/` 下的条形图展示关键对比；
- 用每次运行目录下自动生成的 `training_curves.png` 展示收敛速度与过拟合趋势。
