# 任务 1：Oxford-IIIT Pet 宠物识别

这个目录对应“微调在 ImageNet 上预训练的卷积神经网络实现宠物识别”任务，已经准备好以下实验内容：

- 基线模型：`ResNet-18` 和 `ResNet-34` 的 ImageNet 预训练微调
- 预训练消融：`ResNet-18` 从随机初始化开始训练
- 注意力机制：在基线模型上加入 `SE Block`，得到 `SE-ResNet18`
- 超参数实验：支持比较不同学习率、训练轮数等组合
- 训练可视化：自动保存训练/验证集的 `loss`、`accuracy` 曲线
- 日志平台：支持 `wandb` 或 `swanlab`

## 1. 环境准备

建议使用一个新的 Python 环境，避免本机已有环境中的依赖冲突。

### 方式一：`venv`

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
├── configs/                 # 实验配置文件
├── outputs/                 # 训练输出目录
├── src/pet_cls/             # 核心训练代码
├── tests/                   # 正确性自检脚本
├── train.py                 # 单次训练入口
├── evaluate.py              # 模型评测入口
├── run_experiments.py       # 批量实验入口
├── requirements.txt         # 依赖列表
└── README.md
```

## 3. 已实现的实验

### 3.1 基线模型：ResNet-18 预训练微调

```bash
python train.py --config configs/resnet18_pretrained.yaml
```

### 3.2 基线模型：ResNet-34 预训练微调

```bash
python train.py --config configs/resnet34_pretrained.yaml
```

### 3.3 预训练消融：ResNet-18 随机初始化

```bash
python train.py --config configs/resnet18_scratch.yaml
```

### 3.4 注意力机制：SE-ResNet18

```bash
python train.py --config configs/se_resnet18_pretrained.yaml
```

## 4. 超参数实验

已经准备了几组可以直接运行的超参数配置：

```bash
python run_experiments.py --configs configs/resnet18_pretrained.yaml configs/resnet18_lr_low.yaml configs/resnet18_lr_high.yaml configs/resnet18_longer.yaml
```

如果只想临时修改参数，也可以直接覆盖配置项：

```bash
python train.py --config configs/resnet18_pretrained.yaml --set training.epochs=30 training.backbone_lr=5e-5 training.head_lr=5e-4
```

## 5. 训练日志与曲线

默认情况下，程序会在本地保存训练日志和曲线图片，不依赖外部平台。

每次训练会在 `outputs/<时间戳>_<实验名>/` 下生成：

- `best.pt`：验证集最优模型权重
- `last.pt`：最后一个 epoch 的模型权重
- `metrics.csv`：每个 epoch 的训练/验证指标
- `metrics.jsonl`：逐行 JSON 格式日志
- `summary.json`：实验结果汇总
- `loss_curve.png`：训练集和验证集的 loss 曲线
- `accuracy_curve.png`：训练集和验证集的 accuracy 曲线
- `training_curves.png`：汇总曲线图，可直接放进实验报告
- `data_split.json`：训练/验证划分信息
- `config_snapshot.yaml`：本次实验的配置快照

其中 `training_curves.png` 是最适合直接放报告的图片。

## 6. wandb / swanlab 可视化

如果你想在实验报告中使用在线可视化截图，可以打开对应日志后端。

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
- 训练/验证集：官方 `trainval` split 按类别分层随机切分
- 默认验证集比例：`0.15`
- 输入尺寸：`224 x 224`
- 损失函数：`CrossEntropyLoss`
- 评价指标：`Accuracy`
- 优化器：默认 `AdamW`
- 学习率策略：为预训练微调分别设置 `backbone_lr` 和 `head_lr`

## 8. 模型评测

训练完成后，可以使用下面的命令评测模型：

```bash
python evaluate.py --config configs/resnet18_pretrained.yaml --checkpoint outputs/<your_run>/best.pt --split test
```

## 9. 正确性自检

项目包含一个不依赖真实数据集的 `smoke test`，用随机张量检查以下内容是否正常：

- `resnet18`
- `resnet34`
- `se_resnet18`
- 优化器和学习率调度器
- 单轮训练与评估流程

运行方式：

```bash
python tests/smoke_test.py
```

## 10. 当前环境的已知问题

我在当前机器上检测到两类环境问题：

- `python` 和 `pip` 指向的不是同一个环境
- 现有 Conda 环境中的 `torch` 相关依赖不完整，存在 `dill` / `sympy` 问题

代码里已经加入了轻量兼容处理，能缓解一部分导入错误，但最稳妥的方式仍然是新建一个干净环境，然后重新安装 `requirements.txt` 中的依赖。

## 11. 建议的实验顺序

如果你想更顺手地完成这次作业，建议按下面顺序跑：

1. 先跑 `ResNet-18` 预训练基线
2. 再跑 `ResNet-18` 随机初始化，做预训练消融对比
3. 再跑 `SE-ResNet18`，完成注意力机制对比
4. 最后跑几组不同学习率和 epoch 的配置，补齐超参数分析

这样你最后写实验报告时，结构会比较自然：

- 基线结果
- 超参数分析
- 预训练消融
- 注意力机制对比
