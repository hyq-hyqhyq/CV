# 任务 2：VisDrone 目标检测、视频多目标跟踪与越线计数

这个文件夹提供了一套完整流程，用于完成作业中的任务 2，主要包括：

1. 将 VisDrone 检测标注转换为 YOLO 格式。
2. 基于 VisDrone 微调训练 YOLOv8 检测模型。
3. 生成实验报告可直接使用的 loss / mAP 曲线图。
4. 对测试视频进行目标检测与多目标跟踪，并输出稳定的 Tracking ID。
5. 基于虚拟线实现越线计数。
6. 导出 3-4 帧关键画面，用于遮挡与 ID 跳变分析。

## 目录结构

```text
task2/
|-- configs/
|   |-- botsort_task2.yaml
|   |-- line_count.sample.json
|   `-- visdrone_det.yaml
|-- data/
|-- lib/
|-- outputs/
|-- report_assets/
|-- runs/
`-- scripts/
```

## 环境配置

安装依赖：

```bash
pip install -r task2/requirements.txt
```

建议环境中至少包含以下包：

- `ultralytics`
- `opencv-python`
- `pandas`
- `matplotlib`
- `wandb`

如果你准备重新配置环境，建议使用 `Python 3.10` 或 `Python 3.11`。

## 数据集放置方式

请将下载好的 VisDrone 检测数据集放到下面的位置：

```text
task2/data/VisDrone2019-DET/
|-- VisDrone2019-DET-train/
|   |-- annotations/
|   `-- images/
|-- VisDrone2019-DET-val/
|   |-- annotations/
|   `-- images/
`-- VisDrone2019-DET-test-dev/
    `-- images/
```

当前默认划分方式为：

- 训练集：官方 `VisDrone2019-DET-train`
- 验证集：官方 `VisDrone2019-DET-val`

## 1. 转换 VisDrone 标注

这一步会把官方标注转换为 YOLO 所需的 `.txt` 标签，并写入各个 split 下的 `labels/` 目录。

```bash
python task2/scripts/prepare_visdrone.py --write-yaml
```

如果加上 `--write-yaml`，脚本还会生成一个带绝对路径的数据集配置文件：

```text
task2/configs/visdrone_det.local.yaml
```

## 2. 训练检测模型

示例命令：

```bash
python task2/scripts/train_detector.py --model yolov8n.pt --data task2/configs/visdrone_det.local.yaml --imgsz 960 --epochs 80 --batch 8 --device 0 --optimizer SGD --lr0 0.01 --sync-wandb
```

当前训练封装脚本中的默认实验参数如下：

- 模型：`YOLOv8n`
- 输入尺寸：`960`
- 训练轮数：`80`
- Batch size：`8`
- 优化器：`SGD`
- 初始学习率：`0.01`
- Weight decay：`5e-4`
- Warmup epochs：`3`
- 关闭 Mosaic 的 epoch：`10`
- 随机种子：`42`

损失函数使用 YOLOv8 默认的检测损失，包含：

- `box loss`
- `classification loss`
- `DFL loss`

Ultralytics 在验证阶段会保存并可视化以下指标，本项目也会进一步整理成报告可用曲线：

- `precision`
- `recall`
- `mAP50`
- `mAP50-95`

## 3. 生成训练曲线

训练结束后，脚本会自动在下面目录生成本地 PNG 曲线图：

```text
task2/runs/train/<run_name>/curves/
```

如果你已经有现成的 `results.csv`，也可以单独重新生成：

```bash
python task2/scripts/plot_training_curves.py --results-csv task2/runs/train/visdrone_yolov8n/results.csv --output-dir task2/report_assets/curves
```

默认生成的图包括：

- `loss_curves.png`
- `metric_curves.png`
- `lr_curves.png`

如果训练时加了 `--sync-wandb`，脚本还会把 `results.csv` 中的指标同步到 W&B，方便你截取训练过程曲线作为实验报告插图。

## 4. 视频跟踪与越线计数

你可以先修改 `task2/configs/line_count.sample.json` 中的虚拟线位置，也可以直接在命令行里传入线段坐标。

示例命令：

```bash
python task2/scripts/track_and_count.py --weights task2/runs/train/visdrone_yolov8n/weights/best.pt --source path\\to\\your_video.mp4 --tracker task2/configs/botsort_task2.yaml --line 240 420 1120 420 --show-trails
```

运行结果会保存到：

```text
task2/outputs/tracking/
|-- <video_name>_tracked.mp4
|-- <video_name>_tracking.csv
`-- <video_name>_summary.json
```

生成的标注视频中会包含：

- 检测框
- 类别标签
- 稳定的 Tracking ID
- 虚拟越线
- 当前总越线计数

越线计数逻辑基于以下信息：

- 检测框中心点坐标
- 目标点到虚拟线的带符号距离
- Tracking ID 的连续性

默认实现中，同一个 track 只会在首次穿越该线时计数一次，这样更符合“统计跨越该线的物体总数”的要求。

## 5. 查找遮挡 / 密集交汇片段

可以利用跟踪输出的 CSV，自动筛选出更容易发生遮挡或目标交汇的连续帧窗口：

```bash
python task2/scripts/find_occlusion_candidates.py --tracking-csv task2/outputs/tracking/your_video_tracking.csv --window 4 --top-k 5
```

这个脚本会优先挑出目标密度高、框之间重叠更多的片段，方便你快速定位报告中要分析的 3-4 帧区域。

## 6. 导出报告需要的 3-4 帧画面

当你确定了要分析的帧范围后，可以从带标注的跟踪视频中抽取关键帧：

```bash
python task2/scripts/extract_occlusion_frames.py --video task2/outputs/tracking/your_video_tracked.mp4 --start-frame 180 --count 4 --stride 1
```

脚本会输出：

- 每一帧单独的 PNG 图片
- 一张拼接好的长条图，便于直接插入 PDF 报告

## 建议保留的实验结果材料

这个文件夹中的代码已经帮你准备好了报告常用素材，主要包括：

- 训练集 / 验证集 loss 曲线
- 验证集指标曲线
- 跟踪结果视频
- 跟踪结果 CSV 和统计 JSON
- 含 Tracking ID 的遮挡分析关键帧

## 说明

- VisDrone 是检测任务，核心评价指标通常应以 `mAP50` 和 `mAP50-95` 为主，而不是传统分类任务中的 Accuracy。
- 如果老师在表述里写了 “Accuracy / mAP 曲线”，你在报告里可以使用 `precision`、`recall` 配合 `mAP` 来展示模型性能变化。
- `line_count.sample.json` 只是模板，最终演示前请根据你自己的视频重新调整越线位置。
