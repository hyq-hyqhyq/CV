# 任务 2：VisDrone 检测、视频多目标跟踪与越线计数

本目录对应作业任务 2：使用 VisDrone 无人机航拍检测数据集微调 YOLOv8 检测器，并在测试视频上完成逐帧检测、多目标跟踪、遮挡/ID 分析和越线计数。

## 当前完成状态

| 项目 | 状态 | 证据路径 | 备注 |
| --- | --- | --- | --- |
| VisDrone 数据集转换代码 | 已有 | `task2/scripts/prepare_visdrone.py` | 将 VisDrone 官方标注转换为 YOLO 格式，过滤 ignore region 和非法类别。 |
| YOLOv8 检测训练代码 | 已有 | `task2/scripts/train_detector.py` | 已使用 YOLOv8m 训练，训练日志见 `task2/logs/train_yolov8m_1280_ep200.log`。 |
| 训练好的检测模型 | 已训练，权重未放入 Git | 服务器路径：`/mnt/data/huangyingqian/code/CV/runs/detect/task2/runs/train/visdrone_yolov8m_1280_7gpu_ep200/weights/best.pt` | `.pt` 权重被 `.gitignore` 忽略。报告中的“模型权重网盘链接”还需要补。 |
| 训练指标 | 已有日志 | `task2/logs/train_yolov8m_1280_ep200.log` | 最优模型验证集 mAP50 为 `0.577`，mAP50-95 为 `0.360`。 |
| loss / mAP 曲线图 | 服务器已生成，本地当前缺失 | 服务器日志显示位于 `runs/detect/task2/runs/train/visdrone_yolov8m_1280_7gpu_ep200/curves/` | 需要从服务器复制或重新生成到 `task2/report_assets/curves/`。 |
| 测试视频 | 已有 | `task2/videos/aerial.mp4` | 1920x1080，30 FPS，325 帧，约 10.83 秒，满足 10-30 秒要求。 |
| 检测 + 跟踪视频 | 已有 | `task2/outputs/tracking/aerial_x880_tracked.mp4` | 输出包含 bbox、类别、Tracking ID、轨迹、虚拟线和越线总数。 |
| 跟踪 CSV | 已有 | `task2/outputs/tracking/aerial_tracking.csv` | 记录每帧目标框、类别、置信度、Tracking ID、中心点、计数点和是否越线。 |
| 越线计数结果 | 已有 | `task2/outputs/tracking/aerial_summary.json` | 总越线数 `15`，正向 `9`，反向 `6`。 |
| 遮挡 / ID 跳变连续 3-4 帧截图 | 待补 | 建议输出到 `task2/report_assets/occlusion/` | 可先用候选帧 `302-305` 生成截图，再人工确认 ID 是否跳变。 |
| GitHub repo 链接 | 已有 | `https://github.com/hyq-hyqhyq/CV/tree/pj2` | 报告中可使用该链接。 |
| 模型权重网盘链接 | 待补 | 空 | 需要上传 `best.pt` 后填写。 |

## 目录与关键文件

```text
task2/
|-- configs/
|   |-- visdrone_det.local.yaml
|   |-- bytetrack_task2.yaml
|   `-- line_count.aerial_x880.json
|-- lib/
|   |-- visdrone.py
|   |-- tracking.py
|   `-- plotting.py
|-- scripts/
|   |-- prepare_visdrone.py
|   |-- train_detector.py
|   |-- plot_training_curves.py
|   |-- track_and_count.py
|   |-- find_occlusion_candidates.py
|   `-- extract_occlusion_frames.py
|-- videos/
|   `-- aerial.mp4
|-- outputs/tracking/
|   |-- aerial_x880_tracked.mp4
|   |-- aerial_tracking.csv
|   `-- aerial_summary.json
`-- logs/
    |-- train_yolov8m_1280_ep200.log
    `-- track_aerial_x880.log
```

## 数据集

使用 VisDrone2019-DET 检测数据集，默认路径配置在 `task2/configs/visdrone_det.local.yaml`：

```yaml
path: /mnt/data/huangyingqian/code/CV/task2/data/VisDrone2019-DET
train: VisDrone2019-DET-train/images
val: VisDrone2019-DET-val/images
test: VisDrone2019-DET-test-dev/images
```

类别数为 10：

| class id | class name |
| --- | --- |
| 0 | pedestrian |
| 1 | people |
| 2 | bicycle |
| 3 | car |
| 4 | van |
| 5 | truck |
| 6 | tricycle |
| 7 | awning-tricycle |
| 8 | bus |
| 9 | motor |

训练日志显示使用的划分规模：

| split | images | instances / notes |
| --- | ---: | --- |
| train | 6471 | VisDrone2019-DET-train，日志中显示 `0 backgrounds, 0 corrupt`。 |
| val | 548 | 验证集目标实例总数 `38759`。 |

标注转换逻辑位于 `task2/lib/visdrone.py`。转换时只保留 VisDrone 类别 id `1-10` 且 `score > 0` 的目标框，因此 ignore region、非法类别和无效框不会写入 YOLO 标签。

## 模型结构

本实验使用 `YOLOv8m` 单阶段检测模型，并加载 Ultralytics 官方预训练权重 `yolov8m.pt` 进行微调。

可写入报告的结构信息：

| 项目 | 内容 |
| --- | --- |
| Backbone / Neck / Head | YOLOv8m 默认结构，包含 Conv、C2f、SPPF、Upsample、Concat 和 Detect Head。 |
| 检测头类别数 | 从 COCO 的 80 类改为 VisDrone 的 10 类。日志显示 `Overriding model.yaml nc=80 with nc=10`。 |
| 预训练迁移 | 日志显示 `Transferred 469/475 items from pretrained weights`。 |
| 模型规模 | 训练时 summary：`170 layers, 25,862,110 parameters, 79.1 GFLOPs`。 |
| 验证时 fused 模型 | `93 layers, 25,845,550 parameters, 78.7 GFLOPs`。 |

## 训练设置

训练命令对应日志 `task2/logs/train_yolov8m_1280_ep200.log`，核心配置如下：

| 项目 | 设置 |
| --- | --- |
| model | `yolov8m.pt` |
| data | `task2/configs/visdrone_det.local.yaml` |
| input size | `1280` |
| epochs | 设置 `200`，实际 EarlyStopping 于第 92 epoch 停止 |
| batch size | `28` |
| device | `0,1,2,3,4,5,6`，7 张 RTX 4090 |
| workers | `16` |
| optimizer | `SGD` |
| initial lr | `0.01` |
| final lr factor | `0.01` |
| momentum | `0.937` |
| weight decay | `0.0005` |
| warmup epochs | `3.0` |
| close mosaic | `10` |
| patience | `30` |
| AMP | enabled |
| loss | YOLOv8 默认检测损失：box loss、classification loss、DFL loss |
| metrics | Precision、Recall、mAP50、mAP50-95 |

服务器训练命令参考：

```bash
cd /mnt/data/huangyingqian/code/CV
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python task2/scripts/train_detector.py \
  --model yolov8m.pt \
  --data task2/configs/visdrone_det.local.yaml \
  --imgsz 1280 \
  --epochs 200 \
  --batch 28 \
  --device 0,1,2,3,4,5,6 \
  --workers 16 \
  --optimizer SGD \
  --lr0 0.01 \
  --lrf 0.01 \
  --patience 30 \
  --name visdrone_yolov8m_1280_7gpu_ep200 \
  --exist-ok \
  2>&1 | tee task2/logs/train_yolov8m_1280_ep200.log
```

## 检测结果

最优权重为：

```text
/mnt/data/huangyingqian/code/CV/runs/detect/task2/runs/train/visdrone_yolov8m_1280_7gpu_ep200/weights/best.pt
```

日志显示 EarlyStopping：

```text
Best results observed at epoch 62, best model saved as best.pt.
92 epochs completed in 1.259 hours.
```

最优模型在 VisDrone val 上的检测结果：

| Class | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 548 | 38759 | 0.666 | 0.555 | 0.577 | 0.360 |
| pedestrian | 520 | 8844 | 0.744 | 0.621 | 0.678 | 0.346 |
| people | 482 | 5125 | 0.699 | 0.508 | 0.543 | 0.240 |
| bicycle | 364 | 1287 | 0.555 | 0.386 | 0.402 | 0.203 |
| car | 515 | 14064 | 0.847 | 0.858 | 0.886 | 0.648 |
| van | 421 | 1975 | 0.628 | 0.591 | 0.582 | 0.431 |
| truck | 266 | 750 | 0.663 | 0.517 | 0.544 | 0.380 |
| tricycle | 337 | 1045 | 0.565 | 0.501 | 0.470 | 0.285 |
| awning-tricycle | 220 | 532 | 0.407 | 0.267 | 0.257 | 0.169 |
| bus | 131 | 251 | 0.835 | 0.649 | 0.730 | 0.557 |
| motor | 485 | 4886 | 0.715 | 0.651 | 0.677 | 0.344 |

结果分析可写入报告：

- `car`、`bus`、`motor` 类效果较好，其中 `car` 的 mAP50 达到 `0.886`。
- `bicycle`、`tricycle`、`awning-tricycle` 相对困难，原因通常包括目标小、遮挡多、外观差异大、密集场景中目标框重叠严重。
- `awning-tricycle` 的 recall 仅 `0.267`，说明该类别漏检较明显。

## 测试视频

最终使用航拍测试视频：

```text
task2/videos/aerial.mp4
```

根据跟踪 summary：

| 项目 | 数值 |
| --- | ---: |
| 分辨率 | 1920 x 1080 |
| FPS | 30 |
| 总帧数 | 325 |
| 实际处理帧数 | 325 |
| 时长 | 约 10.83 秒 |
| 解码是否完整 | true |

该视频满足作业要求的 10-30 秒测试视频长度。

## 多目标跟踪与越线计数

最终跟踪配置：

| 项目 | 设置 |
| --- | --- |
| detector | YOLOv8m VisDrone fine-tuned best.pt |
| tracker | ByteTrack |
| tracker config | `task2/configs/bytetrack_task2.yaml` |
| input video | `task2/videos/aerial.mp4` |
| output video | `task2/outputs/tracking/aerial_x880_tracked.mp4` |
| output CSV | `task2/outputs/tracking/aerial_tracking.csv` |
| output summary | `task2/outputs/tracking/aerial_summary.json` |
| imgsz | `1280` |
| conf | `0.15` |
| iou | `0.45` |
| count point | `bottom_center` |

ByteTrack 参数：

```yaml
tracker_type: bytetrack
track_high_thresh: 0.15
track_low_thresh: 0.05
new_track_thresh: 0.15
track_buffer: 90
match_thresh: 0.9
fuse_score: true
```

最终虚拟线配置：

```json
{
  "line": [[880, 340], [880, 840]],
  "dead_zone": 10
}
```

说明：`aerial_tracking.csv` 中保存了 bbox 中心点 `center_x, center_y`。最终计数使用 `bottom_center` 作为穿越判定点，因为车辆与道路的接触点比 bbox 几何中心更稳定；如果报告需要严格表述为“检测框中心点”，可以将命令中的 `--count-point bottom_center` 改为 `--count-point center` 后重新生成结果。

最终服务器命令：

```bash
cd /mnt/data/huangyingqian/code/CV
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

cat > task2/configs/line_count.aerial_x880.json <<'EOF'
{
  "line": [[880, 340], [880, 840]],
  "dead_zone": 10
}
EOF

python task2/scripts/track_and_count.py \
  --weights runs/detect/task2/runs/train/visdrone_yolov8m_1280_7gpu_ep200/weights/best.pt \
  --source task2/videos/aerial.mp4 \
  --tracker task2/configs/bytetrack_task2.yaml \
  --line-config task2/configs/line_count.aerial_x880.json \
  --imgsz 1280 \
  --conf 0.15 \
  --iou 0.45 \
  --device 0 \
  --count-point bottom_center \
  --show-trails \
  --output-dir task2/outputs/tracking \
  --video-name aerial_x880_tracked.mp4 \
  2>&1 | tee task2/logs/track_aerial_x880.log
```

## 跟踪与越线结果

`task2/outputs/tracking/aerial_summary.json` 中的结果：

| 项目 | 数值 |
| --- | ---: |
| processed frames | 325 / 325 |
| total crossings | 15 |
| forward count | 9 |
| backward count | 6 |
| counted track IDs | 15 个 |

越线类别统计：

| class | count |
| --- | ---: |
| car | 10 |
| pedestrian | 3 |
| van | 2 |

`aerial_tracking.csv` 的整体统计：

| 项目 | 数值 |
| --- | ---: |
| CSV rows | 26067 |
| frames with detections | 325 |
| unique tracking IDs | 265 |
| rows marked crossed_line=1 | 15 |

跟踪 CSV 字段包括：

```text
frame, timestamp_sec, track_id, class_id, class_name, confidence,
x1, y1, x2, y2, center_x, center_y,
count_point_x, count_point_y, signed_distance, crossed_line
```

越线计数逻辑：

1. 对每一帧调用 `YOLO.track(..., persist=True)` 得到 bbox、类别、置信度和 Tracking ID。
2. 对每个检测框计算计数点，本次实验使用 `bottom_center`。
3. 计算该点到虚拟线的带符号距离。
4. 设置 `dead_zone=10`，点在线附近小范围抖动时不计入稳定换边。
5. 当同一 Tracking ID 的稳定侧从线的一侧变到另一侧时，记为一次越线。
6. 同一个 Tracking ID 只计数一次，避免同一目标反复抖动造成重复计数。

## 遮挡与 ID 跳变分析

当前状态：**遮挡 / 密集交汇候选帧已经可以从 CSV 中选出，但连续 3-4 帧截图尚未生成到本地。**

根据 `aerial_tracking.csv` 的检测密度和 bbox overlap 粗筛，建议优先使用：

```text
frames 302-305
```

该窗口平均每帧约 `93.5` 个检测目标，平均重叠对数约 `4.5`，适合作为遮挡或密集交汇分析片段。

报告分析思路：

- ByteTrack 主要依赖检测框位置、运动预测和 IoU 匹配维护 ID，本身不使用强 ReID 外观特征。
- 当目标密集、框重叠或短暂遮挡时，检测框可能丢失或发生位置突变，后续匹配可能失败，从而产生 ID 丢失或 ID 跳变。
- 如果遮挡时间短且检测框位置连续，Kalman 预测和 IoU 匹配可以维持原 ID。
- 最终报告需要结合 `frames 302-305` 的截图，人工确认目标 ID 是保持还是跳变。

生成遮挡候选和截图的服务器命令见下方“待补材料生成命令”。

## 当前还缺什么

以下内容建议在写最终报告前补齐：

| 缺失项 | 是否必须 | 说明 |
| --- | --- | --- |
| 训练曲线图片 | 必须 | 报告要求 loss 曲线、mAP 曲线。服务器已生成，但本地当前没有对应 PNG。 |
| 遮挡 / ID 跳变 3-4 帧截图 | 必须 | 报告要求连续 3-4 帧可视化，并分析 ID 是否跳变。 |
| 遮挡分析文字结论 | 必须 | 需要看截图后确认具体 ID 保持或跳变，不能瞎写。 |
| 模型权重网盘链接 | 必须 | `best.pt` 未进入 Git，需要上传网盘后填写链接。 |
| W&B 或 SwanLab 截图 | 视老师要求 | 当前有本地曲线生成逻辑；若必须使用平台截图，需要从 W&B/SwanLab 页面截取。 |
| GitHub repo 链接 | 已有 | `https://github.com/hyq-hyqhyq/CV/tree/pj2`。 |

## 待补材料生成命令

### 1. 生成并复制训练曲线

服务器上运行：

```bash
cd /mnt/data/huangyingqian/code/CV
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

mkdir -p task2/report_assets/curves

python task2/scripts/plot_training_curves.py \
  --results-csv runs/detect/task2/runs/train/visdrone_yolov8m_1280_7gpu_ep200/results.csv \
  --output-dir task2/report_assets/curves

cp runs/detect/task2/runs/train/visdrone_yolov8m_1280_7gpu_ep200/results.png task2/report_assets/curves/
cp runs/detect/task2/runs/train/visdrone_yolov8m_1280_7gpu_ep200/BoxPR_curve.png task2/report_assets/curves/
cp runs/detect/task2/runs/train/visdrone_yolov8m_1280_7gpu_ep200/confusion_matrix.png task2/report_assets/curves/
```

推荐加入 Git：

```bash
git add -f task2/report_assets/curves
git commit -m "Add task2 training curve assets"
git push origin pj2
```

### 2. 生成遮挡候选帧和 3-4 帧截图

服务器上运行：

```bash
cd /mnt/data/huangyingqian/code/CV
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

mkdir -p task2/report_assets/occlusion

python task2/scripts/find_occlusion_candidates.py \
  --tracking-csv task2/outputs/tracking/aerial_tracking.csv \
  --window 4 \
  --top-k 5 \
  --output-json task2/report_assets/occlusion/aerial_occlusion_candidates.json

python task2/scripts/extract_occlusion_frames.py \
  --video task2/outputs/tracking/aerial_x880_tracked.mp4 \
  --start-frame 302 \
  --count 4 \
  --stride 1 \
  --prefix aerial_occlusion_302_305 \
  --output-dir task2/report_assets/occlusion
```

推荐加入 Git：

```bash
git add -f task2/report_assets/occlusion
git commit -m "Add task2 occlusion analysis frames"
git push origin pj2
```

### 3. 上传模型权重并记录网盘链接

需要上传的权重：

```text
/mnt/data/huangyingqian/code/CV/runs/detect/task2/runs/train/visdrone_yolov8m_1280_7gpu_ep200/weights/best.pt
```

上传到网盘后，在最终报告中填写：

```text
模型权重网盘链接：待填写
```

### 4. 如果必须严格使用 bbox center 计数

当前最终结果用的是 `bottom_center`。如需严格使用检测框中心点，可以重新跑：

```bash
cd /mnt/data/huangyingqian/code/CV
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python task2/scripts/track_and_count.py \
  --weights runs/detect/task2/runs/train/visdrone_yolov8m_1280_7gpu_ep200/weights/best.pt \
  --source task2/videos/aerial.mp4 \
  --tracker task2/configs/bytetrack_task2.yaml \
  --line-config task2/configs/line_count.aerial_x880.json \
  --imgsz 1280 \
  --conf 0.15 \
  --iou 0.45 \
  --device 0 \
  --count-point center \
  --show-trails \
  --output-dir task2/outputs/tracking \
  --video-name aerial_x880_center_tracked.mp4 \
  2>&1 | tee task2/logs/track_aerial_x880_center.log
```

## 报告建议写法

最终报告 Task 2 部分建议按以下结构写：

1. 数据集介绍：VisDrone2019-DET，10 类无人机航拍目标，train/val 官方划分。
2. 模型结构：YOLOv8m，COCO 预训练，检测头改为 10 类，参数量和 GFLOPs。
3. 实验设置：imgsz、batch、optimizer、lr、epoch、loss、metrics。
4. 检测结果：放 mAP 表格和训练曲线图。
5. 视频检测与跟踪：说明使用 aerial.mp4、ByteTrack、输出 bbox/class/ID。
6. 越线计数：说明虚拟线坐标、计数点、dead zone、最终 crossing 数。
7. 遮挡与 ID 跳变分析：插入连续 3-4 帧截图，结合 ByteTrack 原理分析 ID 保持或跳变。
8. 附录链接：GitHub repo 链接、模型权重网盘链接。
