# Report Image Package

这个文件夹只保留报告中需要插入的图片。所有图片都在 `output/` 目录下。

| 图片路径 | 内容说明 |
| --- | --- |
| `output/object_a_input_frame.png` | 物体 A 的原始手机视频截图，用于展示真实多视角重建的输入来源。 |
| `output/object_a_2dgs_preview.png` | 物体 A 经过 COLMAP + 2DGS 重建后的单独预览图。 |
| `output/object_b_text_to_3d_preview.png` | 物体 B 使用 threestudio / SDS 文本到 3D 生成后的单独预览图。 |
| `output/object_c_input.png` | 物体 C 的去背景单图输入。 |
| `output/object_c_magic123_montage.png` | 物体 C 使用 Magic123 单图到 3D 生成后的多视角预览图。 |
| `output/background_2dgs_preview.png` | Mip-NeRF 360 背景场景经过 2DGS 重建后的预览图。 |
| `output/fusion_preview.png` | A/B/C 三个物体插入到背景场景后的最终融合预览图。 |
| `output/fusion_keyframe_0022.png` | 融合场景漫游视频关键帧。 |
| `output/fusion_keyframe_0045.png` | 融合场景漫游视频关键帧。 |
| `output/fusion_keyframe_0090.png` | 融合场景漫游视频关键帧。 |
| `output/fusion_keyframe_0135.png` | 融合场景漫游视频关键帧。 |
| `output/loss_curves.png` | 训练 / 优化过程曲线图，用于报告中的实验过程展示。 |
| `output/fusion_flythrough.mp4` | 融合场景的多视角漫游渲染视频。 |

说明：本提交包包含报告图片和最终漫游视频，不包含 mesh、checkpoint、预训练权重或表格。
