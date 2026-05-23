# 常见问题排查

## COLMAP 匹配失败怎么办

- 检查图片是否过少，建议至少 80 张左右。
- 检查相邻视角是否连续，视频抽帧过稀时调小 `--stride`。
- 检查图片是否严重模糊、曝光变化过大或背景太纯。
- 尝试 `sequential_matcher`，尤其是视频连续抽帧数据。
- 确认图片路径中没有特殊字符，COLMAP 输出目录有写入权限。

## 2DGS 显存不足怎么办

- 降低 `--resolution` 或启用更强 downscale。
- 减少 batch、采样点数或训练迭代数，具体参数取决于所使用的 2DGS fork。
- 关闭不必要的 viewer、GUI、WandB 图像记录。
- 先用小分辨率跑通 pipeline，再提高质量。

## 2DGS 背景训练太慢怎么办

- Mip-NeRF 360 场景图像分辨率较高，建议先使用 `--resolution 4` 或更低分辨率。
- 选择 `counter` 或 `garden` 中较容易训练的一组子数据。
- 减少迭代数，例如先跑 3000-7000 iterations 做课程展示版本。
- 只在最终结果阶段使用更高分辨率和更多迭代。

## threestudio 结果出现 Janus face 怎么办

- 调整 prompt，加入 "single front face"、"consistent back view" 等约束。
- 使用更稳定的 SDS 配置或多阶段优化配置。
- 减少过强的正面语义词，增加整体形状描述。
- 调整 guidance scale、camera sampling 范围和训练迭代数。
- 多试几个随机种子，选择最稳定的结果。

## Magic123 背面乱生成怎么办

- 单图到 3D 天然存在背面不确定性，报告中应说明这是方法局限。
- 选择形状简单、背面可推测的物体。
- 使用干净的 foreground PNG，减少边缘误抠。
- 调整 Magic123 的正则项、迭代数和 prompt 描述。
- 若允许，可补充文字 prompt 描述背面材质和形状。

## Blender 导入 mesh 没贴图怎么办

- 检查 OBJ 是否有对应 `.mtl` 文件，贴图图片路径是否仍然有效。
- 对 GLB/GLTF 优先使用 Blender 的 glTF 导入器。
- 确认贴图文件没有被 `.gitignore` 或移动操作漏掉。
- 在 Blender 的 Shader Editor 中检查材质节点是否连接到 Base Color。
- 如果导出工具只给了几何，可以在报告中将 `has_texture` 标为 false，并说明原因。

## 融合后物体漂浮怎么办

- 先运行 `normalize_meshes.py`，统一每个物体的尺度。
- 在 `configs/blender_scene.yaml` 中调整 `object_*_location` 的 z 值。
- 在 Blender 里打开侧视图，观察物体底部是否接触背景桌面或地面。
- 如果 mesh 原点不在底部，先居中归一化，再手动微调摆放高度。

## 渲染视频太慢怎么办

- 降低 `--resolution_x` 和 `--resolution_y`。
- 减少 `--num_frames`，课程展示可先用 120 帧。
- 使用 EEVEE/Workbench 预览渲染，最终再切换到更高质量设置。
- 降低采样数、关闭高成本体积和复杂阴影。
- 检查 mesh 面数，必要时对高面数资产做简化。

