# 期末项目报告大纲

## 1. 任务背景与目标

说明本项目面向多源 3D 资产生成与真实场景融合，目标是结合 2D Gaussian Splatting、SDS text-to-3D、single-image-to-3D 和 Blender 完成一个端到端 pipeline。

## 2. 数据与资产来源

### 2.1 真实物体 A 拍摄数据

描述手机拍摄设备、图片数量、拍摄路径、分辨率、光照条件和 COLMAP 注册情况。

### 2.2 文本生成物体 B prompt

列出 threestudio 使用的 prompt、负面 prompt、随机种子和生成目标。

### 2.3 单图生成物体 C 输入图像

展示原图、去背景结果和 foreground PNG，说明选择该物体的原因。

### 2.4 Mip-NeRF 360 背景场景

说明使用 `counter` 或 `garden` 场景，列出数据来源、图像数量和训练设置。

## 3. 方法

### 3.1 COLMAP 位姿估计

介绍特征提取、特征匹配、稀疏重建和图像去畸变流程。

### 3.2 2D Gaussian Splatting 重建

说明 2DGS 的显式表示、训练输入、优化目标和用于物体/背景重建的设置。

### 3.3 SDS text-to-3D 生成

说明 threestudio 中 SDS loss 的基本思想，以及 text prompt 如何约束 3D 资产生成。

### 3.4 Magic123 single-image-to-3D 生成

说明单图约束、多视图先验和几何/纹理优化流程。

### 3.5 统一 mesh 表达与 Blender 融合

说明为什么将 Gaussian/surfel 表达和生成式 3D 资产统一导出为 textured mesh，并介绍尺度归一化、坐标对齐、空间摆放、光照近似和渲染流程。

## 4. 实验设置

### 4.1 硬件环境

列出 GPU、CPU、内存、CUDA、PyTorch、Blender、COLMAP 和第三方仓库版本。

### 4.2 超参数

给出 object A、background、object B、object C 的迭代数、分辨率、学习率、优化器、loss 和 batch 设置。

### 4.3 训练/生成时间

统计每个组件的训练或生成时间，以及估计显存占用。

## 5. 实验结果

### 5.1 物体 A 结果

展示输入图像、COLMAP 稀疏结果、2DGS 渲染结果和导出 mesh。

### 5.2 物体 B 结果

展示 prompt、生成过程截图、mesh 结果和不同视角渲染。

### 5.3 物体 C 结果

展示输入图、去背景图、Magic123 输出和不同视角渲染。

### 5.4 背景场景结果

展示 Mip-NeRF 360 背景的 2DGS 重建和导出 mesh。

### 5.5 融合渲染结果

展示 Blender 融合场景、preview 图和 flythrough 视频关键帧。

## 6. 质量对比分析

### 6.1 几何准确度

比较各资产的形状完整性、孔洞、噪声和视角一致性。

### 6.2 纹理细节

比较真实纹理、生成纹理、背面纹理和贴图完整性。

### 6.3 计算耗时

用表格比较训练时间、生成时间和显存占用。

### 6.4 失败案例分析

分析 COLMAP 失败、Janus face、单图背面不确定性、mesh 贴图丢失、融合尺度不准等问题。

## 7. 表达形式统一与渲染实现

详细说明 mesh 归一化、Blender 导入、场景摆放、相机路径和视频导出实现。

## 8. 局限性与改进方向

讨论 2DGS mesh extraction 质量、AIGC 资产一致性、物理光照不真实、尺度估计不准确和计算成本等局限，并提出后续改进方向。

## 9. GitHub 链接与模型权重链接

填写 GitHub 仓库地址、云盘权重/输出链接和复现实验说明。

