# Multi-source 3D Asset Generation and Real-scene Fusion with 2DGS and AIGC

This repository is a course-project engineering framework for **Topic 1: multi-source asset generation and real-scene fusion with 2DGS and AIGC**. It does not vendor large third-party repositories, pretrained weights, datasets, videos, or generated meshes. Instead, it provides wrapper scripts, configuration files, Blender scene scripts, evaluation utilities, report tables, and reproducible command templates.

## Project Overview

The project completes four connected parts:

- Real object reconstruction with 2DGS: capture real object A using phone multi-view photos or video, estimate camera poses with COLMAP, and reconstruct the object with 2D Gaussian Splatting.
- Text-to-3D generation with threestudio: generate virtual object B from a text prompt using an SDS-based text-to-3D pipeline.
- Image-to-3D generation with Magic123: generate object C from a single foreground image after background removal.
- Background reconstruction and scene fusion in Blender: reconstruct a Mip-NeRF 360 `counter` or `garden` background with 2DGS, export all assets as textured meshes, align them in Blender, and render a multi-view flythrough video.

## Method Overview

The reconstructed 2DGS background and real object A are originally explicit Gaussian/surfel-like representations, while threestudio and Magic123 usually output meshes or assets that can be exported as textured meshes. To reduce the complexity of directly combining heterogeneous 3D representations, this project converts all assets into a unified **textured mesh** format. Blender is then used for scale normalization, coordinate alignment, spatial placement, approximate lighting, preview rendering, and final flythrough rendering.

## Environment Setup

Recommended environment:

- Ubuntu 22.04
- NVIDIA CUDA GPU, preferably 12 GB VRAM or more
- CUDA and PyTorch versions matching the selected 2DGS, threestudio, and Magic123 repositories
- Blender 3.6+ or Blender 4.x
- COLMAP installed and available as `colmap`

Create the lightweight wrapper environment:

```bash
cd final_project_2dgs_aigc_fusion
conda env create -f environment.yml
conda activate final-project-2dgs-aigc-fusion
pip install -r requirements.txt
```

Install third-party repositories under `external/` by Git submodule or manual clone:

```bash
mkdir -p external

# Example only. Pin commits according to your course environment.
git clone <2dgs-repository-url> external/2d-gaussian-splatting
git clone https://github.com/threestudio-project/threestudio.git external/threestudio
git clone <magic123-repository-url> external/magic123
```

This repository only provides wrappers. It does not include third-party source code, pretrained weights, model checkpoints, datasets, videos, or generated assets.

## Data Preparation

Organize data as follows:

```text
data/
  object_a/
    raw_video/
    frames/
    colmap/
    masks/
  object_c/
    input_image/
    foreground_png/
  mipnerf360/
    counter/
    garden/
```

If you do not already have the Mip-NeRF 360 dataset locally, download only the needed background scene:

```bash
python scripts/download_mipnerf360.py \
  --scene counter \
  --data_root data
```

The script uses the official Mip-NeRF 360 `360_v2.zip` archive and extracts only the selected `counter` or `garden` scene into `data/mipnerf360/<scene>/`. The zip file is removed after extraction by default to save disk space; add `--keep_zip` if you want to keep it.

Object A capture recommendations:

- Capture 80-150 images, or extract a similar number of frames from a video.
- Move around the object for a full 360-degree view.
- Include multiple heights: low, middle, and slightly high viewpoints.
- Use stable, uniform lighting and avoid motion blur.
- Avoid transparent, mirror-like, thin, or strongly reflective objects.
- Keep the background textured enough for COLMAP matching.

Object C single-image recommendations:

- Use a clean background and remove it before running Magic123.
- Keep the whole object visible, not cropped.
- Prefer high resolution and even lighting.
- Avoid transparent, shiny, or heavily occluded objects.

## Commands

All commands below are intended to be copied from the repository root.

### 1. Extract frames for object A

```bash
python scripts/extract_frames.py \
  --video data/object_a/raw_video/object_a.mp4 \
  --out_dir data/object_a/frames \
  --stride 5 \
  --max_frames 150
```

### 2. Prepare COLMAP dataset

```bash
python scripts/prepare_colmap_dataset.py \
  --image_dir data/object_a/frames \
  --out_dir data/object_a/colmap \
  --mode symlink
```

### 3. Run COLMAP

```bash
bash scripts/run_colmap.sh \
  --image_dir data/object_a/colmap/images \
  --output_dir data/object_a/colmap
```

### 4. Train 2DGS for object A

```bash
bash scripts/run_2dgs_train_object_a.sh \
  --data_dir data/object_a/colmap/dense \
  --output_dir outputs/object_a/2dgs \
  --iterations 7000 \
  --resolution 2
```

### 5. Train 2DGS for Mip-NeRF 360 background

Use either `counter` or `garden`:

```bash
python scripts/download_mipnerf360.py \
  --scene counter \
  --data_root data
```

```bash
bash scripts/run_2dgs_train_background.sh \
  --data_dir data/mipnerf360/counter \
  --output_dir outputs/background/2dgs \
  --iterations 7000 \
  --resolution 4
```

### 6. Export 2DGS meshes

2DGS mesh extraction differs across forks. Check `scripts/export_2dgs_mesh.sh` and update `--export_command` or the TODO block for your selected repository version.

```bash
bash scripts/export_2dgs_mesh.sh \
  --asset object_a \
  --checkpoint_dir outputs/object_a/2dgs \
  --out_dir outputs/object_a/mesh

bash scripts/export_2dgs_mesh.sh \
  --asset background \
  --checkpoint_dir outputs/background/2dgs \
  --out_dir outputs/background/mesh
```

### 7. Generate object B with threestudio SDS

```bash
bash scripts/run_threestudio_object_b.sh \
  --prompt "a small stylized ceramic robot toy, glossy white body, blue circular eyes, rounded head, short arms, compact body, cute and minimal design, smooth clean surface, high-quality product design, single object, centered, highly detailed" \
  --output_dir outputs/object_b/threestudio \
  --config_name TODO_choose_valid_threestudio_sds_config
```

After training, export a textured mesh using the export command supported by your threestudio version and place it at:

```text
outputs/object_b/mesh/object_b.obj
```

or:

```text
outputs/object_b/mesh/object_b.glb
```

If GPU memory is insufficient, reduce render resolution, reduce iterations, use a smaller SDS config, or use lower-resolution guidance settings in the selected threestudio config.

### 8. Generate object C with Magic123

```bash
bash scripts/run_magic123_object_c.sh \
  --input_png data/object_c/foreground_png/object_c.png \
  --output_dir outputs/object_c/magic123 \
  --mesh_out outputs/object_c/mesh/object_c.obj \
  --run_command 'TODO_replace_with_your_magic123_command data.image_path={input_png} save_dir={output_dir}'
```

Magic123 has heavy dependencies and pretrained weights. Keep all weights outside this repository and configure them inside `external/magic123` according to that repository's documentation.

### 9. Normalize meshes

```bash
python scripts/normalize_meshes.py \
  --inputs \
    outputs/object_a/mesh/object_a.obj \
    outputs/object_b/mesh/object_b.obj \
    outputs/object_c/mesh/object_c.obj \
    outputs/background/mesh/background.obj \
  --out_dir outputs/normalized_meshes \
  --target_size 1.0
```

### 10. Compose scene in Blender

```bash
blender --background --python blender/compose_scene.py -- \
  --config configs/blender_scene.yaml \
  --out_blend outputs/fusion/fused_scene.blend \
  --preview outputs/fusion/preview.png
```

### 11. Render flythrough video

```bash
blender --background --python blender/render_flythrough.py -- \
  --blend_file outputs/fusion/fused_scene.blend \
  --out_video outputs/final_video/flythrough.mp4 \
  --num_frames 180 \
  --resolution_x 1280 \
  --resolution_y 720
```

### 12. Generate evaluation tables

Create runtime JSON files manually if needed, for example:

```json
{
  "object_a": {
    "training_or_generation_time_minutes": 45,
    "estimated_gpu_memory": "10 GB"
  }
}
```

Then run:

```bash
python scripts/evaluate_assets.py \
  --object_a_mesh outputs/object_a/mesh/object_a.obj \
  --object_b_mesh outputs/object_b/mesh/object_b.obj \
  --object_c_mesh outputs/object_c/mesh/object_c.obj \
  --background_mesh outputs/background/mesh/background.obj \
  --runtime_json outputs/evaluation/runtime_object_a.json \
  --runtime_json outputs/evaluation/runtime_object_b.json \
  --runtime_json outputs/evaluation/runtime_object_c.json \
  --runtime_json outputs/evaluation/runtime_background.json \
  --out_csv outputs/evaluation/asset_metrics.csv \
  --runtime_csv outputs/evaluation/runtime_table.csv

python scripts/collect_training_logs.py \
  --log_dir outputs/object_a/2dgs \
  --log_dir outputs/background/2dgs \
  --log_dir outputs/object_b/threestudio \
  --log_dir outputs/object_c/magic123 \
  --out_dir outputs/evaluation/logs

python scripts/make_report_tables.py \
  --asset_metrics outputs/evaluation/asset_metrics.csv \
  --out_dir report/tables
```

## Outputs

Expected final outputs:

```text
outputs/object_a/
outputs/object_b/
outputs/object_c/
outputs/background/
outputs/fusion/
outputs/final_video/flythrough.mp4
outputs/evaluation/asset_metrics.csv
outputs/evaluation/runtime_table.csv
```

Large outputs are ignored by Git. Upload final videos, meshes, checkpoints, and pretrained weights to cloud storage if required by the course.

## Report Checklist

- [ ] Task background
- [ ] Dataset description
- [ ] Method principles
- [ ] A/B/C asset results
- [ ] Background reconstruction result
- [ ] Fusion rendering result
- [ ] Quality comparison table
- [ ] Training loss curves or optimization curves
- [ ] Hyperparameter table
- [ ] GitHub link
- [ ] Cloud link for weights and large outputs
