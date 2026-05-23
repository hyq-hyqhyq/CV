#!/usr/bin/env python3
"""Compose normalized meshes in Blender, save a fused scene, and render preview."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any


def blender_argv() -> list[str]:
    if "--" in sys.argv:
        return sys.argv[sys.argv.index("--") + 1 :]
    return sys.argv[1:]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/blender_scene.yaml", help="Scene YAML config.")
    parser.add_argument("--out_blend", default="outputs/fusion/fused_scene.blend", help="Output .blend path.")
    parser.add_argument("--preview", default="outputs/fusion/preview.png", help="Preview render path.")
    return parser.parse_args(blender_argv())


def parse_scalar(value: str) -> Any:
    value = value.strip()
    if value.startswith("[") and value.endswith("]"):
        items = [item.strip() for item in value[1:-1].split(",") if item.strip()]
        return [parse_scalar(item) for item in items]
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return data or {}
    except ImportError:
        data: dict[str, Any] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            clean = line.split("#", 1)[0].strip()
            if not clean or ":" not in clean:
                continue
            key, value = clean.split(":", 1)
            data[key.strip()] = parse_scalar(value)
        return data


def resolve_path(project_root: Path, path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else project_root / path


def require_vec(config: dict[str, Any], key: str, length: int = 3) -> list[float]:
    value = config.get(key)
    if not isinstance(value, list) or len(value) != length:
        raise SystemExit(f"Error: config key {key} must be a list of length {length}.")
    return [float(item) for item in value]


def import_mesh(bpy, mesh_path: Path, label: str) -> list[Any]:
    if not mesh_path.is_file():
        raise SystemExit(f"Error: mesh file not found for {label}: {mesh_path}")

    before = {obj.name for obj in bpy.context.scene.objects}
    suffix = mesh_path.suffix.lower()
    if suffix == ".obj":
        if hasattr(bpy.ops.wm, "obj_import"):
            bpy.ops.wm.obj_import(filepath=str(mesh_path))
        elif hasattr(bpy.ops.import_scene, "obj"):
            bpy.ops.import_scene.obj(filepath=str(mesh_path))
        else:
            raise SystemExit("Error: this Blender version has no OBJ importer enabled.")
    elif suffix in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=str(mesh_path))
    elif suffix == ".ply":
        if hasattr(bpy.ops.wm, "ply_import"):
            bpy.ops.wm.ply_import(filepath=str(mesh_path))
        elif hasattr(bpy.ops.import_mesh, "ply"):
            bpy.ops.import_mesh.ply(filepath=str(mesh_path))
        else:
            raise SystemExit("Error: this Blender version has no PLY importer enabled.")
    else:
        raise SystemExit(f"Error: unsupported mesh extension for {label}: {mesh_path.suffix}")

    imported = [obj for obj in bpy.context.scene.objects if obj.name not in before]
    if not imported:
        raise SystemExit(f"Error: Blender imported no objects from {mesh_path}")
    return imported


def parent_objects(bpy, objects: list[Any], parent_name: str):
    parent = bpy.data.objects.new(parent_name, None)
    bpy.context.scene.collection.objects.link(parent)
    for obj in objects:
        obj.parent = parent
    return parent


def set_transform(obj, location: list[float], rotation_degrees: list[float], scale: list[float]) -> None:
    obj.location = location
    obj.rotation_euler = [math.radians(value) for value in rotation_degrees]
    obj.scale = scale


def set_render_engine(bpy) -> None:
    try:
        bpy.context.scene.render.engine = "BLENDER_EEVEE_NEXT"
    except Exception:
        try:
            bpy.context.scene.render.engine = "BLENDER_EEVEE"
        except Exception:
            bpy.context.scene.render.engine = "BLENDER_WORKBENCH"


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_file():
        raise SystemExit(f"Error: Blender scene config not found: {config_path}")

    project_root = config_path.resolve().parent.parent
    config = load_yaml(config_path)

    import bpy

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    background_mesh = resolve_path(project_root, str(config.get("background_mesh", "")))
    object_meshes = {
        "object_a": resolve_path(project_root, str(config.get("object_a_mesh", ""))),
        "object_b": resolve_path(project_root, str(config.get("object_b_mesh", ""))),
        "object_c": resolve_path(project_root, str(config.get("object_c_mesh", ""))),
    }

    bg_parent = parent_objects(bpy, import_mesh(bpy, background_mesh, "background"), "background")
    bg_parent.location = (0.0, 0.0, 0.0)

    for asset_name, mesh_path in object_meshes.items():
        parent = parent_objects(bpy, import_mesh(bpy, mesh_path, asset_name), asset_name)
        set_transform(
            parent,
            require_vec(config, f"{asset_name}_location"),
            require_vec(config, f"{asset_name}_rotation"),
            require_vec(config, f"{asset_name}_scale"),
        )

    light_data = bpy.data.lights.new("key_area_light", type="AREA")
    light_data.energy = 500.0
    light_data.size = 4.0
    light = bpy.data.objects.new("key_area_light", light_data)
    bpy.context.scene.collection.objects.link(light)
    light.location = require_vec(config, "light_location")

    camera_data = bpy.data.cameras.new("camera")
    camera = bpy.data.objects.new("camera", camera_data)
    bpy.context.scene.collection.objects.link(camera)
    camera.location = require_vec(config, "camera_location")
    camera.rotation_euler = [math.radians(value) for value in require_vec(config, "camera_rotation")]
    camera.data.lens = 28
    bpy.context.scene.camera = camera

    resolution = require_vec(config, "render_resolution", length=2)
    bpy.context.scene.render.resolution_x = int(resolution[0])
    bpy.context.scene.render.resolution_y = int(resolution[1])
    bpy.context.scene.render.film_transparent = False
    set_render_engine(bpy)

    out_blend = resolve_path(project_root, args.out_blend)
    preview = resolve_path(project_root, args.preview)
    out_blend.parent.mkdir(parents=True, exist_ok=True)
    preview.parent.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.save_as_mainfile(filepath=str(out_blend))

    bpy.context.scene.render.image_settings.file_format = "PNG"
    bpy.context.scene.render.filepath = str(preview)
    bpy.ops.render.render(write_still=True)

    print(f"Saved fused Blender scene: {out_blend}")
    print(f"Saved preview render: {preview}")


if __name__ == "__main__":
    main()

