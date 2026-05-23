#!/usr/bin/env python3
"""Normalize meshes to a shared origin and bounding-box size using trimesh."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


SUPPORTED_EXTENSIONS = {".obj", ".glb", ".gltf", ".ply", ".stl", ".off"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, help="Input mesh paths.")
    parser.add_argument("--out_dir", required=True, help="Output directory for normalized meshes.")
    parser.add_argument(
        "--target_size",
        type=float,
        default=1.0,
        help="Maximum bounding-box side length after normalization.",
    )
    return parser.parse_args()


def mesh_has_texture(obj: Any) -> bool:
    geometries = []
    if hasattr(obj, "geometry"):
        geometries = list(obj.geometry.values())
    else:
        geometries = [obj]

    for geom in geometries:
        visual = getattr(geom, "visual", None)
        if visual is not None and getattr(visual, "kind", None) == "texture":
            return True
    return False


def get_bounds(obj: Any) -> np.ndarray:
    bounds = np.asarray(obj.bounds, dtype=float)
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all():
        raise ValueError("mesh bounds are invalid")
    return bounds


def apply_normalization(obj: Any, center: np.ndarray, scale: float) -> Any:
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] *= scale
    matrix[:3, 3] = -center * scale
    obj.apply_transform(matrix)
    return obj


def safe_output_name(input_path: Path, out_dir: Path) -> Path:
    stem = input_path.stem
    suffix = input_path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        suffix = ".obj"
    return out_dir / f"{stem}{suffix}"


def main() -> None:
    args = parse_args()
    if args.target_size <= 0:
        raise SystemExit("Error: --target_size must be positive.")

    try:
        import trimesh
    except ImportError as exc:
        raise SystemExit("Error: trimesh is required. Install it with `pip install trimesh`.") from exc

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    normalization_info: dict[str, Any] = {}

    for input_text in args.inputs:
        input_path = Path(input_text)
        if not input_path.is_file():
            raise SystemExit(f"Error: mesh file not found: {input_path}")
        if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise SystemExit(f"Error: unsupported mesh extension: {input_path}")

        mesh = trimesh.load(input_path, force="scene" if input_path.suffix.lower() in {".glb", ".gltf"} else None)
        bounds = get_bounds(mesh)
        extents = bounds[1] - bounds[0]
        max_extent = float(np.max(extents))
        if max_extent <= 0 or not math.isfinite(max_extent):
            raise SystemExit(f"Error: invalid bounding box for mesh: {input_path}")

        center = (bounds[0] + bounds[1]) / 2.0
        scale = args.target_size / max_extent
        has_texture = mesh_has_texture(mesh)
        if not has_texture:
            print(f"Warning: no texture detected for {input_path}; geometry will still be exported.")

        normalized = apply_normalization(mesh.copy(), center, scale)
        out_path = safe_output_name(input_path, out_dir)
        normalized.export(out_path)

        normalization_info[input_path.name] = {
            "input_path": str(input_path),
            "output_path": str(out_path),
            "original_bbox_min": bounds[0].tolist(),
            "original_bbox_max": bounds[1].tolist(),
            "original_bbox_size": extents.tolist(),
            "center_translation_before_scale": (-center).tolist(),
            "scale": scale,
            "target_size": args.target_size,
            "has_texture": has_texture,
        }
        print(f"Normalized {input_path} -> {out_path}")

    info_path = out_dir / "normalization_info.json"
    info_path.write_text(json.dumps(normalization_info, indent=2), encoding="utf-8")
    print(f"Normalization info written to: {info_path}")


if __name__ == "__main__":
    main()

