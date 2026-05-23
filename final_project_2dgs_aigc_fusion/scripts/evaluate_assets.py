#!/usr/bin/env python3
"""Generate CSV metrics for generated/reconstructed mesh assets."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


FIELDS = [
    "asset_name",
    "source_type",
    "input_type",
    "method",
    "output_representation",
    "num_vertices",
    "num_faces",
    "bbox_x",
    "bbox_y",
    "bbox_z",
    "has_texture",
    "training_or_generation_time_minutes",
    "estimated_gpu_memory",
    "geometry_quality_note",
    "texture_quality_note",
    "failure_cases",
]

RUNTIME_FIELDS = [
    "asset_name",
    "method",
    "training_or_generation_time_minutes",
    "estimated_gpu_memory",
]


ASSET_META = {
    "object_a": {
        "source_type": "real_capture",
        "input_type": "multi_view_images_or_video",
        "method": "COLMAP + 2DGS",
    },
    "object_b": {
        "source_type": "aigc_text_to_3d",
        "input_type": "text_prompt",
        "method": "threestudio SDS",
    },
    "object_c": {
        "source_type": "aigc_image_to_3d",
        "input_type": "single_foreground_image",
        "method": "Magic123",
    },
    "background": {
        "source_type": "real_scene_dataset",
        "input_type": "Mip-NeRF 360 images",
        "method": "2DGS",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object_a_mesh", help="Mesh path for object A.")
    parser.add_argument("--object_b_mesh", help="Mesh path for object B.")
    parser.add_argument("--object_c_mesh", help="Mesh path for object C.")
    parser.add_argument("--background_mesh", help="Mesh path for background.")
    parser.add_argument(
        "--runtime_json",
        action="append",
        default=[],
        help="Runtime metadata JSON. Can be passed multiple times.",
    )
    parser.add_argument(
        "--out_csv",
        default="outputs/evaluation/asset_metrics.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--runtime_csv",
        default="outputs/evaluation/runtime_table.csv",
        help="Output runtime summary CSV path.",
    )
    return parser.parse_args()


def load_runtime_json(paths: list[str]) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for text in paths:
        path = Path(text)
        if not path.is_file():
            print(f"Warning: runtime JSON not found, skipping: {path}")
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and "asset_name" in item:
                    merged[str(item["asset_name"])] = item
        elif isinstance(data, dict):
            if "asset_name" in data:
                merged[str(data["asset_name"])] = data
            else:
                for key, value in data.items():
                    if isinstance(value, dict):
                        merged[str(key)] = value
    return merged


def has_texture(mesh: Any) -> bool:
    geometries = list(mesh.geometry.values()) if hasattr(mesh, "geometry") else [mesh]
    for geom in geometries:
        visual = getattr(geom, "visual", None)
        if visual is not None and getattr(visual, "kind", None) == "texture":
            return True
    return False


def mesh_stats(mesh_path: Path) -> dict[str, Any]:
    try:
        import trimesh
    except ImportError as exc:
        raise SystemExit("Error: trimesh is required. Install it with `pip install trimesh`.") from exc

    mesh = trimesh.load(mesh_path, force="scene" if mesh_path.suffix.lower() in {".glb", ".gltf"} else None)
    geometries = list(mesh.geometry.values()) if hasattr(mesh, "geometry") else [mesh]
    num_vertices = sum(len(getattr(geom, "vertices", [])) for geom in geometries)
    num_faces = sum(len(getattr(geom, "faces", [])) for geom in geometries)
    bounds = np.asarray(mesh.bounds, dtype=float)
    bbox = bounds[1] - bounds[0]
    return {
        "num_vertices": num_vertices,
        "num_faces": num_faces,
        "bbox_x": float(bbox[0]),
        "bbox_y": float(bbox[1]),
        "bbox_z": float(bbox[2]),
        "has_texture": has_texture(mesh),
    }


def build_row(asset_name: str, mesh_text: str | None, runtime: dict[str, Any]) -> dict[str, Any]:
    meta = ASSET_META[asset_name]
    row: dict[str, Any] = {
        "asset_name": asset_name,
        "source_type": meta["source_type"],
        "input_type": meta["input_type"],
        "method": meta["method"],
        "output_representation": "textured_mesh",
        "num_vertices": "",
        "num_faces": "",
        "bbox_x": "",
        "bbox_y": "",
        "bbox_z": "",
        "has_texture": "",
        "training_or_generation_time_minutes": runtime.get("training_or_generation_time_minutes", ""),
        "estimated_gpu_memory": runtime.get("estimated_gpu_memory", ""),
        "geometry_quality_note": runtime.get("geometry_quality_note", ""),
        "texture_quality_note": runtime.get("texture_quality_note", ""),
        "failure_cases": runtime.get("failure_cases", ""),
    }

    if not mesh_text:
        print(f"Warning: no mesh argument provided for {asset_name}; metrics left blank.")
        return row

    mesh_path = Path(mesh_text)
    if not mesh_path.is_file():
        print(f"Warning: mesh not found for {asset_name}: {mesh_path}; metrics left blank.")
        return row

    row.update(mesh_stats(mesh_path))
    return row


def main() -> None:
    args = parse_args()
    runtime = load_runtime_json(args.runtime_json)
    rows = [
        build_row("object_a", args.object_a_mesh, runtime.get("object_a", {})),
        build_row("object_b", args.object_b_mesh, runtime.get("object_b", {})),
        build_row("object_c", args.object_c_mesh, runtime.get("object_c", {})),
        build_row("background", args.background_mesh, runtime.get("background", {})),
    ]

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    runtime_csv = Path(args.runtime_csv)
    runtime_csv.parent.mkdir(parents=True, exist_ok=True)
    with runtime_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RUNTIME_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in RUNTIME_FIELDS})

    print(f"Asset metrics written to: {out_csv}")
    print(f"Runtime table written to: {runtime_csv}")


if __name__ == "__main__":
    main()
