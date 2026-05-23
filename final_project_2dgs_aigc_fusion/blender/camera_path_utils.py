"""Camera and bounding-box utilities for Blender flythrough rendering."""

from __future__ import annotations

import math
from typing import Iterable, Sequence


def look_at(obj, target: Sequence[float]) -> None:
    """Rotate a Blender object so its local -Z axis points at target."""
    from mathutils import Vector

    target_vec = Vector(target)
    direction = target_vec - obj.location
    if direction.length == 0:
        return
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def iter_mesh_world_corners(bpy, exclude_types: Iterable[str] = ("CAMERA", "LIGHT")):
    """Yield world-space bounding-box corners for visible mesh-like objects."""
    from mathutils import Vector

    excluded = set(exclude_types)
    for obj in bpy.context.scene.objects:
        if obj.type in excluded or obj.hide_render:
            continue
        if not hasattr(obj, "bound_box") or not obj.bound_box:
            continue
        for corner in obj.bound_box:
            yield obj.matrix_world @ Vector(corner)


def scene_bounds(bpy) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return scene center and extents from renderable object bounding boxes."""
    corners = list(iter_mesh_world_corners(bpy))
    if not corners:
        return (0.0, 0.0, 0.5), (2.0, 2.0, 1.0)

    mins = [min(corner[i] for corner in corners) for i in range(3)]
    maxs = [max(corner[i] for corner in corners) for i in range(3)]
    center = tuple((mins[i] + maxs[i]) / 2.0 for i in range(3))
    extents = tuple(max(maxs[i] - mins[i], 0.001) for i in range(3))
    return center, extents


def circular_camera_location(
    frame: int,
    num_frames: int,
    center: Sequence[float],
    radius: float,
    height: float,
    start_angle_degrees: float = -90.0,
) -> tuple[float, float, float]:
    """Return a smooth circular camera position around center."""
    if num_frames <= 1:
        phase = 0.0
    else:
        phase = (frame - 1) / (num_frames - 1)
    angle = math.radians(start_angle_degrees + phase * 360.0)
    return (
        center[0] + radius * math.cos(angle),
        center[1] + radius * math.sin(angle),
        height,
    )

