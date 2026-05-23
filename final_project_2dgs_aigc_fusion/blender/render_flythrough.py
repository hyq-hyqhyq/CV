#!/usr/bin/env python3
"""Render a circular flythrough video and keyframes from a fused Blender scene."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def blender_argv() -> list[str]:
    if "--" in sys.argv:
        return sys.argv[sys.argv.index("--") + 1 :]
    return sys.argv[1:]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blend_file", required=True, help="Input fused .blend file.")
    parser.add_argument("--out_video", required=True, help="Output MP4 path.")
    parser.add_argument("--num_frames", type=int, default=180, help="Number of animation frames.")
    parser.add_argument("--resolution_x", type=int, default=1280, help="Render width.")
    parser.add_argument("--resolution_y", type=int, default=720, help="Render height.")
    return parser.parse_args(blender_argv())


def ensure_camera(bpy):
    if bpy.context.scene.camera is not None:
        return bpy.context.scene.camera
    camera_data = bpy.data.cameras.new("camera")
    camera = bpy.data.objects.new("camera", camera_data)
    bpy.context.scene.collection.objects.link(camera)
    bpy.context.scene.camera = camera
    return camera


def set_linear_interpolation(camera) -> None:
    if not camera.animation_data or not camera.animation_data.action:
        return
    for fcurve in camera.animation_data.action.fcurves:
        for keyframe in fcurve.keyframe_points:
            keyframe.interpolation = "LINEAR"


def render_keyframes(bpy, keyframes: list[int], keyframe_dir: Path) -> None:
    keyframe_dir.mkdir(parents=True, exist_ok=True)
    bpy.context.scene.render.image_settings.file_format = "PNG"
    for frame in keyframes:
        bpy.context.scene.frame_set(frame)
        bpy.context.scene.render.filepath = str(keyframe_dir / f"keyframe_{frame:04d}.png")
        bpy.ops.render.render(write_still=True)


def configure_video_output(bpy, out_video: Path) -> None:
    bpy.context.scene.render.image_settings.file_format = "FFMPEG"
    bpy.context.scene.render.ffmpeg.format = "MPEG4"
    bpy.context.scene.render.ffmpeg.codec = "H264"
    bpy.context.scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
    bpy.context.scene.render.ffmpeg.ffmpeg_preset = "GOOD"
    bpy.context.scene.render.filepath = str(out_video)


def main() -> None:
    args = parse_args()
    if args.num_frames <= 1:
        raise SystemExit("Error: --num_frames must be greater than 1.")
    if args.resolution_x <= 0 or args.resolution_y <= 0:
        raise SystemExit("Error: render resolution must be positive.")

    blend_file = Path(args.blend_file)
    if not blend_file.is_file():
        raise SystemExit(f"Error: blend file not found: {blend_file}")

    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    import bpy
    from camera_path_utils import circular_camera_location, look_at, scene_bounds

    bpy.ops.wm.open_mainfile(filepath=str(blend_file))

    out_video = Path(args.out_video)
    out_video.parent.mkdir(parents=True, exist_ok=True)
    keyframe_dir = out_video.parent / "keyframes"

    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = args.num_frames
    scene.render.resolution_x = args.resolution_x
    scene.render.resolution_y = args.resolution_y
    scene.render.fps = 24

    camera = ensure_camera(bpy)
    center, extents = scene_bounds(bpy)
    radius = max(max(extents[0], extents[1]) * 1.35, 3.0)
    height = center[2] + max(extents[2] * 0.65, 1.2)
    look_target = (center[0], center[1], center[2] + extents[2] * 0.15)

    for frame in range(1, args.num_frames + 1):
        scene.frame_set(frame)
        camera.location = circular_camera_location(frame, args.num_frames, center, radius, height)
        look_at(camera, look_target)
        camera.keyframe_insert(data_path="location", frame=frame)
        camera.keyframe_insert(data_path="rotation_euler", frame=frame)
    set_linear_interpolation(camera)

    keyframes = sorted({1, args.num_frames // 4, args.num_frames // 2, (args.num_frames * 3) // 4, args.num_frames})
    render_keyframes(bpy, keyframes, keyframe_dir)

    configure_video_output(bpy, out_video)
    bpy.ops.render.render(animation=True)

    print(f"Saved flythrough video: {out_video}")
    print(f"Saved keyframes to: {keyframe_dir}")


if __name__ == "__main__":
    main()

