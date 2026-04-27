from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract 3-4 key frames from a video for occlusion analysis.")
    parser.add_argument("--video", type=Path, required=True, help="Annotated tracking video is recommended.")
    parser.add_argument("--output-dir", type=Path, default=Path("task2/report_assets/occlusion"))
    parser.add_argument("--frames", nargs="+", type=int, default=None, help="Explicit frame ids to export.")
    parser.add_argument("--start-frame", type=int, default=None, help="Start frame when --frames is not used.")
    parser.add_argument("--count", type=int, default=4, help="Number of frames to export from --start-frame.")
    parser.add_argument("--stride", type=int, default=1, help="Frame step size when using --start-frame.")
    parser.add_argument("--prefix", default="occlusion")
    parser.add_argument("--target-height", type=int, default=320)
    return parser


def resolve_frames(explicit_frames: list[int] | None, start_frame: int | None, count: int, stride: int) -> list[int]:
    if explicit_frames:
        return explicit_frames
    if start_frame is None:
        raise ValueError("Provide either --frames or --start-frame")
    return [start_frame + index * stride for index in range(count)]


def read_frame(capture: cv2.VideoCapture, frame_index: int):
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    if not ok:
        raise RuntimeError(f"Failed to read frame {frame_index}")
    return frame


def make_contact_strip(frames: list[tuple[int, np.ndarray]], target_height: int) -> np.ndarray:
    rendered = []
    for frame_index, frame in frames:
        scale = target_height / frame.shape[0]
        resized = cv2.resize(frame, dsize=None, fx=scale, fy=scale)
        canvas = cv2.copyMakeBorder(resized, 40, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(
            canvas,
            f"Frame {frame_index}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        rendered.append(canvas)
    return np.concatenate(rendered, axis=1)


def main() -> None:
    args = build_parser().parse_args()
    frame_ids = resolve_frames(args.frames, args.start_frame, args.count, args.stride)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {args.video}")

    collected_frames: list[tuple[int, np.ndarray]] = []
    for frame_index in frame_ids:
        frame = read_frame(capture, frame_index)
        collected_frames.append((frame_index, frame))
        frame_path = args.output_dir / f"{args.prefix}_frame_{frame_index:05d}.png"
        cv2.imwrite(str(frame_path), frame)
        print(frame_path)

    strip = make_contact_strip(collected_frames, target_height=args.target_height)
    strip_path = args.output_dir / f"{args.prefix}_strip.png"
    cv2.imwrite(str(strip_path), strip)
    print(strip_path)

    capture.release()


if __name__ == "__main__":
    main()
