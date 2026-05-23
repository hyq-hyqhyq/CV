#!/usr/bin/env python3
"""Extract regularly spaced PNG frames from a video for COLMAP/2DGS input."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, help="Input video path.")
    parser.add_argument("--out_dir", required=True, help="Directory for output PNG frames.")
    parser.add_argument("--stride", type=int, default=5, help="Save one frame every N frames.")
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Maximum number of frames to write. Omit for no limit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_path = Path(args.video)
    out_dir = Path(args.out_dir)

    if args.stride <= 0:
        raise SystemExit("Error: --stride must be a positive integer.")
    if args.max_frames is not None and args.max_frames <= 0:
        raise SystemExit("Error: --max_frames must be positive when provided.")
    if not video_path.is_file():
        raise SystemExit(f"Error: video file not found: {video_path}")

    try:
        import cv2
    except ImportError as exc:
        raise SystemExit(
            "Error: OpenCV is required. Install it with `pip install opencv-python`."
        ) from exc

    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Error: failed to open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_idx = 0
    saved = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % args.stride == 0:
            saved += 1
            out_path = out_dir / f"frame_{saved:06d}.png"
            if not cv2.imwrite(str(out_path), frame):
                cap.release()
                raise SystemExit(f"Error: failed to write frame: {out_path}")
            if args.max_frames is not None and saved >= args.max_frames:
                frame_idx += 1
                break

        frame_idx += 1

    cap.release()
    total_text = str(total_frames) if total_frames > 0 else "unknown"
    print(f"Video: {video_path}")
    print(f"Total frames reported by video metadata: {total_text}")
    print(f"Frames decoded before stopping: {frame_idx}")
    print(f"Frames saved: {saved}")
    print(f"Output directory: {out_dir}")


if __name__ == "__main__":
    main()

