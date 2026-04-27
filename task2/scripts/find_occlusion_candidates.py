from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from task2.lib.tracking import iou_xyxy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Suggest dense/occluded frame windows from tracking CSV.")
    parser.add_argument("--tracking-csv", type=Path, required=True)
    parser.add_argument("--window", type=int, default=4, help="Number of consecutive frames to score together.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--iou-threshold", type=float, default=0.15)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser


def score_frame(frame_df: pd.DataFrame, iou_threshold: float) -> dict[str, float]:
    boxes = [
        (float(row.x1), float(row.y1), float(row.x2), float(row.y2))
        for row in frame_df.itertuples(index=False)
    ]
    overlap_pairs = 0
    max_iou = 0.0
    for idx in range(len(boxes)):
        for jdx in range(idx + 1, len(boxes)):
            overlap = iou_xyxy(boxes[idx], boxes[jdx])
            max_iou = max(max_iou, overlap)
            if overlap >= iou_threshold:
                overlap_pairs += 1
    detection_count = len(boxes)
    score = detection_count + overlap_pairs * 2.0 + max_iou * 3.0
    return {
        "detection_count": float(detection_count),
        "overlap_pairs": float(overlap_pairs),
        "max_iou": float(max_iou),
        "score": float(score),
    }


def select_non_overlapping(windows: list[dict], top_k: int) -> list[dict]:
    selected: list[dict] = []
    for candidate in sorted(windows, key=lambda item: item["score"], reverse=True):
        overlap = False
        for existing in selected:
            if not (
                candidate["end_frame"] < existing["start_frame"]
                or candidate["start_frame"] > existing["end_frame"]
            ):
                overlap = True
                break
        if not overlap:
            selected.append(candidate)
        if len(selected) >= top_k:
            break
    return selected


def main() -> None:
    args = build_parser().parse_args()
    dataframe = pd.read_csv(args.tracking_csv)
    per_frame = []
    for frame_id, frame_df in dataframe.groupby("frame"):
        stats = score_frame(frame_df, args.iou_threshold)
        stats["frame"] = int(frame_id)
        per_frame.append(stats)

    frame_stats = pd.DataFrame(per_frame).sort_values("frame").reset_index(drop=True)
    if len(frame_stats) < args.window:
        raise RuntimeError("Tracking CSV is too short for the requested window length")

    window_summaries: list[dict] = []
    for start_idx in range(0, len(frame_stats) - args.window + 1):
        window_df = frame_stats.iloc[start_idx : start_idx + args.window]
        window_summaries.append(
            {
                "start_frame": int(window_df.iloc[0]["frame"]),
                "end_frame": int(window_df.iloc[-1]["frame"]),
                "avg_detections": float(window_df["detection_count"].mean()),
                "avg_overlap_pairs": float(window_df["overlap_pairs"].mean()),
                "avg_max_iou": float(window_df["max_iou"].mean()),
                "score": float(window_df["score"].mean()),
            }
        )

    selected = select_non_overlapping(window_summaries, args.top_k)
    if args.output_json is not None:
        args.output_json.write_text(json.dumps(selected, indent=2, ensure_ascii=False), encoding="utf-8")

    for item in selected:
        print(json.dumps(item, ensure_ascii=False))


if __name__ == "__main__":
    main()
