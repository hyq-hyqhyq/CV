from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
from ultralytics import YOLO

from task2.lib.tracking import (
    TrackState,
    bbox_center,
    bbox_count_point,
    color_for_track,
    draw_virtual_line,
    load_line_definition,
    signed_distance_to_line,
    stable_side,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run YOLO tracking and line crossing counting on a video.")
    parser.add_argument("--weights", type=Path, required=True, help="Trained detector weights, e.g. best.pt")
    parser.add_argument("--source", type=Path, required=True, help="Input video path.")
    parser.add_argument(
        "--tracker",
        type=Path,
        default=Path("task2/configs/botsort_task2.yaml"),
        help="Ultralytics tracker config yaml.",
    )
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--device", default="0")
    parser.add_argument("--line", nargs=4, type=int, default=None, metavar=("X1", "Y1", "X2", "Y2"))
    parser.add_argument("--line-config", type=Path, default=Path("task2/configs/line_count.sample.json"))
    parser.add_argument("--dead-zone", type=float, default=10.0)
    parser.add_argument(
        "--count-point",
        default="bottom_center",
        choices=["center", "bottom_center", "bottom_mid_80"],
        help="Which point of each bbox is used for line crossing. bottom_center is usually better for road scenes.",
    )
    parser.add_argument("--show-trails", action="store_true")
    parser.add_argument("--trail-length", type=int, default=30)
    parser.add_argument("--output-dir", type=Path, default=Path("task2/outputs/tracking"))
    parser.add_argument("--video-name", default=None, help="Annotated video filename. Defaults to <source>_tracked.mp4")
    return parser


def resolve_output_paths(source: Path, output_dir: Path, video_name: str | None) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    annotated_video = output_dir / (video_name or f"{source.stem}_tracked.mp4")
    tracking_csv = output_dir / f"{source.stem}_tracking.csv"
    summary_json = output_dir / f"{source.stem}_summary.json"
    return annotated_video, tracking_csv, summary_json


def main() -> None:
    args = build_parser().parse_args()
    line_config = None
    if args.line is None and args.line_config and args.line_config.exists():
        line_config = args.line_config
    line_start, line_end, dead_zone = load_line_definition(
        line_config=line_config,
        line_points=args.line,
        default_dead_zone=args.dead_zone,
    )
    annotated_video, tracking_csv, summary_json = resolve_output_paths(
        source=args.source,
        output_dir=args.output_dir,
        video_name=args.video_name,
    )

    capture = cv2.VideoCapture(str(args.source))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {args.source}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = cv2.VideoWriter(
        str(annotated_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create video writer: {annotated_video}")

    model = YOLO(str(args.weights))
    track_states: dict[int, TrackState] = {}
    counted_ids: set[int] = set()
    classwise_counts: dict[str, int] = defaultdict(int)
    total_crossings = 0
    forward_count = 0
    backward_count = 0

    with tracking_csv.open("w", newline="", encoding="utf-8") as file_obj:
        fieldnames = [
            "frame",
            "timestamp_sec",
            "track_id",
            "class_id",
            "class_name",
            "confidence",
            "x1",
            "y1",
            "x2",
            "y2",
            "center_x",
            "center_y",
            "count_point_x",
            "count_point_y",
            "signed_distance",
            "crossed_line",
        ]
        writer_csv = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer_csv.writeheader()

        frame_index = 0
        while True:
            has_frame, frame = capture.read()
            if not has_frame:
                break

            results = model.track(
                frame,
                persist=True,
                tracker=str(args.tracker),
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                device=args.device,
                verbose=False,
            )
            result = results[0]
            boxes = result.boxes

            if boxes is not None and len(boxes) > 0:
                xyxy_list = boxes.xyxy.cpu().tolist()
                conf_list = boxes.conf.cpu().tolist() if boxes.conf is not None else [0.0] * len(xyxy_list)
                cls_list = boxes.cls.int().cpu().tolist() if boxes.cls is not None else [-1] * len(xyxy_list)
                id_list = boxes.id.int().cpu().tolist() if boxes.id is not None else [-1] * len(xyxy_list)
                names = result.names

                for bbox, confidence, class_id, track_id in zip(xyxy_list, conf_list, cls_list, id_list):
                    x1, y1, x2, y2 = bbox
                    center = bbox_center(x1, y1, x2, y2)
                    count_point = bbox_count_point(x1, y1, x2, y2, args.count_point)
                    track_label = int(track_id)
                    state = track_states.setdefault(track_label, TrackState()) if track_label >= 0 else TrackState()
                    if state.trail.maxlen != args.trail_length:
                        state.trail = state.trail.__class__(state.trail, maxlen=args.trail_length)
                    state.trail.append(count_point)
                    current_distance = signed_distance_to_line(count_point, line_start, line_end)

                    crossed_line = False
                    current_side = stable_side(current_distance, dead_zone)
                    if track_label >= 0 and current_side != 0:
                        if state.previous_stable_side is None:
                            state.previous_stable_side = current_side
                        elif current_side != state.previous_stable_side:
                            previous_side = state.previous_stable_side
                            state.previous_stable_side = current_side
                            if track_label not in counted_ids:
                                total_crossings += 1
                                counted_ids.add(track_label)
                                crossed_line = True
                                class_name = names.get(class_id, str(class_id))
                                classwise_counts[class_name] += 1
                                if previous_side < 0 < current_side:
                                    forward_count += 1
                                else:
                                    backward_count += 1
                                state.counted = True
                        else:
                            state.previous_stable_side = current_side

                    state.previous_distance = current_distance
                    state.last_frame_seen = frame_index

                    color = color_for_track(track_label)
                    cv2.rectangle(
                        frame,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        color,
                        thickness=2,
                    )
                    label = f"ID {track_label} | {names.get(class_id, class_id)} | {confidence:.2f}"
                    cv2.putText(
                        frame,
                        label,
                        (int(x1), max(24, int(y1) - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.circle(frame, count_point, radius=4, color=color, thickness=-1)

                    if args.show_trails and len(state.trail) >= 2:
                        trail_points = list(state.trail)
                        for start_point, end_point in zip(trail_points[:-1], trail_points[1:]):
                            cv2.line(frame, start_point, end_point, color, thickness=2)

                    writer_csv.writerow(
                        {
                            "frame": frame_index,
                            "timestamp_sec": frame_index / fps,
                            "track_id": track_label,
                            "class_id": class_id,
                            "class_name": names.get(class_id, class_id),
                            "confidence": f"{confidence:.6f}",
                            "x1": f"{x1:.2f}",
                            "y1": f"{y1:.2f}",
                            "x2": f"{x2:.2f}",
                            "y2": f"{y2:.2f}",
                            "center_x": center[0],
                            "center_y": center[1],
                            "count_point_x": count_point[0],
                            "count_point_y": count_point[1],
                            "signed_distance": f"{current_distance:.6f}",
                            "crossed_line": int(crossed_line),
                        }
                    )

            draw_virtual_line(frame, line_start, line_end, total_crossings, forward_count, backward_count)
            writer.write(frame)

            if frame_index % 30 == 0:
                print(f"Processed frame {frame_index + 1}/{total_frames or '?'}")
            frame_index += 1

    capture.release()
    writer.release()

    processed_frames = frame_index
    decode_shortfall_frames = max(0, total_frames - processed_frames) if total_frames else 0
    summary = {
        "source_video": str(args.source.resolve()),
        "annotated_video": str(annotated_video.resolve()),
        "tracking_csv": str(tracking_csv.resolve()),
        "tracker_config": str(args.tracker.resolve()),
        "line": [list(line_start), list(line_end)],
        "dead_zone": dead_zone,
        "count_point": args.count_point,
        "fps": fps,
        "frame_width": width,
        "frame_height": height,
        "total_frames": total_frames,
        "metadata_total_frames": total_frames,
        "processed_frames": processed_frames,
        "decode_shortfall_frames": decode_shortfall_frames,
        "decoded_complete": decode_shortfall_frames == 0,
        "total_crossings": total_crossings,
        "forward_count": forward_count,
        "backward_count": backward_count,
        "counted_track_ids": sorted(counted_ids),
        "classwise_counts": dict(sorted(classwise_counts.items())),
    }
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Annotated video: {annotated_video}")
    print(f"Tracking CSV: {tracking_csv}")
    print(f"Summary JSON: {summary_json}")
    print(f"Processed frames: {processed_frames}/{total_frames or '?'}")
    if decode_shortfall_frames:
        print(
            "WARNING: OpenCV stopped before the container-reported frame count. "
            f"Missing {decode_shortfall_frames} frames; the source video may have a broken or unsupported stream."
        )
    print(f"Total crossings: {total_crossings}")


if __name__ == "__main__":
    main()
