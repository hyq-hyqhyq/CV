from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize pure-detection debug CSV files.")
    parser.add_argument("--input-root", type=Path, default=Path("task2/outputs/predict_debug"))
    parser.add_argument("--output-dir", type=Path, default=Path("task2/report_assets"))
    parser.add_argument("--reference-run", default=None, help="Run directory name used to extract sample frames.")
    parser.add_argument("--source-video", type=Path, default=None, help="Fallback video for frame extraction.")
    parser.add_argument("--max-frames", type=int, default=10)
    return parser


def load_summary(csv_path: Path) -> dict[str, Any]:
    summary_path = csv_path.with_name(csv_path.name.replace("_detections.csv", "_summary.json"))
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    return {}


def find_detection_csvs(input_root: Path) -> list[Path]:
    return sorted(input_root.rglob("*_detections.csv"))


def run_name_for_csv(csv_path: Path, input_root: Path) -> str:
    try:
        relative = csv_path.relative_to(input_root)
    except ValueError:
        return csv_path.parent.name
    if len(relative.parts) > 1:
        return relative.parts[0]
    return csv_path.stem.replace("_detections", "")


def load_counts_and_stats(csv_paths: list[Path], input_root: Path):
    counts_by_run: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    frames_by_run: dict[str, int] = {}
    stats: dict[tuple[str, int, str], list[float]] = defaultdict(lambda: [0.0, 0.0])
    summaries: dict[str, dict[str, Any]] = {}

    for csv_path in csv_paths:
        run_name = run_name_for_csv(csv_path, input_root)
        summary = load_summary(csv_path)
        summaries[run_name] = summary
        frames_by_run[run_name] = int(summary.get("frames_processed") or 0)

        with csv_path.open(newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                frame_id = int(float(row["frame_id"]))
                class_id = int(float(row["class_id"]))
                class_name = row["class_name"]
                confidence = float(row["confidence"])
                counts_by_run[run_name][frame_id] += 1
                key = (run_name, class_id, class_name)
                stats[key][0] += 1.0
                stats[key][1] += confidence
                frames_by_run[run_name] = max(frames_by_run[run_name], frame_id)

    completed_counts: dict[str, dict[int, int]] = {}
    for run_name, counts in counts_by_run.items():
        frame_total = frames_by_run.get(run_name, 0)
        completed_counts[run_name] = {frame_id: counts.get(frame_id, 0) for frame_id in range(1, frame_total + 1)}

    return completed_counts, stats, summaries


def plot_detection_counts(counts_by_run: dict[str, dict[int, int]], output_path: Path) -> Path:
    figure, axis = plt.subplots(figsize=(12, 6))
    for run_name, counts in sorted(counts_by_run.items()):
        frame_ids = sorted(counts)
        values = [counts[frame_id] for frame_id in frame_ids]
        axis.plot(frame_ids, values, linewidth=1.6, label=run_name)
    axis.set_title("Pure Detector Detections Per Frame")
    axis.set_xlabel("frame_id")
    axis.set_ylabel("num_detections")
    axis.grid(True, linestyle="--", alpha=0.35)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    return output_path


def write_class_stats(stats: dict[tuple[str, int, str], list[float]], output_path: Path) -> Path:
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["run_name", "class_id", "class_name", "num_detections", "avg_confidence"],
        )
        writer.writeheader()
        for (run_name, class_id, class_name), (count, confidence_sum) in sorted(stats.items()):
            writer.writerow(
                {
                    "run_name": run_name,
                    "class_id": class_id,
                    "class_name": class_name,
                    "num_detections": int(count),
                    "avg_confidence": confidence_sum / count if count else 0.0,
                }
            )
    return output_path


def choose_reference_run(
    counts_by_run: dict[str, dict[int, int]],
    requested_run: str | None,
) -> str:
    if requested_run and requested_run in counts_by_run:
        return requested_run
    for run_name in counts_by_run:
        lowered = run_name.lower()
        if "conf020" in lowered or "conf0p20" in lowered or lowered.startswith("b"):
            return run_name
    return sorted(counts_by_run)[0]


def select_representative_frames(counts: dict[int, int], max_frames: int) -> list[tuple[str, int, int]]:
    if not counts:
        return []

    ordered = sorted(counts.items())
    values = sorted(count for _, count in ordered)
    median_value = values[len(values) // 2]

    low = sorted(ordered, key=lambda item: (item[1], item[0]))[:3]
    high = sorted(ordered, key=lambda item: (-item[1], item[0]))[:4]
    normal = sorted(ordered, key=lambda item: (abs(item[1] - median_value), item[0]))[:3]

    selected: list[tuple[str, int, int]] = []
    seen: set[int] = set()
    for tag, group in (
        ("low_count_possible_miss", low),
        ("normal_count", normal),
        ("high_count_possible_fragment", high),
    ):
        for frame_id, count in group:
            if frame_id in seen:
                continue
            selected.append((tag, frame_id, count))
            seen.add(frame_id)
            if len(selected) >= max_frames:
                return selected
    return selected


def video_for_run(input_root: Path, run_name: str, summaries: dict[str, dict[str, Any]], source_video: Path | None) -> Path | None:
    output_media = summaries.get(run_name, {}).get("output_media")
    if output_media:
        output_path = Path(output_media)
        if output_path.exists():
            return output_path

    run_dir = input_root / run_name
    candidates = sorted(run_dir.glob("*_predict.mp4"))
    if candidates:
        return candidates[0]

    return source_video if source_video and source_video.exists() else None


def extract_frame(video_path: Path, frame_id: int):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_id - 1))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Failed to read frame {frame_id} from {video_path}")
    return frame


def write_representative_frames(
    input_root: Path,
    output_dir: Path,
    counts_by_run: dict[str, dict[int, int]],
    summaries: dict[str, dict[str, Any]],
    requested_run: str | None,
    source_video: Path | None,
    max_frames: int,
) -> list[Path]:
    frames_dir = output_dir / "predict_debug_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    reference_run = choose_reference_run(counts_by_run, requested_run)
    video_path = video_for_run(input_root, reference_run, summaries, source_video)
    if video_path is None:
        print("No annotated prediction video found; representative frames were not exported.")
        return []

    selected = select_representative_frames(counts_by_run[reference_run], max_frames=max_frames)
    exported: list[Path] = []
    for index, (tag, frame_id, count) in enumerate(selected, start=1):
        frame = extract_frame(video_path, frame_id)
        cv2.putText(
            frame,
            f"{reference_run} | frame {frame_id} | detections {count} | {tag}",
            (24, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        output_path = frames_dir / f"{index:02d}_{tag}_frame_{frame_id:05d}.jpg"
        cv2.imwrite(str(output_path), frame)
        exported.append(output_path)
    return exported


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = find_detection_csvs(args.input_root)
    if not csv_paths:
        raise FileNotFoundError(f"No *_detections.csv files found under {args.input_root}")

    counts_by_run, stats, summaries = load_counts_and_stats(csv_paths, args.input_root)
    count_plot = plot_detection_counts(counts_by_run, args.output_dir / "detection_count_per_frame.png")
    stats_csv = write_class_stats(stats, args.output_dir / "detection_class_stats.csv")
    frames = write_representative_frames(
        input_root=args.input_root,
        output_dir=args.output_dir,
        counts_by_run=counts_by_run,
        summaries=summaries,
        requested_run=args.reference_run,
        source_video=args.source_video,
        max_frames=args.max_frames,
    )

    print(f"Detection count plot: {count_plot}")
    print(f"Class stats CSV: {stats_csv}")
    print("Representative frames:")
    for frame_path in frames:
        print(f"  - {frame_path}")


if __name__ == "__main__":
    main()
