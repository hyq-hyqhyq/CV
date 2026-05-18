from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run YOLO detection only, without tracking.")
    parser.add_argument("--weights", type=Path, required=True, help="YOLO checkpoint path.")
    parser.add_argument("--source", type=Path, required=True, help="Image, image directory, or video path.")
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--classes", nargs="*", type=int, default=None, help="Optional zero-based class ids.")
    parser.add_argument("--output-dir", type=Path, default=Path("task2/outputs/predict_debug"))
    parser.add_argument("--device", default=None, help="Optional CUDA device id, for example 0 or cpu.")
    return parser


def color_for_class(class_id: int) -> tuple[int, int, int]:
    palette = [
        (56, 56, 255),
        (151, 157, 255),
        (31, 112, 255),
        (29, 178, 255),
        (49, 210, 207),
        (10, 249, 72),
        (23, 204, 146),
        (134, 219, 61),
        (52, 147, 26),
        (187, 212, 0),
        (168, 153, 44),
        (255, 194, 0),
        (147, 69, 52),
        (255, 115, 100),
        (236, 24, 0),
        (255, 56, 132),
    ]
    return palette[class_id % len(palette)]


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_SUFFIXES


def is_video(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_SUFFIXES


def iter_images(source: Path) -> list[Path]:
    if source.is_file() and is_image(source):
        return [source]
    if source.is_dir():
        return sorted(path for path in source.rglob("*") if path.is_file() and is_image(path))
    return []


def get_class_name(names: Any, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, f"class_{class_id}"))
    if isinstance(names, list) and 0 <= class_id < len(names):
        return str(names[class_id])
    return f"class_{class_id}"


def predict_one_frame(
    model: YOLO,
    frame,
    conf: float,
    iou: float,
    imgsz: int,
    classes: list[int] | None,
    device: str | None,
):
    results = model.predict(
        source=frame,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        classes=classes,
        device=device,
        verbose=False,
        save=False,
        stream=False,
    )
    return results[0]


def rows_from_result(result, frame_id: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return rows

    xyxy = boxes.xyxy.cpu().numpy()
    confidences = boxes.conf.cpu().numpy()
    class_ids = boxes.cls.cpu().numpy().astype(int)
    for box, confidence, class_id in zip(xyxy, confidences, class_ids, strict=False):
        x1, y1, x2, y2 = [float(value) for value in box]
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        rows.append(
            {
                "frame_id": frame_id,
                "class_id": int(class_id),
                "class_name": get_class_name(result.names, int(class_id)),
                "confidence": float(confidence),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "center_x": x1 + width / 2.0,
                "center_y": y1 + height / 2.0,
                "area": width * height,
            }
        )
    return rows


def draw_detections(frame, rows: list[dict[str, Any]]):
    annotated = frame.copy()
    for row in rows:
        class_id = int(row["class_id"])
        color = color_for_class(class_id)
        x1, y1, x2, y2 = [int(round(row[key])) for key in ("x1", "y1", "x2", "y2")]
        label = f'{row["class_name"]} {row["confidence"]:.2f}'
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        top = max(y1, label_size[1] + baseline + 4)
        cv2.rectangle(
            annotated,
            (x1, top - label_size[1] - baseline - 4),
            (x1 + label_size[0] + 6, top),
            color,
            -1,
        )
        cv2.putText(
            annotated,
            label,
            (x1 + 3, top - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return annotated


def make_output_stem(source: Path, conf: float, iou: float) -> str:
    conf_part = f"conf{conf:.2f}".replace(".", "p")
    iou_part = f"iou{iou:.2f}".replace(".", "p")
    return f"{source.stem}_{conf_part}_{iou_part}"


def write_summary(
    summary_path: Path,
    source: Path,
    weights: Path,
    output_media: Path | None,
    csv_path: Path,
    frame_count: int,
    detection_count: int,
    args: argparse.Namespace,
) -> None:
    payload = {
        "source": str(source),
        "weights": str(weights),
        "output_media": str(output_media) if output_media is not None else None,
        "detections_csv": str(csv_path),
        "frames_processed": frame_count,
        "detections": detection_count,
        "conf": args.conf,
        "iou": args.iou,
        "imgsz": args.imgsz,
        "classes": args.classes,
        "device": args.device,
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def process_video(model: YOLO, args: argparse.Namespace, csv_writer: csv.DictWriter) -> tuple[Path, int, int]:
    capture = cv2.VideoCapture(str(args.source))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {args.source}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_stem = make_output_stem(args.source, args.conf, args.iou)
    output_video = args.output_dir / f"{output_stem}_predict.mp4"
    writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create video writer: {output_video}")

    frame_id = 0
    detection_count = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frame_id += 1
        result = predict_one_frame(
            model=model,
            frame=frame,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            classes=args.classes,
            device=args.device,
        )
        rows = rows_from_result(result, frame_id=frame_id)
        for row in rows:
            csv_writer.writerow(row)
        detection_count += len(rows)
        writer.write(draw_detections(frame, rows))
        if frame_id == 1 or frame_id % 30 == 0:
            print(f"Processed frame {frame_id}, detections={len(rows)}")

    capture.release()
    writer.release()
    return output_video, frame_id, detection_count


def process_images(model: YOLO, args: argparse.Namespace, csv_writer: csv.DictWriter) -> tuple[Path | None, int, int]:
    image_paths = iter_images(args.source)
    if not image_paths:
        raise RuntimeError(f"No image files found: {args.source}")

    output_media: Path | None = None
    detection_count = 0
    for frame_id, image_path in enumerate(image_paths, start=1):
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Skipping unreadable image: {image_path}")
            continue
        result = predict_one_frame(
            model=model,
            frame=image,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            classes=args.classes,
            device=args.device,
        )
        rows = rows_from_result(result, frame_id=frame_id)
        for row in rows:
            row["source_path"] = str(image_path)
            csv_writer.writerow(row)
        detection_count += len(rows)

        relative_name = image_path.name if args.source.is_file() else f"{frame_id:05d}_{image_path.name}"
        output_image = args.output_dir / f"{Path(relative_name).stem}_predict.jpg"
        cv2.imwrite(str(output_image), draw_detections(image, rows))
        output_media = output_image
        print(f"Processed image {image_path}, detections={len(rows)}")

    return output_media, len(image_paths), detection_count


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    classes = args.classes if args.classes else None
    args.classes = classes

    output_stem = make_output_stem(args.source, args.conf, args.iou)
    csv_path = args.output_dir / f"{output_stem}_detections.csv"
    summary_path = args.output_dir / f"{output_stem}_summary.json"
    fieldnames = [
        "frame_id",
        "class_id",
        "class_name",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
        "center_x",
        "center_y",
        "area",
        "source_path",
    ]

    model = YOLO(str(args.weights))
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        if args.source.is_file() and is_video(args.source):
            output_media, frame_count, detection_count = process_video(model, args, writer)
        else:
            output_media, frame_count, detection_count = process_images(model, args, writer)

    write_summary(
        summary_path=summary_path,
        source=args.source,
        weights=args.weights,
        output_media=output_media,
        csv_path=csv_path,
        frame_count=frame_count,
        detection_count=detection_count,
        args=args,
    )
    print(f"Annotated output: {output_media}")
    print(f"Detection CSV: {csv_path}")
    print(f"Summary JSON: {summary_path}")
    print(f"Frames processed: {frame_count}")
    print(f"Total detections: {detection_count}")


if __name__ == "__main__":
    main()
