from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import cv2

from task2.lib.visdrone import CLASS_NAMES, VALID_CATEGORY_IDS, find_image_for_stem


SPLIT_DIRS = {
    "train": "VisDrone2019-DET-train",
    "val": "VisDrone2019-DET-val",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize original VisDrone annotations and converted YOLO labels.")
    parser.add_argument("--dataset-root", type=Path, default=Path("task2/data/VisDrone2019-DET"))
    parser.add_argument("--split", choices=sorted(SPLIT_DIRS), required=True)
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def color_for_category(category_id: int) -> tuple[int, int, int]:
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
    ]
    return palette[(category_id - 1) % len(palette)]


def put_label(image, text: str, x: int, y: int, color: tuple[int, int, int], scale: float = 0.48) -> None:
    label_size, baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    top = max(y, label_size[1] + baseline + 4)
    cv2.rectangle(
        image,
        (x, top - label_size[1] - baseline - 4),
        (x + label_size[0] + 6, top),
        color,
        -1,
    )
    cv2.putText(
        image,
        text,
        (x + 3, top - baseline - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def parse_annotation_file(annotation_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_index, row in enumerate(annotation_path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = [part.strip() for part in row.split(",")]
        if len(parts) < 8:
            rows.append({"line_index": line_index, "invalid": True, "reason": "too_few_columns"})
            continue
        try:
            x, y, w, h, score, category, truncation, occlusion = [float(value) for value in parts[:8]]
        except ValueError:
            rows.append({"line_index": line_index, "invalid": True, "reason": "non_numeric"})
            continue
        category_id = int(category)
        ignored = score <= 0 or category_id not in VALID_CATEGORY_IDS or w <= 0 or h <= 0
        rows.append(
            {
                "line_index": line_index,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "score": score,
                "category_id": category_id,
                "truncation": int(truncation),
                "occlusion": int(occlusion),
                "ignored": ignored,
                "invalid": False,
            }
        )
    return rows


def parse_yolo_labels(label_path: Path, width: int, height: int) -> tuple[list[dict[str, Any]], list[str]]:
    boxes: list[dict[str, Any]] = []
    issues: list[str] = []
    if not label_path.exists():
        return boxes, [f"missing_label:{label_path}"]

    for line_index, row in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = row.split()
        if len(parts) != 5:
            issues.append(f"{label_path.name}:{line_index}:bad_column_count")
            continue
        try:
            class_id = int(float(parts[0]))
            center_x, center_y, norm_w, norm_h = [float(value) for value in parts[1:]]
        except ValueError:
            issues.append(f"{label_path.name}:{line_index}:non_numeric")
            continue
        if class_id < 0 or class_id >= len(CLASS_NAMES):
            issues.append(f"{label_path.name}:{line_index}:bad_class_id:{class_id}")
        if not (0.0 <= center_x <= 1.0 and 0.0 <= center_y <= 1.0 and 0.0 < norm_w <= 1.0 and 0.0 < norm_h <= 1.0):
            issues.append(f"{label_path.name}:{line_index}:bad_normalized_box")

        box_w = norm_w * width
        box_h = norm_h * height
        x1 = center_x * width - box_w / 2.0
        y1 = center_y * height - box_h / 2.0
        boxes.append(
            {
                "line_index": line_index,
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id] if 0 <= class_id < len(CLASS_NAMES) else f"class_{class_id}",
                "x1": x1,
                "y1": y1,
                "x2": x1 + box_w,
                "y2": y1 + box_h,
            }
        )
    return boxes, issues


def clip_xyxy(x: float, y: float, w: float, h: float, width: int, height: int) -> tuple[int, int, int, int] | None:
    x1 = max(0, min(int(round(x)), width - 1))
    y1 = max(0, min(int(round(y)), height - 1))
    x2 = max(0, min(int(round(x + w)), width - 1))
    y2 = max(0, min(int(round(y + h)), height - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def draw_debug_image(
    image,
    annotation_rows: list[dict[str, Any]],
    yolo_boxes: list[dict[str, Any]],
) -> tuple[int, int]:
    height, width = image.shape[:2]
    valid_count = 0
    ignored_count = 0

    for row in annotation_rows:
        if row.get("invalid"):
            continue
        clipped = clip_xyxy(row["x"], row["y"], row["w"], row["h"], width, height)
        if clipped is None:
            continue
        x1, y1, x2, y2 = clipped
        if row["ignored"]:
            ignored_count += 1
            color = (130, 130, 130)
            label = f'ignored c{row["category_id"]} s{row["score"]:.0f}'
            thickness = 1
        else:
            valid_count += 1
            color = color_for_category(row["category_id"])
            label = f'{CLASS_NAMES[row["category_id"] - 1]} occ{row["occlusion"]}'
            thickness = 2
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        put_label(image, label, x1, y1, color)

    for box in yolo_boxes:
        x1 = int(round(box["x1"]))
        y1 = int(round(box["y1"]))
        x2 = int(round(box["x2"]))
        y2 = int(round(box["y2"]))
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 0), 1)

    cv2.rectangle(image, (0, 0), (720, 34), (0, 0, 0), -1)
    cv2.putText(
        image,
        f"original valid={valid_count}, original ignored={ignored_count}, cyan=converted YOLO labels",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return valid_count, ignored_count


def main() -> None:
    args = build_parser().parse_args()
    split_root = args.dataset_root / SPLIT_DIRS[args.split]
    images_dir = split_root / "images"
    annotations_dir = split_root / "annotations"
    labels_dir = split_root / "labels"
    output_dir = args.output_dir or Path(f"task2/report_assets/label_debug_{args.split}")
    output_dir.mkdir(parents=True, exist_ok=True)

    annotation_files = sorted(annotations_dir.glob("*.txt"))
    if not annotation_files:
        raise FileNotFoundError(f"No annotation files found under {annotations_dir}")

    rng = random.Random(args.seed)
    sample_files = rng.sample(annotation_files, k=min(args.num_samples, len(annotation_files)))

    summary_rows: list[dict[str, Any]] = []
    total_valid = 0
    total_ignored = 0
    total_yolo = 0
    yolo_issues: list[str] = []

    for annotation_path in sample_files:
        image_path = find_image_for_stem(images_dir, annotation_path.stem)
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        height, width = image.shape[:2]

        annotation_rows = parse_annotation_file(annotation_path)
        yolo_boxes, issues = parse_yolo_labels(labels_dir / annotation_path.name, width, height)
        valid_count, ignored_count = draw_debug_image(image, annotation_rows, yolo_boxes)
        if len(yolo_boxes) != valid_count:
            issues.append(f"count_mismatch:original_valid={valid_count}:converted_yolo={len(yolo_boxes)}")

        output_path = output_dir / f"{annotation_path.stem}_label_debug.jpg"
        cv2.imwrite(str(output_path), image)

        total_valid += valid_count
        total_ignored += ignored_count
        total_yolo += len(yolo_boxes)
        yolo_issues.extend(issues)
        summary_rows.append(
            {
                "image": str(image_path),
                "annotation": str(annotation_path),
                "debug_image": str(output_path),
                "original_valid_boxes": valid_count,
                "original_ignored_rows": ignored_count,
                "converted_yolo_boxes": len(yolo_boxes),
                "yolo_issues": ";".join(issues),
            }
        )
        print(output_path)

    summary_csv = output_dir / f"{args.split}_label_debug_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "image",
                "annotation",
                "debug_image",
                "original_valid_boxes",
                "original_ignored_rows",
                "converted_yolo_boxes",
                "yolo_issues",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    summary_json = output_dir / f"{args.split}_label_debug_summary.json"
    summary_json.write_text(
        json.dumps(
            {
                "split": args.split,
                "samples": len(sample_files),
                "original_valid_boxes": total_valid,
                "original_ignored_rows": total_ignored,
                "converted_yolo_boxes": total_yolo,
                "yolo_issue_count": len(yolo_issues),
                "first_yolo_issues": yolo_issues[:20],
                "note": "Gray boxes are original ignored rows; cyan boxes are converted YOLO labels.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Summary CSV: {summary_csv}")
    print(f"Summary JSON: {summary_json}")


if __name__ == "__main__":
    main()
