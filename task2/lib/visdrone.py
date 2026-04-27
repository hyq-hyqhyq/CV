from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import yaml

CLASS_NAMES = [
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
]

VALID_CATEGORY_IDS = set(range(1, 11))
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")


def find_image_for_stem(images_dir: Path, stem: str) -> Path:
    for suffix in IMAGE_SUFFIXES:
        candidate = images_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Image file for '{stem}' not found under {images_dir}")


def normalize_box(
    x: float,
    y: float,
    w: float,
    h: float,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float] | None:
    x1 = max(0.0, min(x, image_width - 1))
    y1 = max(0.0, min(y, image_height - 1))
    x2 = max(0.0, min(x + w, image_width))
    y2 = max(0.0, min(y + h, image_height))
    if x2 <= x1 or y2 <= y1:
        return None

    box_w = x2 - x1
    box_h = y2 - y1
    center_x = x1 + box_w / 2.0
    center_y = y1 + box_h / 2.0
    return (
        center_x / image_width,
        center_y / image_height,
        box_w / image_width,
        box_h / image_height,
    )


def parse_annotation_rows(
    rows: Iterable[str],
    image_width: int,
    image_height: int,
) -> list[str]:
    yolo_rows: list[str] = []
    for row in rows:
        parts = [part.strip() for part in row.split(",")]
        if len(parts) < 8:
            continue

        try:
            x, y, w, h, score, category, *_ = [float(value) for value in parts[:8]]
        except ValueError:
            continue

        category_id = int(category)
        if category_id not in VALID_CATEGORY_IDS or score <= 0 or w <= 0 or h <= 0:
            continue

        normalized = normalize_box(x, y, w, h, image_width, image_height)
        if normalized is None:
            continue

        x_center, y_center, norm_w, norm_h = normalized
        yolo_rows.append(
            f"{category_id - 1} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}"
        )
    return yolo_rows


def prepare_split(split_root: Path) -> dict[str, int | str]:
    images_dir = split_root / "images"
    annotations_dir = split_root / "annotations"
    labels_dir = split_root / "labels"

    if not images_dir.exists():
        raise FileNotFoundError(f"Missing images directory: {images_dir}")
    if not annotations_dir.exists():
        raise FileNotFoundError(f"Missing annotations directory: {annotations_dir}")

    labels_dir.mkdir(parents=True, exist_ok=True)
    annotation_files = sorted(annotations_dir.glob("*.txt"))

    image_count = 0
    kept_boxes = 0
    for annotation_file in annotation_files:
        image_path = find_image_for_stem(images_dir, annotation_file.stem)
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Failed to read image: {image_path}")

        height, width = image.shape[:2]
        yolo_rows = parse_annotation_rows(
            annotation_file.read_text(encoding="utf-8").splitlines(),
            image_width=width,
            image_height=height,
        )
        label_path = labels_dir / annotation_file.name
        label_path.write_text("\n".join(yolo_rows), encoding="utf-8")

        image_count += 1
        kept_boxes += len(yolo_rows)

    return {
        "split": split_root.name,
        "images": image_count,
        "boxes": kept_boxes,
        "labels_dir": str(labels_dir),
    }


def write_dataset_yaml(dataset_root: Path, output_yaml: Path) -> Path:
    payload = {
        "path": str(dataset_root.resolve()),
        "train": "VisDrone2019-DET-train/images",
        "val": "VisDrone2019-DET-val/images",
        "test": "VisDrone2019-DET-test-dev/images",
        "names": {index: name for index, name in enumerate(CLASS_NAMES)},
    }
    output_yaml.parent.mkdir(parents=True, exist_ok=True)
    output_yaml.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=False), encoding="utf-8")
    return output_yaml
