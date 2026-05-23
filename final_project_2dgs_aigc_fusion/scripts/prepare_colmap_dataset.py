#!/usr/bin/env python3
"""Prepare a lightweight COLMAP image directory and dataset_info.json."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Iterable


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image_dir", required=True, help="Input directory containing images.")
    parser.add_argument("--out_dir", required=True, help="COLMAP dataset output directory.")
    parser.add_argument(
        "--mode",
        choices=("copy", "symlink", "none"),
        default="symlink",
        help="How to place images under out_dir/images.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files or links in out_dir/images.",
    )
    return parser.parse_args()


def iter_images(image_dir: Path) -> Iterable[Path]:
    for path in sorted(image_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def read_resolution(image_path: Path) -> list[int] | None:
    try:
        from PIL import Image
    except ImportError:
        return None

    try:
        with Image.open(image_path) as image:
            width, height = image.size
        return [width, height]
    except Exception:
        return None


def place_image(src: Path, dst: Path, mode: str, overwrite: bool) -> None:
    if dst.exists() or dst.is_symlink():
        if overwrite:
            dst.unlink()
        else:
            return

    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        rel_src = os.path.relpath(src.resolve(), start=dst.parent.resolve())
        dst.symlink_to(rel_src)


def main() -> None:
    args = parse_args()
    image_dir = Path(args.image_dir)
    out_dir = Path(args.out_dir)
    images_out = out_dir / "images"

    if not image_dir.is_dir():
        raise SystemExit(f"Error: image directory not found: {image_dir}")

    images = list(iter_images(image_dir))
    if not images:
        raise SystemExit(
            f"Error: no images found in {image_dir}. Supported extensions: "
            f"{', '.join(sorted(IMAGE_EXTENSIONS))}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    if args.mode != "none":
        images_out.mkdir(parents=True, exist_ok=True)
        for src in images:
            place_image(src, images_out / src.name, args.mode, args.overwrite)

    first_resolution = read_resolution(images[0])
    info = {
        "image_count": len(images),
        "first_image": str(images[0]),
        "first_image_resolution": first_resolution,
        "input_image_dir": str(image_dir),
        "output_dir": str(out_dir),
        "colmap_image_dir": str(images_out if args.mode != "none" else image_dir),
        "mode": args.mode,
        "extensions": sorted({path.suffix.lower() for path in images}),
    }

    info_path = out_dir / "dataset_info.json"
    info_path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Images found: {len(images)}")
    print(f"Input directory: {image_dir}")
    print(f"COLMAP image directory: {info['colmap_image_dir']}")
    print(f"Dataset info written to: {info_path}")


if __name__ == "__main__":
    main()

