from __future__ import annotations

import argparse
from pathlib import Path

from task2.lib.visdrone import prepare_split, write_dataset_yaml

SPLIT_TO_FOLDER = {
    "train": "VisDrone2019-DET-train",
    "val": "VisDrone2019-DET-val",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert VisDrone DET labels into YOLO format in-place.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("task2/data/VisDrone2019-DET"),
        help="Root folder containing VisDrone2019-DET-train and VisDrone2019-DET-val.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=sorted(SPLIT_TO_FOLDER),
        default=["train", "val"],
        help="Which splits to convert.",
    )
    parser.add_argument(
        "--write-yaml",
        action="store_true",
        help="Also write a local dataset yaml with absolute path for Ultralytics.",
    )
    parser.add_argument(
        "--yaml-output",
        type=Path,
        default=Path("task2/configs/visdrone_det.local.yaml"),
        help="Where to write the local dataset yaml when --write-yaml is enabled.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dataset_root = args.dataset_root.resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    print(f"Preparing VisDrone labels under {dataset_root}")
    for split_name in args.splits:
        split_root = dataset_root / SPLIT_TO_FOLDER[split_name]
        summary = prepare_split(split_root)
        print(
            f"[{summary['split']}] images={summary['images']} boxes={summary['boxes']} "
            f"labels={summary['labels_dir']}"
        )

    if args.write_yaml:
        yaml_path = write_dataset_yaml(dataset_root, args.yaml_output)
        print(f"Dataset yaml written to {yaml_path}")


if __name__ == "__main__":
    main()
