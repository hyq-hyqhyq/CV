#!/usr/bin/env python3
"""Collect exported training logs or create a manual log template for reports."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


TEMPLATE_FIELDS = ["step", "loss", "psnr", "ssim", "lpips", "method", "asset_name"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log_dir",
        action="append",
        default=[],
        help="Directory to search recursively for exported CSV logs. Can be passed multiple times.",
    )
    parser.add_argument(
        "--out_dir",
        default="outputs/evaluation/logs",
        help="Directory for collected logs or manual template.",
    )
    return parser.parse_args()


def copy_csv_logs(log_dirs: list[str], out_dir: Path) -> int:
    copied = 0
    used_names: set[str] = set()
    for text in log_dirs:
        root = Path(text)
        if not root.exists():
            print(f"Warning: log directory not found, skipping: {root}")
            continue
        for csv_path in sorted(root.rglob("*.csv")):
            candidate_name = f"{root.name}_{csv_path.name}"
            if candidate_name in used_names:
                candidate_name = f"{root.name}_{copied:03d}_{csv_path.name}"
            used_names.add(candidate_name)
            dst = out_dir / candidate_name
            shutil.copy2(csv_path, dst)
            copied += 1
            print(f"Copied log: {csv_path} -> {dst}")
    return copied


def write_template(out_dir: Path) -> Path:
    template_path = out_dir / "manual_log_template.csv"
    with template_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TEMPLATE_FIELDS)
        writer.writeheader()
        for method, asset_name in [
            ("2DGS", "object_a"),
            ("2DGS", "background"),
            ("threestudio_SDS", "object_b"),
            ("Magic123", "object_c"),
        ]:
            writer.writerow(
                {
                    "step": "",
                    "loss": "",
                    "psnr": "",
                    "ssim": "",
                    "lpips": "",
                    "method": method,
                    "asset_name": asset_name,
                }
            )
    return template_path


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    copied = copy_csv_logs(args.log_dir, out_dir)
    if copied == 0:
        template_path = write_template(out_dir)
        print(f"No CSV logs found. Manual log template written to: {template_path}")
    else:
        print(f"Collected {copied} CSV log file(s) into: {out_dir}")


if __name__ == "__main__":
    main()

