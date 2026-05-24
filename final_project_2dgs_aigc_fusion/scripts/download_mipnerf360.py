#!/usr/bin/env python3
"""Download and extract a Mip-NeRF 360 scene for background reconstruction."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


DEFAULT_URL = "https://storage.googleapis.com/gresearch/refraw360/360_v2.zip"
DEFAULT_SCENES = ("counter", "garden")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene",
        choices=DEFAULT_SCENES,
        default="counter",
        help="Mip-NeRF 360 scene to extract. This project expects counter or garden.",
    )
    parser.add_argument(
        "--data_root",
        default=None,
        help="Project data root. Defaults to configs/project_paths.yaml data_root or data.",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Scene output directory. Defaults to <data_root>/mipnerf360/<scene>.",
    )
    parser.add_argument(
        "--zip_path",
        default=None,
        help="Local zip path. Defaults to <data_root>/downloads/360_v2.zip.",
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help="Dataset zip URL. Default is the official Mip-NeRF 360 360_v2.zip URL.",
    )
    parser.add_argument(
        "--config",
        default="configs/project_paths.yaml",
        help="Path config YAML used when --data_root is omitted.",
    )
    parser.add_argument("--force_download", action="store_true", help="Re-download the zip file.")
    parser.add_argument("--force_extract", action="store_true", help="Overwrite an existing scene directory.")
    parser.add_argument(
        "--keep_zip",
        action="store_true",
        help="Keep the downloaded zip after extraction. By default it is removed to save disk space.",
    )
    return parser.parse_args()


def parse_simple_yaml(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    data: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        clean = line.split("#", 1)[0].strip()
        if not clean or ":" not in clean:
            continue
        key, value = clean.split(":", 1)
        data[key.strip()] = value.strip().strip('"').strip("'")
    return data


def resolve_project_path(project_root: Path, text: str | None) -> Path | None:
    if not text:
        return None
    path = Path(text)
    return path if path.is_absolute() else project_root / path


def progress_hook(start_time: float):
    def hook(block_count: int, block_size: int, total_size: int) -> None:
        downloaded = block_count * block_size
        if total_size > 0:
            percent = min(downloaded / total_size * 100.0, 100.0)
            elapsed = max(time.time() - start_time, 1e-6)
            speed_mb = downloaded / elapsed / (1024 * 1024)
            sys.stderr.write(
                f"\rDownloading: {percent:6.2f}% "
                f"({downloaded / (1024 ** 3):.2f}/{total_size / (1024 ** 3):.2f} GB, "
                f"{speed_mb:.1f} MB/s)"
            )
        else:
            sys.stderr.write(f"\rDownloading: {downloaded / (1024 ** 2):.1f} MB")
        sys.stderr.flush()

    return hook


def download(url: str, zip_path: Path, force: bool) -> None:
    if zip_path.is_file() and not force:
        print(f"Using existing zip: {zip_path}")
        return

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = zip_path.with_suffix(zip_path.suffix + ".part")
    if tmp_path.exists():
        tmp_path.unlink()

    print(f"Downloading Mip-NeRF 360 archive from: {url}")
    print(f"Temporary file: {tmp_path}")
    start_time = time.time()
    try:
        urllib.request.urlretrieve(url, tmp_path, reporthook=progress_hook(start_time))
    except Exception as exc:
        if tmp_path.exists():
            tmp_path.unlink()
        raise SystemExit(
            "Error: failed to download dataset archive. Check network access, disk space, "
            f"and URL: {url}\n{exc}"
        ) from exc
    finally:
        sys.stderr.write("\n")

    tmp_path.replace(zip_path)
    print(f"Downloaded zip: {zip_path}")


def matching_members(zip_file: zipfile.ZipFile, scene: str) -> list[zipfile.ZipInfo]:
    members = []
    for info in zip_file.infolist():
        parts = PurePosixPath(info.filename).parts
        if scene in parts and not info.is_dir():
            members.append(info)
    return members


def target_relative_path(zip_name: str, scene: str) -> Path:
    parts = PurePosixPath(zip_name).parts
    scene_index = parts.index(scene)
    return Path(*parts[scene_index + 1 :])


def extract_scene(zip_path: Path, scene: str, out_dir: Path, force: bool) -> dict[str, Any]:
    if out_dir.exists():
        if not force:
            print(f"Scene directory already exists, skipping extraction: {out_dir}")
            return {
                "scene": scene,
                "out_dir": str(out_dir),
                "extracted_files": None,
                "skipped_existing": True,
            }
        shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        members = matching_members(zf, scene)
        if not members:
            raise SystemExit(
                f"Error: scene '{scene}' was not found inside {zip_path}. "
                "Use --scene counter or --scene garden with the default 360_v2 archive."
            )

        for index, info in enumerate(members, start=1):
            rel_path = target_relative_path(info.filename, scene)
            dst = out_dir / rel_path
            dst.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, dst.open("wb") as f:
                shutil.copyfileobj(src, f)
            if index % 100 == 0 or index == len(members):
                print(f"Extracted {index}/{len(members)} files", end="\r")
        print()

    return {
        "scene": scene,
        "out_dir": str(out_dir),
        "extracted_files": len(members),
        "skipped_existing": False,
    }


def write_metadata(out_dir: Path, metadata: dict[str, Any]) -> None:
    metadata_path = out_dir / "download_info.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Download metadata written to: {metadata_path}")


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    project_root = config_path.resolve().parent.parent if config_path.parent.name == "configs" else Path.cwd()
    config = parse_simple_yaml(config_path)

    data_root = resolve_project_path(project_root, args.data_root or config.get("data_root") or "data")
    assert data_root is not None
    out_dir = resolve_project_path(project_root, args.out_dir) or data_root / "mipnerf360" / args.scene
    zip_path = resolve_project_path(project_root, args.zip_path) or data_root / "downloads" / "360_v2.zip"

    download(args.url, zip_path, args.force_download)
    metadata = extract_scene(zip_path, args.scene, out_dir, args.force_extract)
    metadata.update(
        {
            "source_url": args.url,
            "zip_path": str(zip_path),
            "downloaded_or_checked_at": datetime.now(timezone.utc).isoformat(),
            "official_project_page": "https://jonbarron.info/mipnerf360/",
        }
    )
    write_metadata(out_dir, metadata)

    if not args.keep_zip:
        try:
            zip_path.unlink()
            print(f"Removed zip to save disk space: {zip_path}")
        except FileNotFoundError:
            pass

    print(f"Ready scene directory: {out_dir}")


if __name__ == "__main__":
    main()

