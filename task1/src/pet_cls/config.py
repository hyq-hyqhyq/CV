from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "experiment": {
        "name": "resnet18_pretrained_baseline",
        "output_dir": "outputs",
        "seed": 42,
    },
    "dataset": {
        "root": "data",
        "download": True,
        "val_ratio": 0.15,
        "image_size": 224,
        "batch_size": 32,
        "num_workers": 2,
        "pin_memory": True,
        "train_resize_scale": [0.7, 1.0],
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    },
    "model": {
        "name": "resnet18",
        "num_classes": 37,
        "pretrained": True,
        "dropout": 0.0,
    },
    "training": {
        "device": "auto",
        "epochs": 20,
        "optimizer": "adamw",
        "backbone_lr": 1e-4,
        "head_lr": 1e-3,
        "weight_decay": 1e-4,
        "momentum": 0.9,
        "scheduler": "cosine",
        "min_lr": 1e-6,
        "label_smoothing": 0.0,
        "amp": True,
        "grad_clip_norm": 0.0,
    },
    "logging": {
        "backend": "none",
        "project": "oxfordiiit-pet",
        "entity": None,
        "run_name": None,
        "log_interval": 20,
    },
    "evaluation": {
        "save_best_only": True,
    },
}


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def _parse_value(raw: str) -> Any:
    return yaml.safe_load(raw)


def _set_by_path(config: dict[str, Any], path: str, value: Any) -> None:
    cursor = config
    keys = path.split(".")
    for key in keys[:-1]:
        if key not in cursor or not isinstance(cursor[key], dict):
            cursor[key] = {}
        cursor = cursor[key]
    cursor[keys[-1]] = value


def load_config(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    user_config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    config = _deep_update(DEFAULT_CONFIG, user_config)
    config["_meta"] = {"config_path": str(config_path.resolve())}
    return config


def apply_overrides(config: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    updated = deepcopy(config)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item}")
        key, raw_value = item.split("=", 1)
        _set_by_path(updated, key, _parse_value(raw_value))
    return updated


def save_config_snapshot(config: dict[str, Any], destination: str | Path) -> None:
    destination = Path(destination)
    destination.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
