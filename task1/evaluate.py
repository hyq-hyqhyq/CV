from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pet_cls.compat import patch_broken_dill

patch_broken_dill()

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved checkpoint.")
    parser.add_argument("--config", type=str, required=True, help="Path to a YAML config file.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint to evaluate.")
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["val", "test"],
        help="Dataset split used for evaluation.",
    )
    parser.add_argument(
        "--set",
        nargs="*",
        default=None,
        help="Override config entries with key=value pairs, e.g. dataset.batch_size=64.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from torch import nn
    from pet_cls.config import apply_overrides, load_config

    try:
        from pet_cls.data import build_dataloaders
        from pet_cls.engine import evaluate
        from pet_cls.models import build_model
        from pet_cls.utils import load_checkpoint, resolve_device
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Missing runtime dependency: {exc}. Please install packages from requirements.txt in a clean environment."
        ) from exc

    config = apply_overrides(load_config(args.config), args.set)
    loaders, data_meta = build_dataloaders(config)
    device = resolve_device(config["training"]["device"])

    model, _ = build_model(config, num_classes=data_meta["num_classes"])
    checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    metrics = evaluate(model, loaders[args.split], criterion, device=device, split_name=args.split)

    print("=" * 72)
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Split      : {args.split}")
    print(f"Loss       : {metrics['loss']:.4f}")
    print(f"Accuracy   : {metrics['acc']:.4f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
