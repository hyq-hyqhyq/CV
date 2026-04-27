from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pet_cls.compat import patch_broken_dill

patch_broken_dill()

import torch
from pet_cls.config import DEFAULT_CONFIG


class RandomPetDataset(torch.utils.data.Dataset):
    def __init__(self, length: int = 12, num_classes: int = 37, image_size: int = 224):
        self.length = length
        self.num_classes = num_classes
        self.image_size = image_size

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        generator = torch.Generator().manual_seed(idx)
        image = torch.rand((3, self.image_size, self.image_size), generator=generator)
        label = idx % self.num_classes
        return image, label


def build_config(model_name: str) -> dict:
    config = {
        "experiment": dict(DEFAULT_CONFIG["experiment"]),
        "dataset": dict(DEFAULT_CONFIG["dataset"]),
        "model": dict(DEFAULT_CONFIG["model"]),
        "training": dict(DEFAULT_CONFIG["training"]),
        "logging": dict(DEFAULT_CONFIG["logging"]),
        "evaluation": dict(DEFAULT_CONFIG["evaluation"]),
    }
    config["model"]["name"] = model_name
    config["model"]["pretrained"] = False
    config["training"]["epochs"] = 1
    config["training"]["amp"] = False
    config["dataset"]["batch_size"] = 2
    config["dataset"]["num_workers"] = 0
    return config


def run_smoke(model_name: str) -> None:
    try:
        from torch import nn
        from torch.utils.data import DataLoader

        from pet_cls.engine import evaluate, train_one_epoch
        from pet_cls.models import build_model
        from pet_cls.optim import build_optimizer, build_scheduler
    except ModuleNotFoundError as exc:
        print(f"[SKIP] smoke_test blocked by missing dependency: {exc}")
        return

    config = build_config(model_name)
    model, _ = build_model(config, num_classes=37)
    device = torch.device("cpu")
    model = model.to(device)

    dataset = RandomPetDataset()
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    train_metrics = train_one_epoch(
        model=model,
        loader=loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epoch=1,
        amp_enabled=False,
        log_interval=10,
    )
    eval_metrics = evaluate(model, loader, criterion, device=device, split_name="smoke")

    if scheduler is not None:
        scheduler.step()

    assert 0.0 <= train_metrics["acc"] <= 1.0
    assert 0.0 <= eval_metrics["acc"] <= 1.0
    assert train_metrics["loss"] >= 0.0
    assert eval_metrics["loss"] >= 0.0
    print(f"[OK] {model_name}: train_acc={train_metrics['acc']:.4f}, eval_acc={eval_metrics['acc']:.4f}")


def main() -> None:
    for model_name in ("resnet18", "se_resnet18"):
        run_smoke(model_name)


if __name__ == "__main__":
    main()
