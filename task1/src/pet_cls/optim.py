from __future__ import annotations

from typing import Any

import torch
from torch import nn


def _parameter_groups(
    model: nn.Module,
    backbone_lr: float,
    head_lr: float,
) -> list[dict[str, Any]]:
    backbone_params = []
    head_params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        # torchvision resnet: "fc.*"
        # timm models commonly use "head.*" or "classifier.*"
        if name.startswith(("fc.", "head.", "classifier.")):
            head_params.append(parameter)
        else:
            backbone_params.append(parameter)

    if not head_params:
        return [{"params": [p for p in model.parameters() if p.requires_grad], "lr": head_lr}]

    groups = []
    if backbone_params:
        groups.append({"params": backbone_params, "lr": backbone_lr})
    groups.append({"params": head_params, "lr": head_lr})
    return groups


def build_optimizer(model: nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    train_cfg = config["training"]
    optimizer_name = str(train_cfg["optimizer"]).lower()
    backbone_lr = float(train_cfg["backbone_lr"])
    head_lr = float(train_cfg["head_lr"])
    weight_decay = float(train_cfg["weight_decay"])
    groups = _parameter_groups(model, backbone_lr=backbone_lr, head_lr=head_lr)

    if optimizer_name == "sgd":
        return torch.optim.SGD(
            groups,
            momentum=float(train_cfg["momentum"]),
            weight_decay=weight_decay,
            nesterov=True,
        )
    if optimizer_name == "adam":
        return torch.optim.Adam(groups, weight_decay=weight_decay)
    if optimizer_name == "adamw":
        return torch.optim.AdamW(groups, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {train_cfg['optimizer']}")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
) -> torch.optim.lr_scheduler.LRScheduler | None:
    train_cfg = config["training"]
    scheduler_name = str(train_cfg["scheduler"]).lower()
    epochs = int(train_cfg["epochs"])
    min_lr = float(train_cfg["min_lr"])

    if scheduler_name == "none":
        return None
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
    if scheduler_name == "step":
        step_size = max(epochs // 3, 1)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    raise ValueError(f"Unsupported scheduler: {train_cfg['scheduler']}")
