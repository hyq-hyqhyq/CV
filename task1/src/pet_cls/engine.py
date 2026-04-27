from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def _autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    amp_enabled: bool,
    log_interval: int,
    grad_clip_norm: float = 0.0,
) -> dict[str, float]:
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled and device.type == "cuda")

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    progress = tqdm(loader, desc=f"Epoch {epoch:03d} train", leave=False)
    for step, (images, targets) in enumerate(progress, start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device, amp_enabled):
            logits = model(images)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        if grad_clip_norm and grad_clip_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * targets.size(0)
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_examples += targets.size(0)

        if step % max(log_interval, 1) == 0 or step == len(loader):
            progress.set_postfix(
                loss=f"{total_loss / max(total_examples, 1):.4f}",
                acc=f"{total_correct / max(total_examples, 1):.4f}",
            )

    return {
        "loss": total_loss / max(total_examples, 1),
        "acc": total_correct / max(total_examples, 1),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    split_name: str,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    progress = tqdm(loader, desc=f"{split_name:>5} eval", leave=False)
    for images, targets in progress:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets)

        total_loss += loss.item() * targets.size(0)
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_examples += targets.size(0)

        progress.set_postfix(
            loss=f"{total_loss / max(total_examples, 1):.4f}",
            acc=f"{total_correct / max(total_examples, 1):.4f}",
        )

    return {
        "loss": total_loss / max(total_examples, 1),
        "acc": total_correct / max(total_examples, 1),
    }
