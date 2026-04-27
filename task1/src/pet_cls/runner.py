from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .config import save_config_snapshot
from .data import build_dataloaders
from .engine import evaluate, train_one_epoch
from .logging_utils import ExperimentLogger
from .models import build_model
from .optim import build_optimizer, build_scheduler
from .plot_utils import save_training_curves
from .utils import (
    ensure_dir,
    load_checkpoint,
    resolve_device,
    save_checkpoint,
    save_json,
    set_seed,
    timestamp,
)


def prepare_run_directory(config: dict[str, Any], project_root: str | Path) -> Path:
    project_root = Path(project_root)
    output_root = ensure_dir(project_root / config["experiment"]["output_dir"])
    run_name = config["logging"]["run_name"] or config["experiment"]["name"]
    run_dir = ensure_dir(output_root / f"{timestamp()}_{run_name}")
    return run_dir


def _checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    config: dict[str, Any],
    epoch: int,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "config": config,
        "epoch": epoch,
        "metrics": metrics,
    }


def run_experiment(config: dict[str, Any], project_root: str | Path) -> tuple[dict[str, Any], Path]:
    project_root = Path(project_root)
    config = deepcopy(config)
    for section, key in (("dataset", "root"), ("experiment", "output_dir")):
        value = Path(config[section][key])
        if not value.is_absolute():
            config[section][key] = str((project_root / value).resolve())

    run_dir = prepare_run_directory(config, project_root=project_root)
    save_config_snapshot(config, run_dir / "config_snapshot.yaml")

    set_seed(int(config["experiment"]["seed"]))
    loaders, data_meta = build_dataloaders(config)
    save_json(data_meta, run_dir / "data_split.json")

    device = resolve_device(config["training"]["device"])
    model, model_meta = build_model(config, num_classes=data_meta["num_classes"])
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=float(config["training"]["label_smoothing"]))
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    logger = ExperimentLogger(config, run_dir)

    history: list[dict[str, Any]] = []
    best_val_acc = -1.0
    best_epoch = -1
    best_checkpoint_path = run_dir / "best.pt"
    last_checkpoint_path = run_dir / "last.pt"

    epochs = int(config["training"]["epochs"])
    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=loaders["train"],
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            amp_enabled=bool(config["training"]["amp"]),
            log_interval=int(config["logging"]["log_interval"]),
            grad_clip_norm=float(config["training"]["grad_clip_norm"]),
        )
        val_metrics = evaluate(
            model=model,
            loader=loaders["val"],
            criterion=criterion,
            device=device,
            split_name="val",
        )

        if scheduler is not None:
            scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "lr_backbone": optimizer.param_groups[0]["lr"],
            "lr_head": optimizer.param_groups[-1]["lr"],
        }
        logger.log_metrics(row, step=epoch)
        history.append(row)

        current_payload = _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            epoch=epoch,
            metrics=row,
        )
        save_checkpoint(current_payload, last_checkpoint_path)

        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            best_epoch = epoch
            save_checkpoint(current_payload, best_checkpoint_path)

    best_state = load_checkpoint(best_checkpoint_path, map_location=device)
    model.load_state_dict(best_state["model_state"])
    test_metrics = evaluate(
        model=model,
        loader=loaders["test"],
        criterion=criterion,
        device=device,
        split_name="test",
    )

    save_training_curves(history, run_dir)
    summary = {
        "experiment_name": config["experiment"]["name"],
        "run_dir": str(run_dir),
        "device": str(device),
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "test_acc": test_metrics["acc"],
        "test_loss": test_metrics["loss"],
        "split_sizes": data_meta["split_sizes"],
        "model_meta": model_meta,
    }
    logger.finalize(summary)
    return summary, run_dir
