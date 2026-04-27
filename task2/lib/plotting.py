from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

LOSS_GROUPS = {
    "loss_curves.png": {
        "title": "Training and Validation Loss",
        "columns": [
            "train/box_loss",
            "train/cls_loss",
            "train/dfl_loss",
            "val/box_loss",
            "val/cls_loss",
            "val/dfl_loss",
        ],
        "ylabel": "Loss",
    },
    "metric_curves.png": {
        "title": "Validation Metrics",
        "columns": [
            "metrics/precision(B)",
            "metrics/recall(B)",
            "metrics/mAP50(B)",
            "metrics/mAP50-95(B)",
        ],
        "ylabel": "Score",
    },
    "lr_curves.png": {
        "title": "Learning Rate Schedule",
        "columns": ["lr/pg0", "lr/pg1", "lr/pg2"],
        "ylabel": "Learning Rate",
    },
}


def load_results_csv(results_csv: Path) -> pd.DataFrame:
    dataframe = pd.read_csv(results_csv)
    dataframe.columns = [column.strip() for column in dataframe.columns]
    return dataframe


def _plot_curves(
    dataframe: pd.DataFrame,
    columns: list[str],
    title: str,
    ylabel: str,
    output_path: Path,
) -> Path | None:
    valid_columns = [column for column in columns if column in dataframe.columns]
    if not valid_columns:
        return None

    figure, axis = plt.subplots(figsize=(10, 6))
    epoch_values = (
        dataframe["epoch"] if "epoch" in dataframe.columns else pd.Series(range(1, len(dataframe) + 1))
    )

    for column in valid_columns:
        axis.plot(epoch_values, dataframe[column], linewidth=2, label=column)

    axis.set_title(title)
    axis.set_xlabel("Epoch")
    axis.set_ylabel(ylabel)
    axis.grid(True, linestyle="--", alpha=0.35)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    return output_path


def generate_training_figures(results_csv: Path, output_dir: Path) -> list[Path]:
    dataframe = load_results_csv(results_csv)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_files: list[Path] = []
    for filename, config in LOSS_GROUPS.items():
        generated_path = _plot_curves(
            dataframe=dataframe,
            columns=config["columns"],
            title=config["title"],
            ylabel=config["ylabel"],
            output_path=output_dir / filename,
        )
        if generated_path is not None:
            generated_files.append(generated_path)
    return generated_files


def sync_results_csv_to_wandb(
    results_csv: Path,
    project: str,
    run_name: str,
    entity: str | None = None,
    config: dict[str, Any] | None = None,
    tags: list[str] | None = None,
) -> None:
    import wandb

    dataframe = load_results_csv(results_csv)
    run = wandb.init(
        project=project,
        name=run_name,
        entity=entity,
        config=config or {},
        tags=tags or [],
    )

    for row_index, (_, row) in enumerate(dataframe.iterrows(), start=1):
        step = int(row["epoch"]) if "epoch" in dataframe.columns else row_index
        payload: dict[str, float] = {}
        for column, value in row.items():
            if column == "epoch" or pd.isna(value):
                continue
            if isinstance(value, (int, float)):
                payload[column] = float(value)
        if payload:
            wandb.log(payload, step=step)

    if "metrics/mAP50(B)" in dataframe.columns:
        run.summary["best_mAP50"] = float(dataframe["metrics/mAP50(B)"].max())
    if "metrics/mAP50-95(B)" in dataframe.columns:
        run.summary["best_mAP50-95"] = float(dataframe["metrics/mAP50-95(B)"].max())
    run.finish()
