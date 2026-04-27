from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


class ExperimentLogger:
    def __init__(self, config: dict[str, Any], run_dir: str | Path):
        self.config = config
        self.run_dir = Path(run_dir)
        self.csv_file = self.run_dir / "metrics.csv"
        self.jsonl_file = self.run_dir / "metrics.jsonl"
        self._csv_handle = self.csv_file.open("w", newline="", encoding="utf-8")
        self._jsonl_handle = self.jsonl_file.open("w", encoding="utf-8")
        self._csv_writer: csv.DictWriter[str] | None = None
        self.backend_name = str(config["logging"]["backend"]).lower()
        self.backend = self._build_backend()

    def _build_backend(self):
        if self.backend_name == "none":
            return None

        if self.backend_name == "wandb":
            try:
                import wandb
            except ImportError:
                print("wandb is not installed; falling back to local logging only.")
                return None

            wandb.init(
                project=self.config["logging"]["project"],
                entity=self.config["logging"]["entity"],
                name=self.config["logging"]["run_name"] or self.config["experiment"]["name"],
                config=self.config,
                dir=str(self.run_dir),
            )
            return wandb

        if self.backend_name == "swanlab":
            try:
                import swanlab
            except ImportError:
                print("swanlab is not installed; falling back to local logging only.")
                return None

            swanlab.init(
                project=self.config["logging"]["project"],
                experiment_name=self.config["logging"]["run_name"] or self.config["experiment"]["name"],
                config=self.config,
            )
            return swanlab

        raise ValueError(f"Unsupported logging backend: {self.config['logging']['backend']}")

    def log_metrics(self, metrics: dict[str, Any], step: int) -> None:
        if self._csv_writer is None:
            self._csv_writer = csv.DictWriter(self._csv_handle, fieldnames=list(metrics.keys()))
            self._csv_writer.writeheader()

        self._csv_writer.writerow(metrics)
        self._csv_handle.flush()
        self._jsonl_handle.write(json.dumps(metrics, ensure_ascii=False) + "\n")
        self._jsonl_handle.flush()

        if self.backend is not None:
            if self.backend_name == "wandb":
                self.backend.log(metrics, step=step)
            elif self.backend_name == "swanlab":
                self.backend.log(metrics, step=step)

    def finalize(self, summary: dict[str, Any]) -> None:
        summary_path = self.run_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

        if self.backend is not None:
            if self.backend_name == "wandb":
                if self.backend.run is not None:
                    self.backend.run.summary.update(summary)
                self.backend.finish()
            elif self.backend_name == "swanlab":
                self.backend.log(summary)
                self.backend.finish()

        self._csv_handle.close()
        self._jsonl_handle.close()
