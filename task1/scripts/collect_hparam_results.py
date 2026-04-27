from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Row:
    output_dir: str
    experiment_name: str
    model_name: str | None
    pretrained: bool | None
    backbone_lr: float | None
    head_lr: float | None
    epochs: int | None
    weight_decay: float | None
    dropout: float | None
    label_smoothing: float | None
    best_epoch: int | None
    best_val_acc: float | None
    test_acc: float | None


def _safe_get(d: dict[str, Any], *keys: str) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _as_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _as_int(x: Any) -> int | None:
    if x is None:
        return None
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def _as_bool(x: Any) -> bool | None:
    if x is None:
        return None
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        if x.lower() in {"true", "1", "yes", "y"}:
            return True
        if x.lower() in {"false", "0", "no", "n"}:
            return False
    return None


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _row_from_run_dir(run_dir: Path) -> Row | None:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return None

    summary = _read_json(summary_path)
    config_path = run_dir / "config_snapshot.yaml"
    config = _read_yaml(config_path) if config_path.exists() else {}

    experiment_name = str(summary.get("experiment_name") or _safe_get(config, "experiment", "name") or run_dir.name)

    model_name = _safe_get(config, "model", "name")
    pretrained = _as_bool(_safe_get(config, "model", "pretrained"))
    backbone_lr = _as_float(_safe_get(config, "training", "backbone_lr"))
    head_lr = _as_float(_safe_get(config, "training", "head_lr"))
    epochs = _as_int(_safe_get(config, "training", "epochs"))
    weight_decay = _as_float(_safe_get(config, "training", "weight_decay"))
    dropout = _as_float(_safe_get(config, "model", "dropout"))
    label_smoothing = _as_float(_safe_get(config, "training", "label_smoothing"))

    best_epoch = _as_int(summary.get("best_epoch"))
    best_val_acc = _as_float(summary.get("best_val_acc"))
    test_acc = _as_float(summary.get("test_acc"))

    return Row(
        output_dir=str(run_dir),
        experiment_name=experiment_name,
        model_name=str(model_name) if model_name is not None else None,
        pretrained=pretrained,
        backbone_lr=backbone_lr,
        head_lr=head_lr,
        epochs=epochs,
        weight_decay=weight_decay,
        dropout=dropout,
        label_smoothing=label_smoothing,
        best_epoch=best_epoch,
        best_val_acc=best_val_acc,
        test_acc=test_acc,
    )


def collect(outputs_dir: Path) -> list[Row]:
    if not outputs_dir.exists():
        return []

    rows: list[Row] = []
    for run_dir in sorted(outputs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        row = _row_from_run_dir(run_dir)
        if row is not None:
            rows.append(row)
    return rows


def write_csv(rows: list[Row], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "output_dir",
        "experiment_name",
        "model_name",
        "pretrained",
        "backbone_lr",
        "head_lr",
        "epochs",
        "weight_decay",
        "dropout",
        "label_smoothing",
        "best_epoch",
        "best_val_acc",
        "test_acc",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "output_dir": r.output_dir,
                    "experiment_name": r.experiment_name,
                    "model_name": r.model_name or "",
                    "pretrained": "" if r.pretrained is None else str(r.pretrained),
                    "backbone_lr": "" if r.backbone_lr is None else f"{r.backbone_lr:.10g}",
                    "head_lr": "" if r.head_lr is None else f"{r.head_lr:.10g}",
                    "epochs": "" if r.epochs is None else str(r.epochs),
                    "weight_decay": "" if r.weight_decay is None else f"{r.weight_decay:.10g}",
                    "dropout": "" if r.dropout is None else f"{r.dropout:.10g}",
                    "label_smoothing": "" if r.label_smoothing is None else f"{r.label_smoothing:.10g}",
                    "best_epoch": "" if r.best_epoch is None else str(r.best_epoch),
                    "best_val_acc": "" if r.best_val_acc is None else f"{r.best_val_acc:.10g}",
                    "test_acc": "" if r.test_acc is None else f"{r.test_acc:.10g}",
                }
            )


def main() -> None:
    task1_root = Path(__file__).resolve().parents[1]
    outputs_dir = task1_root / "outputs"
    out_csv = outputs_dir / "hparam_summary.csv"

    rows = collect(outputs_dir)
    rows_sorted = sorted(rows, key=lambda r: (r.best_val_acc is not None, r.best_val_acc or -1.0), reverse=True)
    write_csv(rows_sorted, out_csv)

    print(f"[OK] Wrote {len(rows_sorted)} rows to {out_csv}")
    print("=" * 72)
    print("Top 10 by best_val_acc")
    print("=" * 72)
    for i, r in enumerate(rows_sorted[:10], start=1):
        bva = "NA" if r.best_val_acc is None else f"{r.best_val_acc:.4f}"
        ta = "NA" if r.test_acc is None else f"{r.test_acc:.4f}"
        print(f"{i:02d}. {r.experiment_name:35s}  best_val_acc={bva}  test_acc={ta}  dir={r.output_dir}")


if __name__ == "__main__":
    main()

