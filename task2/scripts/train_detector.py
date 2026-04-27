from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from task2.lib.plotting import generate_training_figures, sync_results_csv_to_wandb


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a YOLOv8 detector on VisDrone.")
    parser.add_argument("--model", default="yolov8n.pt", help="Model checkpoint or architecture yaml.")
    parser.add_argument("--data", type=Path, default=Path("task2/configs/visdrone_det.yaml"))
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0", help="CUDA device id, cpu, or comma-separated ids.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--optimizer", default="SGD", choices=["SGD", "Adam", "AdamW", "auto"])
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--warmup-epochs", type=float, default=3.0)
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project", type=Path, default=Path("task2/runs/train"))
    parser.add_argument("--name", default="visdrone_yolov8n")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--sync-wandb", action="store_true", help="Upload results.csv metrics to W&B after training.")
    parser.add_argument("--wandb-project", default="cv-task2-visdrone")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument(
        "--curve-dir",
        type=Path,
        default=None,
        help="Where to save local curve PNGs. Defaults to <run_dir>/curves.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model = YOLO(args.model)

    train_kwargs = {
        "data": str(args.data),
        "imgsz": args.imgsz,
        "epochs": args.epochs,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "close_mosaic": args.close_mosaic,
        "patience": args.patience,
        "cache": args.cache,
        "seed": args.seed,
        "project": str(args.project),
        "name": args.name,
        "exist_ok": args.exist_ok,
        "plots": True,
        "save": True,
        "cos_lr": True,
        "val": True,
    }

    print("Starting Ultralytics training with:")
    for key, value in train_kwargs.items():
        print(f"  - {key}: {value}")

    model.train(**train_kwargs)

    run_dir = Path(getattr(model.trainer, "save_dir", args.project / args.name))
    results_csv = run_dir / "results.csv"
    if not results_csv.exists():
        raise FileNotFoundError(f"Training finished but results.csv was not found under {run_dir}")

    curve_dir = args.curve_dir or (run_dir / "curves")
    generated_files = generate_training_figures(results_csv, curve_dir)
    print("Local curve files:")
    for file_path in generated_files:
        print(f"  - {file_path}")

    if args.sync_wandb:
        sync_results_csv_to_wandb(
            results_csv=results_csv,
            project=args.wandb_project,
            run_name=args.wandb_run_name or args.name,
            entity=args.wandb_entity,
            config={
                "model": args.model,
                "data": str(args.data),
                "imgsz": args.imgsz,
                "epochs": args.epochs,
                "batch": args.batch,
                "optimizer": args.optimizer,
                "lr0": args.lr0,
                "lrf": args.lrf,
                "weight_decay": args.weight_decay,
                "warmup_epochs": args.warmup_epochs,
                "close_mosaic": args.close_mosaic,
                "patience": args.patience,
                "seed": args.seed,
            },
            tags=["visdrone", "yolov8", "task2"],
        )
        print("W&B sync complete.")

    print(f"Run directory: {run_dir}")
    print(f"Best weights: {run_dir / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
