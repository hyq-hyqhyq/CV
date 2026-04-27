from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pet_cls.compat import patch_broken_dill

patch_broken_dill()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a list of Oxford-IIIT Pet experiments.")
    parser.add_argument(
        "--configs",
        nargs="+",
        required=True,
        help="One or more YAML config files.",
    )
    parser.add_argument(
        "--set",
        nargs="*",
        default=None,
        help="Shared key=value overrides applied to all experiments.",
    )
    parser.add_argument(
        "--summary-name",
        type=str,
        default="experiment_summary.csv",
        help="File name of the aggregated CSV summary under task1/outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from pet_cls.config import apply_overrides, load_config

    try:
        from pet_cls.runner import run_experiment
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Missing runtime dependency: {exc}. Please install packages from requirements.txt in a clean environment."
        ) from exc

    rows: list[dict[str, object]] = []
    for config_path in args.configs:
        config = apply_overrides(load_config(config_path), args.set)
        summary, _ = run_experiment(config, project_root=PROJECT_ROOT)
        rows.append(
            {
                "config": config_path,
                "experiment_name": summary["experiment_name"],
                "best_epoch": summary["best_epoch"],
                "best_val_acc": summary["best_val_acc"],
                "test_acc": summary["test_acc"],
                "test_loss": summary["test_loss"],
                "run_dir": summary["run_dir"],
            }
        )

    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / args.summary_name
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved aggregated summary to {csv_path}")


if __name__ == "__main__":
    main()
