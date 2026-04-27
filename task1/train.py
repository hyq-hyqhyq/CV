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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Oxford-IIIT Pet classifiers.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to a YAML config file.",
    )
    parser.add_argument(
        "--set",
        nargs="*",
        default=None,
        help="Override config entries with key=value pairs, e.g. training.epochs=30.",
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

    config = load_config(args.config)
    config = apply_overrides(config, args.set)
    summary, run_dir = run_experiment(config, project_root=PROJECT_ROOT)

    print("=" * 72)
    print(f"Run directory : {run_dir}")
    print(f"Best val acc  : {summary['best_val_acc']:.4f}")
    print(f"Test acc      : {summary['test_acc']:.4f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
