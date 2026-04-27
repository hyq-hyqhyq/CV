from __future__ import annotations

import argparse
from pathlib import Path

from task2.lib.plotting import generate_training_figures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate report-ready curve figures from Ultralytics results.csv.")
    parser.add_argument("--results-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("task2/report_assets/curves"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    figure_paths = generate_training_figures(args.results_csv, args.output_dir)
    if not figure_paths:
        raise RuntimeError("No matching metrics were found in results.csv")
    for figure_path in figure_paths:
        print(figure_path)


if __name__ == "__main__":
    main()
