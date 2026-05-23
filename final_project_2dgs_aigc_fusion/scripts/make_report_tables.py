#!/usr/bin/env python3
"""Create Markdown and LaTeX report tables from asset metrics and hyperparameters."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable


ASSET_TABLE_FIELDS = [
    "asset_name",
    "source_type",
    "method",
    "num_vertices",
    "num_faces",
    "bbox_x",
    "bbox_y",
    "bbox_z",
    "has_texture",
    "training_or_generation_time_minutes",
    "estimated_gpu_memory",
]

HYPERPARAMETER_FIELDS = [
    "component",
    "method",
    "iterations",
    "resolution",
    "batch_size",
    "learning_rate",
    "optimizer",
    "loss_function",
    "hardware",
    "runtime",
]


DEFAULT_HYPERPARAMETERS = [
    {
        "component": "object_a",
        "method": "COLMAP + 2DGS",
        "iterations": "7000",
        "resolution": "2",
        "batch_size": "N/A",
        "learning_rate": "see external 2DGS config",
        "optimizer": "see external 2DGS config",
        "loss_function": "2DGS photometric losses",
        "hardware": "TODO",
        "runtime": "TODO",
    },
    {
        "component": "background",
        "method": "2DGS",
        "iterations": "7000",
        "resolution": "4",
        "batch_size": "N/A",
        "learning_rate": "see external 2DGS config",
        "optimizer": "see external 2DGS config",
        "loss_function": "2DGS photometric losses",
        "hardware": "TODO",
        "runtime": "TODO",
    },
    {
        "component": "object_b",
        "method": "threestudio SDS",
        "iterations": "TODO",
        "resolution": "TODO",
        "batch_size": "TODO",
        "learning_rate": "see threestudio config",
        "optimizer": "see threestudio config",
        "loss_function": "SDS loss",
        "hardware": "TODO",
        "runtime": "TODO",
    },
    {
        "component": "object_c",
        "method": "Magic123",
        "iterations": "TODO",
        "resolution": "TODO",
        "batch_size": "TODO",
        "learning_rate": "see Magic123 config",
        "optimizer": "see Magic123 config",
        "loss_function": "image/view prior losses",
        "hardware": "TODO",
        "runtime": "TODO",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--asset_metrics",
        default="outputs/evaluation/asset_metrics.csv",
        help="Input asset metrics CSV.",
    )
    parser.add_argument(
        "--hyperparams_json",
        help="Optional hyperparameter JSON list. If omitted, a TODO template is generated.",
    )
    parser.add_argument("--out_dir", default="report/tables", help="Output directory.")
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise SystemExit(f"Error: asset metrics CSV not found: {path}")
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def select_fields(rows: Iterable[dict[str, str]], fields: list[str]) -> list[dict[str, str]]:
    selected = []
    for row in rows:
        selected.append({field: str(row.get(field, "")) for field in fields})
    return selected


def markdown_table(rows: list[dict[str, str]], fields: list[str]) -> str:
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join(["---"] * len(fields)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row.get(field, "") for field in fields) + " |")
    return "\n".join(lines) + "\n"


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def latex_table(rows: list[dict[str, str]], fields: list[str], caption: str, label: str) -> str:
    column_spec = "l" * len(fields)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\hline",
        " & ".join(latex_escape(field) for field in fields) + r" \\",
        r"\hline",
    ]
    for row in rows:
        lines.append(" & ".join(latex_escape(row.get(field, "")) for field in fields) + r" \\")
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            rf"\caption{{{latex_escape(caption)}}}",
            rf"\label{{{latex_escape(label)}}}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def load_hyperparams(path_text: str | None) -> list[dict[str, str]]:
    if not path_text:
        return DEFAULT_HYPERPARAMETERS
    path = Path(path_text)
    if not path.is_file():
        raise SystemExit(f"Error: hyperparameter JSON not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit("Error: hyperparameter JSON must contain a list of row objects.")
    return [{field: str(item.get(field, "")) for field in HYPERPARAMETER_FIELDS} for item in data]


def write_tables(
    rows: list[dict[str, str]],
    fields: list[str],
    out_dir: Path,
    stem: str,
    caption: str,
    label: str,
) -> None:
    md_path = out_dir / f"{stem}.md"
    tex_path = out_dir / f"{stem}.tex"
    md_path.write_text(markdown_table(rows, fields), encoding="utf-8")
    tex_path.write_text(latex_table(rows, fields, caption, label), encoding="utf-8")
    print(f"Wrote {md_path}")
    print(f"Wrote {tex_path}")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    asset_rows = select_fields(read_csv_rows(Path(args.asset_metrics)), ASSET_TABLE_FIELDS)
    hyper_rows = select_fields(load_hyperparams(args.hyperparams_json), HYPERPARAMETER_FIELDS)

    write_tables(
        asset_rows,
        ASSET_TABLE_FIELDS,
        out_dir,
        "asset_comparison_table",
        "Asset comparison across reconstruction and generation methods.",
        "tab:asset_comparison",
    )
    write_tables(
        hyper_rows,
        HYPERPARAMETER_FIELDS,
        out_dir,
        "hyperparameter_table",
        "Hyperparameter settings for all pipeline components.",
        "tab:hyperparameters",
    )


if __name__ == "__main__":
    main()

