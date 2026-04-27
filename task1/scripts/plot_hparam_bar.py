from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def _as_float(x: Any) -> float | None:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _filter_with_acc(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keep: list[dict[str, Any]] = []
    for r in rows:
        acc = _as_float(r.get("best_val_acc"))
        if acc is None:
            continue
        r2 = dict(r)
        r2["_best_val_acc_float"] = acc
        keep.append(r2)
    return keep


def _save_bar_plot(
    labels: list[str],
    values: list[float],
    title: str,
    out_path: Path,
    xlabel: str = "",
    ylabel: str = "best_val_acc",
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(max(10, 0.6 * len(labels)), 5))
    plt.bar(range(len(labels)), values)
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(0.0, 1.0)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    task1_root = Path(__file__).resolve().parents[1]
    summary_csv = task1_root / "outputs" / "hparam_summary.csv"
    out_dir = task1_root / "outputs" / "hparam_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not summary_csv.exists():
        raise SystemExit(f"Missing {summary_csv}. Run scripts/collect_hparam_results.py first.")

    rows = _filter_with_acc(_read_csv(summary_csv))
    rows_sorted = sorted(rows, key=lambda r: r["_best_val_acc_float"], reverse=True)

    # 1) Top 10 best_val_acc
    top10 = rows_sorted[:10]
    _save_bar_plot(
        labels=[r["experiment_name"] for r in top10],
        values=[r["_best_val_acc_float"] for r in top10],
        title="Top 10 Experiments by best_val_acc",
        out_path=out_dir / "top10_best_val_acc.png",
    )

    # 2) Phase 1 LR experiments
    p1 = [r for r in rows_sorted if str(r.get("experiment_name", "")).startswith("hparam_p1_lr_")]
    p1 = sorted(p1, key=lambda r: r["experiment_name"])
    if p1:
        _save_bar_plot(
            labels=[r["experiment_name"] for r in p1],
            values=[r["_best_val_acc_float"] for r in p1],
            title="Phase 1: Learning Rate Search (best_val_acc)",
            out_path=out_dir / "phase1_lr_best_val_acc.png",
            xlabel="experiment.name",
        )

    # 3) Phase 2 epoch / regularization experiments
    p2 = [r for r in rows_sorted if str(r.get("experiment_name", "")).startswith("hparam_p2_")]
    p2 = sorted(p2, key=lambda r: r["experiment_name"])
    if p2:
        _save_bar_plot(
            labels=[r["experiment_name"] for r in p2],
            values=[r["_best_val_acc_float"] for r in p2],
            title="Phase 2: Epoch & Regularization (best_val_acc)",
            out_path=out_dir / "phase2_reg_epoch_best_val_acc.png",
            xlabel="experiment.name",
        )

    print(f"[OK] Plots saved to {out_dir}")


if __name__ == "__main__":
    main()

