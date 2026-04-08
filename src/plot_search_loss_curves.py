import argparse
import glob
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def load_results(search_dir, search_results_path):
    if os.path.isfile(search_results_path):
        with open(search_results_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        if results:
            normalized = []
            for row in results:
                r = dict(row)
                if "history" not in r or r["history"] is None:
                    hp = r.get("history_path")
                    if hp and os.path.isfile(hp):
                        with open(hp, "r", encoding="utf-8") as hf:
                            r["history"] = json.load(hf)
                normalized.append(r)
            if all("history" in x and x["history"] for x in normalized):
                return normalized

    pattern = os.path.join(search_dir, "run_*_history.json")
    paths = sorted(glob.glob(pattern))
    if not paths:
        return []

    results = []
    for hp in paths:
        base = os.path.basename(hp)
        run_name = base.replace("_history.json", "")
        meta_path = os.path.join(search_dir, f"{run_name}_best_meta.json")
        if not os.path.isfile(meta_path):
            continue
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        with open(hp, "r", encoding="utf-8") as f:
            history = json.load(f)
        row = {
            "run_name": run_name,
            **{k: v for k, v in meta.items() if k not in ("best_val_acc", "best_epoch")},
            "best_val_acc": meta.get("best_val_acc"),
            "best_epoch": meta.get("best_epoch"),
            "history": history,
        }
        results.append(row)
    return results


def _stack_with_nan(curves):
    max_len = max(len(c) for c in curves)
    arr = np.full((len(curves), max_len), np.nan, dtype=float)
    for i, c in enumerate(curves):
        arr[i, : len(c)] = c
    return arr


def _plot_metric_for_param(ax, results, param_name, metric_key):
    grouped = defaultdict(list)
    for row in results:
        if param_name not in row or "history" not in row:
            continue
        hist = row["history"]
        if metric_key not in hist:
            continue
        grouped[row[param_name]].append(hist[metric_key])

    values = sorted(grouped.keys(), key=lambda x: str(x))
    if not values:
        ax.axis("off")
        return

    for val in values:
        curves = grouped[val]
        arr = _stack_with_nan(curves)
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0)
        epochs = np.arange(1, len(mean) + 1)
        ax.plot(epochs, mean, label=f"{val}")
        ax.fill_between(epochs, mean - std, mean + std, alpha=0.15)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric_key)
    ax.set_title(f"{param_name} -> {metric_key} (mean±std)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, loc="upper right")


def main():
    parser = argparse.ArgumentParser(description="Plot grid-search loss curves by each hyperparameter.")
    parser.add_argument("--search_dir", type=str, default="artifacts/search")
    parser.add_argument(
        "--search_results",
        type=str,
        default="artifacts/search/search_results.json",
        help="Optional: full search_results.json (if missing, uses run_*_history.json + *_best_meta.json)",
    )
    parser.add_argument("--out_train", type=str, default="artifacts/gridsearch_loss_curves_train_by_dim.png")
    parser.add_argument("--out_val", type=str, default="artifacts/gridsearch_loss_curves_val_by_dim.png")
    args = parser.parse_args()

    results = load_results(args.search_dir, args.search_results)
    if not results:
        raise FileNotFoundError(
            "没有找到网格搜索的 loss 历史：需要 artifacts/search/search_results.json，"
            "或 run_*_history.json（与 run_*_best_meta.json 配对）。请至少完整运行一次 "
            "`python main.py --mode search`（会写入每个 run 的 history）。仅 checkpoint 无法还原训练曲线。"
        )

    candidate_params = []
    for key in results[0].keys():
        if key in {"run_name", "best_val_acc", "best_epoch", "save_path", "history", "history_path"}:
            continue
        vals = [r.get(key) for r in results]
        if len(set(vals)) > 1:
            candidate_params.append(key)

    if not candidate_params:
        raise ValueError("No varying hyperparameters found in search results.")

    n = len(candidate_params)
    cols = 2
    rows = int(np.ceil(n / cols))

    fig1, axes1 = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    axes1 = np.array(axes1).reshape(rows, cols)
    for i, p in enumerate(candidate_params):
        ax = axes1[i // cols, i % cols]
        _plot_metric_for_param(ax, results, p, "train_loss")
    for j in range(n, rows * cols):
        axes1[j // cols, j % cols].axis("off")
    fig1.tight_layout()
    fig1.savefig(args.out_train, dpi=180)
    plt.close(fig1)

    fig2, axes2 = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    axes2 = np.array(axes2).reshape(rows, cols)
    for i, p in enumerate(candidate_params):
        ax = axes2[i // cols, i % cols]
        _plot_metric_for_param(ax, results, p, "val_loss")
    for j in range(n, rows * cols):
        axes2[j // cols, j % cols].axis("off")
    fig2.tight_layout()
    fig2.savefig(args.out_val, dpi=180)
    plt.close(fig2)

    print(f"saved {args.out_train}")
    print(f"saved {args.out_val}")
    print(f"params: {candidate_params}")


if __name__ == "__main__":
    main()
