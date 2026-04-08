import itertools
import json
import os

from src.train_eval import train_one_run


def grid_search(
    x_train, y_train, x_val, y_val,
    param_grid,
    fixed_params,
    out_dir="artifacts/search"
):
    # Run grid search and keep all experiment metrics.
    os.makedirs(out_dir, exist_ok=True)
    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    results = []
    best = None

    for i, combo in enumerate(itertools.product(*values), start=1):
        params = dict(zip(keys, combo))
        run_name = f"run_{i:03d}"
        save_path = os.path.join(out_dir, f"{run_name}_best.npz")
        full = {**fixed_params, **params, "save_path": save_path}
        print(f"\n[Grid Search] {run_name} params={params}")
        _, history, best_info = train_one_run(
            x_train, y_train, x_val, y_val, **full
        )
        history_path = os.path.join(out_dir, f"{run_name}_history.json")
        with open(history_path, "w", encoding="utf-8") as hf:
            json.dump(history, hf, indent=2)
        row = {
            "run_name": run_name,
            **params,
            "best_val_acc": best_info["acc"],
            "best_epoch": best_info["epoch"],
            "save_path": save_path,
            "history_path": history_path,
            "history": history,
        }
        results.append(row)
        if best is None or row["best_val_acc"] > best["best_val_acc"]:
            best = row

    with open(os.path.join(out_dir, "search_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(out_dir, "best_result.json"), "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)
    return best, results
