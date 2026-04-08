import argparse
import json
import os
from glob import glob

import numpy as np

from src.autograd import get_backend_name, set_backend
from src.data import CLASS_NAMES, load_fashion_mnist
from src.model import MLP3
from src.search import grid_search
from src.train_eval import confusion_matrix, evaluate, predict, train_one_run
from src.visualize import (
    plot_confusion_matrix,
    plot_misclassified_examples,
    save_all_misclassified_images,
    plot_training_curves,
    visualize_first_layer_weights,
)


def build_search_space(batch_size):
    # Use a larger-LR search space for large batches.
    if batch_size >= 1024:
        return {
            "lr": [0.08, 0.12, 0.16],
            "hidden_dim": [256, 384],
            "weight_decay": [5e-5, 1e-4],
            "activation": ["relu", "tanh"],
        }
    if batch_size >= 512:
        return {
            "lr": [0.05, 0.08, 0.1],
            "hidden_dim": [128, 256],
            "weight_decay": [1e-4, 3e-4],
            "activation": ["relu", "tanh"],
        }
    return {
        "lr": [0.03, 0.05],
        "hidden_dim": [128, 256],
        "weight_decay": [1e-4, 5e-4],
        "activation": ["relu", "tanh"],
    }


def resolve_model_config(path, default_hidden_dim=None, default_activation=None):
    # Resolve model config from meta first, then fallback to npz, then defaults.
    hidden_dim = default_hidden_dim
    activation = default_activation
    meta_path = path.replace(".npz", "_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        hidden_dim = meta.get("hidden_dim", hidden_dim)
        activation = meta.get("activation", activation)

    state = np.load(path)
    if hidden_dim is None and "fc1_W" in state.files:
        hidden_dim = int(state["fc1_W"].shape[1])
    if activation is None and "activation" in state.files:
        activation = str(state["activation"]).lower()

    if hidden_dim is None:
        raise ValueError(f"Cannot resolve hidden_dim from checkpoint: {path}")
    if activation is None:
        activation = "relu"
    return hidden_dim, activation


def load_model_from_npz(path, hidden_dim, activation):
    state = np.load(path)
    state_dict = {k: state[k] for k in state.files if k.startswith("fc")}
    if hidden_dim is None:
        hidden_dim = int(state_dict["fc1_W"].shape[1])
    model = MLP3(784, hidden_dim, 10, activation=activation, seed=42)
    model.load_state_dict(state_dict)
    return model


def evaluate_checkpoint(path, x_test, y_test, batch_size, default_hidden_dim=None, default_activation=None):
    # Evaluate one checkpoint and return key metrics.
    hidden_dim, activation = resolve_model_config(path, default_hidden_dim, default_activation)
    model = load_model_from_npz(path, hidden_dim=hidden_dim, activation=activation)
    test_loss, test_acc = evaluate(model, x_test, y_test, batch_size=batch_size)
    return {
        "path": path,
        "hidden_dim": hidden_dim,
        "activation": activation,
        "test_loss": test_loss,
        "test_acc": test_acc,
    }


def main():
    parser = argparse.ArgumentParser(description="Manual autodiff MLP on Fashion-MNIST")
    parser.add_argument("--mode", type=str, default="full", choices=["train", "search", "test", "full"])
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--artifacts_dir", type=str, default="artifacts")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.12)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lr_gamma", type=float, default=0.995)
    parser.add_argument("--hidden_dim", type=int, default=384)
    parser.add_argument("--activation", type=str, default="relu", choices=["relu", "sigmoid", "tanh"])
    parser.add_argument("--backend", type=str, default="cpu", choices=["cpu", "gpu"])
    parser.add_argument("--test_all_search", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_backend(args.backend)
    print(f"Running with backend: {get_backend_name()}")

    os.makedirs(args.artifacts_dir, exist_ok=True)
    x_train, y_train, x_val, y_val, x_test, y_test = load_fashion_mnist(args.data_dir, seed=args.seed)

    best_model_path = os.path.join(args.artifacts_dir, "best_model.npz")
    history_path = os.path.join(args.artifacts_dir, "history.json")

    if args.mode in ["train", "full"]:
        model, history, best = train_one_run(
            x_train, y_train, x_val, y_val,
            hidden_dim=args.hidden_dim,
            activation=args.activation,
            lr=args.lr,
            weight_decay=args.weight_decay,
            lr_gamma=args.lr_gamma,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            save_path=best_model_path,
        )
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        plot_training_curves(history, os.path.join(args.artifacts_dir, "training_curves.png"))
        print(f"Best validation accuracy: {best['acc']:.4f} at epoch {best['epoch']}")

    if args.mode in ["search", "full"]:
        param_grid = build_search_space(args.batch_size)
        fixed = {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr_gamma": args.lr_gamma,
            "seed": args.seed,
        }
        best_result, _ = grid_search(
            x_train, y_train, x_val, y_val,
            param_grid=param_grid,
            fixed_params=fixed,
            out_dir=os.path.join(args.artifacts_dir, "search"),
        )
        print("Best hyperparameter setting:", best_result)

    if args.mode in ["test", "full"]:
        if args.test_all_search:
            search_dir = os.path.join(args.artifacts_dir, "search")
            ckpts = sorted(glob(os.path.join(search_dir, "run_*_best.npz")))
            if len(ckpts) == 0:
                print(f"No search checkpoints found under: {search_dir}")
                return
            all_results = []
            print(f"Evaluating {len(ckpts)} checkpoints from search directory...")
            for i, ckpt in enumerate(ckpts, start=1):
                res = evaluate_checkpoint(
                    ckpt, x_test, y_test, args.batch_size, None, None
                )
                all_results.append(res)
                print(
                    f"[{i:02d}/{len(ckpts)}] {os.path.basename(ckpt)} | "
                    f"acc={res['test_acc']:.4f} | loss={res['test_loss']:.4f} | "
                    f"hidden={res['hidden_dim']} | act={res['activation']}"
                )

            all_results.sort(key=lambda x: x["test_acc"], reverse=True)
            best_row = all_results[0]
            print("\nTop-5 checkpoints by test accuracy:")
            for rank, row in enumerate(all_results[:5], start=1):
                print(
                    f"{rank}. {os.path.basename(row['path'])} | acc={row['test_acc']:.4f} | "
                    f"loss={row['test_loss']:.4f} | hidden={row['hidden_dim']} | act={row['activation']}"
                )

            summary_path = os.path.join(search_dir, "test_ranking.json")
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2)
            print(f"\nBest checkpoint: {best_row['path']}")
            print(f"Ranking saved to: {summary_path}")
            return

        if args.mode == "full":
            # Use best model from search if available.
            search_best_json = os.path.join(args.artifacts_dir, "search", "best_result.json")
            if os.path.exists(search_best_json):
                with open(search_best_json, "r", encoding="utf-8") as f:
                    search_best = json.load(f)
                model = load_model_from_npz(
                    search_best["save_path"], hidden_dim=search_best["hidden_dim"], activation=search_best["activation"]
                )
            else:
                hdim, act = resolve_model_config(best_model_path, args.hidden_dim, args.activation)
                model = load_model_from_npz(best_model_path, hidden_dim=hdim, activation=act)
        else:
            hdim, act = resolve_model_config(best_model_path, args.hidden_dim, args.activation)
            model = load_model_from_npz(best_model_path, hidden_dim=hdim, activation=act)

        test_loss, test_acc = evaluate(model, x_test, y_test, batch_size=args.batch_size)
        y_pred = predict(model, x_test, batch_size=args.batch_size)
        cm = confusion_matrix(y_test, y_pred, num_classes=10)
        print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")
        print("Confusion matrix:")
        print(cm)

        np.savetxt(os.path.join(args.artifacts_dir, "confusion_matrix.csv"), cm, fmt="%d", delimiter=",")
        plot_confusion_matrix(cm, CLASS_NAMES, os.path.join(args.artifacts_dir, "confusion_matrix.png"))
        visualize_first_layer_weights(model.fc1.W.data, os.path.join(args.artifacts_dir, "first_layer_weights.png"))
        plot_misclassified_examples(
            x_test, y_test, y_pred, CLASS_NAMES, os.path.join(args.artifacts_dir, "misclassified_examples.png")
        )
        save_all_misclassified_images(
            x_test, y_test, y_pred, CLASS_NAMES, os.path.join(args.artifacts_dir, "misclassified_all")
        )


if __name__ == "__main__":
    main()
