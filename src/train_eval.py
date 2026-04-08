import copy
import json
import os

import numpy as np

from src.autograd import Tensor, to_numpy
from src.data import iterate_minibatches
from src.model import MLP3
from src.optim import ExponentialLRScheduler, SGD


def one_hot(labels, num_classes):
    y = np.zeros((len(labels), num_classes), dtype=np.float32)
    y[np.arange(len(labels)), labels] = 1.0
    return y


def cross_entropy_loss(logits, y_true_int):
    # Stable softmax CE built from autodiff tensor ops.
    y_oh = Tensor(one_hot(y_true_int, logits.data.shape[1]), requires_grad=False)
    max_per_row = Tensor(logits.data.max(axis=1, keepdims=True), requires_grad=False)
    shifted = logits - max_per_row
    log_probs = shifted - shifted.exp().sum(axis=1, keepdims=True).log()
    return -(y_oh * log_probs).sum() * (1.0 / logits.data.shape[0])


def predict(model, x, batch_size=256):
    preds = []
    for xb, _ in iterate_minibatches(x, np.zeros(len(x), dtype=np.int64), batch_size, shuffle=False):
        logits = model(Tensor(xb, requires_grad=False))
        preds.append(np.argmax(to_numpy(logits.data), axis=1))
    return np.concatenate(preds, axis=0)


def accuracy(y_true, y_pred):
    return float((y_true == y_pred).mean())


def confusion_matrix(y_true, y_pred, num_classes=10):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def train_one_run(
    x_train, y_train, x_val, y_val,
    hidden_dim=256, activation="relu",
    lr=0.05, weight_decay=1e-4, lr_gamma=0.98,
    epochs=20, batch_size=128, seed=42,
    save_path="artifacts/best_model.npz"
):
    model = MLP3(784, hidden_dim, 10, activation=activation, seed=seed)
    optimizer = SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ExponentialLRScheduler(optimizer, gamma=lr_gamma)
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best = {"acc": -1.0, "state": None, "epoch": -1}

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    for epoch in range(1, epochs + 1):
        batch_losses = []
        for xb, yb in iterate_minibatches(x_train, y_train, batch_size, shuffle=True, seed=seed + epoch):
            optimizer.zero_grad()
            logits = model(Tensor(xb, requires_grad=True))
            loss = cross_entropy_loss(logits, yb)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(to_numpy(loss.data)))

        train_loss = float(np.mean(batch_losses))
        val_loss, val_acc = evaluate(model, x_val, y_val, batch_size=batch_size)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best["acc"]:
            best["acc"] = val_acc
            best["state"] = copy.deepcopy(model.state_dict())
            best["epoch"] = epoch
            save_state = {k: to_numpy(v) for k, v in best["state"].items() if k != "activation"}
            save_state["activation"] = best["state"]["activation"]
            np.savez(save_path, **save_state)

        scheduler.step()
        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | lr={optimizer.lr:.6f}"
        )

    meta_path = save_path.replace(".npz", "_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_val_acc": best["acc"],
                "best_epoch": best["epoch"],
                "hidden_dim": hidden_dim,
                "activation": activation,
                "lr": lr,
                "weight_decay": weight_decay,
                "lr_gamma": lr_gamma,
                "epochs": epochs,
                "batch_size": batch_size,
                "seed": seed,
            },
            f,
            indent=2,
        )

    model.load_state_dict(best["state"])
    return model, history, best


def evaluate(model, x, y, batch_size=256):
    losses = []
    all_preds = []
    for xb, yb in iterate_minibatches(x, y, batch_size, shuffle=False):
        logits = model(Tensor(xb, requires_grad=False))
        loss = cross_entropy_loss(logits, yb)
        losses.append(float(to_numpy(loss.data)))
        all_preds.append(np.argmax(to_numpy(logits.data), axis=1))
    y_pred = np.concatenate(all_preds, axis=0)
    return float(np.mean(losses)), accuracy(y, y_pred)
