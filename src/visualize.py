import os

import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(history, save_path):
    # Plot train/val loss and val accuracy curves.
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(epochs, history["train_loss"], label="Train Loss")
    ax1.plot(epochs, history["val_loss"], label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax2 = ax1.twinx()
    ax2.plot(epochs, history["val_acc"], color="green", label="Val Acc")
    ax2.set_ylabel("Accuracy")
    ax2.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(cm, class_names, save_path):
    # Draw confusion matrix as heatmap-like image.
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def visualize_first_layer_weights(weight_matrix, save_path, max_plots=256):
    # Visualize first-layer weights as a 28x28 grid of 16x16 maps.
    # Each small map corresponds to one input pixel (28x28 total),
    # showing its 256 outgoing weights to hidden units (16x16).
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if weight_matrix.shape[0] != 28 * 28:
        raise ValueError("Expected first dimension to be 784 (28x28 input).")
    if weight_matrix.shape[1] < 16 * 16:
        raise ValueError("Expected at least 256 hidden units for 16x16 maps.")

    w = weight_matrix[:, : 16 * 16]
    rows, cols = 28, 28
    fig, axes = plt.subplots(rows, cols, figsize=(28, 28))
    axes = np.array(axes).reshape(rows, cols)
    vmax = float(np.max(np.abs(w)))
    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        ax.axis("off")
        img = w[i].reshape(16, 16)
        ax.imshow(img, cmap="seismic", vmin=-vmax, vmax=vmax)
    fig.suptitle("First Layer Weights (28x28 maps, each 16x16)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_misclassified_examples(x, y_true, y_pred, class_names, save_path, max_items=16):
    # Show a few test errors for error analysis.
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    wrong = np.where(y_true != y_pred)[0]
    n = min(max_items, len(wrong))
    if n == 0:
        return
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    axes = np.array(axes).reshape(rows, cols)
    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        ax.axis("off")
        if i < n:
            idx = wrong[i]
            ax.imshow(x[idx].reshape(28, 28), cmap="gray")
            ax.set_title(f"T:{class_names[y_true[idx]]}\nP:{class_names[y_pred[idx]]}", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def save_all_misclassified_images(x, y_true, y_pred, class_names, out_dir):
    # Save every misclassified sample as an individual image file.
    os.makedirs(out_dir, exist_ok=True)
    wrong = np.where(y_true != y_pred)[0]
    for idx in wrong:
        true_name = class_names[y_true[idx]].replace("/", "_").replace(" ", "_")
        pred_name = class_names[y_pred[idx]].replace("/", "_").replace(" ", "_")
        filename = f"idx_{idx:05d}_true_{y_true[idx]}-{true_name}_pred_{y_pred[idx]}-{pred_name}.png"
        save_path = os.path.join(out_dir, filename)
        fig, ax = plt.subplots(figsize=(2.4, 2.4))
        ax.imshow(x[idx].reshape(28, 28), cmap="gray")
        ax.axis("off")
        ax.set_title(f"T:{class_names[y_true[idx]]} | P:{class_names[y_pred[idx]]}", fontsize=8)
        fig.tight_layout()
        fig.savefig(save_path, dpi=120)
        plt.close(fig)
