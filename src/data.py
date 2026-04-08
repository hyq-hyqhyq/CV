import gzip
import os
import urllib.request

import numpy as np


FASHION_MNIST_URLS = {
    "train_images": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz",
    "train_labels": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz",
    "test_images": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz",
    "test_labels": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz",
}

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def _download(url, target_path):
    # Download file only if not found locally.
    if not os.path.exists(target_path):
        urllib.request.urlretrieve(url, target_path)


def _read_idx_images(path):
    with gzip.open(path, "rb") as f:
        data = f.read()
    images = np.frombuffer(data, dtype=np.uint8, offset=16)
    return images.reshape(-1, 28 * 28).astype(np.float32)


def _read_idx_labels(path):
    with gzip.open(path, "rb") as f:
        data = f.read()
    return np.frombuffer(data, dtype=np.uint8, offset=8).astype(np.int64)


def load_fashion_mnist(data_dir="data", val_ratio=0.1, seed=42):
    os.makedirs(data_dir, exist_ok=True)
    paths = {}
    for key, url in FASHION_MNIST_URLS.items():
        path = os.path.join(data_dir, f"{key}.gz")
        _download(url, path)
        paths[key] = path

    x_train = _read_idx_images(paths["train_images"]) / 255.0
    y_train = _read_idx_labels(paths["train_labels"])
    x_test = _read_idx_images(paths["test_images"]) / 255.0
    y_test = _read_idx_labels(paths["test_labels"])

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x_train))
    x_train = x_train[idx]
    y_train = y_train[idx]

    n_val = int(len(x_train) * val_ratio)
    x_val, y_val = x_train[:n_val], y_train[:n_val]
    x_train, y_train = x_train[n_val:], y_train[n_val:]
    return x_train, y_train, x_val, y_val, x_test, y_test


def iterate_minibatches(x, y, batch_size, shuffle=True, seed=42):
    n = len(x)
    indices = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_idx = indices[start:end]
        yield x[batch_idx], y[batch_idx]
