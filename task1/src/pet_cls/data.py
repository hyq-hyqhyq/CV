from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet


class TransformedSubset(Dataset):
    def __init__(self, dataset: Dataset, indices: list[int], transform: transforms.Compose):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        image, target = self.dataset[self.indices[idx]]
        if self.transform is not None:
            image = self.transform(image)
        return image, int(target)


def stratified_split(labels: list[int], val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    rng = np.random.default_rng(seed)
    buckets: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        buckets[int(label)].append(index)

    train_indices: list[int] = []
    val_indices: list[int] = []
    for label in sorted(buckets):
        class_indices = np.array(buckets[label], dtype=np.int64)
        rng.shuffle(class_indices)
        val_count = max(1, int(round(len(class_indices) * val_ratio)))
        val_indices.extend(class_indices[:val_count].tolist())
        train_indices.extend(class_indices[val_count:].tolist())

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def build_transforms(config: dict[str, Any]) -> tuple[transforms.Compose, transforms.Compose]:
    dataset_cfg = config["dataset"]
    image_size = int(dataset_cfg["image_size"])
    resize_scale = tuple(dataset_cfg["train_resize_scale"])
    mean = dataset_cfg["mean"]
    std = dataset_cfg["std"]

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=resize_scale),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize(int(image_size * 256 / 224)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return train_transform, eval_transform


def build_dataloaders(config: dict[str, Any]) -> tuple[dict[str, DataLoader], dict[str, Any]]:
    dataset_cfg = config["dataset"]
    seed = int(config["experiment"]["seed"])
    train_transform, eval_transform = build_transforms(config)
    data_root = Path(dataset_cfg["root"])

    trainval_base = OxfordIIITPet(
        root=str(data_root),
        split="trainval",
        target_types="category",
        download=bool(dataset_cfg["download"]),
    )
    test_base = OxfordIIITPet(
        root=str(data_root),
        split="test",
        target_types="category",
        download=bool(dataset_cfg["download"]),
    )

    labels = [int(label) for label in getattr(trainval_base, "_labels")]
    train_indices, val_indices = stratified_split(labels, float(dataset_cfg["val_ratio"]), seed)

    datasets = {
        "train": TransformedSubset(trainval_base, train_indices, train_transform),
        "val": TransformedSubset(trainval_base, val_indices, eval_transform),
        "test": TransformedSubset(test_base, list(range(len(test_base))), eval_transform),
    }

    common_loader_kwargs = {
        "batch_size": int(dataset_cfg["batch_size"]),
        "num_workers": int(dataset_cfg["num_workers"]),
        "pin_memory": bool(dataset_cfg["pin_memory"]),
    }

    loaders = {
        "train": DataLoader(datasets["train"], shuffle=True, **common_loader_kwargs),
        "val": DataLoader(datasets["val"], shuffle=False, **common_loader_kwargs),
        "test": DataLoader(datasets["test"], shuffle=False, **common_loader_kwargs),
    }

    num_classes = len(set(labels))
    configured_num_classes = int(config["model"]["num_classes"])
    if configured_num_classes != num_classes:
        raise ValueError(
            f"Config num_classes={configured_num_classes} does not match dataset classes={num_classes}."
        )

    meta = {
        "num_classes": num_classes,
        "split_sizes": {name: len(dataset) for name, dataset in datasets.items()},
        "class_count_from_trainval": num_classes,
        "train_indices": train_indices,
        "val_indices": val_indices,
    }
    return loaders, meta
