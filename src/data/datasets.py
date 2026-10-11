from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .cifar10 import download_cifar10, load_cifar10_raw
from .mnist import download_mnist, load_mnist_raw
from .preprocessing import build_preprocessor
from .splits import load_or_create_split


@dataclass
class DatasetBundle:
    train_x: np.ndarray
    train_y: np.ndarray
    val_x: np.ndarray
    val_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray
    preprocess: Callable[[np.ndarray], np.ndarray]
    manifest: dict


def load_dataset(
    config: dict,
    *,
    include_test: bool = True,
    frozen_preprocess: dict | None = None,
) -> DatasetBundle:
    data_config = config["data"]
    dataset = data_config["name"]
    root = Path(data_config["root"])
    if dataset == "mnist":
        all_x, all_y, test_x, test_y = load_mnist_raw(root, include_test=include_test)
    elif dataset == "cifar10":
        all_x, all_y, test_x, test_y = load_cifar10_raw(root, include_test=include_test)
    else:
        raise ValueError(f"unsupported real dataset: {dataset}")
    train_indices, val_indices, split_meta = load_or_create_split(
        dataset, all_y, int(data_config["val_size"]), int(data_config["split_seed"])
    )
    train_x, train_y = all_x[train_indices], all_y[train_indices]
    val_x, val_y = all_x[val_indices], all_y[val_indices]
    subset_size = config.get("training", {}).get("train_subset_size")
    if subset_size is not None:
        subset_size = int(subset_size)
        if not 1 <= subset_size <= len(train_x):
            raise ValueError("training.train_subset_size is outside the train split")
        train_x, train_y = train_x[:subset_size], train_y[:subset_size]
    preprocess, preprocess_spec = build_preprocessor(
        dataset,
        np.dtype(config["compute"]["dtype"]),
        train_x,
        frozen_spec=frozen_preprocess,
    )
    manifest = {
        "dataset": dataset,
        "split": split_meta,
        "effective_train_size": len(train_x),
        "test_loaded": bool(include_test),
        "train_subset_policy": "prefix_of_fixed_shuffled_split" if subset_size is not None else "full_split",
        "preprocess": preprocess_spec,
    }
    return DatasetBundle(train_x, train_y, val_x, val_y, test_x, test_y, preprocess, manifest)


def batches(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    *,
    rng: np.random.Generator | None = None,
    shuffle: bool = False,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    indices = np.arange(len(x))
    if shuffle:
        if rng is None:
            raise ValueError("shuffle requires an explicit RNG")
        rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        selection = indices[start : start + batch_size]
        yield x[selection], y[selection]


__all__ = [
    "DatasetBundle",
    "batches",
    "download_cifar10",
    "download_mnist",
    "load_dataset",
]
