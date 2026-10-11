from __future__ import annotations

import pickle
import tarfile
from pathlib import Path

import numpy as np

from .mnist import download_file


CIFAR_ARCHIVE = "cifar-10-python.tar.gz"
CIFAR_URL = "https://dataset.bj.bcebos.com/cifar/cifar-10-python.tar.gz"
CIFAR_MD5 = "c58f30108f718f92721af3b95e74349a"


def download_cifar10(root: Path) -> None:
    archive = root / CIFAR_ARCHIVE
    download_file(CIFAR_URL, archive, CIFAR_MD5)
    target = root / "cifar-10-batches-py"
    if not target.exists():
        with tarfile.open(archive, "r:gz") as tar:
            members = tar.getmembers()
            if any(member.name != "cifar-10-batches-py" and not member.name.startswith("cifar-10-batches-py/") for member in members):
                raise ValueError("unexpected path in CIFAR archive")
            tar.extractall(root, filter="data")


def _load_cifar_batch(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as handle:
        record = pickle.load(handle, encoding="bytes")
    data = np.asarray(record[b"data"], dtype=np.uint8)
    labels = np.asarray(record.get(b"labels", record.get(b"fine_labels")), dtype=np.uint8)
    if data.ndim != 2 or data.shape[1] != 3072 or len(data) != len(labels):
        raise ValueError(f"invalid CIFAR batch: {path}")
    if np.any(labels > 9):
        raise ValueError("CIFAR label outside [0,9]")
    return data, labels


def load_cifar10_raw(root: Path, *, include_test: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    batch_root = root / "cifar-10-batches-py"
    if not batch_root.exists():
        raise FileNotFoundError("missing CIFAR-10 batches; run prepare-data --dataset cifar10 --download")
    batches = [_load_cifar_batch(batch_root / f"data_batch_{index}") for index in range(1, 6)]
    train_x = np.concatenate([batch[0] for batch in batches])
    train_y = np.concatenate([batch[1] for batch in batches])
    if len(train_x) != 50000:
        raise ValueError("unexpected CIFAR-10 training sample count")
    if not include_test:
        return train_x, train_y, np.empty((0, 3072), dtype=np.uint8), np.empty(0, dtype=np.uint8)
    test_x, test_y = _load_cifar_batch(batch_root / "test_batch")
    if len(test_x) != 10000:
        raise ValueError("unexpected CIFAR-10 test sample count")
    return train_x, train_y, test_x, test_y
