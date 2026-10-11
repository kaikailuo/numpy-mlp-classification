from __future__ import annotations

import gzip
import hashlib
import json
import pickle
import struct
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import numpy as np


MNIST_FILES = {
    "train-images-idx3-ubyte.gz": ("https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
    "train-labels-idx1-ubyte.gz": ("https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
    "t10k-images-idx3-ubyte.gz": ("https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
    "t10k-labels-idx1-ubyte.gz": ("https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c"),
}
CIFAR_ARCHIVE = "cifar-10-python.tar.gz"
CIFAR_URL = "https://dataset.bj.bcebos.com/cifar/cifar-10-python.tar.gz"
CIFAR_MD5 = "c58f30108f718f92721af3b95e74349a"


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


def file_md5(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, path: Path, md5: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and file_md5(path) == md5:
        return
    temporary = path.with_suffix(path.suffix + ".part")
    urllib.request.urlretrieve(url, temporary)
    actual = file_md5(temporary)
    if actual != md5:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"checksum mismatch for {path.name}: {actual}")
    temporary.replace(path)


def download_mnist(root: Path) -> None:
    for filename, (url, md5) in MNIST_FILES.items():
        download_file(url, root / filename, md5)


def download_cifar10(root: Path) -> None:
    archive = root / CIFAR_ARCHIVE
    download_file(CIFAR_URL, archive, CIFAR_MD5)
    target = root / "cifar-10-batches-py"
    if not target.exists():
        with tarfile.open(archive, "r:gz") as tar:
            members = tar.getmembers()
            if any(
                member.name != "cifar-10-batches-py"
                and not member.name.startswith("cifar-10-batches-py/")
                for member in members
            ):
                raise ValueError("unexpected path in CIFAR archive")
            tar.extractall(root, filter="data")


def _open_maybe_gzip(path: Path):
    return gzip.open(path, "rb") if path.suffix == ".gz" else path.open("rb")


def read_idx_images(path: str | Path) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with _open_maybe_gzip(path) as handle:
        header = handle.read(16)
        if len(header) != 16:
            raise ValueError("truncated IDX image header")
        magic, count, rows, cols = struct.unpack(">IIII", header)
        if magic != 2051 or (rows, cols) != (28, 28):
            raise ValueError(f"invalid IDX images: magic={magic}, shape={(rows, cols)}")
        raw = handle.read()
    expected = count * rows * cols
    if len(raw) != expected:
        raise ValueError(f"IDX image payload size {len(raw)} != {expected}")
    return np.frombuffer(raw, dtype=np.uint8).reshape(count, rows * cols).copy()


def read_idx_labels(path: str | Path) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with _open_maybe_gzip(path) as handle:
        header = handle.read(8)
        if len(header) != 8:
            raise ValueError("truncated IDX label header")
        magic, count = struct.unpack(">II", header)
        if magic != 2049:
            raise ValueError(f"invalid IDX labels magic={magic}")
        raw = handle.read()
    if len(raw) != count:
        raise ValueError(f"IDX label payload size {len(raw)} != {count}")
    labels = np.frombuffer(raw, dtype=np.uint8).copy()
    if np.any(labels > 9):
        raise ValueError("MNIST label outside [0,9]")
    return labels


def _find_mnist_file(root: Path, filename: str) -> Path:
    candidates = [root / filename, root / filename.removesuffix(".gz")]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"missing {filename}; run prepare-data --dataset mnist --download")


def load_mnist_raw(root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_x = read_idx_images(_find_mnist_file(root, "train-images-idx3-ubyte.gz"))
    train_y = read_idx_labels(_find_mnist_file(root, "train-labels-idx1-ubyte.gz"))
    test_x = read_idx_images(_find_mnist_file(root, "t10k-images-idx3-ubyte.gz"))
    test_y = read_idx_labels(_find_mnist_file(root, "t10k-labels-idx1-ubyte.gz"))
    if (len(train_x), len(test_x)) != (60000, 10000) or len(train_x) != len(train_y) or len(test_x) != len(test_y):
        raise ValueError("unexpected MNIST sample counts")
    return train_x, train_y, test_x, test_y


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


def load_cifar10_raw(root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    batch_root = root / "cifar-10-batches-py"
    if not batch_root.exists():
        raise FileNotFoundError("missing CIFAR-10 batches; run prepare-data --dataset cifar10 --download")
    batches = [_load_cifar_batch(batch_root / f"data_batch_{index}") for index in range(1, 6)]
    train_x = np.concatenate([batch[0] for batch in batches])
    train_y = np.concatenate([batch[1] for batch in batches])
    test_x, test_y = _load_cifar_batch(batch_root / "test_batch")
    if (len(train_x), len(test_x)) != (50000, 10000):
        raise ValueError("unexpected CIFAR-10 sample counts")
    return train_x, train_y, test_x, test_y


def stratified_split(labels: np.ndarray, val_size: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if not 0 < val_size < len(labels):
        raise ValueError("val_size must be within dataset")
    classes, counts = np.unique(labels, return_counts=True)
    exact = counts * (val_size / len(labels))
    allocations = np.floor(exact).astype(int)
    remainder = val_size - int(allocations.sum())
    order = np.argsort(-(exact - allocations), kind="stable")
    allocations[order[:remainder]] += 1
    rng = np.random.default_rng(seed)
    val_parts: list[np.ndarray] = []
    train_parts: list[np.ndarray] = []
    for label, allocation in zip(classes, allocations):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        val_parts.append(indices[:allocation])
        train_parts.append(indices[allocation:])
    val_indices = np.concatenate(val_parts)
    train_indices = np.concatenate(train_parts)
    rng.shuffle(val_indices)
    rng.shuffle(train_indices)
    if len(val_indices) != val_size or len(np.intersect1d(train_indices, val_indices)):
        raise AssertionError("invalid split")
    return train_indices, val_indices


def split_path(dataset: str) -> Path:
    return Path("data/splits") / f"{dataset}_split_v1.npz"


def load_or_create_split(dataset: str, labels: np.ndarray, val_size: int, seed: int) -> tuple[np.ndarray, np.ndarray, dict]:
    path = split_path(dataset)
    meta_path = path.with_suffix(".json")
    label_hash = hashlib.sha256(labels.tobytes()).hexdigest()
    if path.exists() and meta_path.exists():
        with np.load(path, allow_pickle=False) as archive:
            train_indices = archive["train_indices"]
            val_indices = archive["val_indices"]
        meta = json.loads(meta_path.read_text())
        if meta.get("label_sha256") != label_hash or meta.get("seed") != seed or meta.get("val_size") != val_size:
            raise ValueError("existing split metadata does not match requested data")
        return train_indices, val_indices, meta
    train_indices, val_indices = stratified_split(labels, val_size, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, train_indices=train_indices, val_indices=val_indices)
    meta = {
        "dataset": dataset,
        "algorithm": "stratified_v1_largest_remainder",
        "seed": seed,
        "val_size": val_size,
        "train_size": len(train_indices),
        "label_sha256": label_hash,
        "train_class_counts": np.bincount(labels[train_indices], minlength=10).tolist(),
        "val_class_counts": np.bincount(labels[val_indices], minlength=10).tolist(),
        "split_sha256": hashlib.sha256(train_indices.tobytes() + val_indices.tobytes()).hexdigest(),
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    return train_indices, val_indices, meta


def _cifar_channel_stats(x: np.ndarray, chunk: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    total = np.zeros(3, dtype=np.float64)
    total_sq = np.zeros(3, dtype=np.float64)
    count = 0
    for start in range(0, len(x), chunk):
        values = x[start : start + chunk].reshape(-1, 3, 1024).astype(np.float64)
        total += values.sum(axis=(0, 2))
        total_sq += np.square(values).sum(axis=(0, 2))
        count += values.shape[0] * values.shape[2]
    mean = total / count
    variance = np.maximum(total_sq / count - mean * mean, 0.0)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def load_dataset(config: dict, *, include_test: bool = True) -> DatasetBundle:
    data_config = config["data"]
    dataset = data_config["name"]
    root = Path(data_config["root"])
    if dataset == "mnist":
        all_x, all_y, test_x, test_y = load_mnist_raw(root)
    elif dataset == "cifar10":
        all_x, all_y, test_x, test_y = load_cifar10_raw(root)
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
    dtype = np.dtype(config["compute"]["dtype"])
    manifest = {
        "dataset": dataset,
        "split": split_meta,
        "effective_train_size": len(train_x),
        "train_subset_policy": "prefix_of_fixed_shuffled_split" if subset_size is not None else "full_split",
    }
    if dataset == "mnist":
        def preprocess(batch: np.ndarray) -> np.ndarray:
            return batch.astype(dtype) / dtype.type(255.0)
        manifest["preprocess"] = {"name": "scale01", "fit_split": "none"}
    else:
        mean, std = _cifar_channel_stats(train_x)
        safe_std = np.maximum(std, 1e-6)
        def preprocess(batch: np.ndarray) -> np.ndarray:
            shaped = batch.astype(dtype).reshape(-1, 3, 1024)
            shaped = (shaped - mean.astype(dtype)[None, :, None]) / safe_std.astype(dtype)[None, :, None]
            return shaped.reshape(-1, 3072)
        manifest["preprocess"] = {
            "name": "channel_standardize",
            "fit_split": "train",
            "mean": mean.tolist(),
            "std": std.tolist(),
        }
    if not include_test:
        test_x = test_x[:0]
        test_y = test_y[:0]
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

