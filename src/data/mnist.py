from __future__ import annotations

import gzip
import hashlib
import struct
import urllib.request
from pathlib import Path

import numpy as np


MNIST_FILES = {
    "train-images-idx3-ubyte.gz": ("https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
    "train-labels-idx1-ubyte.gz": ("https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
    "t10k-images-idx3-ubyte.gz": ("https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
    "t10k-labels-idx1-ubyte.gz": ("https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c"),
}


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


def _find(root: Path, filename: str) -> Path:
    for candidate in (root / filename, root / filename.removesuffix(".gz")):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"missing {filename}; run prepare-data --dataset mnist --download")


def load_mnist_raw(root: Path, *, include_test: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_x = read_idx_images(_find(root, "train-images-idx3-ubyte.gz"))
    train_y = read_idx_labels(_find(root, "train-labels-idx1-ubyte.gz"))
    if len(train_x) != 60000 or len(train_x) != len(train_y):
        raise ValueError("unexpected MNIST training sample count")
    if not include_test:
        return train_x, train_y, np.empty((0, 784), dtype=np.uint8), np.empty(0, dtype=np.uint8)
    test_x = read_idx_images(_find(root, "t10k-images-idx3-ubyte.gz"))
    test_y = read_idx_labels(_find(root, "t10k-labels-idx1-ubyte.gz"))
    if len(test_x) != 10000 or len(test_x) != len(test_y):
        raise ValueError("unexpected MNIST test sample count")
    return train_x, train_y, test_x, test_y
