import gzip
import struct

import numpy as np
import pytest

from a1_mlp.data import read_idx_images, read_idx_labels, stratified_split


def test_idx_readers(tmp_path) -> None:
    images = np.arange(2 * 28 * 28, dtype=np.uint8).reshape(2, 784)
    labels = np.asarray([2, 9], dtype=np.uint8)
    image_path = tmp_path / "images.gz"
    label_path = tmp_path / "labels.gz"
    with gzip.open(image_path, "wb") as handle:
        handle.write(struct.pack(">IIII", 2051, 2, 28, 28))
        handle.write(images.tobytes())
    with gzip.open(label_path, "wb") as handle:
        handle.write(struct.pack(">II", 2049, 2))
        handle.write(labels.tobytes())
    assert np.array_equal(read_idx_images(image_path), images)
    assert np.array_equal(read_idx_labels(label_path), labels)


def test_bad_idx_rejected(tmp_path) -> None:
    path = tmp_path / "bad"
    path.write_bytes(struct.pack(">IIII", 123, 1, 28, 28) + bytes(784))
    with pytest.raises(ValueError):
        read_idx_images(path)


def test_stratified_split_reproducible_and_disjoint() -> None:
    labels = np.repeat(np.arange(10), np.arange(1, 11) + 20)
    train_a, val_a = stratified_split(labels, 30, 20261011)
    train_b, val_b = stratified_split(labels, 30, 20261011)
    assert np.array_equal(train_a, train_b)
    assert np.array_equal(val_a, val_b)
    assert len(val_a) == 30
    assert len(np.intersect1d(train_a, val_a)) == 0
    assert len(np.union1d(train_a, val_a)) == len(labels)

