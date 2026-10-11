from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np


def stratified_split(labels: np.ndarray, val_size: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if not 0 < val_size < len(labels):
        raise ValueError("val_size must be within dataset")
    classes, counts = np.unique(labels, return_counts=True)
    exact = counts * (val_size / len(labels))
    allocations = np.floor(exact).astype(int)
    order = np.argsort(-(exact - allocations), kind="stable")
    allocations[order[: val_size - int(allocations.sum())]] += 1
    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    for label, allocation in zip(classes, allocations):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        val_parts.append(indices[:allocation])
        train_parts.append(indices[allocation:])
    train_indices = np.concatenate(train_parts)
    val_indices = np.concatenate(val_parts)
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
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
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, train_indices=train_indices, val_indices=val_indices)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
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
