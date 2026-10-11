from __future__ import annotations

from collections.abc import Callable

import numpy as np


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


def build_preprocessor(
    dataset: str,
    dtype: np.dtype,
    train_x: np.ndarray,
    *,
    frozen_spec: dict | None = None,
) -> tuple[Callable[[np.ndarray], np.ndarray], dict]:
    dtype = np.dtype(dtype)
    if dataset == "mnist":
        spec = frozen_spec or {"name": "scale01", "fit_split": "none"}
        if spec.get("name") != "scale01":
            raise ValueError("MNIST checkpoint preprocessing is incompatible")

        def preprocess(batch: np.ndarray) -> np.ndarray:
            return batch.astype(dtype) / dtype.type(255.0)

        return preprocess, spec
    if frozen_spec is None:
        mean, std = _cifar_channel_stats(train_x)
        spec = {
            "name": "channel_standardize",
            "fit_split": "effective_train",
            "mean": mean.tolist(),
            "std": std.tolist(),
        }
    else:
        if frozen_spec.get("name") != "channel_standardize":
            raise ValueError("CIFAR-10 checkpoint preprocessing is incompatible")
        spec = frozen_spec
        mean = np.asarray(spec["mean"], dtype=np.float32)
        std = np.asarray(spec["std"], dtype=np.float32)
    safe_std = np.maximum(std, 1e-6)

    def preprocess(batch: np.ndarray) -> np.ndarray:
        shaped = batch.astype(dtype).reshape(-1, 3, 1024)
        shaped = (shaped - mean.astype(dtype)[None, :, None]) / safe_std.astype(dtype)[None, :, None]
        return shaped.reshape(-1, 3072)

    return preprocess, spec
