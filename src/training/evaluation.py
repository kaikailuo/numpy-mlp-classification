from __future__ import annotations

import numpy as np

from src.data.datasets import batches
from src.nn.losses import SoftmaxCrossEntropy
from src.nn.model import MLP

from .metrics import accuracy


def evaluate_arrays(
    model: MLP,
    x: np.ndarray,
    y: np.ndarray,
    preprocess,
    *,
    batch_size: int,
) -> dict:
    if len(x) == 0:
        raise ValueError("cannot evaluate an empty split")
    ce = SoftmaxCrossEntropy()
    total_loss = 0.0
    predictions: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for raw_x, batch_y in batches(x, y, batch_size):
        logits = model.forward(preprocess(raw_x), training=False)
        loss, _ = ce.value_and_grad(logits, batch_y)
        total_loss += loss * len(batch_y)
        predictions.append(np.argmax(logits, axis=1))
        labels.append(batch_y)
    predicted = np.concatenate(predictions)
    actual = np.concatenate(labels)
    return {
        "ce": total_loss / len(y),
        "accuracy": accuracy(predicted, actual),
        "predictions": predicted,
        "labels": actual,
    }
