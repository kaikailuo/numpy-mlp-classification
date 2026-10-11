from __future__ import annotations

import numpy as np


def accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    if predictions.shape != labels.shape:
        raise ValueError("prediction/label shape mismatch")
    return float(np.mean(predictions == labels))


def confusion_matrix(predictions: np.ndarray, labels: np.ndarray, classes: int = 10) -> np.ndarray:
    if predictions.shape != labels.shape:
        raise ValueError("prediction/label shape mismatch")
    matrix = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(matrix, (labels, predictions), 1)
    return matrix

