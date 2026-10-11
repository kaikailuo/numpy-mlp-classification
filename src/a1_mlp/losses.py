from __future__ import annotations

import numpy as np


def _validate(logits: np.ndarray, y: np.ndarray) -> None:
    if logits.ndim != 2 or logits.shape[0] == 0:
        raise ValueError("logits must be a non-empty [B,C] array")
    if y.ndim != 1 or len(y) != len(logits):
        raise ValueError("labels must have shape [B]")
    if not np.issubdtype(y.dtype, np.integer):
        raise TypeError("labels must be integers")
    if np.any(y < 0) or np.any(y >= logits.shape[1]):
        raise ValueError("label outside class range")
    if not np.isfinite(logits).all():
        raise FloatingPointError("non-finite logits")


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exps = np.exp(shifted)
    return exps / np.sum(exps, axis=1, keepdims=True)


class SoftmaxCrossEntropy:
    name = "cross_entropy"

    def value_and_grad(
        self, logits: np.ndarray, y: np.ndarray
    ) -> tuple[float, np.ndarray]:
        _validate(logits, y)
        maximum = np.max(logits, axis=1)
        shifted = logits - maximum[:, None]
        log_partition = maximum + np.log(np.exp(shifted).sum(axis=1))
        loss = np.mean(log_partition - logits[np.arange(len(y)), y])
        probabilities = np.exp(logits - log_partition[:, None])
        probabilities[np.arange(len(y)), y] -= 1.0
        probabilities /= len(y)
        return float(loss), probabilities


class SoftmaxMSE:
    name = "softmax_mse"

    def value_and_grad(
        self, logits: np.ndarray, y: np.ndarray
    ) -> tuple[float, np.ndarray]:
        _validate(logits, y)
        probabilities = softmax(logits)
        difference = probabilities.copy()
        difference[np.arange(len(y)), y] -= 1.0
        loss = 0.5 * np.sum(difference * difference) / len(y)
        grad_probability = difference / len(y)
        projection = np.sum(grad_probability * probabilities, axis=1, keepdims=True)
        grad_logits = probabilities * (grad_probability - projection)
        return float(loss), grad_logits


def make_loss(name: str) -> SoftmaxCrossEntropy | SoftmaxMSE:
    if name == "cross_entropy":
        return SoftmaxCrossEntropy()
    if name == "softmax_mse":
        return SoftmaxMSE()
    raise ValueError(f"unsupported loss: {name}")

