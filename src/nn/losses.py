from __future__ import annotations

from typing import Protocol

import numpy as np


class Loss(Protocol):
    name: str

    def value_and_grad(self, logits: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]: ...


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


def _shifted(logits: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exps = np.exp(shifted)
    partition = np.sum(exps, axis=1, keepdims=True)
    return shifted, exps, partition


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted, exps, partition = _shifted(logits)
    del shifted
    return exps / partition


class SoftmaxCrossEntropy:
    name = "cross_entropy"

    def value_and_grad(self, logits: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
        _validate(logits, y)
        shifted, exps, partition = _shifted(logits)
        log_partition_shifted = np.log(partition[:, 0])
        # Compute entirely in translated coordinates. This avoids cancellation of
        # two large float32 values (logsumexp(original) - original_true_logit).
        losses = log_partition_shifted - shifted[np.arange(len(y)), y]
        probabilities = exps / partition
        probabilities[np.arange(len(y)), y] -= 1.0
        probabilities /= len(y)
        return float(np.mean(losses)), probabilities


class LabelSmoothingCrossEntropy:
    name = "label_smoothing_cross_entropy"

    def __init__(self, smoothing: float = 0.1) -> None:
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing must be in [0,1)")
        self.smoothing = float(smoothing)

    def value_and_grad(self, logits: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
        _validate(logits, y)
        shifted, exps, partition = _shifted(logits)
        log_probabilities = shifted - np.log(partition)
        classes = logits.shape[1]
        targets = np.full_like(logits, self.smoothing / classes)
        targets[np.arange(len(y)), y] += 1.0 - self.smoothing
        loss = -np.sum(targets * log_probabilities) / len(y)
        gradient = (exps / partition - targets) / len(y)
        return float(loss), gradient


class SoftmaxMSE:
    name = "softmax_mse"

    def value_and_grad(self, logits: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
        _validate(logits, y)
        probabilities = softmax(logits)
        difference = probabilities.copy()
        difference[np.arange(len(y)), y] -= 1.0
        loss = 0.5 * np.sum(difference * difference) / len(y)
        grad_probability = difference / len(y)
        projection = np.sum(grad_probability * probabilities, axis=1, keepdims=True)
        return float(loss), probabilities * (grad_probability - projection)


class MulticlassHingeLoss:
    name = "multiclass_hinge"

    def __init__(self, margin: float = 1.0) -> None:
        if margin <= 0:
            raise ValueError("margin must be positive")
        self.margin = float(margin)

    def value_and_grad(self, logits: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
        _validate(logits, y)
        rows = np.arange(len(y))
        correct = logits[rows, y]
        margins = np.maximum(0.0, logits - correct[:, None] + self.margin)
        margins[rows, y] = 0.0
        active = margins > 0
        gradient = active.astype(logits.dtype)
        gradient[rows, y] = -np.sum(active, axis=1)
        gradient /= len(y)
        return float(np.sum(margins) / len(y)), gradient


class FocalLoss:
    name = "focal"

    def __init__(self, gamma: float = 2.0, alpha: float = 1.0) -> None:
        if gamma < 0:
            raise ValueError("gamma must be nonnegative")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        self.gamma = float(gamma)
        self.alpha = float(alpha)

    def value_and_grad(self, logits: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
        _validate(logits, y)
        shifted, exps, partition = _shifted(logits)
        probabilities = exps / partition
        rows = np.arange(len(y))
        log_pt = shifted[rows, y] - np.log(partition[:, 0])
        pt = np.exp(log_pt)
        one_minus = 1.0 - pt
        modulator = np.power(one_minus, self.gamma)
        loss = -self.alpha * np.mean(modulator * log_pt)
        if self.gamma == 0.0:
            factor = np.full_like(pt, self.alpha)
        else:
            factor = self.alpha * (
                modulator - self.gamma * pt * np.power(one_minus, self.gamma - 1.0) * log_pt
            )
        gradient = probabilities.copy()
        gradient[rows, y] -= 1.0
        gradient *= factor[:, None] / len(y)
        return float(loss), gradient


def make_loss(name: str, options: dict[str, float] | None = None) -> Loss:
    options = options or {}
    classes = {
        "cross_entropy": SoftmaxCrossEntropy,
        "softmax_mse": SoftmaxMSE,
        "label_smoothing_cross_entropy": LabelSmoothingCrossEntropy,
        "multiclass_hinge": MulticlassHingeLoss,
        "focal": FocalLoss,
    }
    if name not in classes:
        raise ValueError(f"unsupported loss: {name}")
    return classes[name](**options)
