from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Parameter:
    name: str
    data: np.ndarray
    grad: np.ndarray

    def __post_init__(self) -> None:
        if self.data.shape != self.grad.shape:
            raise ValueError(f"gradient shape mismatch for {self.name}")


class Dense:
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        rng: np.random.Generator,
        name: str,
        initialization: str = "kaiming_normal",
        dtype: np.dtype = np.float32,
    ) -> None:
        if in_features < 1 or out_features < 1:
            raise ValueError("dense dimensions must be positive")
        dtype = np.dtype(dtype)
        if initialization == "kaiming_normal":
            std = np.sqrt(2.0 / in_features)
        elif initialization == "xavier_normal":
            std = np.sqrt(2.0 / (in_features + out_features))
        else:
            raise ValueError(f"unsupported initialization: {initialization}")
        weight = rng.normal(0.0, std, (in_features, out_features)).astype(dtype)
        bias = np.zeros(out_features, dtype=dtype)
        self.weight = Parameter(f"{name}.weight", weight, np.zeros_like(weight))
        self.bias = Parameter(f"{name}.bias", bias, np.zeros_like(bias))
        self._input: np.ndarray | None = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        if x.ndim != 2 or x.shape[1] != self.weight.data.shape[0]:
            raise ValueError(
                f"expected [B,{self.weight.data.shape[0]}], got {x.shape}"
            )
        if not np.isfinite(x).all():
            raise FloatingPointError("non-finite dense input")
        if training:
            self._input = x
        return x @ self.weight.data + self.bias.data

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._input is None:
            raise RuntimeError("Dense.backward called without training forward")
        if grad_output.ndim != 2 or grad_output.shape[1] != self.weight.data.shape[1]:
            raise ValueError("invalid dense output-gradient shape")
        if grad_output.shape[0] != self._input.shape[0]:
            raise ValueError("dense batch mismatch")
        self.weight.grad[...] = self._input.T @ grad_output
        self.bias.grad[...] = grad_output.sum(axis=0)
        grad_input = grad_output @ self.weight.data.T
        if not all(np.isfinite(v).all() for v in (self.weight.grad, self.bias.grad, grad_input)):
            raise FloatingPointError("non-finite dense gradient")
        return grad_input

    def parameters(self) -> list[Parameter]:
        return [self.weight, self.bias]


class ReLU:
    def __init__(self) -> None:
        self._mask: np.ndarray | None = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        if training:
            self._mask = x > 0
        return np.maximum(x, 0)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._mask is None:
            raise RuntimeError("ReLU.backward called without training forward")
        return grad_output * self._mask

    def parameters(self) -> list[Parameter]:
        return []


class Dropout:
    def __init__(self, drop_prob: float, rng: np.random.Generator) -> None:
        if not 0.0 <= drop_prob < 1.0:
            raise ValueError("drop_prob must be in [0, 1)")
        self.drop_prob = float(drop_prob)
        self.keep_prob = 1.0 - self.drop_prob
        self.rng = rng
        self._scaled_mask: np.ndarray | None = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        if not training or self.drop_prob == 0.0:
            if training:
                self._scaled_mask = np.ones_like(x)
            return x
        self._scaled_mask = (
            self.rng.random(x.shape) < self.keep_prob
        ).astype(x.dtype) / self.keep_prob
        return x * self._scaled_mask

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._scaled_mask is None:
            raise RuntimeError("Dropout.backward called without training forward")
        return grad_output * self._scaled_mask

    def parameters(self) -> list[Parameter]:
        return []

