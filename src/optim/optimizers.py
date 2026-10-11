from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.nn.layers import Parameter


def _check(params: list[Parameter]) -> None:
    for parameter in params:
        if parameter.grad.shape != parameter.data.shape:
            raise ValueError(f"gradient shape mismatch: {parameter.name}")
        if not np.isfinite(parameter.grad).all():
            raise FloatingPointError(f"non-finite gradient: {parameter.name}")


@dataclass
class SGD:
    learning_rate: float

    def step(self, params: list[Parameter]) -> None:
        _check(params)
        for parameter in params:
            parameter.data -= self.learning_rate * parameter.grad


@dataclass
class Momentum:
    learning_rate: float
    momentum: float = 0.9
    velocity: dict[str, np.ndarray] = field(default_factory=dict)

    def step(self, params: list[Parameter]) -> None:
        _check(params)
        for parameter in params:
            velocity = self.velocity.setdefault(parameter.name, np.zeros_like(parameter.data))
            velocity *= self.momentum
            velocity += parameter.grad
            parameter.data -= self.learning_rate * velocity


@dataclass
class NesterovMomentum:
    learning_rate: float
    momentum: float = 0.9
    velocity: dict[str, np.ndarray] = field(default_factory=dict)

    def step(self, params: list[Parameter]) -> None:
        """Nesterov's accelerated-gradient buffer formulation.

        b_t = mu b_(t-1) + g_t; theta_t = theta_(t-1) - lr (g_t + mu b_t).
        This is deliberately different from ordinary Momentum's lr*b_t update.
        """
        _check(params)
        for parameter in params:
            velocity = self.velocity.setdefault(parameter.name, np.zeros_like(parameter.data))
            velocity *= self.momentum
            velocity += parameter.grad
            parameter.data -= self.learning_rate * (parameter.grad + self.momentum * velocity)


@dataclass
class AdaGrad:
    learning_rate: float
    eps: float = 1e-8
    squared: dict[str, np.ndarray] = field(default_factory=dict)

    def step(self, params: list[Parameter]) -> None:
        _check(params)
        for parameter in params:
            squared = self.squared.setdefault(parameter.name, np.zeros_like(parameter.data))
            squared += parameter.grad * parameter.grad
            parameter.data -= self.learning_rate * parameter.grad / (np.sqrt(squared) + self.eps)


@dataclass
class RMSProp:
    learning_rate: float
    rho: float = 0.9
    eps: float = 1e-8
    squared: dict[str, np.ndarray] = field(default_factory=dict)

    def step(self, params: list[Parameter]) -> None:
        _check(params)
        for parameter in params:
            squared = self.squared.setdefault(parameter.name, np.zeros_like(parameter.data))
            squared *= self.rho
            squared += (1.0 - self.rho) * parameter.grad * parameter.grad
            parameter.data -= self.learning_rate * parameter.grad / (np.sqrt(squared) + self.eps)


@dataclass
class Adam:
    learning_rate: float
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    first: dict[str, np.ndarray] = field(default_factory=dict)
    second: dict[str, np.ndarray] = field(default_factory=dict)
    step_count: int = 0

    def step(self, params: list[Parameter]) -> None:
        _check(params)
        self.step_count += 1
        for parameter in params:
            first = self.first.setdefault(parameter.name, np.zeros_like(parameter.data))
            second = self.second.setdefault(parameter.name, np.zeros_like(parameter.data))
            first *= self.beta1
            first += (1.0 - self.beta1) * parameter.grad
            second *= self.beta2
            second += (1.0 - self.beta2) * parameter.grad * parameter.grad
            first_hat = first / (1.0 - self.beta1**self.step_count)
            second_hat = second / (1.0 - self.beta2**self.step_count)
            parameter.data -= self.learning_rate * first_hat / (np.sqrt(second_hat) + self.eps)


@dataclass
class Adamax:
    learning_rate: float
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    first: dict[str, np.ndarray] = field(default_factory=dict)
    infinity_norm: dict[str, np.ndarray] = field(default_factory=dict)
    step_count: int = 0

    def step(self, params: list[Parameter]) -> None:
        _check(params)
        self.step_count += 1
        scale = self.learning_rate / (1.0 - self.beta1**self.step_count)
        for parameter in params:
            first = self.first.setdefault(parameter.name, np.zeros_like(parameter.data))
            norm = self.infinity_norm.setdefault(parameter.name, np.zeros_like(parameter.data))
            first *= self.beta1
            first += (1.0 - self.beta1) * parameter.grad
            norm[...] = np.maximum(self.beta2 * norm, np.abs(parameter.grad))
            parameter.data -= scale * first / (norm + self.eps)


def make_optimizer(name: str, learning_rate: float, options: dict[str, float] | None = None):
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    classes = {
        "sgd": SGD,
        "momentum": Momentum,
        "nesterov": NesterovMomentum,
        "adagrad": AdaGrad,
        "rmsprop": RMSProp,
        "adam": Adam,
        "adamax": Adamax,
    }
    if name not in classes:
        raise ValueError(f"unsupported optimizer: {name}")
    return classes[name](learning_rate=learning_rate, **(options or {}))
