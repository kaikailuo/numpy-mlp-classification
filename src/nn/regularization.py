from __future__ import annotations

import numpy as np

from .layers import Parameter


def regularization_loss(
    params: list[Parameter], *, l1_lambda: float = 0.0, l2_lambda: float = 0.0
) -> float:
    _validate(l1_lambda, l2_lambda)
    weights = [parameter.data for parameter in params if parameter.name.endswith(".weight")]
    l1 = l1_lambda * sum(float(np.abs(weight).sum()) for weight in weights)
    l2 = 0.5 * l2_lambda * sum(float(np.square(weight).sum()) for weight in weights)
    return float(l1 + l2)


def add_regularization_grads(
    params: list[Parameter], *, l1_lambda: float = 0.0, l2_lambda: float = 0.0
) -> None:
    _validate(l1_lambda, l2_lambda)
    for parameter in params:
        if not parameter.name.endswith(".weight"):
            continue
        if l1_lambda:
            parameter.grad += l1_lambda * np.sign(parameter.data)
        if l2_lambda:
            parameter.grad += l2_lambda * parameter.data


def _validate(l1_lambda: float, l2_lambda: float) -> None:
    if l1_lambda < 0 or l2_lambda < 0:
        raise ValueError("regularization strengths must be nonnegative")
