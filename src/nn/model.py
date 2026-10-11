from __future__ import annotations

import numpy as np

from .layers import Dense, Dropout, Parameter, ReLU
from .losses import softmax


class MLP:
    def __init__(
        self,
        dims: list[int] | tuple[int, ...],
        *,
        init_seed: int = 42,
        dropout_seed: int = 44,
        dropout: float = 0.0,
        hidden_activation: str = "relu",
        hidden_init: str = "kaiming_normal",
        output_init: str = "xavier_normal",
        dtype: np.dtype = np.float32,
    ) -> None:
        if len(dims) < 2 or any(int(d) <= 0 for d in dims):
            raise ValueError("dims must contain at least two positive dimensions")
        if hidden_activation != "relu":
            raise ValueError(f"unsupported hidden activation: {hidden_activation}")
        self.dims = tuple(int(d) for d in dims)
        self.dtype = np.dtype(dtype)
        self.hidden_activation = hidden_activation
        self.hidden_init = hidden_init
        self.output_init = output_init
        self.dropout = float(dropout)
        init_rng = np.random.default_rng(init_seed)
        dropout_rng = np.random.default_rng(dropout_seed)
        layers: list[object] = []
        for index, (in_dim, out_dim) in enumerate(zip(self.dims[:-1], self.dims[1:])):
            is_output = index == len(self.dims) - 2
            layers.append(
                Dense(
                    in_dim,
                    out_dim,
                    rng=init_rng,
                    name=f"layers.{index}",
                    initialization=output_init if is_output else hidden_init,
                    dtype=self.dtype,
                )
            )
            if not is_output:
                layers.append(ReLU())
                if dropout > 0:
                    layers.append(Dropout(dropout, dropout_rng))
        self.layers = layers

    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        value = x.astype(self.dtype, copy=False) if x.dtype != self.dtype else x
        for layer in self.layers:
            value = layer.forward(value, training=training)
        if not np.isfinite(value).all():
            raise FloatingPointError("non-finite model output")
        return value

    def backward(self, grad_logits: np.ndarray) -> np.ndarray:
        value = grad_logits
        for layer in reversed(self.layers):
            value = layer.backward(value)
        return value

    def parameters(self) -> list[Parameter]:
        params: list[Parameter] = []
        for layer in self.layers:
            params.extend(layer.parameters())
        names = [parameter.name for parameter in params]
        if len(names) != len(set(names)):
            raise RuntimeError("parameter names are not unique")
        return params

    def zero_grad(self) -> None:
        for parameter in self.parameters():
            parameter.grad.fill(0)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return softmax(self.forward(x, training=False))

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self.forward(x, training=False), axis=1)

    @property
    def parameter_count(self) -> int:
        return int(sum(parameter.data.size for parameter in self.parameters()))
