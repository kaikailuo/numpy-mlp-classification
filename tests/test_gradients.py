import numpy as np

from a1_mlp.layers import Dense
from a1_mlp.losses import SoftmaxCrossEntropy
from a1_mlp.model import MLP


def relative_error(a, b):
    return abs(a - b) / max(1e-8, abs(a) + abs(b))


def test_dense_gradient_finite_difference() -> None:
    rng = np.random.default_rng(2)
    layer = Dense(3, 2, rng=rng, name="d", dtype=np.float64)
    x = rng.normal(size=(4, 3))
    upstream = rng.normal(size=(4, 2))
    layer.forward(x, training=True)
    layer.backward(upstream)
    analytic = layer.weight.grad.copy()
    h = 1e-5
    for index in np.ndindex(layer.weight.data.shape):
        original = layer.weight.data[index]
        layer.weight.data[index] = original + h
        plus = np.sum(layer.forward(x, training=False) * upstream)
        layer.weight.data[index] = original - h
        minus = np.sum(layer.forward(x, training=False) * upstream)
        layer.weight.data[index] = original
        numeric = (plus - minus) / (2 * h)
        assert relative_error(numeric, analytic[index]) <= 1e-4 or abs(numeric - analytic[index]) <= 1e-7


def test_full_mlp_gradient_finite_difference() -> None:
    x = np.asarray([[-1.2, 0.7], [0.3, 1.4], [1.2, -0.8]], dtype=np.float64)
    labels = np.asarray([0, 2, 1])
    model = MLP([2, 3, 3], init_seed=13, dtype=np.float64)
    loss = SoftmaxCrossEntropy()
    model.zero_grad()
    value, gradient = loss.value_and_grad(model.forward(x, training=True), labels)
    assert np.isfinite(value)
    model.backward(gradient)
    h = 1e-5
    for parameter in model.parameters():
        analytic = parameter.grad.copy()
        selections = list(np.ndindex(parameter.data.shape))[: min(5, parameter.data.size)]
        for index in selections:
            original = parameter.data[index]
            parameter.data[index] = original + h
            plus = loss.value_and_grad(model.forward(x, training=False), labels)[0]
            parameter.data[index] = original - h
            minus = loss.value_and_grad(model.forward(x, training=False), labels)[0]
            parameter.data[index] = original
            numeric = (plus - minus) / (2 * h)
            assert relative_error(numeric, analytic[index]) <= 1e-4 or abs(numeric - analytic[index]) <= 1e-7

