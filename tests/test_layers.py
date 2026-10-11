import numpy as np
import pytest

from a1_mlp.layers import Dense, Dropout, ReLU


def test_dense_shapes_dtype_and_backward() -> None:
    layer = Dense(3, 2, rng=np.random.default_rng(1), name="x", dtype=np.float32)
    x = np.ones((4, 3), dtype=np.float32)
    output = layer.forward(x, training=True)
    input_grad = layer.backward(np.ones_like(output))
    assert output.shape == (4, 2)
    assert output.dtype == np.float32
    assert input_grad.shape == x.shape
    assert layer.weight.grad.shape == layer.weight.data.shape
    assert np.isfinite(input_grad).all()


def test_dense_rejects_bad_input() -> None:
    layer = Dense(3, 2, rng=np.random.default_rng(1), name="x")
    with pytest.raises(ValueError):
        layer.forward(np.ones((4, 4), dtype=np.float32))
    with pytest.raises(FloatingPointError):
        layer.forward(np.asarray([[1, np.nan, 2]], dtype=np.float32))


def test_relu_zero_derivative() -> None:
    layer = ReLU()
    value = np.asarray([[-1.0, 0.0, 2.0]])
    assert np.array_equal(layer.forward(value), [[0, 0, 2]])
    assert np.array_equal(layer.backward(np.ones_like(value)), [[0, 0, 1]])


def test_dropout_eval_identity_and_rng_unchanged() -> None:
    rng = np.random.default_rng(5)
    before = repr(rng.bit_generator.state)
    layer = Dropout(0.25, rng)
    x = np.ones((100, 10), dtype=np.float64)
    assert layer.forward(x, training=False) is x
    assert repr(rng.bit_generator.state) == before
    output = layer.forward(x, training=True)
    assert set(np.unique(output)).issubset({0.0, 1.0 / 0.75})
    assert np.array_equal(layer.backward(np.ones_like(x)), output)

