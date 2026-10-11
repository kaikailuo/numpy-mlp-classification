import numpy as np

from src.nn.losses import SoftmaxCrossEntropy, SoftmaxMSE, softmax


def finite_difference(loss, logits, labels, index, h=1e-5):
    plus = logits.copy()
    minus = logits.copy()
    plus[index] += h
    minus[index] -= h
    return (loss.value_and_grad(plus, labels)[0] - loss.value_and_grad(minus, labels)[0]) / (2 * h)


def test_stable_cross_entropy_extreme_logits() -> None:
    logits = np.asarray([[1000.0, -1000.0, 0.0], [-1000.0, 1000.0, 0.0]])
    labels = np.asarray([0, 2])
    loss, grad = SoftmaxCrossEntropy().value_and_grad(logits, labels)
    assert np.isfinite(loss)
    assert np.isfinite(grad).all()
    assert np.allclose(softmax(logits).sum(axis=1), 1.0)


def test_float32_large_tied_logits_preserve_log_two() -> None:
    logits = np.asarray([[1e20, 1e20]], dtype=np.float32)
    value, gradient = SoftmaxCrossEntropy().value_and_grad(logits, np.asarray([0]))
    assert np.isclose(value, np.log(2.0), rtol=1e-6)
    assert np.allclose(gradient, [[-0.5, 0.5]])


def test_cross_entropy_gradient() -> None:
    rng = np.random.default_rng(4)
    logits = rng.normal(size=(4, 3))
    labels = np.asarray([0, 2, 1, 0])
    loss = SoftmaxCrossEntropy()
    _, analytic = loss.value_and_grad(logits, labels)
    for index in np.ndindex(logits.shape):
        assert np.isclose(analytic[index], finite_difference(loss, logits, labels, index), atol=1e-7, rtol=1e-4)


def test_softmax_mse_gradient() -> None:
    rng = np.random.default_rng(8)
    logits = rng.normal(size=(3, 4))
    labels = np.asarray([0, 3, 1])
    loss = SoftmaxMSE()
    _, analytic = loss.value_and_grad(logits, labels)
    for index in np.ndindex(logits.shape):
        assert np.isclose(analytic[index], finite_difference(loss, logits, labels, index), atol=1e-7, rtol=1e-4)

