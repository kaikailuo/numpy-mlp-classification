import numpy as np
import pytest

from src.nn.losses import FocalLoss, LabelSmoothingCrossEntropy, MulticlassHingeLoss


def finite_difference(loss, logits, labels, index, h=1e-5):
    plus = logits.copy()
    minus = logits.copy()
    plus[index] += h
    minus[index] -= h
    return (loss.value_and_grad(plus, labels)[0] - loss.value_and_grad(minus, labels)[0]) / (2 * h)


@pytest.mark.parametrize(
    "loss",
    [LabelSmoothingCrossEntropy(0.1), MulticlassHingeLoss(1.0), FocalLoss(2.0, 1.0)],
)
def test_extended_loss_gradients(loss) -> None:
    logits = np.asarray([[0.1, 1.3, -0.7], [1.2, -0.4, 0.3]], dtype=np.float64)
    labels = np.asarray([1, 2])
    value, analytic = loss.value_and_grad(logits, labels)
    assert np.isfinite(value)
    for index in np.ndindex(logits.shape):
        numeric = finite_difference(loss, logits, labels, index)
        assert np.isclose(analytic[index], numeric, atol=1e-7, rtol=1e-4)


@pytest.mark.parametrize("loss", [LabelSmoothingCrossEntropy(0.1), FocalLoss(2.0)])
def test_exponential_losses_are_stable_at_extreme_logits(loss) -> None:
    logits = np.asarray([[1e20, 1e20, -1e20], [-1e20, 1e20, 0]], dtype=np.float32)
    value, gradient = loss.value_and_grad(logits, np.asarray([0, 2]))
    assert np.isfinite(value)
    assert np.isfinite(gradient).all()
