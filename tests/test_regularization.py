import numpy as np

from a1_mlp.layers import Parameter
from a1_mlp.regularization import add_regularization_grads, regularization_loss


def test_l1_l2_loss_and_gradient_weights_only() -> None:
    weight = Parameter("layers.0.weight", np.asarray([[-2.0, 0.0, 3.0]]), np.zeros((1, 3)))
    bias = Parameter("layers.0.bias", np.asarray([2.0]), np.zeros(1))
    value = regularization_loss([weight, bias], l1_lambda=0.1, l2_lambda=0.2)
    assert np.isclose(value, 0.1 * 5 + 0.5 * 0.2 * 13)
    add_regularization_grads([weight, bias], l1_lambda=0.1, l2_lambda=0.2)
    assert np.allclose(weight.grad, [[-0.5, 0.0, 0.7]])
    assert np.array_equal(bias.grad, [0.0])

