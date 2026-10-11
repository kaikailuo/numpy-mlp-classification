import numpy as np

from src.nn.losses import SoftmaxCrossEntropy
from src.nn.model import MLP


def test_v1_forward_and_backward_regression_fixture() -> None:
    x = np.asarray([[0.2, -0.3, 1.1], [-0.7, 0.5, 0.4]], dtype=np.float64)
    labels = np.asarray([1, 0])
    model = MLP([3, 4, 2], init_seed=17, dtype=np.float64)
    logits = model.forward(x, training=True)
    expected_logits = np.asarray([[-0.18423222, 0.52810251], [0.26582271, 0.04125420]])
    assert np.allclose(logits, expected_logits, atol=1e-8)
    value, gradient = SoftmaxCrossEntropy().value_and_grad(logits, labels)
    model.backward(gradient)
    assert np.isclose(value, 0.49313184, atol=1e-8)
    expected_first_weight_grad = np.asarray(
        [[-0.07384147, 0.02018544, 0.0, 0.06347004], [0.11076220, -0.03027816, 0.0, -0.05404929], [-0.40612807, 0.11101993, 0.0, 0.03106333]]
    )
    assert np.allclose(model.parameters()[0].grad, expected_first_weight_grad, atol=1e-8)
