import numpy as np

from src.nn.layers import Parameter
from src.nn.regularization import add_regularization_grads, regularization_loss


def test_regularization_gradient_matches_finite_difference_away_from_l1_zero() -> None:
    values = np.asarray([[-1.2, 0.7], [2.1, -0.4]], dtype=np.float64)
    parameter = Parameter("layers.0.weight", values.copy(), np.zeros_like(values))
    add_regularization_grads([parameter], l1_lambda=0.03, l2_lambda=0.07)
    analytic = parameter.grad.copy()
    h = 1e-6
    for index in np.ndindex(values.shape):
        original = parameter.data[index]
        parameter.data[index] = original + h
        plus = regularization_loss([parameter], l1_lambda=0.03, l2_lambda=0.07)
        parameter.data[index] = original - h
        minus = regularization_loss([parameter], l1_lambda=0.03, l2_lambda=0.07)
        parameter.data[index] = original
        assert np.isclose(analytic[index], (plus - minus) / (2 * h), atol=1e-8)
