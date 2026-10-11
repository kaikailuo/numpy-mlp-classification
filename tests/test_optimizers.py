import numpy as np

from a1_mlp.layers import Parameter
from a1_mlp.optimizers import AdaGrad, Adam, Momentum, SGD


def parameter(name="p"):
    return Parameter(name, np.asarray([1.0, -2.0]), np.asarray([0.5, -0.25]))


def test_sgd_exact_update() -> None:
    p = parameter()
    SGD(0.1).step([p])
    assert np.allclose(p.data, [0.95, -1.975])


def test_momentum_two_steps() -> None:
    p = parameter()
    optimizer = Momentum(0.1, momentum=0.9)
    optimizer.step([p])
    optimizer.step([p])
    assert np.allclose(p.data, [0.855, -1.9275])


def test_adagrad_first_step() -> None:
    p = parameter()
    AdaGrad(0.1, eps=0).step([p])
    assert np.allclose(p.data, [0.9, -1.9])


def test_adam_first_step_bias_correction() -> None:
    p = parameter()
    optimizer = Adam(0.1, eps=0)
    optimizer.step([p])
    assert np.allclose(p.data, [0.9, -1.9])
    assert optimizer.step_count == 1


def test_optimizer_state_is_independent_by_name() -> None:
    first = parameter("a")
    second = parameter("b")
    optimizer = Momentum(0.1)
    optimizer.step([first, second])
    assert optimizer.velocity["a"] is not optimizer.velocity["b"]

