import numpy as np

from src.nn.losses import SoftmaxCrossEntropy
from src.nn.model import MLP
from src.optim.optimizers import SGD


def test_tiny_set_reaches_full_accuracy() -> None:
    rng = np.random.default_rng(7)
    centers = np.asarray([[-2, -2], [2, -2], [0, 2]], dtype=np.float64)
    x = np.concatenate([center + rng.normal(0, 0.15, (8, 2)) for center in centers])
    y = np.repeat(np.arange(3), 8)
    model = MLP([2, 12, 3], init_seed=42, dtype=np.float64)
    loss = SoftmaxCrossEntropy()
    optimizer = SGD(0.1)
    initial = None
    for _ in range(500):
        model.zero_grad()
        value, gradient = loss.value_and_grad(model.forward(x, training=True), y)
        initial = value if initial is None else initial
        model.backward(gradient)
        optimizer.step(model.parameters())
    final, _ = loss.value_and_grad(model.forward(x), y)
    assert final < initial * 0.05
    assert np.mean(model.predict(x) == y) == 1.0

