import numpy as np

from src.data.datasets import DatasetBundle
from src.nn.losses import SoftmaxCrossEntropy
from src.nn.model import MLP
from src.optim.optimizers import Adam
from src.training.checkpoints import load_checkpoint
from src.training.trainer import Trainer


def test_lightweight_end_to_end_training_and_checkpoint(tmp_path) -> None:
    rng = np.random.default_rng(9)
    centers = np.asarray([[-2, -2], [2, -2], [0, 2]], dtype=np.float64)
    x = np.concatenate([center + rng.normal(0, 0.2, (10, 2)) for center in centers])
    y = np.repeat(np.arange(3), 10)
    bundle = DatasetBundle(
        train_x=x,
        train_y=y,
        val_x=x.copy(),
        val_y=y.copy(),
        test_x=np.empty((0, 2)),
        test_y=np.empty(0, dtype=int),
        preprocess=lambda value: value,
        manifest={"split": {"seed": 1, "val_size": 30}, "preprocess": {"name": "identity"}},
    )
    config = {
        "run": {"name": "integration", "output_root": str(tmp_path)},
        "data": {"name": "toy", "root": "none"},
        "model": {"dims": [2, 8, 3], "init_seed": 2, "dropout_seed": 3, "dropout": 0.0, "hidden_activation": "relu", "hidden_init": "kaiming_normal", "output_init": "xavier_normal"},
        "loss": {"name": "cross_entropy", "l1_lambda": 0.0, "l2_lambda": 0.0},
        "optimizer": {"name": "adam", "learning_rate": 0.03},
        "training": {"epochs": 15, "batch_size": 10, "shuffle_seed": 4},
        "compute": {"dtype": "float64", "backend": "cpu"},
        "_config_hash": "integration-test",
    }
    model = MLP([2, 8, 3], init_seed=2, dtype=np.float64)
    result = Trainer().fit(model, bundle, SoftmaxCrossEntropy(), Adam(0.03), config, tmp_path / "best.npz")
    restored, metadata = load_checkpoint(tmp_path / "best.npz")
    assert result.best_val_accuracy >= 0.95
    assert np.mean(restored.predict(x) == y) >= 0.95
    assert metadata["data_manifest"]["preprocess"]["name"] == "identity"
