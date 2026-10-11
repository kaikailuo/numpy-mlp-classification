import numpy as np
import pytest

from a1_mlp.checkpoints import load_checkpoint, save_checkpoint
from a1_mlp.model import MLP


def metadata():
    return {
        "model": {"dims": [2, 4, 3], "init_seed": 2, "dropout_seed": 3, "dropout": 0.0, "dtype": "float64"},
        "dataset": "toy",
    }


def test_checkpoint_roundtrip(tmp_path) -> None:
    model = MLP([2, 4, 3], init_seed=2, dropout_seed=3, dtype=np.float64)
    x = np.asarray([[1.0, 2.0]])
    expected = model.forward(x)
    path = tmp_path / "model.npz"
    save_checkpoint(path, model, metadata())
    restored, restored_metadata = load_checkpoint(path)
    assert restored_metadata == metadata()
    assert np.allclose(restored.forward(x), expected)


def test_invalid_checkpoint_rejected(tmp_path) -> None:
    path = tmp_path / "bad.npz"
    np.savez(path, x=np.ones(1))
    with pytest.raises(ValueError):
        load_checkpoint(path)

