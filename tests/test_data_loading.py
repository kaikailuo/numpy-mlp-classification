import numpy as np

from src.data import datasets


def test_include_test_false_does_not_request_test_files(monkeypatch) -> None:
    calls = []

    def fake_raw(root, *, include_test=True):
        calls.append(include_test)
        x = np.zeros((20, 784), dtype=np.uint8)
        y = np.repeat(np.arange(10), 2).astype(np.uint8)
        return x, y, np.empty((0, 784), dtype=np.uint8), np.empty(0, dtype=np.uint8)

    def fake_split(dataset, labels, val_size, seed):
        return np.arange(10), np.arange(10, 20), {"seed": seed, "val_size": val_size, "train_size": 10}

    monkeypatch.setattr(datasets, "load_mnist_raw", fake_raw)
    monkeypatch.setattr(datasets, "load_or_create_split", fake_split)
    config = {
        "data": {"name": "mnist", "root": "unused", "val_size": 10, "split_seed": 1},
        "compute": {"dtype": "float32"},
    }
    bundle = datasets.load_dataset(config, include_test=False)
    assert calls == [False]
    assert bundle.test_x.shape == (0, 784)
    assert bundle.manifest["test_loaded"] is False
