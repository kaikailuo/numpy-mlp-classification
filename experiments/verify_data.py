from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from a1_mlp.data import CIFAR_MD5, MNIST_FILES, file_md5, load_dataset
from a1_mlp.reporting import write_json


def config(dataset: str) -> dict:
    return {
        "data": {
            "name": dataset,
            "root": f"data/raw/{'cifar10' if dataset == 'cifar10' else 'mnist'}",
            "split_seed": 20261011,
            "val_size": 5000,
        },
        "compute": {"dtype": "float32"},
    }


def verify(dataset: str) -> dict:
    bundle = load_dataset(config(dataset))
    expected_train = 45000 if dataset == "cifar10" else 55000
    expected_dim = 3072 if dataset == "cifar10" else 784
    assert bundle.train_x.shape == (expected_train, expected_dim)
    assert bundle.val_x.shape == (5000, expected_dim)
    assert bundle.test_x.shape == (10000, expected_dim)
    assert bundle.train_y.min() == bundle.val_y.min() == bundle.test_y.min() == 0
    assert bundle.train_y.max() == bundle.val_y.max() == bundle.test_y.max() == 9
    split_file = Path("data/splits") / f"{dataset}_split_v1.npz"
    with np.load(split_file, allow_pickle=False) as archive:
        train_indices = archive["train_indices"]
        val_indices = archive["val_indices"]
    assert len(np.intersect1d(train_indices, val_indices)) == 0
    assert len(np.union1d(train_indices, val_indices)) == expected_train + 5000
    if dataset == "cifar10":
        checksums = {"cifar-10-python.tar.gz": file_md5(Path("data/raw/cifar10/cifar-10-python.tar.gz"))}
        assert next(iter(checksums.values())) == CIFAR_MD5
    else:
        checksums = {
            filename: file_md5(Path("data/raw/mnist") / filename)
            for filename in MNIST_FILES
        }
        assert all(checksums[name] == spec[1] for name, spec in MNIST_FILES.items())
    probe = bundle.preprocess(bundle.train_x[:128])
    assert probe.dtype == np.float32 and np.isfinite(probe).all()
    return {
        "dataset": dataset,
        "status": "passed",
        "shapes": {
            "train": list(bundle.train_x.shape),
            "validation": list(bundle.val_x.shape),
            "test": list(bundle.test_x.shape),
        },
        "checksums": checksums,
        "split": bundle.manifest["split"],
        "preprocess": bundle.manifest["preprocess"],
        "probe": {"dtype": str(probe.dtype), "finite": bool(np.isfinite(probe).all())},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = {dataset: verify(dataset) for dataset in ("mnist", "cifar10")}
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

