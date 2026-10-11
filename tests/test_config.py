from copy import deepcopy

import pytest

from src.config import validate_config


def valid_config():
    return {
        "run": {"name": "x", "output_root": "results"},
        "data": {"name": "mnist", "root": "data/raw/mnist", "val_size": 5000, "split_seed": 1, "preprocess": "scale01"},
        "model": {"dims": [784, 8, 10], "hidden_activation": "relu", "hidden_init": "kaiming_normal", "output_init": "xavier_normal", "dropout": 0.0, "init_seed": 1},
        "loss": {"name": "cross_entropy", "l1_lambda": 0.0, "l2_lambda": 0.0},
        "optimizer": {"name": "sgd", "learning_rate": 0.1},
        "training": {"epochs": 1, "batch_size": 4, "shuffle_seed": 2},
        "compute": {"dtype": "float32", "backend": "cpu"},
    }


def test_supported_model_configuration_is_accepted() -> None:
    validate_config(valid_config())


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [("model", "hidden_activation", "sigmoid"), ("model", "hidden_init", "zeros"), ("optimizer", "name", "magic"), ("loss", "name", "unknown")],
)
def test_unsupported_configuration_is_rejected(section, key, value) -> None:
    config = deepcopy(valid_config())
    config[section][key] = value
    with pytest.raises(ValueError):
        validate_config(config)


def test_unknown_optimizer_option_is_rejected() -> None:
    config = valid_config()
    config["optimizer"]["unused"] = 1
    with pytest.raises(ValueError, match="unsupported optimizer"):
        validate_config(config)
