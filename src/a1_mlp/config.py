from __future__ import annotations

import hashlib
import json
import tomllib
from copy import deepcopy
from pathlib import Path
from typing import Any


REQUIRED_SECTIONS = ("run", "data", "model", "loss", "optimizer", "training", "compute")


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("rb") as handle:
        config = tomllib.load(handle)
    validate_config(config)
    config = deepcopy(config)
    config["_source_path"] = str(path.resolve())
    config["_config_hash"] = config_hash(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    missing = [section for section in REQUIRED_SECTIONS if section not in config]
    if missing:
        raise ValueError(f"missing config sections: {missing}")
    dims = config["model"].get("dims")
    if not isinstance(dims, list) or len(dims) < 2 or any(not isinstance(d, int) or d <= 0 for d in dims):
        raise ValueError("model.dims must be a list of positive integers")
    dataset = config["data"].get("name")
    input_dims = {"mnist": 784, "cifar10": 3072, "toy": 2}
    if dataset not in input_dims or dims[0] != input_dims[dataset] or dims[-1] != 10:
        raise ValueError("model dimensions do not match dataset and 10-class output")
    if config["loss"].get("name") not in {"cross_entropy", "softmax_mse"}:
        raise ValueError("invalid loss.name")
    if config["optimizer"].get("name") not in {"sgd", "momentum", "adagrad", "adam", "rmsprop"}:
        raise ValueError("invalid optimizer.name")
    if float(config["optimizer"].get("learning_rate", 0)) <= 0:
        raise ValueError("optimizer.learning_rate must be positive")
    training = config["training"]
    if int(training.get("epochs", 0)) < 1 or int(training.get("batch_size", 0)) < 1:
        raise ValueError("training epochs and batch_size must be positive")
    dropout = float(config["model"].get("dropout", 0.0))
    if not 0 <= dropout < 1:
        raise ValueError("model.dropout must be in [0,1)")
    for field in ("l1_lambda", "l2_lambda"):
        if float(config["loss"].get(field, 0.0)) < 0:
            raise ValueError(f"loss.{field} must be nonnegative")
    if config["compute"].get("dtype") not in {"float32", "float64"}:
        raise ValueError("compute.dtype must be float32 or float64")
    if int(config["data"].get("val_size", 0)) < 1:
        raise ValueError("data.val_size must be positive")


def public_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if not key.startswith("_")}


def config_hash(config: dict[str, Any]) -> str:
    encoded = json.dumps(public_config(config), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()

