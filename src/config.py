from __future__ import annotations

import hashlib
import json
import tomllib
from copy import deepcopy
from pathlib import Path
from typing import Any


REQUIRED_SECTIONS = ("run", "data", "model", "loss", "optimizer", "training", "compute")
LOSS_OPTIONS = {
    "cross_entropy": set(),
    "softmax_mse": set(),
    "label_smoothing_cross_entropy": {"smoothing"},
    "multiclass_hinge": {"margin"},
    "focal": {"gamma", "alpha"},
}
OPTIMIZER_OPTIONS = {
    "sgd": set(),
    "momentum": {"momentum"},
    "nesterov": {"momentum"},
    "adagrad": {"eps"},
    "rmsprop": {"rho", "eps"},
    "adam": {"beta1", "beta2", "eps"},
    "adamax": {"beta1", "beta2", "eps"},
}


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("rb") as handle:
        config = tomllib.load(handle)
    validate_config(config)
    config = deepcopy(config)
    model = config["model"]
    model.setdefault("hidden_activation", "relu")
    model.setdefault("hidden_init", "kaiming_normal")
    model.setdefault("output_init", "xavier_normal")
    model.setdefault("dropout", 0.0)
    model.setdefault("dropout_seed", 44)
    config["loss"].setdefault("l1_lambda", 0.0)
    config["loss"].setdefault("l2_lambda", 0.0)
    config["_source_path"] = str(path.resolve())
    config["_config_hash"] = config_hash(config)
    return config


def _reject_unknown(section: dict, permitted: set[str], label: str) -> None:
    unknown = set(section) - permitted
    if unknown:
        raise ValueError(f"unsupported {label} options: {sorted(unknown)}")


def validate_config(config: dict[str, Any]) -> None:
    missing = [section for section in REQUIRED_SECTIONS if section not in config]
    if missing:
        raise ValueError(f"missing config sections: {missing}")
    dims = config["model"].get("dims")
    if not isinstance(dims, list) or len(dims) < 2 or any(not isinstance(d, int) or d <= 0 for d in dims):
        raise ValueError("model.dims must be a list of positive integers")
    dataset = config["data"].get("name")
    input_dims = {"mnist": 784, "cifar10": 3072}
    if dataset not in input_dims or dims[0] != input_dims[dataset] or dims[-1] != 10:
        raise ValueError("model dimensions do not match dataset and 10-class output")
    model = config["model"]
    _reject_unknown(
        model,
        {"dims", "hidden_activation", "hidden_init", "output_init", "dropout", "init_seed", "dropout_seed"},
        "model",
    )
    if model.get("hidden_activation", "relu") != "relu":
        raise ValueError("model.hidden_activation currently supports only 'relu'")
    supported_inits = {"kaiming_normal", "xavier_normal"}
    for field, default in (("hidden_init", "kaiming_normal"), ("output_init", "xavier_normal")):
        if model.get(field, default) not in supported_inits:
            raise ValueError(f"model.{field} is unsupported")
    dropout = float(model.get("dropout", 0.0))
    if not 0 <= dropout < 1:
        raise ValueError("model.dropout must be in [0,1)")
    loss = config["loss"]
    loss_name = loss.get("name")
    if loss_name not in LOSS_OPTIONS:
        raise ValueError("invalid loss.name")
    _reject_unknown(loss, {"name", "l1_lambda", "l2_lambda"} | LOSS_OPTIONS[loss_name], "loss")
    for field in ("l1_lambda", "l2_lambda"):
        if float(loss.get(field, 0.0)) < 0:
            raise ValueError(f"loss.{field} must be nonnegative")
    if loss_name == "label_smoothing_cross_entropy" and not 0 <= float(loss.get("smoothing", 0.1)) < 1:
        raise ValueError("loss.smoothing must be in [0,1)")
    if loss_name == "multiclass_hinge" and float(loss.get("margin", 1.0)) <= 0:
        raise ValueError("loss.margin must be positive")
    if loss_name == "focal":
        if float(loss.get("gamma", 2.0)) < 0 or float(loss.get("alpha", 1.0)) <= 0:
            raise ValueError("focal gamma must be nonnegative and alpha positive")
    optimizer = config["optimizer"]
    optimizer_name = optimizer.get("name")
    if optimizer_name not in OPTIMIZER_OPTIONS:
        raise ValueError("invalid optimizer.name")
    _reject_unknown(optimizer, {"name", "learning_rate"} | OPTIMIZER_OPTIONS[optimizer_name], "optimizer")
    if float(optimizer.get("learning_rate", 0)) <= 0:
        raise ValueError("optimizer.learning_rate must be positive")
    for field in ("momentum", "rho", "beta1", "beta2"):
        if field in optimizer and not 0 <= float(optimizer[field]) < 1:
            raise ValueError(f"optimizer.{field} must be in [0,1)")
    if "eps" in optimizer and float(optimizer["eps"]) < 0:
        raise ValueError("optimizer.eps must be nonnegative")
    training = config["training"]
    if int(training.get("epochs", 0)) < 1 or int(training.get("batch_size", 0)) < 1:
        raise ValueError("training epochs and batch_size must be positive")
    if config["compute"].get("dtype") not in {"float32", "float64"}:
        raise ValueError("compute.dtype must be float32 or float64")
    if config["compute"].get("backend", "cpu") != "cpu":
        raise ValueError("compute.backend must be cpu")
    if int(config["data"].get("val_size", 0)) < 1:
        raise ValueError("data.val_size must be positive")
    expected_preprocess = "scale01" if dataset == "mnist" else "channel_standardize"
    if config["data"].get("preprocess", expected_preprocess) != expected_preprocess:
        raise ValueError(f"data.preprocess must be {expected_preprocess}")


def public_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if not key.startswith("_")}


def config_hash(config: dict[str, Any]) -> str:
    encoded = json.dumps(public_config(config), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
