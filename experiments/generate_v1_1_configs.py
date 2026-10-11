"""Generate the predeclared v1.1 validation-only experiment matrix.

This file contains the frozen candidate choices. It never reads test metrics.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "configs/v1_1"


MNIST = {
    "data": {"name": "mnist", "root": "data/raw/mnist", "split_seed": 20261011, "val_size": 5000, "preprocess": "scale01"},
    "model": {"dims": [784, 128, 64, 10], "hidden_activation": "relu", "hidden_init": "kaiming_normal", "output_init": "xavier_normal", "dropout": 0.0, "init_seed": 42, "dropout_seed": 44},
    "loss": {"name": "cross_entropy", "l1_lambda": 0.0, "l2_lambda": 0.0},
    "optimizer": {"name": "sgd", "learning_rate": 0.05},
    "training": {"epochs": 12, "batch_size": 128, "shuffle_seed": 43, "selection": "val_accuracy_then_val_ce", "checkpoint_best": True},
    "compute": {"dtype": "float32", "backend": "cpu"},
}
CIFAR = deepcopy(MNIST)
CIFAR["data"] = {"name": "cifar10", "root": "data/raw/cifar10", "split_seed": 20261011, "val_size": 5000, "preprocess": "channel_standardize"}
CIFAR["model"]["dims"] = [3072, 256, 128, 10]
CIFAR["optimizer"]["learning_rate"] = 0.01
CIFAR["training"]["epochs"] = 15


def candidate(run_id: str, label: str, *, dataset: str = "mnist", model=None, loss=None, optimizer=None, seeds=None) -> dict:
    config = deepcopy(MNIST if dataset == "mnist" else CIFAR)
    config["run"] = {"name": run_id.lower().replace("-", "_"), "label": label, "output_root": "results/formal_v1_1"}
    if model:
        config["model"].update(model)
    if loss:
        config["loss"].update(loss)
    if optimizer:
        config["optimizer"] = optimizer
    if seeds:
        config["model"]["init_seed"] = seeds
        config["model"]["dropout_seed"] = seeds + 2
        config["training"]["shuffle_seed"] = seeds + 1
    return {"run_id": run_id, "config": config}


MATRIX = [
    candidate("L-LS-0.1", "标签平滑 CE (epsilon=0.1)", loss={"name": "label_smoothing_cross_entropy", "smoothing": 0.1}),
    candidate("L-HINGE-1", "多类 Hinge (margin=1)", loss={"name": "multiclass_hinge", "margin": 1.0}),
    candidate("L-FOCAL-G2", "Focal (gamma=2)", loss={"name": "focal", "gamma": 2.0, "alpha": 1.0}),
    candidate("R-L1-1E6", "L1 lambda=1e-6", loss={"l1_lambda": 1e-6}),
    candidate("R-L1-1E4", "L1 lambda=1e-4", loss={"l1_lambda": 1e-4}),
    candidate("R-L2-1E5", "L2 lambda=1e-5", loss={"l2_lambda": 1e-5}),
    candidate("R-L2-1E3", "L2 lambda=1e-3", loss={"l2_lambda": 1e-3}),
    candidate("R-DROP-0.1", "Dropout p=0.1", model={"dropout": 0.1}),
    candidate("R-DROP-0.4", "Dropout p=0.4", model={"dropout": 0.4}),
    candidate("R-ELASTIC", "Elastic Net L1=1e-5 L2=1e-4", loss={"l1_lambda": 1e-5, "l2_lambda": 1e-4}),
    candidate("R-L2DROP", "L2=1e-4 + Dropout=0.2", model={"dropout": 0.2}, loss={"l2_lambda": 1e-4}),
    candidate("O-RMS-0.001", "RMSProp lr=0.001", optimizer={"name": "rmsprop", "learning_rate": 0.001, "rho": 0.9, "eps": 1e-8}),
    candidate("O-RMS-0.01", "RMSProp lr=0.01", optimizer={"name": "rmsprop", "learning_rate": 0.01, "rho": 0.9, "eps": 1e-8}),
    candidate("O-NAG-0.01", "Nesterov lr=0.01", optimizer={"name": "nesterov", "learning_rate": 0.01, "momentum": 0.9}),
    candidate("O-NAG-0.05", "Nesterov lr=0.05", optimizer={"name": "nesterov", "learning_rate": 0.05, "momentum": 0.9}),
    candidate("O-AMAX-0.001", "Adamax lr=0.001", optimizer={"name": "adamax", "learning_rate": 0.001, "beta1": 0.9, "beta2": 0.999, "eps": 1e-8}),
    candidate("O-AMAX-0.003", "Adamax lr=0.003", optimizer={"name": "adamax", "learning_rate": 0.003, "beta1": 0.9, "beta2": 0.999, "eps": 1e-8}),
    candidate("C-MOM", "CIFAR-10 Momentum", dataset="cifar10", optimizer={"name": "momentum", "learning_rate": 0.01, "momentum": 0.9}),
    candidate("C-ADAM", "CIFAR-10 Adam", dataset="cifar10", optimizer={"name": "adam", "learning_rate": 0.001, "beta1": 0.9, "beta2": 0.999, "eps": 1e-8}),
    candidate("C-L2", "CIFAR-10 L2=1e-4", dataset="cifar10", loss={"l2_lambda": 1e-4}),
    candidate("C-DROP", "CIFAR-10 Dropout=0.2", dataset="cifar10", model={"dropout": 0.2}),
    candidate("S-B-M-101", "MNIST baseline seed=101", seeds=101),
    candidate("S-B-M-202", "MNIST baseline seed=202", seeds=202),
    candidate("S-O-MOM-101", "Momentum lr=0.05 seed=101", optimizer={"name": "momentum", "learning_rate": 0.05, "momentum": 0.9}, seeds=101),
    candidate("S-O-MOM-202", "Momentum lr=0.05 seed=202", optimizer={"name": "momentum", "learning_rate": 0.05, "momentum": 0.9}, seeds=202),
    candidate("S-L-FOCAL-101", "Focal gamma=2 seed=101", loss={"name": "focal", "gamma": 2.0, "alpha": 1.0}, seeds=101),
    candidate("S-L-FOCAL-202", "Focal gamma=2 seed=202", loss={"name": "focal", "gamma": 2.0, "alpha": 1.0}, seeds=202),
    candidate("S-R-L2DROP-101", "L2+Dropout seed=101", model={"dropout": 0.2}, loss={"l2_lambda": 1e-4}, seeds=101),
    candidate("S-R-L2DROP-202", "L2+Dropout seed=202", model={"dropout": 0.2}, loss={"l2_lambda": 1e-4}, seeds=202),
    candidate("S-C-ADAM-101", "CIFAR-10 Adam seed=101", dataset="cifar10", optimizer={"name": "adam", "learning_rate": 0.001, "beta1": 0.9, "beta2": 0.999, "eps": 1e-8}, seeds=101),
    candidate("S-C-ADAM-202", "CIFAR-10 Adam seed=202", dataset="cifar10", optimizer={"name": "adam", "learning_rate": 0.001, "beta1": 0.9, "beta2": 0.999, "eps": 1e-8}, seeds=202),
]


def toml_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return "[" + ", ".join(toml_value(item) for item in value) + "]"
    return repr(value)


def render(config: dict) -> str:
    sections = ["run", "data", "model", "loss", "optimizer", "training", "compute"]
    lines = []
    for section in sections:
        lines.append(f"[{section}]")
        lines.extend(f"{key} = {toml_value(value)}" for key, value in config[section].items())
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    expected = {f"{item['run_id']}.toml" for item in MATRIX}
    unexpected = {path.name for path in OUTPUT.glob("*.toml")} - expected
    if unexpected:
        raise RuntimeError(f"refusing to leave unexpected v1.1 configs: {sorted(unexpected)}")
    for item in MATRIX:
        (OUTPUT / f"{item['run_id']}.toml").write_text(render(item["config"]))
    freeze = {
        "version": "v1.1",
        "protocol": "All candidates are frozen before v1.1 test evaluation; training and selection use train/validation only.",
        "historical_test_disclosure": "v1.0 test results were already known and may have informed broad method choices; no v1.1 candidate is selected using a new test result.",
        "selection_rule": "highest validation accuracy, then lowest validation CE, then earlier epoch",
        "split_seed": 20261011,
        "repeat_rule": "Each representative family uses seeds 42 (base), 101, 202; the data split remains fixed.",
        "candidate_count": len(MATRIX),
        "run_ids": [item["run_id"] for item in MATRIX],
        "test_evaluation_status": "not_started",
    }
    selection_path = ROOT / "experiments/selection_v1_1.json"
    if selection_path.exists():
        existing = json.loads(selection_path.read_text())
        if existing.get("run_ids") != freeze["run_ids"]:
            raise RuntimeError("existing v1.1 selection file has a different frozen matrix")
        if not existing.get("selected_checkpoints"):
            selection_path.write_text(json.dumps(freeze, indent=2, ensure_ascii=False) + "\n")
    else:
        selection_path.write_text(json.dumps(freeze, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"generated": len(MATRIX), "output": str(OUTPUT)}, indent=2))


if __name__ == "__main__":
    main()
