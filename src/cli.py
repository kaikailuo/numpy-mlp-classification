from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

from src.config import load_config, public_config
from src.data.cifar10 import download_cifar10
from src.data.datasets import load_dataset
from src.data.mnist import download_mnist
from src.nn.losses import make_loss
from src.nn.model import MLP
from src.optim.optimizers import make_optimizer
from src.training.checkpoints import load_checkpoint
from src.training.evaluation import evaluate_arrays
from src.training.metrics import confusion_matrix
from src.training.trainer import Trainer
from src.utils.io import write_json, write_metrics
from src.utils.plotting import plot_confusion, plot_history
from src.utils.reproducibility import environment_snapshot


def _optimizer_from_config(config: dict):
    optimizer_config = config["optimizer"]
    options = {key: value for key, value in optimizer_config.items() if key not in {"name", "learning_rate"}}
    return make_optimizer(optimizer_config["name"], float(optimizer_config["learning_rate"]), options)


def _loss_from_config(config: dict):
    loss_config = config["loss"]
    options = {key: value for key, value in loss_config.items() if key not in {"name", "l1_lambda", "l2_lambda"}}
    return make_loss(loss_config["name"], options)


def _new_model(config: dict) -> MLP:
    model = config["model"]
    return MLP(
        model["dims"],
        init_seed=int(model.get("init_seed", 42)),
        dropout_seed=int(model.get("dropout_seed", 44)),
        dropout=float(model.get("dropout", 0.0)),
        hidden_activation=model.get("hidden_activation", "relu"),
        hidden_init=model.get("hidden_init", "kaiming_normal"),
        output_init=model.get("output_init", "xavier_normal"),
        dtype=np.dtype(config["compute"]["dtype"]),
    )


def _run_directory(config: dict, explicit: str | None = None) -> Path:
    if explicit:
        path = Path(explicit)
        if path.exists():
            raise FileExistsError(f"refusing to overwrite run directory: {path}")
        return path
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path(config["run"]["output_root"]) / f"{timestamp}_{config['run']['name']}_{config['_config_hash'][:8]}"


def _minimal_data_config(dataset: str, root: Path) -> dict:
    return {
        "data": {"name": dataset, "root": str(root), "split_seed": 20261011, "val_size": 5000},
        "compute": {"dtype": "float32"},
    }


def command_prepare(args: argparse.Namespace) -> int:
    root = Path(args.root)
    if args.dataset == "mnist":
        if args.download:
            download_mnist(root)
    elif args.download:
        download_cifar10(root)
    data = load_dataset(_minimal_data_config(args.dataset, root), include_test=True)
    print(json.dumps({
        "dataset": args.dataset,
        "train": list(data.train_x.shape),
        "validation": list(data.val_x.shape),
        "test": list(data.test_x.shape),
        "manifest": data.manifest,
    }, indent=2))
    return 0


def command_train(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    run_dir = _run_directory(config, args.output)
    run_dir.mkdir(parents=True)
    write_json(run_dir / "resolved_config.json", public_config(config))
    shutil.copyfile(args.config, run_dir / "source_config.toml")
    write_json(run_dir / "environment.json", environment_snapshot())
    log_path = run_dir / "console.log"
    try:
        data = load_dataset(config, include_test=False)
        write_json(run_dir / "data_manifest.json", data.manifest)
        model = _new_model(config)
        loss = _loss_from_config(config)
        optimizer = _optimizer_from_config(config)
        start = time.perf_counter()
        with log_path.open("w") as log:
            old_stdout = sys.stdout
            sys.stdout = _Tee(old_stdout, log)
            try:
                print(f"run_dir={run_dir}")
                print(f"parameter_count={model.parameter_count}")
                result = Trainer().fit(model, data, loss, optimizer, config, run_dir / "best.npz")
            finally:
                sys.stdout = old_stdout
        write_metrics(run_dir / "metrics.csv", result.history)
        plot_history(result.history, run_dir / "figures")
        summary = {
            "status": "completed",
            "run_id": run_dir.name,
            "dataset": config["data"]["name"],
            "label": config["run"].get("label", config["run"]["name"]),
            "config_hash": config["_config_hash"],
            "parameter_count": model.parameter_count,
            "best_epoch": result.best_epoch,
            "best_val_accuracy": result.best_val_accuracy,
            "best_val_ce": result.best_val_ce,
            "optimizer_steps": result.optimizer_steps,
            "samples_seen": result.samples_seen,
            "wall_seconds": result.wall_seconds,
            "command_wall_seconds": time.perf_counter() - start,
        }
        write_json(run_dir / "summary.json", summary)
        print(json.dumps(summary, indent=2))
        return 0
    except Exception as error:
        failure = {
            "status": "failed",
            "run_id": run_dir.name,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        write_json(run_dir / "summary.json", failure)
        with log_path.open("a") as log:
            log.write(failure["traceback"])
        raise


class _Tee:
    def __init__(self, *streams) -> None:
        self.streams = streams

    def write(self, value: str) -> int:
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def command_benchmark(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    data = load_dataset(config, include_test=False)
    size = min(int(args.samples), len(data.train_x))
    batch_size = int(config["training"]["batch_size"])
    model = _new_model(config)
    loss = _loss_from_config(config)
    optimizer = _optimizer_from_config(config)
    rng = np.random.default_rng(int(config["training"]["shuffle_seed"]))
    order = np.arange(size)
    timings: list[float] = []
    samples = 0
    for step in range(int(args.steps) + 5):
        start = (step * batch_size) % size
        if start + batch_size > size:
            rng.shuffle(order)
            start = 0
        selection = order[start : start + batch_size]
        batch_x = data.preprocess(data.train_x[selection])
        batch_y = data.train_y[selection]
        tick = time.perf_counter()
        model.zero_grad()
        _, grad = loss.value_and_grad(model.forward(batch_x, training=True), batch_y)
        model.backward(grad)
        optimizer.step(model.parameters())
        elapsed = time.perf_counter() - tick
        if step >= 5:
            timings.append(elapsed)
            samples += len(selection)
    result = {
        "dataset": config["data"]["name"],
        "steps": len(timings),
        "batch_size": batch_size,
        "samples": samples,
        "wall_seconds": sum(timings),
        "samples_per_second": samples / sum(timings),
        "mean_step_seconds": float(np.mean(timings)),
        "p95_step_seconds": float(np.percentile(timings, 95)),
        "estimated_full_epoch_seconds_train_only": len(data.train_x) / (samples / sum(timings)),
        "parameter_count": model.parameter_count,
    }
    print(json.dumps(result, indent=2))
    if args.output:
        write_json(args.output, result)
    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    model, metadata = load_checkpoint(args.checkpoint)
    manifest = metadata["data_manifest"]
    resolved = metadata.get("resolved_config")
    data_config = resolved or {
        "data": {
            "name": metadata["dataset"],
            "root": metadata["data_root"],
            "split_seed": manifest["split"]["seed"],
            "val_size": manifest["split"]["val_size"],
        },
        "compute": {"dtype": metadata["model"]["dtype"]},
    }
    data = load_dataset(
        data_config,
        include_test=args.split == "test",
        frozen_preprocess=manifest["preprocess"],
    )
    if args.split == "train":
        x, y = data.train_x, data.train_y
    elif args.split == "validation":
        x, y = data.val_x, data.val_y
    else:
        x, y = data.test_x, data.test_y
    metrics = evaluate_arrays(model, x, y, data.preprocess, batch_size=args.batch_size)
    result = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "dataset": metadata["dataset"],
        "split": args.split,
        "sample_count": len(y),
        "ce": metrics["ce"],
        "accuracy": metrics["accuracy"],
        "confusion_matrix": confusion_matrix(metrics["predictions"], metrics["labels"]).tolist(),
        "preprocess": manifest["preprocess"],
    }
    print(json.dumps(result, indent=2))
    if args.output:
        write_json(args.output, result)
        plot_confusion(np.asarray(result["confusion_matrix"]), Path(args.output).with_name(Path(args.output).stem + "_confusion"), f"{metadata['dataset'].upper()} {args.split} 混淆矩阵")
    return 0


def command_toy(args: argparse.Namespace) -> int:
    rng = np.random.default_rng(7)
    centers = np.asarray([[-2, -2], [2, -2], [0, 2]], dtype=np.float64)
    x = np.concatenate([center + rng.normal(0, 0.15, (8, 2)) for center in centers])
    y = np.repeat(np.arange(3), 8)
    model = MLP([2, 12, 3], init_seed=42, dtype=np.float64)
    loss = make_loss("cross_entropy")
    optimizer = make_optimizer("sgd", 0.1)
    first_loss = None
    for _ in range(args.steps):
        model.zero_grad()
        value, gradient = loss.value_and_grad(model.forward(x, training=True), y)
        first_loss = value if first_loss is None else first_loss
        model.backward(gradient)
        optimizer.step(model.parameters())
    final_logits = model.forward(x, training=False)
    final_loss, _ = loss.value_and_grad(final_logits, y)
    result = {
        "dataset": "toy_three_gaussian_clusters",
        "steps": args.steps,
        "initial_loss": first_loss,
        "final_loss": final_loss,
        "accuracy": float(np.mean(np.argmax(final_logits, axis=1) == y)),
    }
    print(json.dumps(result, indent=2))
    if args.output:
        write_json(args.output, result)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NumPy-only MLP experiments")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare-data")
    prepare.add_argument("--dataset", choices=("mnist", "cifar10"), required=True)
    prepare.add_argument("--root", required=True)
    prepare.add_argument("--download", action="store_true")
    prepare.set_defaults(func=command_prepare)
    train = subparsers.add_parser("train")
    train.add_argument("--config", required=True)
    train.add_argument("--output")
    train.set_defaults(func=command_train)
    benchmark = subparsers.add_parser("benchmark")
    benchmark.add_argument("--config", required=True)
    benchmark.add_argument("--steps", type=int, default=100)
    benchmark.add_argument("--samples", type=int, default=5000)
    benchmark.add_argument("--output")
    benchmark.set_defaults(func=command_benchmark)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--split", choices=("train", "validation", "test"), required=True)
    evaluate.add_argument("--batch-size", type=int, default=512)
    evaluate.add_argument("--output")
    evaluate.set_defaults(func=command_evaluate)
    toy = subparsers.add_parser("toy-overfit")
    toy.add_argument("--steps", type=int, default=500)
    toy.add_argument("--output")
    toy.set_defaults(func=command_toy)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
