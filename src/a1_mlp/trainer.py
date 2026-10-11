from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .checkpoints import save_checkpoint
from .data import DatasetBundle, batches
from .losses import SoftmaxCrossEntropy
from .metrics import accuracy
from .model import MLP
from .regularization import add_regularization_grads, regularization_loss


@dataclass
class FitResult:
    history: list[dict]
    best_epoch: int
    best_val_accuracy: float
    best_val_ce: float
    optimizer_steps: int
    samples_seen: int
    wall_seconds: float


def evaluate_arrays(
    model: MLP,
    x: np.ndarray,
    y: np.ndarray,
    preprocess,
    *,
    batch_size: int,
) -> dict:
    if len(x) == 0:
        raise ValueError("cannot evaluate an empty split")
    ce = SoftmaxCrossEntropy()
    total_loss = 0.0
    predictions: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for raw_x, batch_y in batches(x, y, batch_size):
        logits = model.forward(preprocess(raw_x), training=False)
        loss, _ = ce.value_and_grad(logits, batch_y)
        total_loss += loss * len(batch_y)
        predictions.append(np.argmax(logits, axis=1))
        labels.append(batch_y)
    predicted = np.concatenate(predictions)
    actual = np.concatenate(labels)
    return {
        "ce": total_loss / len(y),
        "accuracy": accuracy(predicted, actual),
        "predictions": predicted,
        "labels": actual,
    }


class Trainer:
    def fit(
        self,
        model: MLP,
        data: DatasetBundle,
        loss,
        optimizer,
        config: dict,
        checkpoint_path: str | Path,
    ) -> FitResult:
        training = config["training"]
        loss_config = config["loss"]
        batch_size = int(training["batch_size"])
        epochs = int(training["epochs"])
        rng = np.random.default_rng(int(training["shuffle_seed"]))
        l1_lambda = float(loss_config.get("l1_lambda", 0.0))
        l2_lambda = float(loss_config.get("l2_lambda", 0.0))
        history: list[dict] = []
        best_key: tuple[float, float, int] | None = None
        best_epoch = -1
        optimizer_steps = 0
        samples_seen = 0
        run_start = time.perf_counter()
        max_steps = training.get("max_steps_per_epoch")
        for epoch in range(1, epochs + 1):
            epoch_start = time.perf_counter()
            objective_total = 0.0
            data_loss_total = 0.0
            reg_loss_total = 0.0
            epoch_samples = 0
            for step, (raw_x, batch_y) in enumerate(
                batches(data.train_x, data.train_y, batch_size, rng=rng, shuffle=True), start=1
            ):
                if max_steps is not None and step > int(max_steps):
                    break
                batch_x = data.preprocess(raw_x)
                model.zero_grad()
                logits = model.forward(batch_x, training=True)
                data_loss, grad_logits = loss.value_and_grad(logits, batch_y)
                model.backward(grad_logits)
                reg_loss = regularization_loss(
                    model.parameters(), l1_lambda=l1_lambda, l2_lambda=l2_lambda
                )
                add_regularization_grads(
                    model.parameters(), l1_lambda=l1_lambda, l2_lambda=l2_lambda
                )
                optimizer.step(model.parameters())
                if not all(np.isfinite(parameter.data).all() for parameter in model.parameters()):
                    raise FloatingPointError(f"non-finite parameter at epoch={epoch}, step={step}")
                size = len(batch_y)
                data_loss_total += data_loss * size
                reg_loss_total += reg_loss * size
                objective_total += (data_loss + reg_loss) * size
                epoch_samples += size
                samples_seen += size
                optimizer_steps += 1
            train_metrics = evaluate_arrays(
                model, data.train_x, data.train_y, data.preprocess, batch_size=max(batch_size, 512)
            )
            val_metrics = evaluate_arrays(
                model, data.val_x, data.val_y, data.preprocess, batch_size=max(batch_size, 512)
            )
            row = {
                "epoch": epoch,
                "optimizer_steps": optimizer_steps,
                "samples_seen": samples_seen,
                "train_data_loss": data_loss_total / epoch_samples,
                "train_reg_loss": reg_loss_total / epoch_samples,
                "train_total_loss": objective_total / epoch_samples,
                "train_ce": train_metrics["ce"],
                "train_accuracy": train_metrics["accuracy"],
                "val_ce": val_metrics["ce"],
                "val_accuracy": val_metrics["accuracy"],
                "epoch_seconds": time.perf_counter() - epoch_start,
            }
            history.append(row)
            key = (-row["val_accuracy"], row["val_ce"], epoch)
            if best_key is None or key < best_key:
                best_key = key
                best_epoch = epoch
                metadata = {
                    "format_version": 1,
                    "model": {
                        "dims": list(model.dims),
                        "init_seed": int(config["model"]["init_seed"]),
                        "dropout_seed": int(config["model"].get("dropout_seed", 44)),
                        "dropout": float(config["model"].get("dropout", 0.0)),
                        "dtype": str(model.dtype),
                    },
                    "dataset": config["data"]["name"],
                    "data_root": config["data"]["root"],
                    "data_manifest": data.manifest,
                    "best_epoch": epoch,
                    "selection": "val_accuracy_then_val_ce_then_earlier_epoch",
                    "val_accuracy": row["val_accuracy"],
                    "val_ce": row["val_ce"],
                    "config_hash": config["_config_hash"],
                }
                save_checkpoint(checkpoint_path, model, metadata)
            print(
                f"epoch={epoch:02d} train_acc={row['train_accuracy']:.4f} "
                f"val_acc={row['val_accuracy']:.4f} val_ce={row['val_ce']:.4f} "
                f"seconds={row['epoch_seconds']:.2f}",
                flush=True,
            )
        best_row = history[best_epoch - 1]
        return FitResult(
            history=history,
            best_epoch=best_epoch,
            best_val_accuracy=best_row["val_accuracy"],
            best_val_ce=best_row["val_ce"],
            optimizer_steps=optimizer_steps,
            samples_seen=samples_seen,
            wall_seconds=time.perf_counter() - run_start,
        )

