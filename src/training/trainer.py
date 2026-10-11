from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.config import public_config
from src.data.datasets import DatasetBundle, batches
from src.nn.regularization import add_regularization_grads, regularization_loss

from .checkpoints import save_checkpoint
from .evaluation import evaluate_arrays


@dataclass
class FitResult:
    history: list[dict]
    best_epoch: int
    best_val_accuracy: float
    best_val_ce: float
    optimizer_steps: int
    samples_seen: int
    wall_seconds: float


class Trainer:
    def fit(
        self,
        model,
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
            if epoch_samples == 0:
                raise RuntimeError("training epoch produced no batches")
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
                "train_batch_data_loss": data_loss_total / epoch_samples,
                "train_regularization_loss": reg_loss_total / epoch_samples,
                "train_batch_total_objective": objective_total / epoch_samples,
                "train_eval_ce": train_metrics["ce"],
                "train_accuracy": train_metrics["accuracy"],
                "val_ce": val_metrics["ce"],
                "val_accuracy": val_metrics["accuracy"],
                "train_val_accuracy_gap": train_metrics["accuracy"] - val_metrics["accuracy"],
                "epoch_seconds": time.perf_counter() - epoch_start,
            }
            history.append(row)
            key = (-row["val_accuracy"], row["val_ce"], epoch)
            if best_key is None or key < best_key:
                best_key = key
                best_epoch = epoch
                model_config = config["model"]
                metadata = {
                    "format_version": 2,
                    "model": {
                        "dims": list(model.dims),
                        "init_seed": int(model_config["init_seed"]),
                        "dropout_seed": int(model_config.get("dropout_seed", 44)),
                        "dropout": float(model_config.get("dropout", 0.0)),
                        "hidden_activation": model.hidden_activation,
                        "hidden_init": model.hidden_init,
                        "output_init": model.output_init,
                        "dtype": str(model.dtype),
                    },
                    "dataset": config["data"]["name"],
                    "data_root": config["data"]["root"],
                    "data_manifest": data.manifest,
                    "resolved_config": public_config(config),
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
