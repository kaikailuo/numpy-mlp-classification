from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def write_json(path: str | Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_metrics(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("cannot write empty metrics")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_history(rows: list[dict], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [row["train_accuracy"] for row in rows], label="train")
    axes[0].plot(epochs, [row["val_accuracy"] for row in rows], label="validation")
    axes[0].set(xlabel="Epoch", ylabel="Accuracy", title="Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(epochs, [row["train_data_loss"] for row in rows], label="train objective")
    axes[1].plot(epochs, [row["val_ce"] for row in rows], label="validation CE")
    axes[1].set(xlabel="Epoch", ylabel="Loss", title="Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "learning_curves.png", dpi=180)
    plt.close(figure)


def plot_confusion(matrix: np.ndarray, path: str | Path, title: str) -> None:
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set(xlabel="Predicted", ylabel="True", title=title)
    axis.set_xticks(range(matrix.shape[1]))
    axis.set_yticks(range(matrix.shape[0]))
    figure.colorbar(image, ax=axis, fraction=0.046)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_comparison(records: list[dict], path: str | Path, title: str = "Validation accuracy") -> None:
    labels = [record["label"] for record in records]
    values = [record["best_val_accuracy"] for record in records]
    figure, axis = plt.subplots(figsize=(max(8, len(labels) * 0.8), 4.8))
    bars = axis.bar(labels, values, color="#4472C4")
    axis.set(ylabel="Best validation accuracy", title=title)
    axis.set_ylim(max(0, min(values) - 0.08), min(1.0, max(values) + 0.04))
    axis.tick_params(axis="x", rotation=35)
    for bar, value in zip(bars, values):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 0.003, f"{value:.3f}", ha="center", fontsize=8)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)

