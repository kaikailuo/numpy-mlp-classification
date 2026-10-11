from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = ["#2F5597", "#C55A11", "#548235", "#7030A0", "#A5A5A5", "#00A6A6", "#C00000"]


def configure_chinese_style() -> None:
    plt.rcParams.update(
        {
            "font.sans-serif": ["PingFang SC", "Hiragino Sans GB", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "lines.linewidth": 1.8,
            "figure.dpi": 120,
        }
    )


def _save_both(figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")


def plot_history(rows: list[dict], output_dir: str | Path) -> None:
    configure_chinese_style()
    output_dir = Path(output_dir)
    epochs = [row["epoch"] for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axes[0].plot(epochs, [row["train_accuracy"] for row in rows], label="训练准确率", color=COLORS[0])
    axes[0].plot(epochs, [row["val_accuracy"] for row in rows], label="验证准确率", color=COLORS[1])
    axes[0].set(xlabel="Epoch", ylabel="准确率", title="训练与验证准确率")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(epochs, [row["train_batch_data_loss"] for row in rows], label="训练批次数据损失", color=COLORS[0])
    axes[1].plot(epochs, [row["train_regularization_loss"] for row in rows], label="正则化损失", color=COLORS[2])
    axes[1].plot(epochs, [row["train_batch_total_objective"] for row in rows], label="训练总目标", color=COLORS[3])
    axes[1].plot(epochs, [row["val_ce"] for row in rows], label="验证交叉熵", color=COLORS[1])
    axes[1].set(xlabel="Epoch", ylabel="损失", title="训练目标与验证交叉熵")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    _save_both(figure, output_dir / "learning_curves")
    plt.close(figure)


def plot_confusion(matrix: np.ndarray, path: str | Path, title: str) -> None:
    configure_chinese_style()
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set(xlabel="预测类别", ylabel="真实类别", title=title)
    axis.set_xticks(range(matrix.shape[1]))
    axis.set_yticks(range(matrix.shape[0]))
    figure.colorbar(image, ax=axis, fraction=0.046)
    figure.tight_layout()
    _save_both(figure, Path(path).with_suffix(""))
    plt.close(figure)
