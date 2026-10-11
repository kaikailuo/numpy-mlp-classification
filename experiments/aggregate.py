"""Safely aggregate complete v1.0 + v1.1 lightweight evidence."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
import tomllib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.utils.plotting import COLORS, configure_chinese_style


ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "results/baseline_v1"
V11 = ROOT / "results/evidence/v1_1"
DEFAULT_OUTPUT = ROOT / "results/expanded"


def read_run(run_dir: Path, version: str) -> tuple[dict, list[dict]]:
    required = ("summary.json", "source_config.toml", "metrics.csv", "data_manifest.json")
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{run_dir} missing {missing}")
    summary = json.loads((run_dir / "summary.json").read_text())
    with (run_dir / "source_config.toml").open("rb") as handle:
        config = tomllib.load(handle)
    with (run_dir / "metrics.csv").open(newline="") as handle:
        history = list(csv.DictReader(handle))
    if not history:
        raise ValueError(f"empty metrics: {run_dir}")
    best_epoch = int(summary["best_epoch"])
    best = next(row for row in history if int(row["epoch"]) == best_epoch)
    loss = config["loss"]
    model = config["model"]
    optimizer = config["optimizer"]
    training = config["training"]
    test_path = run_dir / "test_evaluation.json"
    test = json.loads(test_path.read_text()) if test_path.exists() else {}
    record = {
        "run_id": run_dir.name,
        "version": version,
        "repeat": run_dir.name.startswith("S-"),
        "label": summary.get("label", config["run"].get("label", run_dir.name)),
        "dataset": config["data"]["name"],
        "model_dims": "-".join(str(value) for value in model["dims"]),
        "loss": loss["name"],
        "label_smoothing": loss.get("smoothing", 0.0),
        "hinge_margin": loss.get("margin", 0.0),
        "focal_gamma": loss.get("gamma", 0.0),
        "l1_lambda": loss.get("l1_lambda", 0.0),
        "l2_lambda": loss.get("l2_lambda", 0.0),
        "dropout": model.get("dropout", 0.0),
        "optimizer": optimizer["name"],
        "learning_rate": optimizer["learning_rate"],
        "batch_size": training["batch_size"],
        "epochs": training["epochs"],
        "init_seed": model.get("init_seed", 42),
        "shuffle_seed": training.get("shuffle_seed", 43),
        "best_epoch": best_epoch,
        "best_val_accuracy": float(summary["best_val_accuracy"]),
        "best_val_ce": float(summary["best_val_ce"]),
        "best_train_accuracy": float(best["train_accuracy"]),
        "train_val_accuracy_gap": float(best["train_accuracy"]) - float(summary["best_val_accuracy"]),
        "optimizer_steps": int(summary["optimizer_steps"]),
        "samples_seen": int(summary["samples_seen"]),
        "wall_seconds": float(summary["wall_seconds"]),
        "parameter_count": int(summary["parameter_count"]),
        "config_hash": summary["config_hash"],
        "status": summary["status"],
        "test_accuracy": test.get("accuracy"),
        "test_ce": test.get("ce"),
    }
    threshold = next((row for row in history if float(row["val_accuracy"]) >= 0.97), None)
    record["steps_to_val_accuracy_0.97"] = int(threshold["optimizer_steps"]) if threshold else None
    return record, history


def load_all(allow_partial: bool) -> tuple[list[dict], dict[str, list[dict]]]:
    expected_v1 = {path.stem for path in (ROOT / "configs/formal").glob("*.toml")}
    freeze = json.loads((ROOT / "experiments/selection_v1_1.json").read_text())
    expected_v11 = set(freeze["run_ids"])
    actual_v1 = {path.name for path in V1.iterdir() if path.is_dir()} if V1.exists() else set()
    actual_v11 = {path.name for path in V11.iterdir() if path.is_dir()} if V11.exists() else set()
    missing = {"v1": sorted(expected_v1 - actual_v1), "v1_1": sorted(expected_v11 - actual_v11)}
    if any(missing.values()) and not allow_partial:
        raise RuntimeError(f"refusing incomplete aggregation; missing={missing}")
    records: list[dict] = []
    histories: dict[str, list[dict]] = {}
    for version, root, expected in (("v1.0", V1, expected_v1), ("v1.1", V11, expected_v11)):
        if not root.exists():
            continue
        for run_dir in sorted(path for path in root.iterdir() if path.is_dir() and path.name in expected):
            record, history = read_run(run_dir, version)
            records.append(record)
            histories[record["run_id"]] = history
    failed = [record["run_id"] for record in records if record["status"] != "completed"]
    if failed and not allow_partial:
        raise RuntimeError(f"refusing aggregation with failed runs: {failed}")
    if not records:
        raise RuntimeError("no run records found")
    return records, histories


def save_figure(figure, stage: Path, name: str) -> None:
    figure.tight_layout()
    figure.savefig(stage / f"{name}.png", dpi=300, bbox_inches="tight")
    figure.savefig(stage / f"{name}.pdf", bbox_inches="tight")
    plt.close(figure)


def bar_chart(records: list[dict], stage: Path, name: str, title: str) -> None:
    figure, axis = plt.subplots(figsize=(max(7.5, 0.8 * len(records)), 4.8))
    values = [float(record["best_val_accuracy"]) for record in records]
    bars = axis.bar([record["run_id"] for record in records], values, color=COLORS[0])
    axis.set(title=title, xlabel="实验配置", ylabel="最佳验证准确率")
    axis.tick_params(axis="x", rotation=32)
    axis.grid(axis="y", alpha=0.25)
    axis.set_ylim(max(0, min(values) - 0.05), min(1.0, max(values) + 0.025))
    for bar, value in zip(bars, values):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 0.002, f"{value:.3f}", ha="center", fontsize=8)
    save_figure(figure, stage, name)


def curves(run_ids: list[str], histories: dict[str, list[dict]], stage: Path, name: str, title: str, *, x: str = "epoch") -> None:
    figure, axis = plt.subplots(figsize=(8.5, 5.0))
    for index, run_id in enumerate(run_ids):
        rows = histories[run_id]
        axis.plot([float(row[x]) for row in rows], [float(row["val_accuracy"]) for row in rows], marker="o", markersize=2.5, label=run_id, color=COLORS[index % len(COLORS)])
    axis.set(title=title, xlabel="优化器更新步数" if x == "optimizer_steps" else "Epoch", ylabel="验证准确率")
    axis.grid(alpha=0.25)
    axis.legend(ncol=2)
    save_figure(figure, stage, name)


def objective_panels(run_ids: list[str], histories: dict[str, list[dict]], stage: Path) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True)
    for axis, run_id in zip(axes.flat, run_ids):
        rows = histories[run_id]
        key = "train_batch_data_loss" if "train_batch_data_loss" in rows[0] else "train_data_loss"
        axis.plot([int(row["epoch"]) for row in rows], [float(row[key]) for row in rows], color=COLORS[0])
        axis.set_title(run_id)
        axis.set_xlabel("Epoch")
        axis.set_ylabel("自身定义的训练数据损失")
        axis.grid(alpha=0.25)
    axes.flat[-1].axis("off")
    figure.suptitle("五类损失的训练曲线（纵轴定义与尺度不同，不直接比较数值大小）")
    save_figure(figure, stage, "loss_training_objectives")


def repeated_statistics(records: list[dict]) -> list[dict]:
    groups = {
        "MNIST 基准": ["B-M", "S-B-M-101", "S-B-M-202"],
        "Momentum": ["O-MOM-0.05", "S-O-MOM-101", "S-O-MOM-202"],
        "Focal": ["L-FOCAL-G2", "S-L-FOCAL-101", "S-L-FOCAL-202"],
        "L2+Dropout": ["R-L2DROP", "S-R-L2DROP-101", "S-R-L2DROP-202"],
        "CIFAR Adam": ["C-ADAM", "S-C-ADAM-101", "S-C-ADAM-202"],
    }
    by_id = {record["run_id"]: record for record in records}
    output = []
    for family, run_ids in groups.items():
        values = np.asarray([by_id[run_id]["best_val_accuracy"] for run_id in run_ids], dtype=float)
        output.append({
            "family": family,
            "run_ids": run_ids,
            "raw_validation_accuracy": values.tolist(),
            "mean_validation_accuracy": float(values.mean()),
            "std_validation_accuracy": float(values.std(ddof=1)),
            "n": len(values),
        })
    return output


def write_csv(path: Path, records: list[dict]) -> None:
    fields = list(records[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    records, histories = load_all(args.allow_partial)
    configure_chinese_style()
    by_id = {record["run_id"]: record for record in records}
    with tempfile.TemporaryDirectory(prefix="a1-v11-aggregate-", dir=output.parent) as temporary:
        stage = Path(temporary)
        write_csv(stage / "all_runs_v1_1.csv", records)
        (stage / "all_runs_v1_1.json").write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n")
        loss_ids = ["B-M", "L-MSE", "L-LS-0.1", "L-HINGE-1", "L-FOCAL-G2"]
        bar_chart([by_id[run_id] for run_id in loss_ids], stage, "loss_validation_accuracy", "五类损失函数：最佳验证准确率")
        curves(loss_ids, histories, stage, "loss_validation_convergence", "五类损失函数：验证准确率收敛")
        objective_panels(loss_ids, histories, stage)
        for prefix, title, ids in (
            ("regularization_l1", "L1 强度对比", ["B-M", "R-L1-1E6", "R-L1", "R-L1-1E4"]),
            ("regularization_l2", "L2 强度对比", ["B-M", "R-L2-1E5", "R-L2", "R-L2-1E3"]),
            ("regularization_dropout", "Dropout 强度对比", ["B-M", "R-DROP-0.1", "R-DROP", "R-DROP-0.4"]),
        ):
            bar_chart([by_id[run_id] for run_id in ids], stage, prefix, title)
        reg_ids = ["B-M", "R-L1", "R-L2", "R-DROP", "R-ELASTIC", "R-L2DROP"]
        bar_chart([by_id[run_id] for run_id in reg_ids], stage, "regularization_methods", "基础与组合正则化方法")
        figure, axis = plt.subplots(figsize=(8.5, 4.8))
        axis.bar(reg_ids, [by_id[run_id]["train_val_accuracy_gap"] for run_id in reg_ids], color=COLORS[2])
        axis.set(title="最佳 Epoch 的训练–验证准确率差距", xlabel="正则化配置", ylabel="Train − Validation Accuracy")
        axis.tick_params(axis="x", rotation=30)
        axis.grid(axis="y", alpha=0.25)
        save_figure(figure, stage, "regularization_train_validation_gap")
        optimizer_candidates = [record for record in records if not record["repeat"] and record["dataset"] == "mnist" and (record["run_id"] == "B-M" or record["run_id"].startswith("O-"))]
        optimizer_candidates.sort(key=lambda record: record["run_id"])
        bar_chart(optimizer_candidates, stage, "optimizer_candidates", "七类优化器及学习率候选")
        selected = {}
        for record in optimizer_candidates:
            current = selected.get(record["optimizer"])
            key = (record["best_val_accuracy"], -record["best_val_ce"], -record["best_epoch"])
            if current is None or key > (current["best_val_accuracy"], -current["best_val_ce"], -current["best_epoch"]):
                selected[record["optimizer"]] = record
        optimizer_best = [selected[name] for name in ("sgd", "momentum", "adagrad", "rmsprop", "adam", "nesterov", "adamax")]
        bar_chart(optimizer_best, stage, "optimizer_best", "七类优化器的验证集最佳候选")
        curves([record["run_id"] for record in optimizer_best], histories, stage, "optimizer_steps_convergence", "优化器最佳候选：更新步数与收敛", x="optimizer_steps")
        figure, axis = plt.subplots(figsize=(7.5, 5))
        for index, record in enumerate(optimizer_best):
            axis.scatter(record["wall_seconds"], record["best_val_accuracy"], s=60, color=COLORS[index])
            axis.annotate(record["run_id"], (record["wall_seconds"], record["best_val_accuracy"]), xytext=(4, 4), textcoords="offset points", fontsize=8)
        axis.set(title="训练耗时与最佳验证准确率", xlabel="训练耗时（秒）", ylabel="最佳验证准确率")
        axis.grid(alpha=0.25)
        save_figure(figure, stage, "optimizer_time_accuracy")
        batch_ids = ["D-ONLINE", "D-MINI", "D-BATCH"]
        write_csv(stage / "batch_strategies.csv", [by_id[run_id] for run_id in batch_ids])
        bar_chart([by_id[run_id] for run_id in ["B-M", "B-C"]], stage, "dataset_baselines", "MNIST 与 CIFAR-10 展平像素 MLP 基准")
        cifar_ids = ["B-C", "C-MOM", "C-ADAM", "C-L2", "C-DROP"]
        bar_chart([by_id[run_id] for run_id in cifar_ids], stage, "cifar_extensions", "CIFAR-10 代表性扩展")
        repeats = repeated_statistics(records)
        (stage / "repeat_statistics.json").write_text(json.dumps(repeats, indent=2, ensure_ascii=False) + "\n")
        repeat_csv = [{
            "family": row["family"],
            "run_ids": ";".join(row["run_ids"]),
            "raw_validation_accuracy": ";".join(f"{value:.6f}" for value in row["raw_validation_accuracy"]),
            "mean_validation_accuracy": row["mean_validation_accuracy"],
            "std_validation_accuracy": row["std_validation_accuracy"],
            "n": row["n"],
        } for row in repeats]
        write_csv(stage / "repeat_statistics.csv", repeat_csv)
        figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
        for axis, subset, title in (
            (axes[0], repeats[:4], "MNIST 代表性配置"),
            (axes[1], repeats[4:], "CIFAR-10 Adam"),
        ):
            means = [row["mean_validation_accuracy"] for row in subset]
            stds = [row["std_validation_accuracy"] for row in subset]
            axis.errorbar([row["family"] for row in subset], means, yerr=stds, fmt="o", capsize=5, color=COLORS[0])
            axis.set(title=title, xlabel="配置族", ylabel="验证准确率（均值 ± 样本标准差）")
            axis.set_ylim(min(np.asarray(means) - np.asarray(stds)) - 0.005, max(np.asarray(means) + np.asarray(stds)) + 0.005)
            axis.tick_params(axis="x", rotation=20)
            axis.grid(axis="y", alpha=0.25)
        save_figure(figure, stage, "repeat_errorbars")
        (stage / "optimizer_selection_v1_1.json").write_text(json.dumps(selected, indent=2, ensure_ascii=False) + "\n")
        selected_test_ids = ["L-HINGE-1", "R-DROP-0.1", "O-NAG-0.05", "C-MOM"]
        selected_tests = [{
            "run_id": run_id,
            "dataset": by_id[run_id]["dataset"],
            "validation_accuracy": by_id[run_id]["best_val_accuracy"],
            "validation_ce": by_id[run_id]["best_val_ce"],
            "test_accuracy": by_id[run_id]["test_accuracy"],
            "test_ce": by_id[run_id]["test_ce"],
        } for run_id in selected_test_ids]
        if any(row["test_accuracy"] is None for row in selected_tests):
            raise RuntimeError("selected v1.1 test evaluation is incomplete")
        (stage / "selected_test_metrics_v1_1.json").write_text(json.dumps(selected_tests, indent=2, ensure_ascii=False) + "\n")
        write_csv(stage / "selected_test_metrics_v1_1.csv", selected_tests)
        manifest = {
            "status": "partial" if args.allow_partial else "complete",
            "historical_runs": sum(record["version"] == "v1.0" for record in records),
            "new_runs": sum(record["version"] == "v1.1" for record in records),
            "repeat_runs": sum(record["repeat"] for record in records),
            "total_runs": len(records),
            "source_roots": [str(V1.relative_to(ROOT)), str(V11.relative_to(ROOT))],
        }
        (stage / "aggregate_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        output.mkdir(parents=True, exist_ok=True)
        for path in stage.iterdir():
            os.replace(path, output / path.name)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
