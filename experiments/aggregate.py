from __future__ import annotations

import csv
import json
import tomllib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from a1_mlp.reporting import plot_comparison, write_json


FORMAL = Path("results/formal")
OUTPUT = Path("results/aggregate")


def load_records() -> list[dict]:
    records: list[dict] = []
    for summary_path in sorted(FORMAL.glob("*/summary.json")):
        summary = json.loads(summary_path.read_text())
        run_dir = summary_path.parent
        record = {"run_id": run_dir.name, **summary, "run_dir": str(run_dir)}
        config_path = run_dir / "source_config.toml"
        if config_path.exists():
            with config_path.open("rb") as handle:
                config = tomllib.load(handle)
            record.update(
                {
                    "loss": config["loss"]["name"],
                    "l1_lambda": config["loss"].get("l1_lambda", 0.0),
                    "l2_lambda": config["loss"].get("l2_lambda", 0.0),
                    "dropout": config["model"].get("dropout", 0.0),
                    "optimizer": config["optimizer"]["name"],
                    "learning_rate": config["optimizer"]["learning_rate"],
                    "epochs": config["training"]["epochs"],
                    "batch_size": config["training"]["batch_size"],
                    "train_subset_size": config["training"].get("train_subset_size", "full"),
                }
            )
        test_path = run_dir / "test_evaluation.json"
        if test_path.exists():
            test = json.loads(test_path.read_text())
            record["test_accuracy"] = test["accuracy"]
            record["test_ce"] = test["ce"]
        records.append(record)
    return records


def write_csv(records: list[dict]) -> None:
    fields = [
        "run_id", "label", "dataset", "status", "loss", "l1_lambda", "l2_lambda",
        "dropout", "optimizer", "learning_rate", "epochs", "batch_size",
        "train_subset_size", "best_epoch", "best_val_accuracy", "best_val_ce",
        "optimizer_steps", "samples_seen", "wall_seconds", "test_accuracy", "test_ce",
        "parameter_count", "config_hash",
    ]
    with (OUTPUT / "all_runs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def plot_curves(run_ids: list[str], path: Path, metric: str, title: str, x_axis: str = "epoch") -> None:
    figure, axis = plt.subplots(figsize=(8.5, 5))
    for run_id in run_ids:
        metrics_path = FORMAL / run_id / "metrics.csv"
        if not metrics_path.exists():
            continue
        with metrics_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        x = [float(row[x_axis]) for row in rows]
        y = [float(row[metric]) for row in rows]
        axis.plot(x, y, marker="o", markersize=2.5, label=run_id)
    axis.set(xlabel=x_axis.replace("_", " ").title(), ylabel=metric.replace("_", " ").title(), title=title)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def best_optimizer(records: list[dict]) -> dict[str, dict]:
    candidates = [
        record for record in records
        if record["run_id"] == "B-M" or record["run_id"].startswith("O-")
    ]
    selected: dict[str, dict] = {}
    for record in candidates:
        algorithm = record["optimizer"]
        current = selected.get(algorithm)
        key = (record["best_val_accuracy"], -record["best_val_ce"], -record["best_epoch"])
        if current is None:
            selected[algorithm] = record
        else:
            current_key = (current["best_val_accuracy"], -current["best_val_ce"], -current["best_epoch"])
            if key > current_key:
                selected[algorithm] = record
    return selected


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records = load_records()
    if any(record.get("status") != "completed" for record in records):
        print("warning: failed/incomplete runs are present and retained")
    write_csv(records)
    write_json(OUTPUT / "all_runs.json", records)
    selection = best_optimizer(records)
    write_json(
        OUTPUT / "optimizer_selection.json",
        {name: {key: value for key, value in record.items() if key != "run_dir"} for name, record in selection.items()},
    )
    by_id = {record["run_id"]: record for record in records}
    groups = {
        "loss_regularization": ["B-M", "L-MSE", "R-L1", "R-L2", "R-DROP"],
        "optimizer_candidates": [
            "B-M", "O-SGD-0.1", "O-MOM-0.01", "O-MOM-0.05",
            "O-ADA-0.01", "O-ADA-0.05", "O-ADAM-0.001", "O-ADAM-0.003",
        ],
        "batch_strategies": ["D-ONLINE", "D-MINI", "D-BATCH"],
    }
    for name, run_ids in groups.items():
        existing = [run_id for run_id in run_ids if run_id in by_id]
        plot_comparison(
            [{"label": run_id, "best_val_accuracy": by_id[run_id]["best_val_accuracy"]} for run_id in existing],
            OUTPUT / f"{name}_accuracy.png",
            title=name.replace("_", " ").title(),
        )
        plot_curves(existing, OUTPUT / f"{name}_curves.png", "val_accuracy", f"{name.replace('_', ' ').title()}: validation accuracy")
    plot_curves(groups["optimizer_candidates"], OUTPUT / "optimizer_by_steps.png", "val_accuracy", "Optimizer comparison by update count", "optimizer_steps")
    test_records = [
        {"label": record["run_id"], "best_val_accuracy": record["test_accuracy"]}
        for record in records if "test_accuracy" in record
    ]
    plot_comparison(test_records, OUTPUT / "frozen_test_accuracy.png", title="Frozen checkpoint test accuracy")
    write_json(
        OUTPUT / "final_test_metrics.json",
        [
            {
                "run_id": record["run_id"],
                "dataset": record["dataset"],
                "validation_accuracy": record["best_val_accuracy"],
                "test_accuracy": record["test_accuracy"],
                "test_ce": record["test_ce"],
            }
            for record in records if "test_accuracy" in record
        ],
    )
    print(json.dumps({"runs": len(records), "optimizer_selection": {k: v["run_id"] for k, v in selection.items()}}, indent=2))


if __name__ == "__main__":
    main()

