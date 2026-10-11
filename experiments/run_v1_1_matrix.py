"""Execute the frozen v1.1 matrix without overwriting any run."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from src.utils.io import write_json


ROOT = Path(__file__).resolve().parents[1]
FORMAL = ROOT / "results/formal_v1_1"
EVIDENCE = ROOT / "results/evidence/v1_1"
SELECTION = ROOT / "experiments/selection_v1_1.json"
LIGHT_FILES = (
    "console.log",
    "data_manifest.json",
    "environment.json",
    "metrics.csv",
    "resolved_config.json",
    "source_config.toml",
    "summary.json",
    "test_evaluation.json",
)


def export_evidence(run_id: str) -> None:
    source = FORMAL / run_id
    target = EVIDENCE / run_id
    target.mkdir(parents=True, exist_ok=True)
    for name in LIGHT_FILES:
        path = source / name
        if path.exists():
            shutil.copy2(path, target / name)
    for path in source.glob("test_evaluation_confusion.*"):
        shutil.copy2(path, target / path.name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", help="Optional exact run IDs")
    args = parser.parse_args()
    freeze = json.loads(SELECTION.read_text())
    run_ids = freeze["run_ids"]
    if args.only:
        unknown = set(args.only) - set(run_ids)
        if unknown:
            raise ValueError(f"unknown run IDs: {sorted(unknown)}")
        run_ids = [run_id for run_id in run_ids if run_id in args.only]
    status_path = ROOT / (
        "results/verification/v1_1/matrix_status_subset.json"
        if args.only
        else "results/verification/v1_1/matrix_status.json"
    )
    FORMAL.mkdir(parents=True, exist_ok=True)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    records = []
    for index, run_id in enumerate(run_ids, start=1):
        config = ROOT / "configs/v1_1" / f"{run_id}.toml"
        output = FORMAL / run_id
        if output.exists():
            summary_path = output / "summary.json"
            status = json.loads(summary_path.read_text()).get("status") if summary_path.exists() else "missing_summary"
            export_evidence(run_id)
            records.append({"run_id": run_id, "status": status, "action": "preserved_existing"})
            print(f"[{index}/{len(run_ids)}] preserve {run_id}: {status}", flush=True)
            continue
        print(f"[{index}/{len(run_ids)}] start {run_id}", flush=True)
        start = time.perf_counter()
        completed = subprocess.run(
            [sys.executable, "-m", "src.cli", "train", "--config", str(config), "--output", str(output)],
            cwd=ROOT,
            check=False,
        )
        export_evidence(run_id)
        status = "completed" if completed.returncode == 0 else "failed"
        records.append({"run_id": run_id, "status": status, "returncode": completed.returncode, "runner_seconds": time.perf_counter() - start})
        write_json(status_path, {"runs": records, "requested": len(run_ids)})
        if completed.returncode:
            print(f"[{index}/{len(run_ids)}] failed {run_id}; continuing to retain evidence", flush=True)
        else:
            print(f"[{index}/{len(run_ids)}] completed {run_id}", flush=True)
    result = {
        "requested": len(run_ids),
        "completed": sum(record["status"] == "completed" for record in records),
        "failed": sum(record["status"] != "completed" for record in records),
        "runs": records,
    }
    write_json(status_path, result)
    print(json.dumps(result, indent=2), flush=True)
    if result["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
