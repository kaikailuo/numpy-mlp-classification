"""Copy immutable lightweight v1.0 evidence without checkpoints."""

from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/formal"
TARGET = ROOT / "results/baseline_v1"
FILES = ("console.log", "data_manifest.json", "environment.json", "metrics.csv", "resolved_config.json", "source_config.toml", "summary.json", "test_evaluation.json", "validation_evaluation.json")


def main() -> None:
    run_dirs = sorted(path for path in SOURCE.iterdir() if path.is_dir())
    if len(run_dirs) != 16:
        raise RuntimeError(f"expected 16 immutable v1.0 runs, found {len(run_dirs)}")
    for run_dir in run_dirs:
        target = TARGET / run_dir.name
        target.mkdir(parents=True, exist_ok=True)
        for name in FILES:
            source = run_dir / name
            if source.exists():
                destination = target / name
                if destination.exists() and destination.read_bytes() != source.read_bytes():
                    raise RuntimeError(f"refusing to overwrite differing v1.0 evidence: {destination}")
                shutil.copy2(source, destination)
    print(f"exported {len(run_dirs)} v1.0 lightweight run records to {TARGET}")


if __name__ == "__main__":
    main()
