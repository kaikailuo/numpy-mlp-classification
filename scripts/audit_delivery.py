"""LEGACY v1.0 delivery audit; use scripts/audit_v1_1.py for v1.1."""

from __future__ import annotations

import csv
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORMAL = ROOT / "results/formal"
VERIFY = ROOT / "results/verification"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def pytest_totals(path: Path) -> dict[str, int]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    totals = {name: sum(int(suite.attrib.get(name, 0)) for suite in suites) for name in ("tests", "failures", "errors", "skipped")}
    return totals


def main() -> None:
    required_run_files = {
        "console.log",
        "data_manifest.json",
        "environment.json",
        "metrics.csv",
        "resolved_config.json",
        "source_config.toml",
        "summary.json",
    }
    run_dirs = sorted(path for path in FORMAL.iterdir() if path.is_dir())
    require(len(run_dirs) == 16, f"expected 16 formal runs, found {len(run_dirs)}")

    run_ids: list[str] = []
    test_ids: list[str] = []
    for run_dir in run_dirs:
        run_ids.append(run_dir.name)
        missing = sorted(name for name in required_run_files if not (run_dir / name).is_file())
        require(not missing, f"{run_dir.name} missing {missing}")
        summary = json.loads((run_dir / "summary.json").read_text())
        require(summary["status"] == "completed", f"incomplete run: {run_dir.name}")
        frozen_config = ROOT / "configs/formal" / f"{run_dir.name}.toml"
        require(frozen_config.read_bytes() == (run_dir / "source_config.toml").read_bytes(), f"config snapshot mismatch: {run_dir.name}")
        if (run_dir / "test_evaluation.json").exists():
            test_ids.append(run_dir.name)

    selection = json.loads((ROOT / "experiments/final_selection.json").read_text())
    selected_ids = [item["run_id"] for item in selection["selected_checkpoints"]]
    require(selection["test_access_before_freeze"] is False, "selection freeze flag is invalid")
    require(sorted(selected_ids) == sorted(test_ids), "test evaluations differ from frozen selection")

    with (ROOT / "results/aggregate/all_runs.csv").open(newline="") as handle:
        aggregate = list(csv.DictReader(handle))
    require(len(aggregate) == 16, "aggregate run count is not 16")
    require(sorted(row["run_id"] for row in aggregate) == run_ids, "aggregate run inventory mismatch")
    require(all(row["status"] == "completed" for row in aggregate), "aggregate includes incomplete runs")

    tests = json.loads((ROOT / "results/aggregate/final_test_metrics.json").read_text())
    test_by_id = {row["run_id"]: row for row in tests}
    require(len(test_by_id) == 10, "expected ten frozen test records")
    expected = {
        "B-M": (0.9690, 0.9683, 0.1055964),
        "B-C": (0.5150, 0.5039, 1.4424832),
        "O-MOM-0.05": (0.9808, 0.9795, 0.0774),
    }
    for run_id, (val_acc, test_acc, test_ce) in expected.items():
        row = test_by_id[run_id]
        require(abs(row["validation_accuracy"] - val_acc) < 5e-7, f"validation metric drift: {run_id}")
        require(abs(row["test_accuracy"] - test_acc) < 5e-7, f"test metric drift: {run_id}")
        require(abs(row["test_ce"] - test_ce) < 5e-4, f"test CE drift: {run_id}")

    gradient = json.loads((VERIFY / "gradient_check.json").read_text())
    tiny = json.loads((VERIFY / "tiny_overfit.json").read_text())
    data_check = json.loads((VERIFY / "data_validation.json").read_text())
    require(gradient["status"] == "passed", "gradient check did not pass")
    full_mlp_max_error = max(item["max_abs_error"] for item in gradient["full_mlp"].values())
    require(full_mlp_max_error < 1e-7, "full-MLP gradient error too large")
    require(tiny["accuracy"] == 1.0, "tiny overfit did not reach 100%")
    require(data_check["mnist"]["status"] == data_check["cifar10"]["status"] == "passed", "data validation did not pass")

    junit = VERIFY / "final_pytest.xml"
    require(junit.is_file(), "final pytest evidence is missing")
    pytest_result = pytest_totals(junit)
    require(pytest_result["tests"] == 21, f"expected 21 tests, got {pytest_result['tests']}")
    require(pytest_result["failures"] == pytest_result["errors"] == 0, "final pytest run failed")

    for item in selection["selected_checkpoints"]:
        checkpoint = ROOT / item["checkpoint"]
        require(checkpoint.is_file() and checkpoint.stat().st_size > 0, f"missing frozen checkpoint: {item['run_id']}")

    record = {
        "status": "passed",
        "formal_run_count": len(run_dirs),
        "frozen_test_count": len(test_ids),
        "formal_run_ids": run_ids,
        "pytest": pytest_result,
        "gradient_max_abs_error": full_mlp_max_error,
        "tiny_overfit_accuracy": tiny["accuracy"],
    }
    output = VERIFY / "repository_v1_evidence_audit.json"
    output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
