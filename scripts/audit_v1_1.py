from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from src.utils.io import write_json


ROOT = Path(__file__).resolve().parents[1]
VERIFY = ROOT / "results/verification/v1_1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pytest_totals(path: Path) -> dict[str, int]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    return {name: sum(int(suite.attrib.get(name, 0)) for suite in suites) for name in ("tests", "failures", "errors", "skipped")}


def main() -> None:
    require(not (ROOT / "src/a1_mlp").exists(), "legacy src/a1_mlp directory still exists")
    v1_dirs = sorted(path for path in (ROOT / "results/baseline_v1").iterdir() if path.is_dir())
    v11_dirs = sorted(path for path in (ROOT / "results/evidence/v1_1").iterdir() if path.is_dir())
    require(len(v1_dirs) == 16, f"expected 16 v1.0 evidence runs, found {len(v1_dirs)}")
    require(len(v11_dirs) == 31, f"expected 31 v1.1 evidence runs, found {len(v11_dirs)}")
    required = {"summary.json", "source_config.toml", "resolved_config.json", "data_manifest.json", "metrics.csv", "environment.json", "console.log"}
    for run_dir in v1_dirs + v11_dirs:
        missing = required - {path.name for path in run_dir.iterdir() if path.is_file()}
        require(not missing, f"{run_dir.name} missing {sorted(missing)}")
        summary = json.loads((run_dir / "summary.json").read_text())
        require(summary["status"] == "completed", f"incomplete run: {run_dir.name}")
    matrix = json.loads((VERIFY / "matrix_status.json").read_text())
    require(matrix["requested"] == matrix["completed"] == 31 and matrix["failed"] == 0, "matrix status is not 31/31")
    selection = json.loads((ROOT / "experiments/selection_v1_1.json").read_text())
    require(selection["test_evaluation_status"] == "completed_once_after_freeze", "v1.1 test protocol not finalized")
    selected_ids = {item["run_id"] for item in selection["selected_checkpoints"]}
    require(selected_ids == {"L-HINGE-1", "R-DROP-0.1", "O-NAG-0.05", "C-MOM"}, "unexpected selected test IDs")
    for run_id in selected_ids:
        require((ROOT / "results/evidence/v1_1" / run_id / "test_evaluation.json").is_file(), f"missing test evidence: {run_id}")
    tests = pytest_totals(VERIFY / "pytest.xml")
    clean_tests = pytest_totals(VERIFY / "clean_env_pytest.xml")
    require(tests["tests"] == clean_tests["tests"] == 40, "expected 40 tests")
    require(tests["failures"] == tests["errors"] == clean_tests["failures"] == clean_tests["errors"] == 0, "pytest failure")
    gradient = json.loads((VERIFY / "gradient_check.json").read_text())
    require(gradient["status"] == "passed", "gradient check failed")
    max_gradient_error = max(
        [item["max_abs_error"] for item in gradient["losses"].values()]
        + [gradient["regularization"]["max_abs_error"]]
        + [item["max_abs_error"] for item in gradient["full_mlp"].values()]
    )
    require(max_gradient_error < 1e-7, "gradient error exceeds tolerance")
    aggregate = json.loads((ROOT / "results/expanded/aggregate_manifest.json").read_text())
    require(aggregate["status"] == "complete" and aggregate["total_runs"] == 47, "aggregate is incomplete")
    required_charts = {
        "loss_validation_accuracy", "loss_validation_convergence", "loss_training_objectives",
        "regularization_l1", "regularization_l2", "regularization_dropout", "regularization_methods",
        "regularization_train_validation_gap", "optimizer_candidates", "optimizer_best",
        "optimizer_steps_convergence", "optimizer_time_accuracy", "dataset_baselines",
        "cifar_extensions", "repeat_errorbars",
    }
    for stem in required_charts:
        require((ROOT / "results/expanded" / f"{stem}.png").is_file(), f"missing PNG: {stem}")
        require((ROOT / "results/expanded" / f"{stem}.pdf").is_file(), f"missing PDF: {stem}")
    clean = json.loads((VERIFY / "clean_environment_reproduction.json").read_text())
    require(clean["status"] == clean["lightweight_reproduction"]["status"] == "passed", "reproduction evidence failed")
    aggregate_safety = json.loads((VERIFY / "aggregate_safety_check.json").read_text())
    require(aggregate_safety["status"] == "passed" and aggregate_safety["preexisting_output_preserved"], "aggregate safety check failed")
    report = ROOT / "2024211735-罗开炜-实验A1-实验报告.pdf"
    require(report.is_file() and report.read_bytes().startswith(b"%PDF-"), "final report PDF is missing or invalid")
    require(not (ROOT / "output").exists() and not (ROOT / "report").exists(), "legacy report directories still exist")
    retained_checkpoints = []
    for name in ("final_selection.json", "selection_v1_1.json"):
        frozen = json.loads((ROOT / "experiments" / name).read_text())
        retained_checkpoints.extend(item["checkpoint"] for item in frozen["selected_checkpoints"])
    retained_checkpoints.append("results/reproduction/v1_1/B-M-clean/best.npz")
    for relative in retained_checkpoints:
        require((ROOT / relative).is_file(), f"missing retained checkpoint: {relative}")
    record = {
        "status": "passed",
        "historical_runs": len(v1_dirs),
        "new_runs": len(v11_dirs),
        "repeat_runs": 10,
        "failed_runs": 0,
        "pytest": tests,
        "clean_env_pytest": clean_tests,
        "gradient_max_abs_error": max_gradient_error,
        "selected_v1_1_test_count": len(selected_ids),
        "chart_pairs": len(required_charts),
        "final_report": {
            "path": str(report.relative_to(ROOT)),
            "bytes": report.stat().st_size,
            "sha256": sha256(report),
        },
        "retained_checkpoint_count": len(retained_checkpoints),
        "clean_environment_reproduction": "passed",
        "lightweight_reproduction": "passed",
        "incomplete_aggregate_safety": "passed",
    }
    write_json(VERIFY / "repository_delivery_audit.json", record)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
