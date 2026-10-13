from __future__ import annotations

import hashlib
import json
import tempfile
import zipfile
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
ARCHIVE = DIST / "numpy-mlp-classification.zip"
PACKAGE_ROOT = "numpy-mlp-classification"
REPORT_NAME = "2024211735-罗开炜-实验A1-实验报告.pdf"

TOP_FILES = ["README.md", "pyproject.toml", "uv.lock", ".gitignore", REPORT_NAME]
TREES = [
    "src",
    "tests",
    "configs",
    "experiments",
    "scripts",
    "docs",
    "data/splits",
    "results/baseline_v1",
    "results/evidence",
    "results/expanded",
    "results/verification",
    "results/benchmarks",
    "results/aggregate",
    "results/formal",
    "results/formal_v1_1",
    "results/reproduction",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def retained_checkpoints() -> set[Path]:
    checkpoints = {ROOT / "results/reproduction/v1_1/B-M-clean/best.npz"}
    for name in ("final_selection.json", "selection_v1_1.json"):
        selection = json.loads((ROOT / "experiments" / name).read_text())
        checkpoints.update(ROOT / item["checkpoint"] for item in selection["selected_checkpoints"])
    return checkpoints


def excluded(path: Path, checkpoints: set[Path]) -> bool:
    parts = set(path.parts)
    if parts & {"__pycache__", ".pytest_cache", ".DS_Store", ".mypy_cache", ".ruff_cache", ".idea", ".vscode"}:
        return True
    if path.name in {".env", "credentials.json", "secrets.json", "id_rsa", "id_ed25519"} or path.name.startswith(".env."):
        return True
    if path.suffix in {".pyc", ".pyo", ".zip", ".whl", ".pem", ".key", ".tmp", ".bak"} or path.name.endswith("~"):
        return True
    return path.suffix == ".npz" and "results" in path.relative_to(ROOT).parts and path not in checkpoints


def collect() -> list[Path]:
    checkpoints = retained_checkpoints()
    files = [ROOT / relative for relative in TOP_FILES]
    for relative in TREES:
        root = ROOT / relative
        if not root.exists():
            raise FileNotFoundError(f"missing package tree: {root}")
        files.extend(path for path in root.rglob("*") if path.is_file() and not excluded(path, checkpoints))
    files.append(ROOT / "results/README.md")
    unique = sorted(set(files), key=lambda path: path.relative_to(ROOT).as_posix())
    missing = [path for path in unique if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing package inputs: {missing}")
    return unique


def validate_members(members: list[str]) -> None:
    required = {
        f"{PACKAGE_ROOT}/README.md",
        f"{PACKAGE_ROOT}/src/nn/model.py",
        f"{PACKAGE_ROOT}/src/nn/losses.py",
        f"{PACKAGE_ROOT}/src/optim/optimizers.py",
        f"{PACKAGE_ROOT}/configs/v1_1/O-NAG-0.05.toml",
        f"{PACKAGE_ROOT}/results/expanded/all_runs_v1_1.csv",
        f"{PACKAGE_ROOT}/results/expanded/repeat_statistics.json",
        f"{PACKAGE_ROOT}/results/verification/v1_1/final_audit_v1_1.json",
        f"{PACKAGE_ROOT}/docs/v1.1_实验结果与方法说明.md",
        f"{PACKAGE_ROOT}/{REPORT_NAME}",
    }
    missing = sorted(required - set(members))
    if missing:
        raise AssertionError(f"archive is missing required entries: {missing}")
    forbidden = ("data/raw/", ".venv/", "__pycache__/", ".pytest_cache/", "tmp/", "output/", "report/", "dist/")
    offenders = [name for name in members if any(token in name for token in forbidden)]
    if offenders:
        raise AssertionError(f"archive contains forbidden entries: {offenders[:10]}")
    if any("src/a1_mlp/" in name for name in members):
        raise AssertionError("archive contains the removed a1_mlp package")
    for name in members:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts:
            raise AssertionError(f"unsafe archive path: {name}")


def main() -> None:
    DIST.mkdir(parents=True, exist_ok=True)
    files = collect()
    with zipfile.ZipFile(ARCHIVE, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            archive.write(path, f"{PACKAGE_ROOT}/{path.relative_to(ROOT).as_posix()}")
    with zipfile.ZipFile(ARCHIVE) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise AssertionError(f"CRC failure: {bad}")
        members = archive.namelist()
        validate_members(members)
        with tempfile.TemporaryDirectory(prefix="a1-v11-zip-check-") as temp_dir:
            archive.extractall(temp_dir)
            extracted = Path(temp_dir) / PACKAGE_ROOT
            for path in files:
                relative = path.relative_to(ROOT)
                if sha256(extracted / relative) != sha256(path):
                    raise AssertionError(f"extracted file differs: {relative}")
    digest = sha256(ARCHIVE)
    checksum_path = DIST / "numpy-mlp-classification.sha256"
    checksum_path.write_text(f"{digest}  {ARCHIVE.name}\n")
    record = {
        "status": "passed",
        "archive": str(ARCHIVE.relative_to(ROOT)),
        "bytes": ARCHIVE.stat().st_size,
        "sha256": digest,
        "file_count": len(members),
        "raw_data_included": False,
        "checkpoints_included": True,
        "checkpoint_count": len(retained_checkpoints()),
        "final_report": REPORT_NAME,
        "crc_check": "passed",
        "extraction_check": "passed",
    }
    (DIST / "submission_manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
