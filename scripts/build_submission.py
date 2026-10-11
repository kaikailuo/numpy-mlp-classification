from __future__ import annotations

import hashlib
import json
import tempfile
import zipfile
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
ARCHIVE = DIST / "A1_NumPy_MLP_Submission.zip"
PACKAGE_ROOT = "A1_NumPy_MLP_Submission"

TOP_FILES = ["README.md", "pyproject.toml", "uv.lock", ".gitignore"]
TREES = [
    "src",
    "tests",
    "configs",
    "experiments",
    "report",
    "scripts",
    "docs",
    "data/splits",
    "results/aggregate",
    "results/verification",
    "results/benchmarks",
    "output/pdf",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def excluded(path: Path) -> bool:
    parts = set(path.parts)
    return bool(parts & {"__pycache__", ".pytest_cache", ".DS_Store"}) or path.suffix in {".pyc", ".pyo", ".npz"}


def collect() -> list[Path]:
    files = [ROOT / relative for relative in TOP_FILES]
    for relative in TREES:
        files.extend(path for path in (ROOT / relative).rglob("*") if path.is_file() and not excluded(path))
    files.append(ROOT / "results/README.md")
    for run_dir in sorted((ROOT / "results/formal").iterdir()):
        if run_dir.is_dir():
            files.extend(path for path in run_dir.iterdir() if path.is_file() and path.name != "best.npz" and not excluded(path))
    unique = sorted(set(files), key=lambda path: path.relative_to(ROOT).as_posix())
    missing = [path for path in unique if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing package inputs: {missing}")
    return unique


def validate_members(members: list[str]) -> None:
    required = {
        f"{PACKAGE_ROOT}/README.md",
        f"{PACKAGE_ROOT}/pyproject.toml",
        f"{PACKAGE_ROOT}/uv.lock",
        f"{PACKAGE_ROOT}/src/a1_mlp/model.py",
        f"{PACKAGE_ROOT}/tests/test_gradients.py",
        f"{PACKAGE_ROOT}/configs/formal/B-M.toml",
        f"{PACKAGE_ROOT}/experiments/final_selection.json",
        f"{PACKAGE_ROOT}/results/aggregate/all_runs.csv",
        f"{PACKAGE_ROOT}/results/verification/final_delivery_audit.json",
        f"{PACKAGE_ROOT}/output/pdf/A1_NumPy_MLP_Report.pdf",
    }
    missing = sorted(required - set(members))
    if missing:
        raise AssertionError(f"archive is missing required entries: {missing}")
    forbidden = ("data/raw/", ".venv/", "__pycache__/", ".pytest_cache/", "tmp/", "best.npz")
    offenders = [name for name in members if any(token in name for token in forbidden)]
    if offenders:
        raise AssertionError(f"archive contains forbidden entries: {offenders[:10]}")
    for name in members:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts:
            raise AssertionError(f"unsafe archive path: {name}")


def main() -> None:
    DIST.mkdir(parents=True, exist_ok=True)
    files = collect()
    with zipfile.ZipFile(ARCHIVE, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            relative = path.relative_to(ROOT).as_posix()
            archive.write(path, f"{PACKAGE_ROOT}/{relative}")

    with zipfile.ZipFile(ARCHIVE) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise AssertionError(f"CRC failure: {bad}")
        members = archive.namelist()
        validate_members(members)
        with tempfile.TemporaryDirectory(prefix="a1-zip-check-") as temp_dir:
            archive.extractall(temp_dir)
            extracted_root = Path(temp_dir) / PACKAGE_ROOT
            if not (extracted_root / "README.md").is_file():
                raise AssertionError("extracted package validation failed")

    digest = sha256(ARCHIVE)
    checksum_path = DIST / "A1_NumPy_MLP_Submission.sha256"
    checksum_path.write_text(f"{digest}  {ARCHIVE.name}\n")
    record = {
        "status": "passed",
        "archive": str(ARCHIVE.relative_to(ROOT)),
        "bytes": ARCHIVE.stat().st_size,
        "sha256": digest,
        "file_count": len(members),
        "raw_data_included": False,
        "checkpoints_included": False,
        "crc_check": "passed",
        "extraction_check": "passed",
    }
    (DIST / "submission_manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
