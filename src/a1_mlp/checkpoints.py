from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np

from .model import MLP


def save_checkpoint(path: str | Path, model: MLP, metadata: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {parameter.name: parameter.data for parameter in model.parameters()}
    descriptor = json.dumps(metadata, sort_keys=True).encode("utf-8")
    arrays["__metadata_utf8__"] = np.frombuffer(descriptor, dtype=np.uint8)
    fd, temporary_name = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(path: str | Path) -> tuple[MLP, dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        if "__metadata_utf8__" not in archive:
            raise ValueError("checkpoint metadata missing")
        metadata = json.loads(archive["__metadata_utf8__"].tobytes().decode("utf-8"))
        model_info = metadata["model"]
        model = MLP(
            model_info["dims"],
            init_seed=int(model_info["init_seed"]),
            dropout_seed=int(model_info["dropout_seed"]),
            dropout=float(model_info["dropout"]),
            dtype=np.dtype(model_info["dtype"]),
        )
        expected = {parameter.name for parameter in model.parameters()}
        present = set(archive.files) - {"__metadata_utf8__"}
        if expected != present:
            raise ValueError(f"checkpoint parameter mismatch: expected {expected}, got {present}")
        for parameter in model.parameters():
            value = archive[parameter.name]
            if value.shape != parameter.data.shape or value.dtype != parameter.data.dtype:
                raise ValueError(f"invalid checkpoint tensor: {parameter.name}")
            parameter.data[...] = value
    return model, metadata

