from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np


def environment_snapshot() -> dict:
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        git_commit = None
    try:
        blas = str(np.show_config(mode="dicts"))
    except TypeError:
        blas = "see console output from numpy.show_config"
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "blas": blas,
        "thread_environment": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
        },
        "git_commit": git_commit,
        "cwd": str(Path.cwd()),
    }

