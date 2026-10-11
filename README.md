# A1 NumPy MLP image classification

This repository contains a reproducible, NumPy-only multilayer perceptron for
MNIST and CIFAR-10. Forward and backward propagation, stable losses,
regularization, optimizers, batching, checkpointing, and metrics are implemented
without a deep-learning framework or automatic differentiation.

The completed study contains 16 formal runs. Model and hyperparameter choices
were made from the validation split, written to
`experiments/final_selection.json`, and frozen before the ten reported test-set
evaluations. The test set was not used to launch or select another run.

## Verified headline results

| Result | Validation accuracy | Test accuracy | Test CE |
|---|---:|---:|---:|
| MNIST baseline, CE + SGD | 96.90% | 96.83% | 0.1056 |
| CIFAR-10 baseline, CE + SGD | 51.50% | 50.39% | 1.4425 |
| Best validation-selected optimizer, Momentum | 98.08% | 97.95% | 0.0774 |

The required additional studies cover cross-entropy versus Softmax-MSE,
L1/L2/Dropout, four optimizer families, and online/mini/full-batch update
strategies. Full tables, figures, protocol details, caveats, and the complete
run inventory are in `output/pdf/A1_NumPy_MLP_Report.pdf`.

## Repository layout

```text
src/a1_mlp/                 NumPy implementation and CLI
tests/                      unit, gradient, stability, and integration tests
configs/formal/             immutable TOML configurations for all 16 runs
experiments/                data checks, aggregation, selection, gradient check
data/splits/                deterministic split metadata (raw data excluded)
results/formal/             full local run evidence and checkpoints (Git-ignored)
results/aggregate/          aggregate tables and figures
results/verification/       correctness and data-validation evidence
report/generate_report.py   report generator backed by real result artifacts
output/pdf/                 final rendered report
scripts/                    delivery audit and ZIP builder
docs/                       design baseline, decisions, progress, acceptance
```

## Environment

The project requires CPython 3.12 and uses `uv` for an isolated, locked
environment. The report dependency group is separate from the NumPy training
runtime.

```bash
UV_CACHE_DIR=/tmp/a1-uv-cache uv sync --locked --group dev --group report
MPLCONFIGDIR=/tmp/a1-mpl UV_CACHE_DIR=/tmp/a1-uv-cache \
  uv run --group dev pytest -q
```

## Data preparation

These commands download and checksum the official-format archives, create a
stratified 55,000/5,000 MNIST split and 45,000/5,000 CIFAR-10 split, and fit
CIFAR-10 normalization statistics from the training subset only.

```bash
UV_CACHE_DIR=/tmp/a1-uv-cache uv run python -m a1_mlp.cli prepare-data \
  --dataset mnist --root data/raw/mnist --download
UV_CACHE_DIR=/tmp/a1-uv-cache uv run python -m a1_mlp.cli prepare-data \
  --dataset cifar10 --root data/raw/cifar10 --download
MPLCONFIGDIR=/tmp/a1-mpl UV_CACHE_DIR=/tmp/a1-uv-cache \
  uv run python experiments/verify_data.py
```

Raw datasets are intentionally excluded from Git and the submission ZIP.

## Reproducing training

Run one baseline into a new directory:

```bash
MPLCONFIGDIR=/tmp/a1-mpl UV_CACHE_DIR=/tmp/a1-uv-cache \
  uv run python -m a1_mlp.cli train \
  --config configs/formal/B-M.toml --output results/reproduced/B-M
```

Run the complete predeclared matrix without overwriting the supplied evidence:

```bash
for config in configs/formal/*.toml; do
  run_id="$(basename "$config" .toml)"
  MPLCONFIGDIR=/tmp/a1-mpl UV_CACHE_DIR=/tmp/a1-uv-cache \
    uv run python -m a1_mlp.cli train \
    --config "$config" --output "results/reproduced/$run_id"
done
```

The full matrix is CPU-friendly on the recorded host, but the existing
`results/formal/` directories are the evidence used in the report and should
not be overwritten. Checkpoints contain the resolved configuration and can be
evaluated explicitly, for example:

```bash
MPLCONFIGDIR=/tmp/a1-mpl UV_CACHE_DIR=/tmp/a1-uv-cache \
  uv run python -m a1_mlp.cli evaluate \
  --checkpoint results/formal/B-M/best.npz --split validation \
  --output results/reproduced/B-M-validation.json
```

Do not use `--split test` for hyperparameter selection. The already executed
test protocol and exact checkpoint list are recorded in
`experiments/final_selection.json`.

## Regenerating tables, figures, report, and package

No retraining is needed for these steps:

```bash
MPLCONFIGDIR=/tmp/a1-mpl UV_CACHE_DIR=/tmp/a1-uv-cache \
  uv run python experiments/aggregate.py
UV_CACHE_DIR=/tmp/a1-uv-cache \
  uv run --group report python report/generate_report.py
UV_CACHE_DIR=/tmp/a1-uv-cache \
  uv run --group report python scripts/audit_delivery.py
UV_CACHE_DIR=/tmp/a1-uv-cache \
  uv run python scripts/build_submission.py
```

The report was compiled to a 12-page A4 PDF and every rendered page was
visually checked for clipped text, table overflow, figure placement, formulas,
and agreement with the underlying JSON/CSV evidence. The package builder
rejects raw datasets, caches, checkpoints, and temporary files and verifies the
ZIP after writing it.

## Reproducibility and evidence

- `docs/A1 技术方案设计 v1.0.md` is the frozen design baseline.
- `docs/stage_acceptance.md` maps S0-S5 gates to actual artifacts.
- `docs/decisions.md` records budget and data-source decisions.
- `docs/final_audit.md` contains the final consistency and packaging audit.
- `results/aggregate/all_runs.csv` contains all 16 runs, including weak results.
- `results/verification/` contains test, gradient, toy-overfit, and data checks.

Random seeds, split hashes, data checksums, configuration hashes, environment
manifests, epoch histories, and console logs are retained. Most experiments use
one seed, so differences are descriptive rather than estimates of statistical
significance. CIFAR-10 is deliberately treated as flattened pixels; its result
is a NumPy-MLP baseline and should not be compared with convolutional models as
if the architectures were equivalent.

## AI assistance disclosure

Codex/ChatGPT assisted with implementation, debugging, test construction,
experiment orchestration, aggregation, and report generation. All numerical
claims come from locally executed commands and persisted artifacts. Human work
remaining is limited to checking the course portal's current file-naming,
size/count, AI-disclosure, and submission rules and performing the actual
upload. No repository was published and no teaching-cloud submission was made.

