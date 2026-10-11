# A1 NumPy MLP image classification

This repository implements a reusable multilayer perceptron with NumPy only and
runs reproducible MNIST and CIFAR-10 classification experiments. Neural-network
forward/backward propagation, losses, regularization, optimizers and metrics do
not depend on an automatic-differentiation or deep-learning framework.

The project is currently under active autonomous execution. The final version
of this file will include verified metrics, full reproduction commands, the
report and submission-package paths.

## Environment

```bash
UV_CACHE_DIR=/tmp/a1-uv-cache uv sync --group dev
UV_CACHE_DIR=/tmp/a1-uv-cache uv run pytest -q
```

## Data preparation

```bash
UV_CACHE_DIR=/tmp/a1-uv-cache uv run python -m a1_mlp.cli prepare-data \
  --dataset mnist --root data/raw/mnist --download
UV_CACHE_DIR=/tmp/a1-uv-cache uv run python -m a1_mlp.cli prepare-data \
  --dataset cifar10 --root data/raw/cifar10 --download
```

## Training

```bash
UV_CACHE_DIR=/tmp/a1-uv-cache uv run python -m a1_mlp.cli train \
  --config configs/mnist_baseline.toml
```

See `docs/A1 技术方案设计 v1.0.md` for the frozen mathematical and experimental
protocol. Raw data and large run checkpoints are excluded from Git and from the
final source ZIP.

