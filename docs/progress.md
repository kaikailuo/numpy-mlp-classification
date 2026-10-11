# A1 project progress

Last updated: 2026-10-11 (Asia/Shanghai)

## Current stage

S2 - MNIST complete baseline training and validation.

## Completed

- S0: Read and accepted `A1 技术方案设计 v1.0.md` as the technical baseline.
- Recorded initial host facts: macOS 15.7.3 x86_64, 8 logical CPUs, about 101.6 GB free disk.
- Initialized the Python package, configuration, test and evidence directory layout.
- S1: Implemented Dense/ReLU/Dropout/MLP, stable CE and Softmax-MSE, L1/L2, SGD/Momentum/AdaGrad/Adam/RMSProp, data loaders, deterministic splits, checkpoints, trainer, CLI and plots.
- S1 gate: 21 tests passed; finite-difference checks passed; toy set reached 100% accuracy; both real datasets passed checksum/shape/split/preprocess validation.
- Benchmarks: MNIST 80,669 samples/s and CIFAR-10 17,337 samples/s for 100 training steps on the current host.

## Remaining

- S2: MNIST benchmark, full training, validation/checkpoint/figures.
- S3: CIFAR-10 benchmark and full training.
- S4: Required loss, regularization, optimizer and batch-strategy experiments.
- S5: Frozen test evaluation, aggregate results, report PDF, ZIP and final audit.

## Next action

Run the full 12-epoch MNIST baseline, inspect its metrics/checkpoint/plots, and only then accept S2.

