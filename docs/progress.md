# A1 project progress

Last updated: 2026-10-11 (Asia/Shanghai)

## Current stage

S5 - Complete. Final report, reproducibility materials, audit, and submission
package have been generated and verified.

## Completed

- S0: Read and accepted `A1 技术方案设计 v1.0.md` as the technical baseline.
- Recorded initial host facts: macOS 15.7.3 x86_64, 8 logical CPUs, about 101.6 GB free disk.
- Initialized the Python package, configuration, test and evidence directory layout.
- S1: Implemented Dense/ReLU/Dropout/MLP, stable CE and Softmax-MSE, L1/L2, SGD/Momentum/AdaGrad/Adam/RMSProp, data loaders, deterministic splits, checkpoints, trainer, CLI and plots.
- S1 gate: 21 tests passed; finite-difference checks passed; toy set reached 100% accuracy; both real datasets passed checksum/shape/split/preprocess validation.
- Benchmarks: MNIST 80,669 samples/s and CIFAR-10 17,337 samples/s for 100 training steps on the current host.
- S2: MNIST baseline completed; best validation accuracy 0.9690 at epoch 10, validation CE 0.11423.
- S3: CIFAR-10 baseline completed; best validation accuracy 0.5150 at epoch 14, validation CE 1.44490.
- S4: completed all 14 required comparison runs in addition to the two baselines; all 16 formal runs are retained under `results/formal/`.
- Froze `experiments/final_selection.json` before opening the test split and evaluated exactly the ten listed checkpoints.
- Frozen test results include MNIST baseline 0.9683, CIFAR-10 baseline 0.5039, and validation-selected Momentum 0.9795 accuracy.
- Aggregated every formal run and generated comparison tables and figures under `results/aggregate/`.
- S5: generated the 12-page paper-style PDF, rendered all 12 pages, and visually checked text, formulas, tables, plots, and reported values.
- Final verification: 21 tests passed, full-MLP finite-difference maximum absolute error was 2.6609159320400977e-11, and the toy set reached 100% training accuracy.
- Exported all 16 immutable formal TOML configurations under `configs/formal/` and completed the README reproduction guide.
- Built and CRC/extraction-validated `dist/A1_NumPy_MLP_Submission.zip`; raw data, caches, and large checkpoints are excluded.
- No S0-S4 training was repeated during S5 recovery, and no decision was changed after test evaluation.

## Remaining

No project implementation or experiment work remains. The course portal's
current naming, size/count, AI-disclosure, and upload requirements require
manual confirmation immediately before submission.

## Next action

Human action only: review the final PDF and ZIP, confirm the teaching-cloud
rules, and upload the artifacts. No public repository or remote submission has
been performed.

