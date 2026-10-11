# Stage acceptance record

All evidence in this file must come from commands actually executed in this repository.

## S0 - Accepted

- Baseline: `docs/A1 技术方案设计 v1.0.md`
- Evidence: document exists in the initial Git commit `650dc9a`.
- Scope override: the user's latest execution prompt authorizes continuous S0-S5 progress without per-stage approval.

## S1 - Accepted

- Environment: uv-managed CPython 3.12.13; NumPy 2.5.4; macOS 15.7.3 x86_64; 8 logical CPUs.
- Tests: 21 passed in 0.33 s. JUnit evidence: `results/verification/pytest.xml`.
- Finite differences: Dense, CE, Softmax-MSE and full MLP checks all passed the frozen tolerance through the test suite.
- Tiny overfit: loss 1.1721364772 -> 0.0011530956 after 500 steps; accuracy 1.0. Evidence: `results/verification/tiny_overfit.json`.
- MNIST: official file checksums accepted; 55,000/5,000/10,000; split SHA-256 `69e9c4a5d2bdf9ac85395a8ef11819be4bd88a78b1dc733be641ba751644317f`.
- CIFAR-10: archive MD5 `c58f30108f718f92721af3b95e74349a`; 45,000/5,000/10,000; split SHA-256 `3aa61de0e8e9e4bae6e2712fbf67a29c10d3c792c192f40106360e5b7e076110`.
- Real-data evidence: `results/verification/data_validation.json`.
- Actual commands: `uv sync --group dev`, `uv run pytest -q`, `python -m a1_mlp.cli toy-overfit`, both `prepare-data` commands, `experiments/verify_data.py`, and both 100-step benchmarks.
- Result: T-01 through T-11 implementation tests needed at this stage pass; real loaders and isolation metadata pass. S1 gate accepted internally.

## S2 - Accepted

- Run: `results/formal/B-M`; full 55,000-sample train split, 12 epochs, 660,000 samples seen.
- Best checkpoint: epoch 10 by the frozen selection rule; validation accuracy 0.9690; validation CE 0.11422997.
- Reload check: `validation_evaluation.json` exactly matches the checkpoint-selection metrics.
- Artifacts: config snapshot, environment, manifest, metrics CSV, atomic NPZ, console log, learning curves and validation confusion matrix all exist.

## S3 - Accepted

- Run: `results/formal/B-C`; full 45,000-sample train split, 15 epochs, 675,000 samples seen.
- Best checkpoint: epoch 14; validation accuracy 0.5150; validation CE 1.44490325.
- Reload check: `validation_evaluation.json` exactly matches the checkpoint-selection metrics.
- Interpretation: the flattened-pixel MLP shows a growing train/validation gap; this is retained as a real limitation rather than hidden.
- Artifacts: the same complete run bundle as S2 exists, including learning curves and confusion matrix.

## S4 - Accepted

- Formal inventory: 16/16 runs have `status=completed`; no weak or unfavorable run was removed. Evidence: `results/aggregate/all_runs.csv` and every `results/formal/*/summary.json`.
- Loss comparison: B-M cross-entropy validation accuracy 0.9690 versus L-MSE Softmax-MSE 0.9368 under matched architecture, seed, batch order, learning rate, and epoch budget.
- Regularization comparison: R-L1 0.9684, R-L2 0.9682, and R-DROP 0.9666 validation accuracy. The modest effect is reported rather than overstated.
- Optimizer candidates: validation-selected representatives were SGD 0.1 (0.9756), Momentum 0.05 (0.9808), AdaGrad 0.01 (0.9746 with lower CE than the tied candidate), and Adam 0.003 (0.9784). Evidence: `results/aggregate/optimizer_selection.json`.
- Batch strategy: on the disclosed fixed 5,000-sample subset, B=1, B=128, and B=5000 each saw 50,000 examples but made 50,000, 400, and 10 updates; validation accuracy was 0.9168, 0.8902, and 0.3948 respectively. This is documented as an update-regime study, not equal-compute tuning.
- Every formal run retains its source and resolved configuration, environment, data manifest, epoch CSV, checkpoint, console log, summary, and figures. The 16 source TOMLs were also exported to `configs/formal/` for reproduction.
- Result: both required additional topics (optimizer comparison and batch strategy), plus loss and regularization comparisons, are complete. S4 accepted internally.

## S5 - Accepted

- Test isolation: `experiments/final_selection.json` has `test_access_before_freeze=false`. The ten directories containing `test_evaluation.json` exactly match its ten selected checkpoints; no batch-subset or rejected optimizer candidate was tested for reselection.
- Frozen test metrics: B-M 0.9683 accuracy / 0.105596 CE; B-C 0.5039 / 1.442483; validation-selected O-MOM-0.05 0.9795 / 0.0774. Complete results: `results/aggregate/final_test_metrics.json`.
- Final correctness run: 21 tests passed with zero failures/errors/skips. Evidence: `results/verification/final_pytest.xml`.
- Delivery audit: 16 completed formal runs, ten frozen tests, matching config snapshots, passed real-data validation, 100% tiny-set fit, full-MLP maximum absolute gradient error 2.6609159320400977e-11. Evidence: `results/verification/final_delivery_audit.json`.
- Report: `output/pdf/A1_NumPy_MLP_Report.pdf`, A4, 12 pages, unencrypted. All pages were rendered to PNG and visually inspected; tables, formulas, plots, captions, commands, and metric values fit without clipping or overlap.
- Reproduction: README, locked dependencies, all formal TOMLs, aggregation/report/audit/package scripts, and explicit no-overwrite commands are present.
- Package: `dist/A1_NumPy_MLP_Submission.zip` passed ZIP CRC and clean extraction checks and contains source, tests, configs, documentation, lightweight experiment evidence, aggregate figures, and the PDF. Raw datasets, caches, and `best.npz` checkpoints are intentionally excluded; the full checkpoints remain in the local workspace.
- Result: S5 and the complete S0-S5 project are accepted internally. Only course-portal rule confirmation and the actual upload remain human actions.

