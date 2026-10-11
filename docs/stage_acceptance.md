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

## S2 - In progress

## S3 - Pending

## S4 - Pending

## S5 - Pending

