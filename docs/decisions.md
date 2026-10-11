# Technical decisions

## D-001 - Python runtime

The host default is Python 3.14.7, while the frozen design specifies Python 3.12. The project therefore declares `>=3.12,<3.13` and lets `uv` provision an isolated 3.12 runtime.

## D-002 - Formal experiment budget

No formal budget reduction has been made yet. It will be selected only after the required MNIST and CIFAR-10 100-step benchmarks. P0/P1 coverage will be preserved; P2 is optional.

## D-003 - CIFAR-10 download mirror

The University of Toronto origin transferred at roughly 60-70 KB/s. A suspected TensorFlow/Keras Google Cloud mirror returned HTTP 404 and was rejected. The project therefore uses PaddlePaddle's documented Baidu Object Storage mirror of the original Python archive. Acceptance still requires the University of Toronto/PyTorch published MD5 `c58f30108f718f92721af3b95e74349a`; incomplete attempts were retained only as `.part` files and never loaded.

## D-004 - Formal training budget after benchmark

The 100-step measurements were 80,669 samples/s for MNIST and 17,337 samples/s for CIFAR-10 (training steps only). Both frozen baseline networks are comfortably below the 45-minute/run governance line. Therefore P0 uses the full 55k and 45k training splits with the original 12 and 15 epochs. All P1 loss/regularization/optimizer runs also use the full MNIST training split. Only the explicitly controlled batch-strategy experiment uses a disclosed fixed 5,000-sample training subset, because B=1 otherwise changes runtime by orders of magnitude.

## D-005 - P1 completion and P2 scope

The required P0/P1 matrix completed well within the measured CPU budget: two
full-dataset baselines, loss and three regularization variants, seven additional
optimizer candidates, and three controlled batch strategies. Optional P2
extensions (multi-seed statistics, RMSProp search, early stopping, or a
convolutional model) were not added because they are outside the completion
criteria and would dilute the controlled one-factor comparisons. This is a
declared limitation, not an omitted or failed result.

## D-006 - Frozen test protocol

Checkpoint selection was serialized to `experiments/final_selection.json`
before the test split was evaluated. Exactly ten representative checkpoints
were tested: both dataset baselines, every required loss/regularization model,
and one validation-selected candidate per optimizer family. Rejected optimizer
candidates and the 5,000-sample batch-strategy models remain validation-only.
No run or hyperparameter was launched in response to test accuracy.

## D-007 - Report and submission packaging

The final paper is generated from persisted CSV/JSON evidence and fails early
if the expected 16-run/10-test inventory is incomplete. Report dependencies are
isolated in the locked `report` dependency group. The source ZIP includes
lightweight logs, metrics, manifests, figures, tests, configs, source, and the
PDF, but excludes raw datasets, caches, temporary renderings, and large
checkpoints. This keeps the package auditable without making it unsuitable for
typical teaching-cloud limits; the original checkpoints remain under
`results/formal/` in the workspace.

