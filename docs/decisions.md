# Technical decisions

## D-001 - Python runtime

The host default is Python 3.14.7, while the frozen design specifies Python 3.12. The project therefore declares `>=3.12,<3.13` and lets `uv` provision an isolated 3.12 runtime.

## D-002 - Formal experiment budget

No formal budget reduction has been made yet. It will be selected only after the required MNIST and CIFAR-10 100-step benchmarks. P0/P1 coverage will be preserved; P2 is optional.

## D-003 - CIFAR-10 download mirror

The University of Toronto origin transferred at roughly 60-70 KB/s. A suspected TensorFlow/Keras Google Cloud mirror returned HTTP 404 and was rejected. The project therefore uses PaddlePaddle's documented Baidu Object Storage mirror of the original Python archive. Acceptance still requires the University of Toronto/PyTorch published MD5 `c58f30108f718f92721af3b95e74349a`; incomplete attempts were retained only as `.part` files and never loaded.

## D-004 - Formal training budget after benchmark

The 100-step measurements were 80,669 samples/s for MNIST and 17,337 samples/s for CIFAR-10 (training steps only). Both frozen baseline networks are comfortably below the 45-minute/run governance line. Therefore P0 uses the full 55k and 45k training splits with the original 12 and 15 epochs. All P1 loss/regularization/optimizer runs also use the full MNIST training split. Only the explicitly controlled batch-strategy experiment uses a disclosed fixed 5,000-sample training subset, because B=1 otherwise changes runtime by orders of magnitude.

