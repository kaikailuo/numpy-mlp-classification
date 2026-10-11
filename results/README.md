# Results directory

Each run writes an immutable directory containing its resolved configuration,
environment, data manifest, epoch metrics, best checkpoint, summary and figures.

- `formal/`: immutable local v1.0 runs and checkpoints (Git ignored).
- `formal_v1_1/`: immutable local v1.1 runs and checkpoints (Git ignored).
- `baseline_v1/`: checkpoint-free lightweight copies of all 16 historical runs.
- `evidence/v1_1/`: checkpoint-free lightweight copies of all 31 v1.1 runs.
- `aggregate/`: untouched v1.0 aggregate.
- `expanded/`: v1.1 aggregate over all 47 runs, including CSV/JSON and PNG/PDF charts.
- `verification/v1_1/`: tests, gradients, matrix status, clean reproduction, and final audit.

The tracked lightweight evidence is sufficient to regenerate `expanded/`
without raw datasets, full run directories, or retraining. The v1.0 and v1.1
aggregate paths are deliberately separate.

