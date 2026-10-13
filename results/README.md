# Results directory

Each run writes an immutable directory containing its resolved configuration,
environment, data manifest, epoch metrics, best checkpoint, summary and figures.

- `formal/`: complete v1.0 run evidence, with the 10 frozen selected checkpoints tracked.
- `formal_v1_1/`: complete v1.1 run evidence, with the 4 frozen selected checkpoints tracked.
- `baseline_v1/`: checkpoint-free lightweight copies of all 16 historical runs.
- `evidence/v1_1/`: checkpoint-free lightweight copies of all 31 v1.1 runs.
- `aggregate/`: untouched v1.0 aggregate.
- `expanded/`: v1.1 aggregate over all 47 runs, including CSV/JSON and PNG/PDF charts.
- `verification/v1_1/`: tests, gradients, matrix status, clean reproduction, and final audit.

The tracked lightweight evidence is sufficient to regenerate `expanded/`
without raw datasets, full run directories, or retraining. The v1.0 and v1.1
aggregate paths are deliberately separate.


`reproduction/v1_1/B-M-clean/` includes the clean reproduction evidence and its checkpoint. Other candidate/repeated-seed weights remain local and ignored; their metrics, logs, configuration snapshots and figures are tracked. `data/splits/` also tracks the small index NPZ files and JSON metadata, while raw datasets are excluded.
