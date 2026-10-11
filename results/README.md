# Results directory

Each run writes an immutable directory containing its resolved configuration,
environment, data manifest, epoch metrics, best checkpoint, summary and figures.
Raw run directories are intentionally ignored by Git because checkpoints can be
large. `results/aggregate/` is generated from those run directories.

