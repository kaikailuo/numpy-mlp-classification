# Final delivery audit

Audit date: 2026-10-11 (Asia/Shanghai)

## Scope and preservation

S5 recovery began from the existing S0-S4 artifacts. The 16 formal run
directories, their checkpoints, and the ten existing frozen test evaluations
were treated as immutable evidence. No training run was repeated, no checkpoint
was replaced, and no test result was used to modify the frozen selection.

## Executed checks

- `uv sync --locked --group dev --group report` completed with CPython 3.12.
- `pytest -q --junitxml=results/verification/final_pytest.xml` executed 21 tests: 21 passed, zero failed/errors/skipped.
- `scripts/audit_delivery.py` verified 16 completed runs, all required run files, nonempty checkpoints, byte-identical exported/source TOMLs, and exact equality between the frozen selection and the ten tested runs.
- Aggregate CSV/JSON inventories contain all 16 runs and retain the low-performing full-batch and Softmax-MSE results.
- Real-data validation, dataset hashes, disjoint deterministic splits, full-MLP finite differences, checkpoint reload behavior, and the 100%-accuracy toy overfit remain passed.
- Headline report values were checked against `final_test_metrics.json`: MNIST 96.83%, CIFAR-10 50.39%, selected Momentum 97.95% test accuracy.
- The report is an unencrypted 12-page A4 PDF. All 12 pages were rendered at 120 DPI and visually inspected in full-page contact sheets, with dense table/command pages also inspected at original resolution. No clipping, overlap, missing glyph, broken figure, or inconsistent value was found.
- `scripts/build_submission.py` created the ZIP, ran `ZipFile.testzip()`, extracted it into a fresh temporary directory, checked mandatory members, and rejected raw data, caches, path traversal, temporary files, and checkpoints.

Machine-readable audit evidence is in
`results/verification/final_delivery_audit.json`; package size, file count,
SHA-256, and extraction status are in `dist/submission_manifest.json` and
`dist/A1_NumPy_MLP_Submission.sha256`.

## Final artifacts

- Paper: `output/pdf/A1_NumPy_MLP_Report.pdf`
- Submission: `dist/A1_NumPy_MLP_Submission.zip`
- Checksum: `dist/A1_NumPy_MLP_Submission.sha256`
- Source and reproduction guide: `README.md`
- Complete local checkpoints/evidence: `results/formal/`

## Manual confirmation still required

The course portal rules were not available in the project. Before uploading, a
human must confirm the required filename, maximum file size, allowed number of
attachments, whether the PDF must be separate from the ZIP, and the exact AI
assistance disclosure format. The actual upload/submission is intentionally not
performed by this project.
