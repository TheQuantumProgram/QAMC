# Teleportation feedback control

The correct teleportation circuit and the circuit with the missing Z correction use the same final assertion. For each circuit the report records both terminal X-basis distributions, the four feedback branches, and the checker and reference judgments.

The [JSON report](reference_results.json) is the original output of this
command, run from the repository root:

```sh
OPENBLAS_NUM_THREADS=1 python validation/feedback_control/run_feedback_comparison.py --output runs/feedback-reference.json
```

The archived report has SHA-256 `5d609e947bd4905ba7d30692f697fd5b97d9022a48b97d19038481271d1a7165`.
The [run manifest](../../experiment_data/paper_results/manifest.json) records
the measured checker revision.
