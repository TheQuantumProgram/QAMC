# Full-state validation for RQ1

The full-state reference and QAMC evaluate 35 component judgments across the 22 models in Table 1. The archived report includes the measurements on which each judgment is based and their individual model and property source hashes.

The [JSON report](reference_results.json) is the original output of this
command, run from the repository root:

```sh
OPENBLAS_NUM_THREADS=1 python validation/rq1_property_audit/verify_rq1.py --output runs/rq1-reference.json
```

The archived report has SHA-256 `09230c5e76ecf2a6bce2bbffa5a7e846447b300f060a2ff6bfe51b5d328b6f5e`.
The [run manifest](../../experiment_data/paper_results/manifest.json) records
the measured checker revision.
