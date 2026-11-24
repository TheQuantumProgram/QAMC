# Measurements reported in the paper

[The run manifest](manifest.json) identifies the source commits, commands,
software environment and SHA-256 hashes of the archived CSVs. The runs used
the same Intel Core i7-13700HX system and NumPy/OpenBLAS configuration.
Each original CSV below is the file emitted by the experiment driver.
Run `python validation/verify_paper_data.py` from the repository root to check the
source and data hashes and recompute every table value and efficiency statistic.

| Output | Original measurements | How the paper uses the measurements |
| --- | --- | --- |
| [Table 1](rq1_table.csv) and [Table 2](rq2_table.csv) | [28 circuit and hybrid runs](raw/rq1_rq2_measurements.csv) | One run per model. The paper rounds time to 0.1 ms and memory to 0.001 MiB. A check mark indicates agreement with the expected judgment. |
| [Figure 5 data](rq3_scalability.csv) | [640 individual trials](raw/rq3_trials.csv) | For every circuit family, size and representation, the plot uses median time and retained-array size across five successful runs. |
| [Table 3](rq4_large_scale.csv) | [24 GHZ, BV, adder and Simon runs](raw/rq4_other_families.csv) and [six Grover runs](raw/rq4_grover.csv) | One run per family and size. The table uses the reported run time, assertion-checking time, their sum and retained-array size. |

The `question`, `row` and `model` columns of the small-model measurements
identify each table row and its source relative to `experiment_data/models/`.
The RQ1 model manifest and full-state judgments are under
[the RQ1 reference report](../../validation/rq1_property_audit/reference_results.json).
For RQ2, [the feedback-control reference report](../../validation/feedback_control/reference_results.json)
records the full-state results for the original and faulty teleportation circuits.

The Grover rows use the same checker and Grover experiment implementation as
the other RQ4 measurements. Their recorded commit precedes the addition of
individual RQ3 trial logging. The two revisions are identified in the manifest.

Run `python scripts/write_tables.py` and `python scripts/plot_figure5.py`
from the repository root to regenerate the tables and Figure 5 from the archived
data. New measurements can be collected using the commands in the
[repository README](../../README.md).
