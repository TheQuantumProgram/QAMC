# QAMC

QAMC is the checker for *QAMC: An Abstraction-Aware Quantum Program Model Checking Framework*. A model specifies quantum operations, measurements, classical control, and the groups of qubits to represent at each stage. The checker executes the model and evaluates its assertions on the resulting branches.

## Install and run

From the repository root, with Python 3.10 or later:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install ".[plot]"
qam-check examples/ghz_ccx_measure_control.qam
```

The six-qubit example in Section 3.5 checks that the first three qubits have equal measurement outcomes and that the output pair has probabilities (5/8, 1/8, 1/8, 1/8) in computational basis order. Both assertions should print `PASS`. To install the checker alone, use `python -m pip install .`.

## Run the paper experiments

Run the following commands from the repository root with the environment activated.

| Experiment | Command | Expected result |
| --- | --- | --- |
| RQ1 · Correctness | `python validation/rq1_property_audit/verify_rq1.py --output runs/rq1.json` | The checker and the independent full-state calculation agree with all 22 expected judgments, including six violations. |
| RQ1 and RQ2 · Model measurements | `qam-evaluate --measure-models --output runs/rq1-rq2.csv` | The CSV records time, retained memory, and the actual and expected judgments for the 22 circuit models and six hybrid programs. |
| RQ2 · Teleportation feedback | `python validation/feedback_control/run_feedback_comparison.py --output runs/teleportation.json` | The original passes. Omitting the conditional Z correction causes the same assertion to fail. |
| RQ3 · Efficiency | `qam-evaluate --rq3-scalability --rq3-sizes 5-20 --rq3-repeats 5 --rq3-results-dir runs/rq3` | Four families, each checked with a full and an abstract model at 16 sizes. |
| RQ4 · Scalability | `qam-evaluate --rq4-large-scale --rq4-sizes 50,100,200,300,400,500 --rq4-results-dir runs/rq4` | Five families, each checked at six sizes. |

To try the generators on a smaller workload, pass `--rq3-sizes 5` or `--rq4-sizes 50` to the corresponding command. RQ4 generates its Grover benchmarks at the requested sizes and checks the local witness-subspace property.

## Rebuild the published tables and figure

```sh
python scripts/write_tables.py
python scripts/plot_figure5.py
```

The scripts read the [archived data](experiment_data/paper_results/) and write Tables 1–3 and Figure 5 under `runs/`. The RQ1 and RQ2 CSVs preserve the reported table values. The RQ3 CSV contains one summary per configuration over five timed runs, and the RQ4 CSV contains one measurement per configuration. New runs measure the released checker on the local machine. The archived CSVs preserve the timings used in the paper.
