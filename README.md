# QAMC

QAMC is the checker for *QAMC: An Abstraction-Aware Quantum Program Model Checking Framework*. It reads a model of a quantum program, executes its quantum and classically controlled operations, and checks assertions at the positions named in the model.

## Install and run

The paper experiments used CPython 3.10.12 and NumPy 2.2.6 on Linux. From a checkout of this repository, install the checker and the plotting dependencies in a virtual environment:

```sh
git clone https://github.com/TheQuantumProgram/QAMC.git
cd QAMC
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-reproduction.txt ".[plot]"
qam-check examples/ghz_ccx_measure_control.qam
```

The last command runs the six-qubit example in Section 3.5. Both assertions should report `PASS`. The first checks agreement among three qubits, and the second checks the distribution of the output pair. To install the checker alone on Python 3.10 or later, use `python -m pip install .`. On systems with [uv](https://docs.astral.sh/uv/), `uv venv --python 3.10 .venv` and `uv pip install --python .venv/bin/python -r requirements-reproduction.txt ".[plot]"` provide the equivalent pinned installation.

The command `qam-check path/to/model.qam` checks the assertions in a model. Use `--properties path/to/properties.qprop` to include a separate property file. Exit statuses are 0 for success, 1 for validation failure or insufficient joint-state information, 2 for a violated property, and 3 for an undefined check.

## Reproduce the paper experiments

Run the commands below from the repository root with the environment activated. Outputs go under `runs/`. The archived inputs, measurements and rendering data are under [experiment_data](experiment_data/README.md).

| Paper result | Command | What to check |
| --- | --- | --- |
| RQ1, Table 1 | `python validation/rq1_property_audit/verify_rq1.py --output runs/rq1.json` | The 35 component judgments for the 22 rows agree with the full-state reference. Six models have expected violations. |
| RQ1 and RQ2, Tables 1 and 2 | `qam-evaluate --measure-models --output runs/rq1-rq2.csv` | One measurement for each of 22 circuit models and six hybrid programs. Compare each actual judgment with its expected judgment. |
| RQ2 feedback control | `python validation/feedback_control/run_feedback_comparison.py --output runs/teleportation.json` | Independent full-state calculations confirm the correct output of teleportation and the violation when the Z correction is omitted. |
| RQ3, Figure 5 | `qam-evaluate --rq3-scalability --rq3-sizes 5-20 --rq3-repeats 5 --rq3-results-dir runs/rq3` | 640 individual timings in `rq3_trials.csv`. The 128 rows of `rq3_scalability.csv` summarize five runs per family, size and representation. |
| RQ4, Table 3 | `qam-evaluate --rq4-large-scale --rq4-sizes 50,100,200,300,400,500 --rq4-results-dir runs/rq4` | All 30 configurations meet their assertions. Each row records execution time, assertion-checking time and retained quantum-array memory. |

For a short first run of the scaling experiments, use `--rq3-sizes 5` or `--rq4-sizes 50`. To check the six-qubit example against an independent calculation, run `python validation/verify_running_example.py`. To check the local Grover density matrices against a full ten-qubit reference, run `python validation/verify_grover_density.py`.

## Models, implementation and independent reference

- [Checker implementation](src/qamc/) includes the language parser, validator, execution engines, numerical backends and assertion evaluator. The [full-state reference](src/qamc/experiments/modeling_reference.py) shares the parsed model and computes gate evolution, measurement and reduced states independently of the checker backends.
- [RQ1 and RQ2 models](experiment_data/models/) include their property files. The [RQ1 mapping](validation/rq1_property_audit/manifest.json) identifies the model, properties and expected result for each Table 1 row. The [teleportation negative control](validation/feedback_control/models/) is supplied alongside its comparison script.
- [The scaling-model archive](experiment_data/scaling_models.zip) contains all 128 RQ3 and 30 RQ4 inputs. Extract it with `python -m zipfile -e experiment_data/scaling_models.zip runs/models`. Rebuild the same archive with `python scripts/generate_scaling_models.py --output runs/scaling_models.zip`. The exporter calls the [RQ3](src/qamc/experiments/rq3.py) and [RQ4](src/qamc/experiments/rq4.py) generators used in the timed experiments.

## Rebuild the paper tables and figures

```sh
python scripts/write_tables.py
python scripts/plot_figure5.py
```

These commands use the recorded measurements to write the three LaTeX tables and Figure 5 under `runs/`. The [measurement archive](experiment_data/paper_results/) includes individual runs, the reported summaries and their provenance. Run `python validation/verify_paper_data.py` to verify the source and data hashes and recompute the reported statistics. The [run manifest](experiment_data/paper_results/manifest.json) identifies the exact measurement commits and commands, including the six Grover runs that used the same Grover implementation. Fresh runs produce machine-dependent timings. The paper counts the bytes of the retained quantum arrays at each generated instance. A shared array counts once per instance under this metric.

[Figures 1–4](figures/README.md) have editable sources and separate build instructions. They can be built with `python scripts/build_method_figures.py` using draw.io Desktop and pdfLaTeX.
