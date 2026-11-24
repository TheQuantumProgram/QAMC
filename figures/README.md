# Figures 1–4

| Figure | Contents | Editable source |
| --- | --- | --- |
| 1 | Model structure | [draw.io diagram](qam_model_structure.drawio) |
| 2 | Program listing and six-qubit circuit | [Figure layout and program](figure2.tex), [Quantikz circuit](qam_modeling_example_circuit.tex) |
| 3 | State classes and execution of the example | [draw.io diagram](qam_modeling_example_models.drawio) |
| 4 | Framework architecture | [draw.io diagram](qam_framework.drawio) |

From the repository root, run:

```sh
python scripts/build_method_figures.py
```

The script writes cropped vector PDFs to `runs/figure1.pdf` through `runs/figure4.pdf`. Figures 1, 3 and 4 use the draw.io Desktop PDF exporter. Install draw.io Desktop and place its executable on `PATH`, or supply its path with `--drawio`. The diagrams were exported with version 29.0.3 on Windows. Font availability and renderer versions can affect the layout.

Figure 2 uses pdfLaTeX with the `standalone`, `quantikz`, `libertine`, and `newtx` packages. To build just the program and circuit, run `python scripts/build_method_figures.py --figures 2`. The accompanying executable model is [ghz_ccx_measure_control.qam](../examples/ghz_ccx_measure_control.qam).

The script for [Figure 5](../scripts/plot_figure5.py) plots the RQ3 measurement data.
