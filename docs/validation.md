# Validation of the corrected implementation

The checks below were performed on the corrected code, not inferred from the original notebook outputs.

- **16 numerical/protocol tests passed.** These cover the grouped split, preservation of duplicate weighting, matched initial weights and minibatch sequences, repeatable training, scoped dataset caching, SGD/Adam agreement with PyTorch, correct averaging of evaluation batches, no test access during training, model checkpoint round-trips, explicit test-report guards, count-weighted likelihood/gradient equivalence, smoothing, learned-blend finite differences, decay boundaries, dev-target labels and paired-summary statistics.
- **Both notebooks executed from fresh local IPython processes:** eight code cells in `bigram.ipynb` and nine in `mlp.ipynb`, with no cell errors. Their verified outputs and plots are embedded in the notebooks. The bigram notebook contains two plots; the MLP notebook contains three.
- **24 real-corpus pilot runs completed:** two contexts × two activations × two initialisations × three seeds, 10,000 updates each. The protocol, split, histories, summaries, figures and final model checkpoints are included under `results/initialization-pilot/`.
- **No new real-corpus test loss was evaluated.** Unit tests exercise the explicit test-evaluation command only on a small generated fixture corpus.
- **Historical outputs are preserved separately.** The earlier figures and saved numerical records are under `figs/legacy/` and `results/legacy/`, with corrected interpretation in `docs/legacy-study.md`.

The tested environment used Python 3.12.14, PyTorch 2.8.0+cpu, NumPy 2.3.5, matplotlib 3.10.8 and one PyTorch CPU thread. Exact environment and core-code hashes are included in each pilot run's JSON.

The workspace prevented a normal Jupyter kernel from opening its local sockets. Notebook validation therefore used the socket-free `scripts/check_notebooks.py --in-process` path, with a separate Python process for each notebook. Standard Jupyter execution remains the default checker and the added GitHub workflow's path; that hosted workflow has not been run here.

The included pilot does not validate the final-loss claims of a 200k/500k-step study. Longer fixed-budget paired comparisons, the full configuration study and the decay-timing study are implemented and documented but are not represented as completed experiments.
