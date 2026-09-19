# Validation of the corrected implementation

The checks below were performed on the corrected code, not inferred from the original notebook outputs.

- **22 numerical/protocol tests passed.** These cover the grouped split, preservation of duplicate weighting, matched initial weights and minibatch sequences, repeatable training, scoped dataset caching, SGD/Adam agreement with PyTorch, correct averaging of evaluation batches, no test access during training, model checkpoint round-trips, explicit test-report guards, count-weighted likelihood/gradient equivalence, smoothing, learned-blend finite differences, decay boundaries, dev-target labels and paired-summary statistics.
- **Both notebooks executed from fresh local IPython processes:** nine code cells in `bigram.ipynb` and thirteen in `mlp.ipynb`, with no cell errors. Their verified outputs and plots are embedded in the notebooks. Both notebooks contain two plots.
- **24 real-corpus pilot runs completed:** two contexts × two activations × two initialisations × three seeds, 10,000 updates each. The protocol, split, histories, summaries, figures and final model checkpoints are included under `results/initialization-pilot/`.
- **No new real-corpus test loss was evaluated.** Unit tests exercise the explicit test-evaluation command only on a small generated fixture corpus.
- **Historical outputs are preserved separately.** The earlier figures and saved numerical records are under `figs/legacy/` and `results/legacy/`, with corrected interpretation in `docs/legacy-study.md`.

The notebooks now write out the data construction, forward pass, loss and
parameter updates directly. Six additional tests compare those calculations
with the command-line implementation, including split identity, minibatch pairing,
SGD/Adam updates, checkpoint compatibility, sampling and bigram likelihoods.
Across both activations, both initialisations and both optimisers, the compared
parameters and losses match exactly.

Two 10,000-update pilot runs (`unscaled-bs4-blend-s0` and
`scaled-bs3-tanh-s0`) were also reproduced from the notebook implementation.
Their final parameter tensors, non-timing loss histories and diagnostics match
the saved records exactly. Original pilot files were not overwritten.

The notebook's optional `RUN_TRAINING=True` path was exercised at a short
20-update budget for all 24 configurations. Protocols, checkpoints, result
files, summaries and plots were checked under `results/local/`; these execution
checks do not replace the included 10,000-update pilot or count as new findings.

The tested environment used Python 3.12.14, PyTorch 2.8.0+cpu, NumPy 2.3.5, matplotlib 3.10.8 and one PyTorch CPU thread. Exact environment and core-code hashes are included in each pilot run's JSON.

The workspace prevented a normal Jupyter kernel from opening its local sockets. Notebook validation therefore used the socket-free `scripts/check_notebooks.py --in-process` path, with a separate Python process for each notebook. Standard Jupyter execution remains the default checker and the added GitHub workflow's path; that hosted workflow has not been run here.

The included pilot does not validate the final-loss claims of a 200k/500k-step study. Longer fixed-budget paired comparisons, the full configuration study and the decay-timing study are implemented and documented but are not represented as completed experiments.
