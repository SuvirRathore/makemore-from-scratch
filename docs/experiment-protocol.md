# Experiment Protocol

## Prediction Target and Partition

The target is next-character prediction in name spellings not seen during training. All copies of an exact spelling are assigned to the same partition. Duplicate records retain their original frequency weighting within that partition.

`load_corpus` sorts unique spellings, shuffles with a local Python RNG seeded with 42, and allocates 80/10/10 percent of the unique spellings. Record counts can differ slightly from those percentages. Each name contributes its characters and one end token to the loss. The start and end marker is `.`; the vocabulary is fixed from the corpus alphabet.

Every study saves its complete group membership, source-file SHA-256, vocabulary, record counts, target counts, split seed and split identity. Historical results from the earlier row split are retained separately. They are not reference scores for the corrected partition.

## Randomness and Matched Comparisons

Every parameter tensor has a deterministic named RNG stream derived from the model seed using SHA-256. Optional blend coefficients have a separate stream. Minibatch selection and sampling also have separate streams.

At a fixed seed and model shape, tanh and blend share identical initial embeddings, weights and biases and identical minibatch indices. Changing context size leaves the target examples in the same order and preserves the initial values of identical-shaped tensors. The input matrix changes shape and is not claimed to be identical. Sharing draws also does not imply the two activation functions produce identical initial outputs.

The default CPU study uses one PyTorch thread. Its environment and code hashes are recorded. Fixed seeds make results repeatable within the tested environment; bitwise agreement across different hardware or library builds is not guaranteed.

## Initialisation Intervention

The unscaled initialisation uses standard-normal embeddings, matrices and biases. The scaled variant uses the same base draws, multiplies the first-layer matrix by $(5/3)/\sqrt{\mathrm{fan\_in}}$, sets both biases to zero, and multiplies the output matrix by $0.01/\sqrt{\mathrm{hidden\_width}}$. Embeddings retain their standard-normal distribution.

The same hidden gain is used for both tanh and blend to hold that choice fixed. It is a reference intervention, not an assertion of optimal scaling for every activation. The intervention changes several aspects of initialisation together; it is not an isolated test of hidden-weight scaling alone.

Initial and final diagnostics measure pre-activation mean and standard deviation, mean tanh derivative, the fraction of values with tanh magnitude above 0.99, and logit standard deviation on the same first 2,048 training examples. These are activation-value statistics, not a count of permanently saturated neurons. Gradient norms are recorded at validation checkpoints.

## Metrics and Timing

The model optimises minibatch cross-entropy. Final train and dev NLL are separately evaluated over all target characters in their partitions, in natural-log units. Chunked evaluation sums losses and divides by the number of examples, including a possibly shorter final chunk.

Each history entry states the number of completed updates. Step zero is evaluated before training. Later dev evaluations occur after the stated updates; each training-window mean is labelled separately and is not a dev score. Learning-rate decay occurs after an absolute number of completed updates: a decay step of 100,000 first changes update 100,001.

Time-to-target is the first scheduled dev evaluation at or below a pre-specified target. It is not interpolated, not a smoothed training-loss threshold, and not a claim of sustained attainment. Timing includes initial diagnostics, training and dev evaluation, but excludes data/model construction and saving. The total run duration additionally includes final training-set evaluation and final diagnostics. Compare runs with the same evaluation cadence, device and thread count. Full-run seconds divided by steps are not a direct measurement of time-to-target.

## Frozen Study Plans

The runner writes every configuration and the split manifest before training any run in a study. Existing output directories are refused, preventing silent replacement of a protocol or its results. Completed runs are saved independently, so earlier results survive a later run failure. A failed study records the failing run and exception. There is no implicit retry, auto-selection or resume from a different checkpoint.

| Suite | Design per seed | Purpose |
| --- | --- | --- |
| `smoke` | Context 3/4, tanh/blend, chosen initialisation | Short engineering verification; four runs per seed. |
| `main` | Ten configurations, including SGD/Adam with and without decay, context changes, and a shorter baseline | Exploratory controls. The short baseline uses 40% of the requested budget and proportionally earlier decay. |
| `paired` | Tanh/blend, context 3, chosen initialisation, identical budget | Main activation comparison; two runs per seed. |
| `initialization` | Unscaled/scaled × context 3/4 × tanh/blend | Initialisation intervention; eight runs per seed. |
| `decay` | Decay at 25%, 50%, 75%, or none, fixed budget | Compare dev-loss attainment and compute; four runs per seed. |

The included pilot uses the `initialization` design at 10,000 updates, three seeds, 1,000-update evaluation cadence, and decay at 5,000 updates. It is an early-training study. Explanations suggested by this pilot require longer-budget checks and, for causal separation, more targeted interventions.

For the longer study, the predictions to test are:

1. Scaled initialisation reduces initial tanh saturation and puts initial NLL near the uniform baseline when output logits are close to zero.
2. Any early benefit of the blend under unscaled initialisation shrinks under the scaled reference initialisation.
3. Context four's effect on dev loss can depend on activation and initialisation; it is not assumed that only the blend can use extra context.
4. An earlier decay may reach a pre-specified dev target with fewer updates or less time. No particular compute saving is assumed beforehand.

The pilot outcomes are already visible. Pre-specifying a later protocol cannot retroactively make the pilot confirmatory. Record any revisions to these hypotheses before launching the longer study.

## Statistical Summaries

Matched comparisons are grouped by identical split, budget, schedule, optimiser, architecture dimensions and initialisation. Only the activation and per-seed labels vary. Unmatched outcomes are not silently treated as pairs. The saved differences are $D_s=L_{\mathrm{blend},s}-L_{\mathrm{tanh},s}$, so negative favours blend.

Report each $D_s$, its mean, sample standard deviation, and standard error $s_D/\sqrt n$. No sample standard deviation or standard error is reported for a single pair. These summaries quantify seed variation conditional on the protocol. They do not quantify data-split uncertainty or correct for exploring many configurations. The code does not use a seed range as a significance threshold or colour results as statistically better/worse.

A smaller train/dev gap is not by itself better generalisation; inspect dev NLL and training NLL separately. Comparisons of activation functions can change both optimisation and representational capacity. The blend has one extra trainable parameter per hidden neuron.

## Checkpoints and Test Evaluation

Each run saves a model checkpoint with its configuration, vocabulary and split identity, plus a JSON record with its checksum, histories, diagnostics, timing and RNG stream seeds. Loading uses PyTorch's `weights_only=True` mode. The checkpoint captures the final trained model; it does not include optimiser state for resuming training.

The training API has no test-evaluation option. Notebook Run All uses only train/dev data. Once dev-based selection is finished, the separate `test` subcommand evaluates an explicitly selected checkpoint, verifies the split and vocabulary, and records the checkpoint checksum. It refuses to overwrite an existing report. The report filename is not a global access control: avoiding repeated test-guided selection remains an experimental discipline.

The original record-split test set was previously evaluated. It cannot be described as historically untouched, and the new grouped partition is also not an independent new data collection. None of the new real-corpus runs in this revision evaluates the new test partition.
