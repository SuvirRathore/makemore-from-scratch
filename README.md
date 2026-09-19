# makemore-from-scratch

**Character-level language models, explicit training code and experiments that test why a result improves.**

I implemented count-based and neural bigram models and a character MLP following [Karpathy's makemore series](https://karpathy.ai/zero-to-hero.html), then investigated optimisation schedules, context length and learned activation mixtures. The project develops a model, checks a baseline, tests proposed improvements, and revises explanations when controls or additional seeds change the result.

PyTorch supplies automatic differentiation. The models, parameter initialisation and SGD/Adam updates are written explicitly. Implementing autodiff itself was the focus of my earlier [micrograd project](https://github.com/SuvirRathore/micrograd-from-scratch).

## Overview

- **Three model implementations:** transition counts with smoothing, neural bigram logits, and an MLP with learned character embeddings.
- **Experimental controls:** splits grouped by name spelling; separate RNG streams for shared parameters, activation coefficients, minibatches and sampling; fresh model and optimiser state for each run.
- **Numerical checks:** handwritten SGD and bias-corrected Adam agree with PyTorch reference optimisers; tests also cover likelihood calculations, gradient checks, matched minibatches and checkpoint round-trips.
- **An executed initialisation pilot:** 24 runs covering two contexts, two activations, two initialisations and three matched seeds. At this short budget, the blend's advantage under unscaled initialisation reverses after scaling.
- **Inspectable evidence:** saved protocols, split membership, complete dev histories, diagnostics, per-seed differences, timings and model checkpoints. Test evaluation is a separate explicit action.

The [historical study](docs/legacy-study.md) records the earlier schedule confound and seed-dependent activation result. Its old split and scores are preserved with their limitations. Current results use the corrected protocol below.

## What I Implemented

| File | Role |
| --- | --- |
| [bigram.ipynb](bigram.ipynb) | Builds transition counts, explains the MLE and smoothing, trains neural logits, separates NLL from regularisation, and generates samples. |
| [mlp.ipynb](mlp.ipynb) | Explains the MLP forward pass and optimiser, inspects initialisation, loads or reruns the pilot, and examines paired outcomes and samples. |
| [makemore.py](makemore.py) | Shared data, model, handwritten optimiser, training, numerical evaluation, diagnostics and checkpoint code. |
| [experiments.py](experiments.py) | Freezes study configurations before execution, runs comparisons, saves reports and plots, and performs explicitly requested final test evaluation. |
| [tests/test_makemore.py](tests/test_makemore.py) | Numerical and protocol checks covering the errors that would invalidate these comparisons. |

The default MLP uses 10-dimensional character embeddings and a 200-unit hidden layer. Three-character tanh has 11,897 parameters; the blend adds 200. Four-character tanh has 13,897 parameters. Each blended neuron learns $f(z)=\alpha\tanh(z)+(1-\alpha)\operatorname{ReLU}(z)$ with $\alpha=\operatorname{sigmoid}(a)$, training $a$ alongside the weights.

The bigram notebook distinguishes three objects: unsmoothed count probabilities are the MLE; additive smoothing changes that estimator; L2 regularisation changes the neural optimisation objective. Count smoothing and penalising logits are not mathematically equivalent. Neural data NLL and its regularisation penalty are reported separately.

## Data and Evaluation

The dataset has 32,033 records and 29,494 distinct name spellings. Every copy of a spelling is assigned to one split before character examples are constructed. Duplicate records retain their frequency weighting within that split.

| Split | Distinct spellings | Records | Next-character targets |
| --- | ---: | ---: | ---: |
| Train | 23,595 | 25,640 | 182,666 |
| Dev | 2,949 | 3,196 | 22,801 |
| Test | 2,950 | 3,197 | 22,679 |

The split is fixed by seed 42 and saved with source and membership hashes. There is no spelling overlap between partitions. Loss is mean next-character NLL in natural-log units, including the end-of-name token. Lower is better.

Every run records an actual pre-training evaluation at step zero and full-dev evaluations at fixed intervals after updates. Minibatch training losses are separately labelled. Time-to-target uses the first **observed dev-loss** crossing, with evaluation overhead included; it is not an interpolated crossing or a claim of sustained performance.

The current real-data runs do not evaluate the test partition. The historical corpus has already been used in earlier experiments, so this should not be presented as an entirely new, historically unseen dataset.

## Current Findings: an Initialisation Pilot

The included pilot fixes batch size 32, SGD learning rate 0.1, tenfold decay after 5,000 updates, evaluation every 1,000 updates, and a **10,000-update** budget. Each setting uses seeds 0, 1 and 2, matched for shared weights and minibatches. These are early-training results, not completed 200k/500k-step replications.

| Initialisation | Context | Mean tanh dev NLL | Mean blend dev NLL | Mean blend − tanh | SE of paired difference |
| --- | ---: | ---: | ---: | ---: | ---: |
| Unscaled | 3 | 2.4159 | 2.3469 | −0.0690 | 0.0139 |
| Unscaled | 4 | 2.4015 | 2.3257 | −0.0758 | 0.0046 |
| Scaled | 3 | 2.2124 | 2.2186 | +0.0062 | 0.0016 |
| Scaled | 4 | 2.1875 | 2.1924 | +0.0049 | 0.0008 |

![Individual matched-seed activation differences in the pilot](results/initialization-pilot/paired_differences.png)

**Observation:** the blend has lower mean dev loss with unscaled initialisation, while tanh has lower mean dev loss with the scaled reference initialisation. Scaled tanh also benefits from the wider context in these runs. The evidence therefore does not support a general claim that only the blend can exploit four-character context.

The fraction of initially saturated tanh activation values falls from approximately 64%/69% at context three/four to approximately 12% in both scaled settings. Scaled initial dev losses are close to the uniform baseline $\log(27)\approx3.296$.

**Interpretation:** initialisation materially affects the early activation comparison. The intervention changes hidden scaling, biases and output scaling together, so it does not identify saturation as the sole cause. Longer budgets and more targeted controls remain necessary to assess final-loss behaviour.

There are only three seeds per setting. The reported standard error describes seed variation conditional on this split and protocol; it is not a universal noise band, a correction for model selection, or proof of a general activation ranking. A smaller train/dev gap alone is not a generalisation improvement.

Full values, configurations and observations are in the [pilot report](results/initialization-pilot/README.md), [protocol](results/initialization-pilot/protocol.json) and [machine-readable summary](results/initialization-pilot/summary.json).

<details>
<summary>Dev curves by updates and elapsed time</summary>

![Dev loss versus completed updates](results/initialization-pilot/dev_by_updates.png)

![Dev loss versus elapsed time](results/initialization-pilot/dev_by_time.png)

Elapsed time is specific to the recorded CPU environment. Compare common targets and evaluation cadences; a method requiring fewer updates need not be faster in seconds.

</details>

## Changes from the Original Study

The original study showed why controls matter: adding a missing decay schedule reduced the recorded Adam–SGD gap, and additional seeds weakened an apparent activation gain. The code and documentation now also correct the issues found during review:

- Duplicate spellings are grouped before splitting.
- Optional activation parameters cannot change the minibatch RNG sequence.
- Absolute decay steps and actual completed-update counts are recorded.
- Per-seed differences replace significance labels based on a three-seed range.
- Shared code replaces duplicated notebook training functions, and results survive outside notebook memory.
- Historical test NLL 2.1320 is correctly attributed to the three-character tanh baseline, not the four-character blend with dev NLL 2.0877.

The [historical report](docs/legacy-study.md) retains the original figures and scores. They are not directly comparable to current-protocol losses.

## Setup and Checks

Use Python 3.11 or later; the recorded validation environment uses Python 3.12 and CPU PyTorch 2.8.0. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
jupyter notebook
```

Open either notebook. `bigram.ipynb` trains its small count/neural examples. `mlp.ipynb` loads the included pilot by default; set `RUN_TRAINING=True` to create a fresh pilot in `results/local/`. Neither notebook's Run All evaluates the test partition.

Execute both notebooks in fresh kernels with `python scripts/check_notebooks.py`. In environments that cannot start kernel sockets, `python scripts/check_notebooks.py --in-process` executes each in a fresh local IPython process. The automated workflow runs the numerical tests and notebook execution.

The [validation record](docs/validation.md) states exactly what was executed and which longer studies remain unrun.

## Running the Study

Every command writes a protocol and split manifest before training. Choose a new output directory: existing directories are deliberately not overwritten. Results include final checkpoints, JSON histories, paired summaries and plots.

Reproduce the included 24-run pilot:

```bash
python experiments.py run --suite initialization --steps 10000 --seeds 0 1 2 --eval-every 1000 --output results/local/initialization-pilot
```

Run a fixed-budget, ten-seed activation comparison (20 runs, 4 million updates):

```bash
python experiments.py run --suite paired --steps 200000 --seeds 0 1 2 3 4 5 6 7 8 9 --initialization scaled --output results/local/paired-200k
```

Run the ten configuration controls with a 500k main budget (the shorter baseline uses 200k):

```bash
python experiments.py run --suite main --steps 500000 --seeds 0 --initialization unscaled --output results/local/main-500k
```

Compare decay timing at a fixed 80k budget (40 runs, 3.2 million updates):

```bash
python experiments.py run --suite decay --steps 80000 --seeds 0 1 2 3 4 5 6 7 8 9 --eval-every 1000 --output results/local/decay-80k
```

`--suite initialization` also supports longer budgets and additional seeds; eight runs are scheduled per seed. These longer studies are implemented but have not been run in the included results. BatchNorm, manual backpropagation and a WaveNet-style model remain future work.

Regenerate reports from saved JSON with `python experiments.py summarize STUDY_DIRECTORY`. Generate names using `python experiments.py sample PATH_TO_MODEL_PT`. Only after dev-based selection is complete, run `python experiments.py test --checkpoint PATH_TO_SELECTED_MODEL_PT --output PATH_TO_TEST_REPORT_JSON`. The test command verifies split identity and refuses to overwrite a report; avoiding repeated test-guided selection still requires discipline.

The [full protocol](docs/experiment-protocol.md) documents the initialisation intervention, timing boundaries, statistical interpretation and checkpoint limitations.

Data: [names.txt](names.txt), attributed to the original [makemore repository](https://github.com/karpathy/makemore). Code: [MIT licence](LICENSE).
