"""Small, explicit character models and a reproducible CPU experiment harness.

PyTorch supplies autodiff; the models and SGD/Adam updates are implemented here.
Importing this module neither trains a model nor evaluates the test partition.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import random
import time

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent


def write_json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def stream_seed(seed: int, stream: str) -> int:
    """Stable named streams, independent of Python's process-randomised hash()."""
    return int.from_bytes(hashlib.sha256(f"{seed}:{stream}".encode()).digest()[:8],
                          "big") % (2**63 - 1)


def generator(seed: int, stream: str) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(stream_seed(seed, stream))


@dataclass
class Corpus:
    train: tuple[str, ...]
    dev: tuple[str, ...]
    test: tuple[str, ...]
    chars: tuple[str, ...]
    manifest: dict

    @property
    def stoi(self) -> dict[str, int]:
        return {character: i for i, character in enumerate(self.chars)}

    def examples(self, partition: str, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        if partition not in ("train", "dev", "test"):
            raise ValueError("partition must be train, dev or test")
        return build_dataset(getattr(self, partition), block_size, self.stoi)


def load_corpus(path: str | Path = ROOT / "names.txt", split_seed: int = 42) -> Corpus:
    """Split distinct spellings 80/10/10; preserve duplicate-record weighting.

    Every copy of a name belongs to one partition. Character examples are built
    only after partitioning. The complete membership and source hash are saved.
    """
    raw = Path(path).read_bytes()
    words = raw.decode("utf-8").splitlines()
    if not words or any(not w or w.strip() != w or "." in w for w in words):
        raise ValueError("Expected nonempty names without whitespace padding or '.'")
    unique = sorted(set(words))
    random.Random(split_seed).shuffle(unique)
    first, second = int(0.8 * len(unique)), int(0.9 * len(unique))
    groups = {"train": set(unique[:first]), "dev": set(unique[first:second]),
              "test": set(unique[second:])}
    if any(not group for group in groups.values()):
        raise ValueError("Not enough distinct names for three nonempty partitions")
    partitions = {key: tuple(w for w in words if w in group)
                  for key, group in groups.items()}
    chars = (".", *sorted(set("".join(words))))
    manifest = {
        "schema_version": 1, "policy": "group_by_exact_spelling_keep_duplicates",
        "split_seed": split_seed, "source_sha256": hashlib.sha256(raw).hexdigest(),
        "vocabulary": list(chars),
        "groups": {key: sorted(group) for key, group in groups.items()},
        "counts": {key: {"records": len(partitions[key]), "unique_names": len(group),
                         "targets": sum(len(w) + 1 for w in partitions[key])}
                   for key, group in groups.items()},
    }
    manifest["split_id"] = digest(manifest)
    return Corpus(**partitions, chars=chars, manifest=manifest)


def build_dataset(words: tuple[str, ...] | list[str], block_size: int,
                  stoi: dict[str, int]) -> tuple[torch.Tensor, torch.Tensor]:
    if block_size < 1:
        raise ValueError("block_size must be positive")
    inputs, targets = [], []
    for word in words:
        context = [0] * block_size
        for character in word + ".":
            index = stoi[character]
            inputs.append(context)
            targets.append(index)
            context = context[1:] + [index]
    if not inputs:
        raise ValueError("Cannot build examples from an empty partition")
    return torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)


@dataclass(frozen=True)
class Config:
    block_size: int = 3
    dimension: int = 10
    hidden_neurons: int = 200
    batch_size: int = 32
    steps: int = 200_000
    learning_rate: float = 0.1
    decay_step: int | None = 100_000
    decay_factor: float = 0.1
    activation: str = "tanh"
    optimizer: str = "sgd"
    initialization: str = "scaled"
    seed: int = 0
    batch_seed: int | None = None
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    eval_every: int = 5_000
    eval_batch_size: int = 4096
    dev_targets: tuple[float, ...] = (2.4, 2.2, 2.15, 2.13)

    def __post_init__(self) -> None:
        for field in ("block_size", "dimension", "hidden_neurons", "batch_size",
                      "steps", "eval_every", "eval_batch_size"):
            if not isinstance(getattr(self, field), int) or getattr(self, field) < 1:
                raise ValueError(f"{field} must be a positive integer")
        if self.activation not in ("tanh", "relu", "blend", "linear"):
            raise ValueError("Unknown activation")
        if self.optimizer not in ("sgd", "adam"):
            raise ValueError("Unknown optimizer")
        if self.initialization not in ("unscaled", "scaled"):
            raise ValueError("Unknown initialization")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not 0 < self.decay_factor <= 1 or not math.isfinite(self.eps) or self.eps <= 0:
            raise ValueError("Invalid decay_factor or eps")
        if not (0 <= self.beta1 < 1 and 0 <= self.beta2 < 1):
            raise ValueError("Adam betas must be in [0, 1)")
        if self.decay_step is not None and (not isinstance(self.decay_step, int)
                                            or self.decay_step < 0):
            raise ValueError("decay_step must be a nonnegative integer or None")
        if any(not math.isfinite(x) or x <= 0 for x in self.dev_targets):
            raise ValueError("dev_targets must be finite and positive")

    def lr_at(self, completed_steps: int) -> float:
        """Rate for the next update, after completed_steps completed updates."""
        factor = self.decay_factor if (self.decay_step is not None
                                       and completed_steps >= self.decay_step) else 1.0
        return self.learning_rate * factor


class MLP:
    def __init__(self, config: Config, vocab_size: int, dtype=torch.float32):
        self.config, self.vocab_size = config, vocab_size
        fan_in, width = config.block_size * config.dimension, config.hidden_neurons
        shapes = {"C": (vocab_size, config.dimension), "W1": (fan_in, width),
                  "b1": (width,), "W2": (width, vocab_size), "b2": (vocab_size,)}
        self.tensors = {
            name: torch.randn(shape, generator=generator(config.seed, name), dtype=dtype)
            for name, shape in shapes.items()
        }
        if config.initialization == "scaled":
            # Apply the SAME scaling to tanh and blend for a controlled comparison.
            self.tensors["W1"] *= (5 / 3) / math.sqrt(fan_in)
            self.tensors["b1"].zero_()
            self.tensors["W2"] *= 0.01 / math.sqrt(width)
            self.tensors["b2"].zero_()
        if config.activation == "blend":
            self.tensors["alpha_raw"] = torch.empty(width, dtype=dtype).uniform_(
                -1, 1, generator=generator(config.seed, "alpha"))
        for tensor in self.tensors.values():
            tensor.requires_grad_(True)

    def parameters(self) -> list[torch.Tensor]:
        return list(self.tensors.values())

    def preactivation(self, X: torch.Tensor) -> torch.Tensor:
        p = self.tensors
        return p["C"][X].reshape(len(X), -1) @ p["W1"] + p["b1"]

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        h = self.preactivation(X)
        if self.config.activation == "tanh":
            h = h.tanh()
        elif self.config.activation == "relu":
            h = h.relu()
        elif self.config.activation == "blend":
            a = self.tensors["alpha_raw"].sigmoid()
            h = a * h.tanh() + (1 - a) * h.relu()
        return h @ self.tensors["W2"] + self.tensors["b2"]

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = None


class Optimizer:
    """Handwritten SGD / bias-corrected Adam, with no weight decay."""

    def __init__(self, parameters: list[torch.Tensor], config: Config):
        self.parameters, self.config, self.step_count = parameters, config, 0
        self.m = [torch.zeros_like(p) for p in parameters]
        self.v = [torch.zeros_like(p) for p in parameters]

    @torch.no_grad()
    def step(self, lr: float) -> None:
        self.step_count += 1
        c = self.config
        for i, p in enumerate(self.parameters):
            if p.grad is None:
                raise ValueError("Missing gradient")
            if c.optimizer == "sgd":
                p.add_(p.grad, alpha=-lr)
            else:
                self.m[i].mul_(c.beta1).add_(p.grad, alpha=1 - c.beta1)
                self.v[i].mul_(c.beta2).addcmul_(p.grad, p.grad, value=1 - c.beta2)
                m_hat = self.m[i] / (1 - c.beta1**self.step_count)
                v_hat = self.v[i] / (1 - c.beta2**self.step_count)
                p.addcdiv_(m_hat, v_hat.sqrt() + c.eps, value=-lr)


@torch.no_grad()
def evaluate(model: MLP, data: tuple[torch.Tensor, torch.Tensor],
             batch_size: int = 4096) -> float:
    X, Y = data
    total = 0.0
    for start in range(0, len(Y), batch_size):
        total += F.cross_entropy(model(X[start:start + batch_size]),
                                 Y[start:start + batch_size], reduction="sum").item()
    return total / len(Y)


@torch.no_grad()
def diagnostics(model: MLP, X: torch.Tensor) -> dict:
    h = model.preactivation(X)
    derivative = 1 - h.tanh().square()
    return {"examples": len(X), "preactivation_mean": h.mean().item(),
            "preactivation_std": h.std(unbiased=False).item(),
            "tanh_derivative_mean": derivative.mean().item(),
            "tanh_saturated_fraction_abs_gt_0.99": (h.tanh().abs() > 0.99).float().mean().item(),
            "logit_std": model(X).std(unbiased=False).item()}


def environment() -> dict:
    return {"python": platform.python_version(), "torch": torch.__version__,
            "numpy": version("numpy"), "matplotlib": version("matplotlib"),
            "platform": platform.platform(), "threads": torch.get_num_threads(),
            "device": "cpu", "dtype": "float32",
            "core_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def first_dev_hits(history: list[dict], thresholds: tuple[float, ...]) -> dict:
    """First observed post-update dev hit; not sustained attainment."""
    return {str(target): next(({"step": row["step"], "seconds": row["seconds"],
                                "dev_nll": row["dev_nll"]}
                               for row in history if row["dev_nll"] <= target), None)
            for target in thresholds}


@dataclass
class TrainingRun:
    model: MLP
    result: dict


def train_and_eval(config: Config, corpus: Corpus, run_dir: str | Path | None = None,
                   datasets: dict | None = None) -> TrainingRun:
    """Train on train, select on dev. There is deliberately no test flag.

    Time-to-dev-target includes diagnostics and scheduled dev evaluations from
    immediately before initial diagnostics; it excludes data/model construction
    and final serialisation. Initial step-zero evaluation is timed as well.
    """
    destination = Path(run_dir) if run_dir is not None else None
    if destination is not None:
        destination.mkdir(parents=True, exist_ok=False)
    cache = datasets if datasets is not None else {}
    cache_key = (corpus.manifest["split_id"], config.block_size)
    if cache_key not in cache:
        cache[cache_key] = (corpus.examples("train", config.block_size),
                            corpus.examples("dev", config.block_size))
    train, dev = cache[cache_key]
    model = MLP(config, len(corpus.chars))
    optimizer = Optimizer(model.parameters(), config)
    batch_seed = config.seed if config.batch_seed is None else config.batch_seed
    batches = generator(batch_seed, "minibatches")
    X, Y = train
    start = time.perf_counter()
    diagnostic_X = X[:min(len(X), 2048)]
    initial = diagnostics(model, diagnostic_X)
    initial_dev = evaluate(model, dev, config.eval_batch_size)
    if not math.isfinite(initial_dev):
        raise FloatingPointError("Non-finite initial dev loss")
    history = [{"step": 0, "seconds": time.perf_counter() - start,
                "dev_nll": initial_dev, "minibatch_nll_mean": None,
                "window_start_step": None, "learning_rate": config.lr_at(0),
                "gradient_l2": None}]
    running_loss, window_start = 0.0, 1
    for step in range(1, config.steps + 1):
        ix = torch.randint(len(Y), (config.batch_size,), generator=batches)
        loss = F.cross_entropy(model(X[ix]), Y[ix])
        if not math.isfinite(loss.item()):
            raise FloatingPointError(f"Non-finite training loss at update {step}")
        model.zero_grad()
        loss.backward()
        checkpoint = step % config.eval_every == 0 or step == config.steps
        grad_norm = (sum(p.grad.square().sum().item() for p in model.parameters()) ** 0.5
                     if checkpoint else None)
        lr = config.lr_at(step - 1)
        optimizer.step(lr)
        running_loss += loss.item()
        if checkpoint:
            dev_nll = evaluate(model, dev, config.eval_batch_size)
            if not math.isfinite(dev_nll):
                raise FloatingPointError(f"Non-finite dev loss at update {step}")
            history.append({"step": step, "seconds": time.perf_counter() - start,
                            "dev_nll": dev_nll,
                            "minibatch_nll_mean": running_loss / (step - window_start + 1),
                            "window_start_step": window_start, "learning_rate": lr,
                            "gradient_l2": grad_norm})
            running_loss, window_start = 0.0, step + 1
    final_train = evaluate(model, train, config.eval_batch_size)
    result = {
        "schema_version": 1, "config": asdict(config), "environment": environment(),
        "split_id": corpus.manifest["split_id"],
        "source_sha256": corpus.manifest["source_sha256"],
        "n_params": sum(p.numel() for p in model.parameters()),
        "initial_diagnostics": initial, "final_diagnostics": diagnostics(model, diagnostic_X),
        "train_nll": final_train, "dev_nll": history[-1]["dev_nll"],
        "seconds": time.perf_counter() - start, "history": history,
        "dev_target_hits": first_dev_hits(history, config.dev_targets),
        "timing_scope": "CPU training, initial diagnostics, dev evaluation; total also includes final train evaluation",
        "alpha": (model.tensors["alpha_raw"].sigmoid().detach().tolist()
                  if config.activation == "blend" else None),
        "rng_seeds": {"parameters": {name: stream_seed(config.seed, name)
                                     for name in ("C", "W1", "b1", "W2", "b2", "alpha")},
                      "minibatches": stream_seed(batch_seed, "minibatches")},
    }
    if destination is not None:
        save_run(destination, model, result, corpus)
    return TrainingRun(model, result)


def save_run(destination: Path, model: MLP, result: dict, corpus: Corpus) -> None:
    state = {name: p.detach().clone() for name, p in model.tensors.items()}
    torch.save({"config": asdict(model.config), "vocabulary": list(corpus.chars),
                "split_id": corpus.manifest["split_id"],
                "source_sha256": corpus.manifest["source_sha256"], "state": state},
               destination / "model.pt")
    result["checkpoint_sha256"] = hashlib.sha256((destination / "model.pt").read_bytes()).hexdigest()
    write_json(destination / "result.json", result)


def load_model(path: str | Path) -> tuple[MLP, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = Config(**checkpoint["config"])
    model = MLP(config, len(checkpoint["vocabulary"]))
    if set(checkpoint["state"]) != set(model.tensors):
        raise ValueError("Checkpoint parameters do not match configuration")
    with torch.no_grad():
        for name, p in model.tensors.items():
            if p.shape != checkpoint["state"][name].shape:
                raise ValueError("Checkpoint parameter shape mismatch")
            p.copy_(checkpoint["state"][name])
    return model, checkpoint


def evaluate_test_checkpoint(checkpoint_path: str | Path, corpus: Corpus,
                             output_path: str | Path) -> dict:
    """Explicit final action on a chosen checkpoint, outside notebook Run All.

    Refuse to overwrite an existing test report. This prevents accidental reruns,
    not intentional repeated testing under other filenames; discipline is needed.
    """
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"Test report already exists: {output}")
    model, checkpoint = load_model(checkpoint_path)
    if checkpoint["split_id"] != corpus.manifest["split_id"]:
        raise ValueError("Checkpoint and evaluation split differ")
    if checkpoint["vocabulary"] != list(corpus.chars):
        raise ValueError("Checkpoint and corpus vocabularies differ")
    result = {"checkpoint": str(checkpoint_path),
              "checkpoint_sha256": hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest(),
              "config": checkpoint["config"], "split_id": checkpoint["split_id"],
              "test_nll": evaluate(model, corpus.examples("test", model.config.block_size),
                                    model.config.eval_batch_size)}
    write_json(output, result)
    return result


@torch.no_grad()
def sample_names(model: MLP, chars: tuple[str, ...] | list[str], n: int = 10,
                 seed: int = 123, max_length: int = 30) -> list[dict]:
    if n < 1 or max_length < 1 or len(chars) != model.vocab_size:
        raise ValueError("Invalid sampling arguments")
    g, output = generator(seed, "sampling"), []
    for _ in range(n):
        context, name, terminated = [0] * model.config.block_size, [], False
        for _ in range(max_length + 1):
            probabilities = model(torch.tensor([context])).softmax(dim=-1)[0]
            index = torch.multinomial(probabilities, 1, generator=g).item()
            if index == 0:
                terminated = True
                break
            if len(name) == max_length:
                break
            name.append(chars[index])
            context = context[1:] + [index]
        output.append({"name": "".join(name), "terminated": terminated})
    return output


def transition_counts(words: tuple[str, ...] | list[str], stoi: dict[str, int]) -> torch.Tensor:
    counts = Counter((stoi[a], stoi[b]) for word in words
                     for a, b in zip("." + word, word + "."))
    matrix = torch.zeros((len(stoi), len(stoi)), dtype=torch.float64)
    for (a, b), count in counts.items():
        matrix[a, b] = count
    return matrix


def count_probabilities(counts: torch.Tensor, smoothing: float = 0.0) -> torch.Tensor:
    if not math.isfinite(smoothing) or smoothing < 0 or counts.ndim != 2:
        raise ValueError("Invalid count probabilities arguments")
    if (counts < 0).any() or counts.shape[0] != counts.shape[1]:
        raise ValueError("Expected a square nonnegative count matrix")
    adjusted = counts + smoothing
    totals = adjusted.sum(dim=1, keepdim=True)
    # An entirely unobserved conditioning row has no identified MLE; choose uniform.
    return torch.where(totals > 0, adjusted / totals.clamp_min(1e-300),
                       torch.full_like(adjusted, 1 / counts.shape[1]))


def bigram_nll(probabilities: torch.Tensor, counts: torch.Tensor) -> float:
    mask = counts > 0
    if counts.sum() <= 0:
        raise ValueError("No evaluation transitions")
    return -(counts[mask] * probabilities[mask].log()).sum().item() / counts.sum().item()


def train_neural_bigram(counts: torch.Tensor, steps: int = 200, lr: float = 50.0,
                        l2: float = 0.01, seed: int = 0) -> dict:
    """Exact full-batch objective, aggregated by transition counts.

    This equals averaging over the expanded one-hot training examples. Report
    data NLL, the penalty and their sum separately, including after the last step.
    """
    if steps < 1 or lr <= 0 or l2 < 0 or counts.sum() <= 0:
        raise ValueError("Invalid neural bigram training arguments")
    W = torch.randn(counts.shape, generator=generator(seed, "bigram"),
                    dtype=counts.dtype, requires_grad=True)
    weights = counts / counts.sum()
    history = []
    def terms():
        nll = -(weights * W.log_softmax(dim=1)).sum()
        penalty = l2 * W.square().mean()
        return nll, penalty, nll + penalty
    for step in range(steps):
        nll, penalty, objective = terms()
        if step % 10 == 0:
            history.append({"step": step, "nll": nll.item(),
                            "penalty": penalty.item(), "objective": objective.item()})
        W.grad = None
        objective.backward()
        with torch.no_grad():
            W.add_(W.grad, alpha=-lr)
    with torch.no_grad():
        nll, penalty, objective = terms()
        history.append({"step": steps, "nll": nll.item(),
                        "penalty": penalty.item(), "objective": objective.item()})
    return {"weights": W.detach(), "probabilities": W.detach().softmax(dim=1),
            "nll": nll.item(), "penalty": penalty.item(), "objective": objective.item(),
            "history": history, "config": {"steps": steps, "lr": lr, "l2": l2, "seed": seed}}
