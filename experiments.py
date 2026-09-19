"""Reproducible experiment plans, reports, plots and explicit final evaluation."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

import torch

from makemore import (Config, ROOT, digest, environment, evaluate_test_checkpoint,
                      load_corpus, load_model, sample_names, train_and_eval, write_json)


def make_plan(suite: str, steps: int, seeds: list[int], initialization: str = "scaled",
              eval_every: int = 5000) -> dict[str, Config]:
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Provide nonempty, distinct seed labels")
    if steps < 2:
        raise ValueError("Study needs at least two updates")
    base = Config(steps=steps, decay_step=steps // 2, initialization=initialization,
                  eval_every=eval_every)
    plan = {}
    for seed in seeds:
        c = replace(base, seed=seed)
        if suite == "smoke":
            for block_size in (3, 4):
                for activation in ("tanh", "blend"):
                    plan[f"bs{block_size}-{activation}-s{seed}"] = replace(
                        c, block_size=block_size, activation=activation)
        elif suite == "main":
            short_steps = max(1, int(0.4 * steps))
            plan[f"short-budget-s{seed}"] = replace(c, steps=short_steps,
                                                     decay_step=short_steps // 2)
            plan[f"sgd-decay-s{seed}"] = c
            plan[f"sgd-constant-0.1-s{seed}"] = replace(c, decay_step=None)
            plan[f"sgd-constant-0.05-s{seed}"] = replace(c, decay_step=None, learning_rate=.05)
            plan[f"adam-constant-s{seed}"] = replace(c, optimizer="adam", learning_rate=.01,
                                                      decay_step=None)
            plan[f"adam-decay-s{seed}"] = replace(c, optimizer="adam", learning_rate=.01)
            for block_size in (3, 4, 5):
                plan[f"blend-bs{block_size}-s{seed}"] = replace(c, activation="blend",
                                                                 block_size=block_size)
            plan[f"tanh-bs4-s{seed}"] = replace(c, block_size=4)
        elif suite == "paired":
            for activation in ("tanh", "blend"):
                plan[f"{activation}-s{seed}"] = replace(c, activation=activation)
        elif suite == "initialization":
            for init in ("unscaled", "scaled"):
                for block_size in (3, 4):
                    for activation in ("tanh", "blend"):
                        name = f"{init}-bs{block_size}-{activation}-s{seed}"
                        plan[name] = replace(c, initialization=init, block_size=block_size,
                                              activation=activation)
        elif suite == "decay":
            for fraction in (0.25, 0.5, 0.75, None):
                label = "none" if fraction is None else str(int(100 * fraction))
                plan[f"decay-{label}-s{seed}"] = replace(
                    c, decay_step=None if fraction is None else int(steps * fraction))
        else:
            raise ValueError(f"Unknown suite: {suite}")
    return plan


def paired_summaries(results: dict[str, dict]) -> list[dict]:
    """Group only identical protocols, then subtract within each seed.

    The reported SE describes seed variability conditional on this split and
    protocol. It is not an adjustment for exploratory model selection.
    """
    groups = {}
    for name, result in results.items():
        config = result["config"].copy()
        activation = config.pop("activation")
        seed = config.pop("seed")
        batch_seed = config.pop("batch_seed")
        if activation not in ("tanh", "blend"):
            continue
        # None means minibatches are deterministically coupled to the seed label.
        config["batch_seed_policy"] = "same_as_model_seed" if batch_seed is None else batch_seed
        protocol = {"config": config, "split_id": result["split_id"]}
        group = groups.setdefault(digest(protocol), {"protocol": protocol, "seeds": {}})
        pair = group["seeds"].setdefault(seed, {})
        if activation in pair:
            raise ValueError(f"Duplicate {activation} result for the same protocol/seed: {name}")
        pair[activation] = {"name": name, "dev_nll": result["dev_nll"]}
    summaries = []
    for group in groups.values():
        pairs = [{"seed": seed, "tanh_dev": pair["tanh"]["dev_nll"],
                  "blend_dev": pair["blend"]["dev_nll"],
                  "difference_blend_minus_tanh": pair["blend"]["dev_nll"] - pair["tanh"]["dev_nll"]}
                 for seed, pair in sorted(group["seeds"].items()) if set(pair) == {"tanh", "blend"}]
        if not pairs:
            continue
        differences = [p["difference_blend_minus_tanh"] for p in pairs]
        sd = statistics.stdev(differences) if len(pairs) > 1 else None
        summaries.append({"protocol": group["protocol"], "pairs": pairs, "n": len(pairs),
                          "mean_difference": statistics.mean(differences), "sample_sd": sd,
                          "standard_error": sd / math.sqrt(len(pairs)) if sd is not None else None,
                          "interpretation": "negative favours blend; SE is conditional seed variation"})
    return summaries


def write_report(directory: Path, results: dict[str, dict], title: str) -> None:
    pairs = paired_summaries(results)
    write_json(directory / "summary.json", {"runs": results, "paired_comparisons": pairs})
    lines = [f"# {title}", "", "These results use a split grouped by name spelling. "
             "All scores below are train/dev; the test partition was not evaluated.", "",
             "| Run | Context | Activation | Init | Steps | Parameters | Train NLL | Dev NLL | Seconds |",
             "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, r in results.items():
        c = r["config"]
        lines.append(f"| {name} | {c['block_size']} | {c['activation']} | {c['initialization']} | "
                     f"{c['steps']} | {r['n_params']} | {r['train_nll']:.6f} | "
                     f"{r['dev_nll']:.6f} | {r['seconds']:.2f} |")
    lines += ["", "## Matched activation comparisons", "",
              "Differences are blend minus tanh. Negative values favour blend. "
              "The standard error quantifies seed variation at this fixed split and protocol; "
              "it is not a significance filter for exploratory selection.", ""]
    for pair in pairs:
        c = pair["protocol"]["config"]
        se = "not estimable from one pair" if pair["standard_error"] is None else f"{pair['standard_error']:.6f}"
        lines += [f"- Context {c['block_size']}, {c['initialization']}, {c['steps']} updates: "
                  f"n = {pair['n']}, mean difference = {pair['mean_difference']:.6f}, SE = {se}."]
    lines += ["", "## First observed dev-loss targets", "",
              "Hits are observed at scheduled evaluations, after the stated number of updates. "
              "They are not interpolated or claims of sustained attainment. Elapsed time includes "
              "initial diagnostics and dev evaluations but excludes setup and saving.", "",
              "| Run | Target dev NLL | First observed step | Seconds |",
              "| --- | ---: | ---: | ---: |"]
    for name, r in results.items():
        for target, hit in r["dev_target_hits"].items():
            fields = "Not reached | —" if hit is None else f"{hit['step']} | {hit['seconds']:.2f}"
            lines.append(f"| {name} | {target} | {fields} |")
    (directory / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_results(directory: Path, results: dict[str, dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for xkey, xlabel, filename in (("step", "Completed updates", "dev_by_updates.png"),
                                   ("seconds", "Elapsed seconds (includes dev evaluation)", "dev_by_time.png")):
        fig, ax = plt.subplots(figsize=(9, 5))
        for name, result in results.items():
            rows = result["history"]
            ax.plot([r[xkey] for r in rows], [r["dev_nll"] for r in rows], label=name)
        ax.set(xlabel=xlabel, ylabel="Dev NLL (nats per next character)")
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(directory / filename, dpi=140)
        plt.close(fig)
    pairs = paired_summaries(results)
    if pairs:
        fig, ax = plt.subplots(figsize=(8, 4))
        for i, group in enumerate(pairs):
            diffs = [p["difference_blend_minus_tanh"] for p in group["pairs"]]
            offsets = [(j - (len(diffs)-1)/2) * min(.04, .5 / len(diffs)) for j in range(len(diffs))]
            ax.scatter([i + x for x in offsets], diffs, color="tab:blue", alpha=.75)
            ax.scatter([i], [group["mean_difference"]], marker="D", color="black", s=35)
        ax.axhline(0, color="gray", linewidth=1)
        ax.set_xticks(range(len(pairs)), [f"bs{g['protocol']['config']['block_size']}\n"
                                          f"{g['protocol']['config']['initialization']}\n"
                                          f"{g['protocol']['config']['steps']} updates" for g in pairs])
        ax.set_ylabel("Dev NLL: blend minus tanh (negative favours blend)")
        ax.set_title("Individual matched-seed differences; black diamonds show means")
        fig.tight_layout()
        fig.savefig(directory / "paired_differences.png", dpi=140)
        plt.close(fig)


def run_study(plan: dict[str, Config], corpus, output: str | Path,
              suite: str, make_plots: bool = True) -> dict[str, dict]:
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=False)
    # Freeze the exact protocol BEFORE inspecting any outcomes from this run.
    write_json(destination / "protocol.json", {
        "schema_version": 1, "suite": suite, "created_unix_seconds": time.time(),
        "configs": {name: asdict(c) for name, c in plan.items()},
        "environment": environment(), "split_id": corpus.manifest["split_id"],
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "role": "engineering smoke check" if suite == "smoke" else "predefined follow-up protocol",
    })
    write_json(destination / "split.json", corpus.manifest)
    results, cache = {}, {}
    for name, config in plan.items():
        print(f"Starting {name}: {config.steps} updates", flush=True)
        try:
            run = train_and_eval(config, corpus, destination / name, datasets=cache)
        except Exception as error:
            write_json(destination / "failure.json", {"run": name, "type": type(error).__name__,
                                                       "message": str(error)})
            raise
        results[name] = run.result
        write_report(destination, results, f"{suite} study")
        print(f"  train {run.result['train_nll']:.4f}, dev {run.result['dev_nll']:.4f}, "
              f"{run.result['seconds']:.1f}s", flush=True)
    if make_plots:
        plot_results(destination, results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    run = sub.add_parser("run", help="Freeze a protocol and run train/dev experiments")
    run.add_argument("--suite", choices=["smoke", "main", "paired", "initialization", "decay"], required=True)
    run.add_argument("--steps", type=int, default=200000)
    run.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    run.add_argument("--initialization", choices=["scaled", "unscaled"], default="scaled")
    run.add_argument("--eval-every", type=int, default=5000)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--data", type=Path, default=ROOT / "names.txt")
    run.add_argument("--split-seed", type=int, default=42)
    run.add_argument("--threads", type=int, default=1)
    summary = sub.add_parser("summarize", help="Rebuild reports and plots from saved run JSON")
    summary.add_argument("directory", type=Path)
    test = sub.add_parser("test", help="Explicitly evaluate ONE already-selected checkpoint")
    test.add_argument("--checkpoint", type=Path, required=True)
    test.add_argument("--output", type=Path, required=True)
    test.add_argument("--data", type=Path, default=ROOT / "names.txt")
    test.add_argument("--split-seed", type=int, default=42)
    sample = sub.add_parser("sample", help="Generate names from a saved checkpoint")
    sample.add_argument("checkpoint", type=Path)
    sample.add_argument("--seed", type=int, default=123)
    sample.add_argument("--count", type=int, default=10)
    args = parser.parse_args()
    torch.set_num_threads(getattr(args, "threads", 1))
    torch.use_deterministic_algorithms(True)
    if args.action == "run":
        corpus = load_corpus(args.data, args.split_seed)
        plan = make_plan(args.suite, args.steps, args.seeds, args.initialization, args.eval_every)
        run_study(plan, corpus, args.output, args.suite)
    elif args.action == "summarize":
        files = sorted(args.directory.glob("*/result.json"))
        if not files:
            parser.error("No completed runs found")
        results = {f.parent.name: json.loads(f.read_text()) for f in files}
        write_report(args.directory, results, "Recorded study")
        plot_results(args.directory, results)
    elif args.action == "test":
        result = evaluate_test_checkpoint(args.checkpoint, load_corpus(args.data, args.split_seed), args.output)
        print(json.dumps(result, indent=2))
    else:
        model, checkpoint = load_model(args.checkpoint)
        print(json.dumps(sample_names(model, checkpoint["vocabulary"], args.count, args.seed), indent=2))


if __name__ == "__main__":
    main()
