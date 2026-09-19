import json
import math
from pathlib import Path
import tempfile
import unittest
from collections import Counter
from dataclasses import replace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from makemore import (Config, MLP, Optimizer, bigram_nll, count_probabilities,
                      diagnostics, evaluate, evaluate_test_checkpoint, first_dev_hits,
                      generator, load_corpus, load_model, sample_names,
                      train_and_eval, train_neural_bigram)
from experiments import make_plan, paired_summaries


torch.set_num_threads(1)


class CorrectnessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.names = [a + b for a in "abcdef" for b in "abcdef"]
        self.names += self.names[:10] * 2
        self.data = self.root / "names.txt"
        self.data.write_text("\n".join(self.names) + "\n")
        self.corpus = load_corpus(self.data)
        self.config = Config(dimension=3, hidden_neurons=8, batch_size=4,
                             steps=8, decay_step=4, eval_every=4, eval_batch_size=7)

    def test_split_groups_duplicates_preserves_records_and_is_stable(self):
        c = self.corpus
        self.assertEqual(Counter(c.train + c.dev + c.test), Counter(self.names))
        for a, b in [(c.train, c.dev), (c.train, c.test), (c.dev, c.test)]:
            self.assertFalse(set(a) & set(b))
        self.assertEqual(c.manifest, load_corpus(self.data).manifest)
        self.assertNotEqual(c.manifest["split_id"], load_corpus(self.data, 43).manifest["split_id"])

    def test_context_sizes_keep_targets_in_same_order(self):
        X3, Y3 = self.corpus.examples("train", 3)
        X4, Y4 = self.corpus.examples("train", 4)
        self.assertTrue(torch.equal(Y3, Y4))
        self.assertTrue(torch.equal(X3, X4[:, -3:]))

    def test_shared_weights_match_between_activations_and_contexts(self):
        tanh = MLP(self.config, len(self.corpus.chars))
        blend = MLP(replace(self.config, activation="blend"), len(self.corpus.chars))
        wider = MLP(replace(self.config, block_size=4), len(self.corpus.chars))
        for name in tanh.tensors:
            self.assertTrue(torch.equal(tanh.tensors[name], blend.tensors[name]), name)
        for name in ("C", "b1", "W2", "b2"):
            self.assertTrue(torch.equal(tanh.tensors[name], wider.tensors[name]), name)

    def test_training_uses_identical_minibatches_for_paired_activations(self):
        original_randint = torch.randint
        observed = []
        def capture(*args, **kwargs):
            indices = original_randint(*args, **kwargs)
            observed.append(indices.clone())
            return indices
        with patch("torch.randint", side_effect=capture):
            train_and_eval(self.config, self.corpus)
            train_and_eval(replace(self.config, activation="blend"), self.corpus)
        self.assertEqual(len(observed), 2 * self.config.steps)
        self.assertTrue(all(torch.equal(a, b) for a, b in
                            zip(observed[:self.config.steps], observed[self.config.steps:])))

    def test_optimizer_steps_match_pytorch_reference(self):
        for method in ("sgd", "adam"):
            with self.subTest(optimizer=method):
                c = replace(self.config, optimizer=method, beta1=.8, beta2=.95, eps=1e-7)
                p = torch.tensor([.2, -1.1, 2.3], dtype=torch.float64, requires_grad=True)
                q = p.detach().clone().requires_grad_()
                ours = Optimizer([p], c)
                reference = (torch.optim.SGD([q], lr=.1) if method == "sgd" else
                             torch.optim.Adam([q], lr=.1, betas=(c.beta1, c.beta2), eps=c.eps))
                for step in range(10):
                    p.grad = q.grad = None
                    ((p - step / 10).square().sum()).backward()
                    ((q - step / 10).square().sum()).backward()
                    rate = .1 if step < 4 else .01
                    reference.param_groups[0]["lr"] = rate
                    ours.step(rate)
                    reference.step()
                    torch.testing.assert_close(p, q, rtol=1e-12, atol=1e-12)

    def test_evaluation_weights_last_batch_by_its_size(self):
        X = torch.arange(11)[:, None]
        Y = torch.arange(11) % 3
        logits = torch.randn((11, 3), generator=generator(0, "test"), dtype=torch.float64)
        class FakeModel:
            def __call__(self, batch):
                return logits[batch[:, 0]]
        actual = evaluate(FakeModel(), (X, Y), batch_size=4)
        self.assertAlmostEqual(actual, F.cross_entropy(logits, Y).item(), places=12)

    def test_train_does_not_request_test_data_and_reports_final_state(self):
        calls = []
        original = self.corpus.examples
        def record(partition, block_size):
            calls.append(partition)
            if partition == "test":
                raise AssertionError("Training evaluated test data")
            return original(partition, block_size)
        with patch.object(self.corpus, "examples", side_effect=record):
            run = train_and_eval(self.config, self.corpus)
        self.assertEqual(calls, ["train", "dev"])
        self.assertEqual([row["step"] for row in run.result["history"]], [0, 4, 8])
        self.assertAlmostEqual(run.result["dev_nll"], evaluate(run.model, original("dev", 3), 7))
        self.assertNotIn("test_nll", run.result)

    def test_repeated_runs_reproduce_losses_and_weights(self):
        first = train_and_eval(self.config, self.corpus)
        second = train_and_eval(self.config, self.corpus)
        self.assertEqual(first.result["dev_nll"], second.result["dev_nll"])
        self.assertEqual(first.result["train_nll"], second.result["train_nll"])
        for name in first.model.tensors:
            self.assertTrue(torch.equal(first.model.tensors[name], second.model.tensors[name]))

    def test_cache_is_scoped_to_split_identity(self):
        shared = {}
        train_and_eval(self.config, self.corpus, datasets=shared)
        other = load_corpus(self.data, 43)
        cached = train_and_eval(self.config, other, datasets=shared)
        fresh = train_and_eval(self.config, other)
        self.assertEqual(cached.result["dev_nll"], fresh.result["dev_nll"])

    def test_checkpoint_roundtrip_and_explicit_test_guard(self):
        folder = self.root / "run"
        original = train_and_eval(self.config, self.corpus, folder)
        restored, checkpoint = load_model(folder / "model.pt")
        for name in original.model.tensors:
            self.assertTrue(torch.equal(original.model.tensors[name], restored.tensors[name]))
        self.assertEqual(sample_names(original.model, self.corpus.chars),
                         sample_names(restored, checkpoint["vocabulary"]))
        with self.assertRaises(ValueError):
            evaluate_test_checkpoint(folder / "model.pt", load_corpus(self.data, 43), self.root / "bad.json")
        report = self.root / "test.json"
        test = evaluate_test_checkpoint(folder / "model.pt", self.corpus, report)
        self.assertTrue(math.isfinite(test["test_nll"]))
        with self.assertRaises(FileExistsError):
            evaluate_test_checkpoint(folder / "model.pt", self.corpus, report)
        with self.assertRaises(FileExistsError):
            train_and_eval(self.config, self.corpus, folder)

    def test_count_weighted_objective_and_gradient_match_expanded_examples(self):
        counts = torch.tensor([[2., 1.], [1., 3.]], dtype=torch.float64)
        W = torch.tensor([[.5, -.4], [-.3, .8]], dtype=torch.float64, requires_grad=True)
        aggregated = -(counts / counts.sum() * W.log_softmax(1)).sum()
        grad_aggregated, = torch.autograd.grad(aggregated, W)
        rows, labels = [], []
        for a in range(2):
            for b in range(2):
                rows.extend([a] * int(counts[a, b]))
                labels.extend([b] * int(counts[a, b]))
        expanded = F.cross_entropy(F.one_hot(torch.tensor(rows), 2).double() @ W,
                                   torch.tensor(labels))
        grad_expanded, = torch.autograd.grad(expanded, W)
        torch.testing.assert_close(aggregated, expanded)
        torch.testing.assert_close(grad_aggregated, grad_expanded)

    def test_bigram_smoothing_and_reported_final_objective(self):
        counts = torch.tensor([[0., 3.], [2., 1.]], dtype=torch.float64)
        mle = count_probabilities(counts)
        smooth = count_probabilities(counts, 1.)
        torch.testing.assert_close(smooth.sum(1), torch.ones(2, dtype=torch.float64))
        unseen = torch.tensor([[1., 0.], [0., 0.]], dtype=torch.float64)
        self.assertTrue(math.isinf(bigram_nll(mle, unseen)))
        self.assertTrue(math.isfinite(bigram_nll(smooth, unseen)))
        run = train_neural_bigram(counts, steps=50, lr=1., l2=.01)
        self.assertAlmostEqual(run["nll"], bigram_nll(run["probabilities"], counts), places=12)
        self.assertAlmostEqual(run["objective"], run["nll"] + run["penalty"], places=12)
        self.assertEqual(run["history"][-1]["step"], 50)
        self.assertEqual(run["history"][-1]["objective"], run["objective"])

    def test_initialization_and_blend_gradient(self):
        model = MLP(replace(self.config, activation="blend"), len(self.corpus.chars), dtype=torch.float64)
        X, Y = self.corpus.examples("train", 3)
        initial_nll = evaluate(model, (X, Y), 7)
        self.assertLess(abs(initial_nll - math.log(len(self.corpus.chars))), .03)
        loss = F.cross_entropy(model(X[:4]), Y[:4])
        loss.backward()
        parameter = model.tensors["alpha_raw"]
        analytical = parameter.grad[0].item()
        with torch.no_grad():
            original = parameter[0].item()
            h = 1e-5
            parameter[0] = original + h
            plus = F.cross_entropy(model(X[:4]), Y[:4]).item()
            parameter[0] = original - h
            minus = F.cross_entropy(model(X[:4]), Y[:4]).item()
            parameter[0] = original
        self.assertAlmostEqual(analytical, (plus - minus)/(2*h), places=8)

    def test_decay_boundary_and_dev_threshold_labels(self):
        self.assertEqual(self.config.lr_at(3), .1)
        self.assertAlmostEqual(self.config.lr_at(4), .01)
        history = [{"step": 0, "seconds": .1, "dev_nll": 3.3},
                   {"step": 10, "seconds": 1., "dev_nll": 2.3},
                   {"step": 20, "seconds": 2., "dev_nll": 2.5}]
        hits = first_dev_hits(history, (2.4, 2.2))
        self.assertEqual(hits["2.4"]["step"], 10)
        self.assertIsNone(hits["2.2"])

    def test_paired_summary_matches_protocol_and_uses_correct_se(self):
        results = {}
        for seed, difference in enumerate((-.01, .02, .005)):
            for activation, value in [("tanh", 2.2), ("blend", 2.2 + difference)]:
                results[f"{activation}-{seed}"] = {
                    "config": self.config.__dict__ | {"seed": seed, "activation": activation},
                    "split_id": "fixture", "dev_nll": value}
        report, = paired_summaries(results)
        self.assertEqual(report["n"], 3)
        self.assertAlmostEqual(report["mean_difference"], .005)
        self.assertAlmostEqual(report["sample_sd"], .015)
        self.assertAlmostEqual(report["standard_error"], .015 / math.sqrt(3))
        results["blend-2"]["config"]["steps"] += 1
        report, = paired_summaries(results)
        self.assertEqual(report["n"], 2)

    def test_plans_hold_comparison_budgets_fixed(self):
        paired = make_plan("paired", 200000, [0, 1, 2])
        self.assertEqual(len(paired), 6)
        self.assertEqual({c.steps for c in paired.values()}, {200000})
        self.assertEqual({c.decay_step for c in paired.values()}, {100000})
        factorial = make_plan("initialization", 100, [0])
        self.assertEqual(len(factorial), 8)
        decay = make_plan("decay", 80000, [0])
        self.assertEqual({c.decay_step for c in decay.values()}, {20000, 40000, 60000, None})


if __name__ == "__main__":
    unittest.main()
