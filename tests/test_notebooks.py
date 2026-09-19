"""Check the actual notebook calculations against the command-line implementation."""

import contextlib
from dataclasses import asdict
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from makemore import (Config, bigram_nll, count_probabilities, load_corpus,
                      load_model, sample_names, train_and_eval, train_neural_bigram)

ROOT = Path(__file__).resolve().parents[1]


def notebook_definitions(name):
    """Load definition cells only; never run the pilot, sampling or save cells."""
    notebook = json.loads((ROOT / name).read_text())
    namespace = {}
    with contextlib.redirect_stdout(io.StringIO()):
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code' and 'definitions' in cell['metadata'].get('tags', []):
                exec(compile(''.join(cell['source']), f"{name}:{cell['id']}", 'exec'), namespace)
    return namespace


class NotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mlp = notebook_definitions('mlp.ipynb')
        cls.bigram = notebook_definitions('bigram.ipynb')
        cls.corpus = load_corpus(ROOT / 'names.txt')
        cls.cache = {}

    def test_notebook_split_matches_saved_protocol(self):
        saved = json.loads((ROOT / 'results/initialization-pilot/split.json').read_text())
        self.assertEqual(self.mlp['split'], saved)
        self.assertEqual(self.bigram['split'], saved)
        self.assertEqual(tuple(self.mlp['train_words']), self.corpus.train)
        self.assertEqual(tuple(self.bigram['dev_words']), self.corpus.dev)

    def test_mlp_updates_match_cli_for_both_initializations_activations_and_optimizers(self):
        for initialization in ['unscaled', 'scaled']:
            for activation in ['tanh', 'blend']:
                for optimizer in ['sgd', 'adam']:
                    with self.subTest(initialization=initialization, activation=activation, optimizer=optimizer):
                        c = Config(dimension=3, hidden_neurons=8, batch_size=4, steps=8,
                                   learning_rate=.02, decay_step=4, eval_every=3,
                                   initialization=initialization, activation=activation,
                                   optimizer=optimizer, seed=19, batch_seed=7)
                        settings = asdict(c)
                        settings['iters'] = settings.pop('steps')
                        settings['lr'] = settings.pop('learning_rate')
                        settings['lr_decay_factor'] = settings.pop('decay_factor')
                        actual = self.mlp['train_and_eval'](**settings)
                        expected = train_and_eval(c, self.corpus, datasets=self.cache)
                        for name, parameter in expected.model.tensors.items():
                            self.assertTrue(torch.equal(actual['state'][name], parameter.detach()), name)
                        self.assertEqual(actual['config'], asdict(c))
                        for field in ['train_nll', 'dev_nll', 'initial_diagnostics', 'final_diagnostics']:
                            self.assertEqual(actual[field], expected.result[field], field)
                        for a, b in zip(actual['history'], expected.result['history']):
                            self.assertEqual({k: v for k, v in a.items() if k != 'seconds'},
                                             {k: v for k, v in b.items() if k != 'seconds'})

    def test_notebook_pairs_have_identical_minibatches_and_no_test_examples(self):
        original = torch.randint
        batches = []
        def record(*args, **kwargs):
            ix = original(*args, **kwargs)
            batches.append(ix.clone())
            return ix
        build_dataset = self.mlp['build_dataset']
        def check_words(words, block_size):
            self.assertIsNot(words, self.mlp['test_words'])
            return build_dataset(words, block_size)
        self.mlp['_split_cache'].clear()
        with patch.object(torch, 'randint', side_effect=record), \
             patch.dict(self.mlp, {'build_dataset': check_words}):
            for activation in ['tanh', 'blend']:
                out = self.mlp['train_and_eval'](iters=4, dimension=3, hidden_neurons=8,
                                                eval_every=2, seed=2, activation=activation)
                self.assertNotIn('test_nll', out)
        self.assertEqual(len(batches), 8)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(batches[:4], batches[4:])))

    def test_notebook_checkpoints_load_in_existing_cli(self):
        out = self.mlp['train_and_eval'](iters=4, dimension=3, hidden_neurons=8,
                                        eval_every=2, activation='blend')
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / 'run'
            result = self.mlp['save_run'](folder, out)
            model, checkpoint = load_model(folder / 'model.pt')
            for name, parameter in model.tensors.items():
                self.assertTrue(torch.equal(parameter.detach(), out['state'][name]))
            self.assertEqual(result['split_id'], self.corpus.manifest['split_id'])
            self.assertEqual(result['environment']['implementation'], 'mlp.ipynb')
            self.assertEqual(self.mlp['sample_names'](checkpoint, n=5),
                             sample_names(model, checkpoint['vocabulary'], n=5))
            with self.assertRaises(FileExistsError):
                self.mlp['save_run'](folder, out)

    def test_notebook_sampling_matches_existing_pilot(self):
        for activation in ['tanh', 'blend']:
            path = ROOT / f'results/initialization-pilot/scaled-bs3-{activation}-s0/model.pt'
            model, checkpoint = load_model(path)
            self.assertEqual(self.mlp['sample_names'](checkpoint, n=10, seed=123),
                             sample_names(model, checkpoint['vocabulary'], n=10, seed=123))

    def test_direct_bigram_loss_and_updates_match_cli(self):
        N = self.bigram['N_train']
        for smoothing in [0., 1.]:
            actual = self.bigram['count_probabilities'](N, smoothing)
            expected = count_probabilities(N, smoothing)
            self.assertTrue(torch.equal(actual, expected))
            self.assertEqual(self.bigram['bigram_nll'](actual, self.bigram['N_dev']),
                             bigram_nll(expected, self.bigram['N_dev']))
        for l2 in [0., .01]:
            actual = self.bigram['train_neural_bigram'](N, steps=20, l2=l2)
            expected = train_neural_bigram(N, steps=20, l2=l2)
            self.assertTrue(torch.equal(actual['weights'], expected['weights']))
            self.assertEqual(actual['history'], expected['history'])
            for field in ['nll', 'penalty', 'objective']:
                self.assertEqual(actual[field], expected[field])


if __name__ == '__main__':
    unittest.main()
