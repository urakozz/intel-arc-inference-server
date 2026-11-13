#!/usr/bin/env python3
"""The engine-tap gate compares the same anchors and never pools away a corpus regression."""
import os
import sys
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != os.path.dirname(os.path.abspath(__file__))]
import importlib.util
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('dflash_engine_accept', Path(__file__).with_name('dflash_engine_accept.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class AcceptanceTest(unittest.TestCase):
    def baseline_fixture(self, root):
        class Tensor(list):
            def tolist(self): return list(self)
        class Dump:
            def __init__(self, path, **kwargs): self.name = Path(path).stem.split('.')[0]
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def metadata(self): return {'name': self.name, 'tap_from': '0'}
            def get_tensor(self, name):
                return Tensor(range(10)) if name == 'ids' else Tensor(range(1, 11))
        rows, drafts, corpora = [{'run': {'format': 'int4'}}], [], {}
        for name in ('a4', 'code', 'prose', 'cjk', 'opencode'):
            (root / (name + '.taps.safetensors')).touch()
            corpora[name] = {}
            for k in range(1, 8):
                corpora[name][str(k)] = {'engine_E': float(k + 1)}
                for p in (0, 1):
                    ids = list(range(p + 1, p + k + 1))
                    rows.append({'dump': name, 'p': p, 'k': k, 'ids': ids})
                    drafts.append({'name': name, 'p': p, 'k': k, 'engine': ids})
        (root/'engine.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        (root/'baseline.json').write_text(json.dumps({'corpora': corpora, 'drafts': drafts, 'pass': True}))
        return Dump, rows

    def test_precision_gate_reuses_original_engine_baseline_without_loading_a_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dump, _ = self.baseline_fixture(root)
            with patch.dict(sys.modules, {'safetensors': SimpleNamespace(safe_open=dump)}), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertTrue(m.compare_baseline(tmp, str(root/'baseline.json'), str(root/'report.json')))
            report = json.loads((root/'report.json').read_text())
            self.assertEqual(report['corpora']['opencode']['7']['engine_E'], 8.0)
            self.assertEqual(report['corpora']['opencode']['7']['gap'], 0.0)

    def test_precision_gate_does_not_pool_away_a_smaller_width_regression(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dump, rows = self.baseline_fixture(root)
            for r in rows[1:]:
                if r['dump'] == 'code' and r['k'] == 1: r['ids'] = [99]
            (root/'engine.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with patch.dict(sys.modules, {'safetensors': SimpleNamespace(safe_open=dump)}), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertFalse(m.compare_baseline(tmp, str(root/'baseline.json'), str(root/'report.json')))
            report = json.loads((root/'report.json').read_text())
            self.assertEqual(report['corpora']['code']['1']['gap'], 1.0)
            self.assertEqual(report['corpora']['code']['7']['gap'], 0.0)

    def test_precision_gate_rejects_missing_anchors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, rows = self.baseline_fixture(root)
            (root/'engine.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows[:-1]))
            with self.assertRaisesRegex(ValueError, 'anchors'):
                m.compare_baseline(tmp, str(root/'baseline.json'), str(root/'report.json'))

    def test_prefix_acceptance_stops_on_first_rejection(self):
        counted, accepted = m.prefix_counts([10, 11, 12, 13], [11, 99, 13], [11, 12, 13, 14], 0, 0)
        self.assertEqual(counted, [1, 1, 0])
        self.assertEqual(accepted, [1, 0, 0])

    def test_recording_divergence_censors_future_rows(self):
        counted, accepted = m.prefix_counts([10, 99, 12], [11, 12], [11, 12, 13], 0, 0)
        self.assertEqual(counted, [1, 0])
        self.assertEqual(accepted, [1, 0])

    def test_nonzero_tap_from(self):
        counted, accepted = m.prefix_counts([8, 9, 10, 11], [11, 12], [11, 12], 2, 2)
        self.assertEqual(counted, [1, 1])
        self.assertEqual(accepted, [1, 1])

    def test_expected_tokens_uses_conditional_prefix_rates(self):
        self.assertAlmostEqual(m.expected([10, 5, 2], [5, 2, 1]), 1.8)

    def test_no_observations_is_not_a_pass(self):
        with self.assertRaises(ValueError):
            m.expected([0], [0])

    def test_each_width_uses_its_own_noncausal_reference(self):
        class Tensor(list):
            def long(self): return self
            def tolist(self): return list(self)

        class Dump:
            def __init__(self, path, **kwargs): self.name = Path(path).stem.split('.')[0]
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def metadata(self): return {'name': self.name, 'tap_from': '0', 'n': '9', 'layers': '5'}
            def get_tensor(self, name):
                return Tensor(range(10 if name == 'ids' else 11, 19 if name == 'ids' else 20))

        widths, batches = [], []
        def draft(anchor, p, k, ctx):
            widths.append(k)
            return SimpleNamespace(ids=Tensor([11 + p] if k == 1 else [99] * k))

        def batch(d, head, anchors, ps, k, ctx, masks):
            batches.append((len(anchors), k))
            return {'full': Tensor([draft(a, p, k, ctx).ids for a, p in zip(anchors, ps)])}

        drafter = SimpleNamespace(context_kv=lambda *args: None, draft_block=draft, head_logits=None)
        module = SimpleNamespace(reference_drafter=lambda *args: drafter,
                                 _load=lambda name: SimpleNamespace(draft_batch=batch))
        spec = SimpleNamespace(loader=SimpleNamespace(exec_module=lambda *args: None))
        torch = SimpleNamespace(set_num_threads=lambda n: None, inference_mode=contextlib.nullcontext,
                                arange=lambda a, b: range(a, b))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [{'run': {}}]
            for name in ('a4', 'code'):
                (root / (name + '.taps.safetensors')).touch()
                rows.extend({'dump': name, 'p': p, 'k': k, 'ids': [11 + p] if k == 1 else [99] * k,
                             'accepted': int(k == 1)} for p in (0, 1) for k in range(1, 8))
            (root / 'engine.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
            with patch.dict(sys.modules, {'torch': torch, 'safetensors': SimpleNamespace(safe_open=Dump)}), \
                 patch.object(m.importlib.util, 'spec_from_file_location', return_value=spec), \
                 patch.object(m.importlib.util, 'module_from_spec', return_value=module), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertTrue(m.evaluate(tmp, '', '', str(root / 'report.json'), 1))
            report = json.loads((root / 'report.json').read_text())
            self.assertEqual(report['corpora']['code']['1']['reference_E'], 2.0)
            self.assertEqual(batches, [(2, k) for k in range(1, 8)] * 2)
            self.assertEqual(widths, [k for k in range(1, 8) for _ in range(3)] +
                             [k for k in range(1, 8) for _ in range(2)])


if __name__ == '__main__':
    unittest.main()
