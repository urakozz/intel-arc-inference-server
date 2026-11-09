"""3i's allocation/protocol CLI checks; --plan must not initialize a GPU."""
import json
from pathlib import Path
import subprocess
import sys
import unittest

PROBE = str(Path(sys.argv.pop(1)).resolve())


class ProbeCliTest(unittest.TestCase):
    def run_probe(self, *args):
        return subprocess.run([PROBE, *args], capture_output=True, text=True, timeout=5)

    def plan(self, *args):
        p = self.run_probe('--plan', *args)
        self.assertEqual(p.returncode, 0, p.stderr)
        return [json.loads(x[5:]) for x in p.stdout.splitlines() if x.startswith('PLAN ')]

    def test_default_has_distinct_64_gu_weights(self):
        rows = self.plan('--shape', 'gu')
        self.assertEqual(len(rows), 1)
        p = rows[0]
        self.assertEqual(p['weight_mode'], 'layers')
        self.assertEqual(p['weight_source'], 'synthetic')
        self.assertEqual(p['model_layers'], 64)
        self.assertEqual(p['weight_copies'], 64)
        self.assertEqual(p['weight_bytes_per_copy'], 94699520)
        self.assertEqual(p['distinct_weight_bytes'], 6060769280)
        self.assertEqual(p['measured_linears'], 64)
        self.assertFalse(p['extrapolated'])

    def test_all_shape_footprints_and_layouts(self):
        rows = self.plan()
        self.assertEqual([(r['shape'], r['model_layers'], r['distinct_weight_bytes']) for r in rows],
                         [('gu', 64, 6060769280), ('down', 64, 3030384640),
                          ('qkvz', 48, 2139095040), ('out', 64, 1069547520),
                          ('faqkv', 16, 623902720)])
        self.assertEqual([r['layout'] for r in rows], [0, 0, 1, 0, 0])
        self.assertTrue(all(r['weight_copies'] == r['measured_linears'] == r['model_layers'] for r in rows))

    def test_resident_projection_is_explicit(self):
        p, = self.plan('--shape', 'gu', '--weights', 'resident')
        self.assertEqual(p['weight_copies'], 1)
        self.assertEqual(p['measured_linears'], 1)
        self.assertEqual(p['distinct_weight_bytes'], 94699520)
        self.assertEqual(p['model_layers'], 64)
        self.assertTrue(p['extrapolated'])

    def test_explicit_layers_and_shape_subset(self):
        rows = self.plan('--shape', 'gu,qkvz', '--weights', 'layers', '--rows', '237,250',
                         '--q4', 'q8b1s4split,q8b1s4slm')
        self.assertEqual([r['shape'] for r in rows], ['gu', 'qkvz'])
        self.assertTrue(all(not r['extrapolated'] for r in rows))

    def test_invalid_mode(self):
        p = self.run_probe('--plan', '--weights', 'typo')
        self.assertEqual(p.returncode, 2)
        self.assertIn('--weights', p.stderr)

    def test_invalid_rows(self):
        for value in ('0', '1025', '-1', '', ',', '237junk', '237,237', '237,,250'):
            with self.subTest(value=value):
                p = self.run_probe('--plan', '--rows', value)
                self.assertEqual(p.returncode, 2)
                self.assertIn('--rows', p.stderr)

    def test_invalid_rounds(self):
        for value in ('0', '-1', '7junk'):
            with self.subTest(value=value):
                p = self.run_probe('--plan', '--rounds', value)
                self.assertEqual(p.returncode, 2)
                self.assertIn('--rounds', p.stderr)

    def test_invalid_shape(self):
        for value in ('typo', 'gu,typo', '', 'gu,,down', 'gu,gu'):
            with self.subTest(value=value):
                p = self.run_probe('--plan', '--shape', value)
                self.assertEqual(p.returncode, 2)
                self.assertIn('--shape', p.stderr)

    def test_invalid_tiles(self):
        for value in ('q0b1s1', 'q8b1s0', 'q999999999b1s1', 'q8b1s3slm', ','):
            with self.subTest(value=value):
                p = self.run_probe('--plan', '--q4', value)
                self.assertEqual(p.returncode, 2)
                self.assertIn('--q4', p.stderr)

    def test_negative_control_requires_distinct_weights(self):
        p = self.run_probe('--negative-check', '--weights', 'resident')
        self.assertEqual(p.returncode, 2)
        self.assertIn('negative', p.stderr)


if __name__ == '__main__':
    unittest.main()
