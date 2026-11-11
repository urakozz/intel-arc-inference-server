#!/usr/bin/env python3
"""The engine-tap gate compares the same anchors and never pools away a corpus regression."""
import os
import sys
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != os.path.dirname(os.path.abspath(__file__))]
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('dflash_engine_accept', Path(__file__).with_name('dflash_engine_accept.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class AcceptanceTest(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()
