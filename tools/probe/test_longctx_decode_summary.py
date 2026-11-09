import copy
import math
import unittest

from summarize_longctx_decode import validate, summarize


def fixture():
    return [dict(kv=kv, round=r, depth=d, M=m, capacity=150272,
                 launches=774 if m == 1 else 784, attn_launches=48 if m == 1 else 49,
                 wall_ms=20.0, gpu_ms=18.0, prep_ms=.1, core_ms=6.0,
                 reduce_ms=.2, head_attn_ms=0 if m == 1 else .01,
                 rows_per_s=m * 50.0)
            for kv in ('bf16', 'int8') for r in range(3)
            for d in (4096, 32768, 81920, 150000) for m in (1, 4)]


class SummaryTest(unittest.TestCase):
    def test_complete_matrix_and_share(self):
        rows = fixture()
        validate(rows)
        table = summarize(rows)
        self.assertEqual(len(table), 16)
        self.assertAlmostEqual(table[0]['attention_ms'], 6.3)
        self.assertAlmostEqual(table[0]['attention_share_pct'], 35.0)
        self.assertAlmostEqual(table[0]['other_ms'], 11.7)

    def test_missing_arm(self):
        with self.assertRaisesRegex(ValueError, 'matrix'):
            validate(fixture()[:-1])

    def test_duplicate(self):
        rows = fixture()
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            validate(rows + rows[:1])

    def test_nonfinite(self):
        for bad in (math.nan, math.inf, -1):
            rows = fixture()
            rows[0]['wall_ms'] = bad
            with self.assertRaises(ValueError):
                validate(rows)

    def test_launch_and_category_accounting(self):
        for field, value in (('attn_launches', 32), ('launches', 0),
                             ('core_ms', 0), ('head_attn_ms', 2),
                             ('gpu_ms', 1), ('rows_per_s', 1)):
            rows = copy.deepcopy(fixture())
            rows[0][field] = value
            with self.assertRaises(ValueError, msg=field):
                validate(rows)


if __name__ == '__main__':
    unittest.main()
