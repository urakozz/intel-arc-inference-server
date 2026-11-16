"""Tests for lookup_runs.py --price.

    python3 tools/spec/test_lookup_runs.py   (or python3 -m pytest tools/spec)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lookup_runs as lr  # noqa: E402

# Flat costs: plain 10 ms, 16 / 32 / 64 rows 20 / 30 / 50 ms, no replay. An MTP iteration is 10 x 1.48, a confirmed
# entry adds 10 x 0.26 for the head's two drafts.
FLAT = {'keys': [0, 1000], '1': [10, 10], '16': [20, 20], '32': [30, 30], '64': [50, 50], 'replay': 0.0}
PROMPT = list(range(300))


def test_copy_walks_each_ladder():
    # out copies the prompt from 50 on. Pending 50 matches nothing (no candidate): one MTP iteration (51..54); then the
    # candidate 55.. agrees with the head and enters hot.
    out = PROMPT[50:170]
    mtp, confirm = 10 * 1.48, 10 * 0.26
    # 16 only: eight 16-row verifies (16 ids each, the last one cut at the end).
    ms, ids, n = lr.price(PROMPT, out, FLAT, 16, 'earn')
    assert ids == 119 and n[16] == 8 and n['mtp'] == 1 and n['confirmed'] == 1
    assert abs(ms - (mtp + confirm + 8 * 20)) < 1e-9
    # earn: 16, 32, 64 kept whole, then 64 again.
    ms, _, n = lr.price(PROMPT, out, FLAT, 64, 'earn')
    assert (n[16], n[32], n[64]) == (1, 1, 2)
    assert abs(ms - (mtp + confirm + 20 + 30 + 2 * 50)) < 1e-9
    # jump: the source holds > 63 more ids, so 64 rows right after the first 16-row verify kept whole.
    ms, _, n = lr.price(PROMPT, out, FLAT, 64, 'jump')
    assert (n[16], n[32], n[64]) == (1, 0, 2)
    assert abs(ms - (mtp + confirm + 20 + 2 * 50)) < 1e-9


def test_no_copy_is_mtp_alone():
    out = list(range(5000, 5041))   # 40 ids after the first, none in the context: ten MTP iterations everywhere
    for _, maxw, ladder in lr.LADDERS:
        ms, ids, n = lr.price(PROMPT, out, FLAT, maxw, ladder)
        assert ids == 40 and dict(n) == {'mtp': 10}
        assert abs(ms - 10 * 10 * 1.48) < 1e-9


if __name__ == '__main__':
    test_copy_walks_each_ladder()
    test_no_copy_is_mtp_alone()
    print('ok')
