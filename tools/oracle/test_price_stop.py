#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Commercial
"""Checks for price_stop.py (the DFlash 2 stop's pricer). CPU, tiny synthetic anchors, seconds.

    test_price_stop.py [test_name ...]

- the Calib twin of server::DraftCalib gives the same q as the C++ class: the pinned rates below are
  the ones tests/server/step_k_test.cc asserts on the same observe sequences (unseeded prior,
  seeded prior at a weight, the decay, a too-few-samples cell falling back);
- the seed table: conditional per-(position, bin) rates, censored anchors skipped, the seed split
  (a4 + code + chat seed; agentic / code-kv8 / prose held out);
- the pricing: the cost stop and fixed K = 7 on two tiny anchors, observations up to the first
  rejection.
"""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

import importlib.util  # noqa: E402


def _load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ps = _load("price_stop")


def close(a, b, eps=1e-12):
    assert abs(a - b) <= eps, f"{a} != {b}"


def test_calib_twin():
    # Unseeded (the stated probability is the prior, weight 10, decay 0.98) - the same sequence
    # step_k_test.cc's test_calib drives.
    c = ps.Calib()
    close(c.rate(0, 0.6), 0.6)                    # no data: the stated probability
    c.observe(0, 0.6, True)
    close(c.rate(0, 0.6), (1.0 + 10 * 0.6) / (1.0 + 10))          # 7/11
    c.observe(0, 0.6, True)
    close(c.rate(0, 0.6), (0.98 * 1 + 1 + 6) / (0.98 * 1 + 1 + 10))
    c.observe(0, 0.6, False)
    close(c.rate(0, 0.6), (0.98 * 1.98 + 6) / (0.98 * 1.98 + 1 + 10))
    close(c.rate(1, 0.6), 0.6)                     # positions are separate
    # Seeded at weight 30, decay 0.995: the cell's measured rate is the prior where it has enough
    # samples (step_k_test.cc's test_seed drives the same sequence).
    seed = [[(0.0, 0)] * ps.BINS for _ in range(ps.POSITIONS)]
    seed[0][ps.bin_of(0.6)] = (0.9, 100)
    s = ps.Calib(seed, 30.0, 0.995)
    close(s.rate(0, 0.6), 0.9)                     # zero observations: the table
    s.observe(0, 0.6, True)
    close(s.rate(0, 0.6), (1.0 + 30 * 0.9) / (1.0 + 30))          # 28/31
    s.observe(0, 0.6, False)
    close(s.rate(0, 0.6), (0.995 + 30 * 0.9) / (0.995 + 1 + 30))
    # A cell with too few samples falls back to the stated probability.
    seed2 = [[(0.0, 0)] * ps.BINS for _ in range(ps.POSITIONS)]
    seed2[0][ps.bin_of(0.2)] = (0.9, ps.SEED_MIN_N - 1)
    s2 = ps.Calib(seed2, 30.0, 0.995)
    close(s2.rate(0, 0.2), 0.2)


def test_seed_table():
    anchors = [
        ("a4/x", "a4", [0.9, 0.6, 0.4], 2, False),    # positions 0,1 accepted, 2 tested and rejected
        ("a4/y", "a4", [0.95, 0.6], 0, False),        # position 0 tested and rejected
        ("a4/z", "a4", [0.9], 1, True),               # censored: skipped
        ("code/golden", "code", [0.9, 0.9], 1, False),
        ("prose/4096", "prose", [0.9], 1, False),     # held out: not in the seed
    ]
    t = ps.seed_table(anchors)
    assert t[0][ps.bin_of(0.9)] == (1.0, 2)          # a4/x accepted + code accepted
    assert t[0][ps.bin_of(0.95)] == (0.0, 1)
    assert t[1][ps.bin_of(0.6)] == (1.0, 1)
    assert t[1][ps.bin_of(0.9)] == (0.0, 1)          # code: position 1 tested and rejected
    assert t[2][ps.bin_of(0.4)] == (0.0, 1)
    assert sum(n for _, n in t[2]) == 1              # the censored and held-out anchors add nothing
    assert sum(n for row in t for _, n in row) == 6


def test_price_stop():
    table = ps.TABLES["one card"]
    verify, draft = table
    anchors = [
        ("code/golden", "code", [0.99] * 7, 7, False),
        ("code/golden", "code", [0.01] * 7, 0, False),
    ]
    stop, k7 = ps.price_stop(anchors, table, ps.Calib())
    # First anchor: every rate 0.99, E(K) grows, the best K is 7: ids 8, cost verify[7] + draft.
    # Second anchor: rates 0.01, the best K is 1: ids 1, cost verify[1] + draft.
    close(stop["code"][0], 8.0 + 1.0)
    close(stop["code"][1], verify[7] + draft + verify[1] + draft)
    close(k7["code"][0], 8.0 + 1.0)
    close(k7["code"][1], 2 * (verify[7] + draft))


TESTS = {name[5:]: fn for name, fn in list(globals().items()) if name.startswith("test_")}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tests", nargs="*", choices=sorted(TESTS))
    args = ap.parse_args()
    names = args.tests or sorted(TESTS)
    for name in names:
        TESTS[name]()
        print(f"{name}: ok")
    print("test_price_stop: all cases passed")
