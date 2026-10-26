"""Tests for lookup_accept.py.

    python3 tools/spec/test_lookup_accept.py   (or python3 -m pytest tools/spec)
"""
import json
import os
import random
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lookup_accept as la  # noqa: E402


def matcher(seq, max_match=64):
    lk = la.Lookup(max_match)
    lk.extend(seq)
    return lk


def test_matcher_vs_brute_force():
    rng = random.Random(7)
    cases = 0
    for alphabet in (2, 3, 5, 50):
        for max_match in (4, 9, 64):
            for _ in range(30):
                s = [rng.randrange(alphabet) for _ in range(rng.randrange(1, 120))]
                # Repetitive variants: a copied span, a run.
                if rng.random() < 0.5 and len(s) > 10:
                    a = rng.randrange(len(s) - 5)
                    s += s[a:a + rng.randrange(3, 40)]
                lk = la.Lookup(max_match)
                for t, tok in enumerate(s):
                    lk.append(tok)
                    for n in (2, 3, 6):
                        if n > max_match:
                            continue
                        got = lk.match(n)
                        want = la.brute_match(s[:t + 1], n, max_match)
                        assert got[0] == want[0], (s[:t + 1], n, max_match, got, want)
                        if got[0] is not None:
                            assert got[1] == want[1]
                        cases += 1
    assert cases > 10000


def test_most_recent_on_ties():
    # "a b X" twice, then "a b": both earlier "a b" match with length 2; the later one wins.
    lk = matcher([1, 2, 7, 1, 2, 8, 1, 2])
    drafts, m = lk.propose(1, 2)
    assert (drafts, m) == ([8], 2)


def test_longest_beats_recent():
    # "c a b" early continues with 7; a later "a b" continues with 8; context ends "c a b".
    lk = matcher([3, 1, 2, 7, 9, 1, 2, 8, 9, 3, 1, 2])
    drafts, m = lk.propose(1, 2)
    assert (drafts, m) == ([7], 3)


def test_no_match_and_min_n():
    lk = matcher([1, 2, 3, 4, 5])
    assert lk.propose(3, 2) == ([], 0)
    lk = matcher([1, 2, 3, 9, 2, 3])
    assert lk.propose(3, 2)[0] == [9, 2, 3]   # the continuation runs into the suffix itself
    assert lk.propose(3, 3)[0] == []


def test_overlapping_run():
    lk = matcher([5, 4, 4, 4])
    drafts, m = lk.propose(5, 2)
    assert drafts == [4, 4, 4, 4, 4] and m == 2
    lk = matcher([1, 2, 1, 2, 1])   # period 2
    assert lk.propose(4, 2)[0] == [2, 1, 2, 1]


def test_boundary_stops_drafts_and_matches():
    lk = la.Lookup()
    lk.extend([1, 2, 3])
    lk.boundary()
    lk.extend([1, 2])
    assert lk.propose(4, 2)[0] == [3]   # stops at the boundary
    lk.extend([3])
    lk2 = la.Lookup()
    lk2.extend([1, 2])
    lk2.boundary()
    lk2.extend([2])   # [boundary, 2] never matches
    assert lk2.propose(2, 2)[0] == []


def test_simulate_copy_and_fresh():
    prompt = list(range(100, 160))
    # Output = an exact copy of the prompt's tail: every draft is right.
    seq = {"prompt_ids": prompt, "out_ids": prompt[20:50]}
    v = la.VERIFY_EST
    r = la.simulate(seq, 3, 2, v)
    # x_0 = 120 pending; context ends ...159, 120: no match, one plain step. From x = 121
    # on, "120 121" matches the prompt and every verify keeps its 3 drafts, except the last,
    # whose drafts run past the recorded output.
    assert r.plain == 1 and r.verifies == 8 and r.accepted == r.drafted - 3
    assert abs(r.verified_ids / r.verifies - 4.0) < 0.6
    # Output of fresh ids: no verify at all, plain cost.
    seq2 = {"prompt_ids": prompt, "out_ids": list(range(500, 530))}
    r2 = la.simulate(seq2, 3, 2, v)
    assert r2.verifies == 0 and r2.cost == 30 and r2.ids == 30


def test_simulate_protocol_counts():
    # Hand-traced: prompt [1 2 3 4], out [1 2 3 9]. x_0 = 1 pending; context [1 2 3 4 1]:
    # match "4 1"? no; "1" alone is below n = 2 -> plain (emit 1). Context [.. 1 2]: match
    # "1 2" (len 2) -> drafts [3 4 1]; out after 2 is 3, 9 -> j = 1, emit 2 (2, 3). Pending 9:
    # context [.. 1 2 3 9]: no match -> plain. Total 4 ids, cost 1 + v[3] + 1.
    seq = {"prompt_ids": [1, 2, 3, 4], "out_ids": [1, 2, 3, 9]}
    v = la.VERIFY_EST
    r = la.simulate(seq, 3, 2, v)
    assert (r.ids, r.verifies, r.plain, r.accepted, r.drafted) == (4, 1, 2, 1, 3)
    assert abs(r.cost - (2 + v[3])) < 1e-9


def test_mtp_auto_projection():
    k, rate = la.mtp_auto_rate([0.92, 0.87, 0.82], la.VERIFY_INT8, la.MTP_DRAFT_INT8)
    e3 = 1 + 0.92 + 0.92 * 0.87 + 0.92 * 0.87 * 0.82
    assert k == 3 and abs(rate - e3 / (1.79 + 0.38)) < 1e-9
    # Low acceptance: plain wins.
    assert la.mtp_auto_rate([0.1, 0.1, 0.1], la.VERIFY_INT8, la.MTP_DRAFT_INT8) == (0, 1.0)


def test_adaptive_k_mirror():
    # server::AdaptiveK's documented boundaries on the int8 table (spec 8 §10): K = 0 below
    # alpha 0.31, K = 1 to 0.81, K = 3 above.
    p = la.AdaptiveK(la.VERIFY_INT8, la.MTP_DRAFT_INT8, 3)
    assert p.best(0.30) == 0 and p.best(0.5) == 1 and p.best(0.85) == 3
    # With a free draft the boundaries move down (lookup).
    q = la.AdaptiveK(la.VERIFY_INT8, [0.0] * 3, 3)
    assert q.best(0.2) == 1 and q.best(0.8) == 3


def test_load_logs_and_seed():
    with tempfile.TemporaryDirectory() as d:
        for i, (p, o) in enumerate([([1, 2, 3], [4, 5]), ([1, 2, 3, 4, 5, 6], [7])], 1):
            with open(os.path.join(d, f"{i:06d}.json"), "w") as f:
                json.dump({"prompt_ids": p, "out_ids": o}, f)
        seqs = la.load_logs(d, seed_session=1)
        assert len(seqs) == 2 and seqs[0]["seed"] == [] and seqs[1]["seed"] == [[4, 5]]
        assert la.main(["--log", d, "--mc-seeds", "2", "--max-k", "3", "--seed-session", "1",
                        "--verify-cost", "1,1.18,1.56,1.79"]) == 0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: ok")
