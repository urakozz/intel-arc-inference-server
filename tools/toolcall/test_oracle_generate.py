"""oracle_generate.py's MoE greedy loop (Ornith / K2-Horizon, spec 15e / 18d) without a model.

    python3 tools/toolcall/test_oracle_generate.py     (pytest also collects it, where installed)

A fake step records the (ids, pos) it is fed - the prompt at 0, then one id a step at the next
position, as a KV-cached decode is fed - and answers a scripted next id.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oracle_generate import greedy  # noqa: E402


def scripted(script):
    fed = []

    def step(ids, pos):
        fed.append((list(ids), pos))
        nxt = script[len(fed) - 1] if len(fed) - 1 < len(script) else 7
        return [1.0 if i == nxt else 0.0 for i in range(300)]   # a "logits row": argmax = nxt
    return step, fed


def argmax(row):
    return max(range(len(row)), key=lambda i: (row[i], -i))


def test_feeds_the_cache_like_decode():
    step, fed = scripted([10, 11, 12, 13])
    out = greedy(step, [5, 6, 7], 4, set(), argmax)
    assert out == [10, 11, 12, 13]
    assert fed == [([5, 6, 7], 0), ([10], 3), ([11], 4), ([12], 5)]   # the last id is never fed


def test_stops_at_eos_and_keeps_it():
    step, fed = scripted([10, 299, 12])
    assert greedy(step, [1, 2], 8, {1, 299}, argmax) == [10, 299]
    assert len(fed) == 2


def test_n_caps():
    step, _ = scripted([3] * 20)
    assert greedy(step, [1], 5, {250019}, argmax) == [3, 3, 3, 3, 3]


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for n, f in tests:
        f()
        print(f"ok  {n}")
    print(f"{len(tests)} tests passed")
