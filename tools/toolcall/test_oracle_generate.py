"""oracle_generate.py's MoE greedy loop (Ornith / K2-Horizon, spec 15e / 18d) without a model.

    python3 tools/toolcall/test_oracle_generate.py     (pytest also collects it, where installed)

A fake step records the (ids, pos) it is fed - the prompt at 0, then one id a step at the next
position, as a KV-cached decode is fed - and answers a scripted next id.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oracle_generate import MOE_TYPES, greedy, greedy_batched, parse_args  # noqa: E402


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


def test_batched_feeds_like_greedy():
    """greedy_batched hands each sequence exactly greedy()'s feeds (prompt at 0, then each new id at
    the next position) whatever the batch, refills a freed slot at once, stops each at its own EOS
    / cap, and reports each the moment it ends."""
    import random
    rng = random.Random(3)
    prompts = {f"q{i}": [rng.randrange(1, 300) for _ in range(n)] for i, n in enumerate([4, 1, 9, 2, 6, 3])}
    caps = dict(zip(prompts, [5, 9, 1, 7, 3, 12]))
    eos = {13, 17}

    def next_id(name, fed):                  # a deterministic "model": the next id from the history
        h = sum(fed) * 31 + len(fed) * 7 + int(name[1:]) * 101
        return 13 if h % 11 == 0 else (17 if h % 13 == 0 else 20 + h % 250)

    want, want_feeds = {}, {}
    for name, p in prompts.items():
        fed_all, feeds = [], []

        def step(ids, pos, fed_all=fed_all, feeds=feeds, name=name):
            assert pos == len(fed_all)
            fed_all.extend(ids)
            feeds.append((list(ids), pos))
            nxt = next_id(name, fed_all)
            return [1.0 if i == nxt else 0.0 for i in range(300)]
        want[name] = greedy(step, p, caps[name], eos, argmax)
        want_feeds[name] = feeds
    assert any(v and v[-1] in eos for v in want.values()) and any(len(v) == caps[k] for k, v in want.items())
    for batch in (1, 2, 4, 6):
        got, feeds, done_order, hist = {}, {k: [] for k in prompts}, [], {}

        def step_many(items):
            assert 1 <= len(items) <= batch
            rows = []
            for ids, pos, st in items:
                name = st["name"]
                assert pos == len(hist[name])
                hist[name].extend(ids)
                feeds[name].append((list(ids), pos))
                nxt = next_id(name, hist[name])
                rows.append([1.0 if i == nxt else 0.0 for i in range(300)])
            return rows
        names = iter(prompts)

        def new_state():
            k = next(names)
            hist[k] = []
            return {"name": k}

        def on_done(k, o):
            got[k] = list(o)
            done_order.append(k)
        greedy_batched(step_many, new_state, [(k, p, caps[k]) for k, p in prompts.items()], batch, eos, argmax, on_done)
        assert got == want, (batch, got, want)
        assert feeds == want_feeds, batch
        assert sorted(done_order) == sorted(prompts)


def test_compare_ref():
    """compare_ref.py: equal runs, a flip (first differing position, both gaps), gaps that differ."""
    import tempfile
    import compare_ref

    def put(d, name, ids, gaps=None):
        with open(os.path.join(d, f"{name}.bf16.ids"), "w") as f:
            f.write(" ".join(map(str, ids)) + "\n")
        if gaps is not None:
            with open(os.path.join(d, f"{name}.bf16.gap"), "w") as f:
                f.write(" ".join("%.9g" % g for g in gaps) + "\n")
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        put(a, "x", [1, 2, 3], [0.5, 0.25, 1.0])
        put(b, "x", [1, 2, 3], [0.5, 0.25, 1.0])
        put(a, "y", [4, 5, 6, 7], [2.0, 0.0078125, 3.0, 1.0])
        put(b, "y", [4, 5, 9], [2.0, 0.015625, 0.5])
        put(a, "z", [8], [1.5])
        put(b, "z", [8], [1.25])
        put(a, "w", [1, 1])
        put(b, "w", [1, 1])
        same, diff, lines = compare_ref.compare(a, b, ["w", "x", "y", "z"])
        assert (same, diff) == (2, 2), lines
        y = next(s for s in lines if " y:" in s)
        assert "first differs at new id 2 (A 6 / B 9; 4 vs 3 ids)" in y and "A gap 3" in y and "B gap 0.5" in y, y
        assert "GAPS DIFFER" in next(s for s in lines if " z:" in s)
        assert sorted(compare_ref.names_in(a)) == ["w", "x", "y", "z"]


def test_args_kolibri():
    """Spec 20e: --model kolibri / --device anywhere around the three paths; Kolibri takes the MoE loop."""
    a = parse_args(["--model", "kolibri", "--device", "cuda", "/snap", "set", "out", "--new-tokens", "256"])
    assert (a["model"], a["device"], a["snapshot"], a["set_dir"], a["out"], a["new_tokens"]) == \
        ("kolibri", "cuda", "/snap", "set", "out", 256)
    b = parse_args(["/snap", "set", "out"])
    assert (b["model"], b["device"], b["new_tokens"], b["batch"], b["keep"]) == (None, "cpu", 192, "1", "none")
    c = parse_args(["/snap", "set", "out", "--batch", "auto", "--resident", "auto"])
    assert (c["batch"], c["keep"]) == ("auto", "auto")
    assert parse_args(["--batch", "6", "/s", "a", "b"])["batch"] == "6"
    assert "kolibri1" in MOE_TYPES
    for bad in (["--model", "llama", "/s", "a", "b"], ["/s", "a"], ["--device"], ["--batch", "0", "/s", "a", "b"],
                ["--resident", "all", "/s", "a", "b"]):
        try:
            parse_args(bad)
        except SystemExit:
            continue
        raise AssertionError(bad)


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for n, f in tests:
        f()
        print(f"ok  {n}")
    print(f"{len(tests)} tests passed")
