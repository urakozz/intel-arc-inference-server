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


def _fake_set(d, ids):
    """A one-scenario set dir (manifest + ids, SHA-256 checked as oracle_generate.py checks it)."""
    import hashlib
    import json
    raw = (" ".join(map(str, ids)) + "\n").encode()
    with open(os.path.join(d, "s1.ids"), "wb") as f:
        f.write(raw)
    with open(os.path.join(d, "manifest.json"), "w") as f:
        json.dump([{"name": "s1", "ids": len(ids), "sha256": hashlib.sha256(raw).hexdigest()}], f)


def test_args_qwen4exp():
    """Spec 21e: --model qwen4exp; qwen4_exp (and a text-only export's qwen4_exp_text) take the MoE loop; the memory
    plan reads its config (layer types, the 4 streams, 512 experts)."""
    import oracle_generate as og
    a = parse_args(["--model", "qwen4exp", "/snap", "set", "out", "--new-tokens", "192"])
    assert (a["model"], a["new_tokens"]) == ("qwen4exp", 192)
    assert "qwen4_exp" in MOE_TYPES and "qwen4_exp_text" in MOE_TYPES
    assert og.MODELS["qwen4exp"] == ("qwen4_exp", "qwen4_exp_text")
    cfg = {"model_type": "qwen4_exp", "text_config": {
        "num_hidden_layers": 48, "hidden_size": 2560, "vocab_size": 248320, "hc_count": 4, "hc_lowrank": 320,
        "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 12,
        "num_attention_heads": 24, "num_key_value_heads": 2, "head_dim": 256, "linear_num_value_heads": 48,
        "linear_num_key_heads": 16, "linear_value_head_dim": 128, "linear_key_head_dim": 128, "num_experts": 512,
        "moe_intermediate_size": 640, "shared_expert_intermediate_size": 640, "index_head_dim": 128}}
    plan = og.batch_plan("qwen4_exp", cfg, [2000, 2800], 192, 60 * og.GiB, "auto", "auto")
    assert 1 <= plan["batch"] <= 2 and "12 QSA layers" in plan["text"] and "36" in plan["text"], plan["text"]
    mm = og.mem_model("qwen4_exp", cfg)
    assert len(mm["dense"]) == 48 and mm["experts"][0] == 512 * 3 * 640 * 2560 * 2


def test_model_check_qwen4exp():
    """Spec 21e: `--model qwen4exp` refuses a snapshot of another model_type before anything loads."""
    import json
    import tempfile
    import oracle_generate as og
    with tempfile.TemporaryDirectory() as snap, tempfile.TemporaryDirectory() as st, \
            tempfile.TemporaryDirectory() as out:
        with open(os.path.join(snap, "config.json"), "w") as f:
            json.dump({"model_type": "kolibri1"}, f)
        _fake_set(st, [1, 2, 3])
        saved = sys.argv
        sys.argv = ["oracle_generate.py", "--model", "qwen4exp", snap, st, out]
        try:
            og.main()
        except SystemExit as e:
            assert "not qwen4_exp or qwen4_exp_text" in str(e), e
        else:
            raise AssertionError("--model qwen4exp ran on a kolibri1 snapshot")
        finally:
            sys.argv = saved


def test_qwen4exp_tiny_one_scenario():
    """Spec 21e: the tiny model (Q4EXP_TINY_SNAP: a snapshot directory with a tokenizer.json - the original's - beside
    the tiny's weights; transformers 5.19.0 first on PYTHONPATH) runs one scenario to its .bf16.{ids,txt,gap}.
    Skipped (printed) without the environment: run it in agnes-ref-img with the qwen4exp site."""
    snap = os.environ.get("Q4EXP_TINY_SNAP")
    if not snap:
        print("    skipped: Q4EXP_TINY_SNAP is unset")
        return
    import tempfile
    import oracle_generate as og
    with tempfile.TemporaryDirectory() as st, tempfile.TemporaryDirectory() as out:
        _fake_set(st, [760, 72103, 506, 37119, 557, 11012, 3213, 310, 6512, 279, 61789, 272])
        saved = sys.argv
        sys.argv = ["oracle_generate.py", "--model", "qwen4exp", snap, st, out, "--new-tokens", "6"]
        try:
            og.main()
        finally:
            sys.argv = saved
        for x in ("ids", "txt", "gap"):
            assert os.path.exists(os.path.join(out, f"s1.bf16.{x}")), x
        ids = open(os.path.join(out, "s1.bf16.ids")).read().split()
        assert 1 <= len(ids) <= 6, ids


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for n, f in tests:
        f()
        print(f"ok  {n}")
    print(f"{len(tests)} tests passed")
