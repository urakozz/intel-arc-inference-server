"""oracle_generate.py's MoE greedy loop (Ornith / K2-Horizon, spec 15e / 18d) without a model.

    python3 tools/toolcall/test_oracle_generate.py     (pytest also collects it, where installed)

A fake step records the (ids, pos) it is fed - the prompt at 0, then one id a step at the next
position, as a KV-cached decode is fed - and answers a scripted next id.
"""
import importlib.util
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


def test_mask_args():
    """Spec 22 P0.8: --expert-mask FILE (qwen4exp only) and the outputs' run tag .mask-<tag>."""
    import oracle_generate as og
    a = parse_args(["--model", "qwen4exp", "/snap", "set", "out", "--expert-mask", "/m/mask-reap-0.5.safetensors"])
    assert a["expert_mask"] == "/m/mask-reap-0.5.safetensors"
    assert og.mask_tag("/m/mask-reap-0.5.safetensors") == "mask-reap-0.5"
    assert og.mask_tag("/m/coding.safetensors") == "mask-coding"
    assert parse_args(["/snap", "set", "out"])["expert_mask"] is None


def test_qwen4exp_mask_reaches_reference():
    """Spec 22 P0.8: the mask reaches the reference - on a tiny Qwen3.8-Flash-Next (16 experts, top 4) the batched
    path's greedy ids with a mask that drops routed experts (keeping >= top_k a layer) differ from the unmasked ids,
    and no dropped expert is routed. Needs transformers 5.19.0 first on PYTHONPATH (the qwen4exp site): skipped
    (printed) otherwise."""
    try:
        import transformers
        ok = transformers.__version__ == "5.19.0"
    except ImportError:
        ok = False
    if not ok:
        print("    skipped: transformers 5.19.0 is not on PYTHONPATH (the qwen4exp site)")
        return
    import tempfile
    import torch
    from safetensors.torch import save_file
    import oracle_generate as og
    with tempfile.TemporaryDirectory() as snap:
        spec = importlib.util.spec_from_file_location(
            "tog_make_tiny", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "oracle", "qwen4exp_make_tiny.py"))
        mk = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mk)
        mk.make(snap, hidden=128, layer_types="lllq", experts=16, topk=4, form="intel", seed=29)
        prompt = [760, 72103, 506, 37119, 557, 11012, 3213, 310, 6512, 279]
        r0 = og.moe_runner(snap, "qwen4_exp")
        L, E, k = 4, 16, 4
        routed = [dict() for _ in range(L)]

        def count(i):
            def fn(_m, _a, out):          # observes only: a forward hook returning a value would replace the output
                for e in out[2].reshape(-1).tolist():
                    routed[i][e] = routed[i].get(e, 0) + 1
            return fn
        hooks = [layer.mlp.gate.register_forward_hook(count(i)) for i, layer in enumerate(r0.ref.model.layers)]
        out0 = {}
        og.greedy_batched(r0.step_many, r0.new_state, [("s", prompt, 6)], 2, set(), r0.argmax,
                          on_done=lambda n, o: out0.__setitem__(n, list(o)))
        for h in hooks:
            h.remove()
        keep = torch.ones(L, E, dtype=torch.uint8)
        for l in range(L):
            for e, _ in sorted(routed[l].items(), key=lambda x: -x[1])[:E - k - 2]:
                keep[l, e] = 0
        mpath = os.path.join(snap, "..", os.path.basename(snap) + "-mask-test.safetensors")
        save_file({"keep": keep, "mtp_keep": torch.ones(E, dtype=torch.uint8)}, mpath,
                  metadata={"format": "spec22-mask-1", "kept_fraction": str(float(keep.float().mean())),
                            "criterion": "test", "per_layer": "test", "calibration": "[]", "model": "tiny",
                            "checkpoint_revision": "test"})
        try:
            r1 = og.moe_runner(snap, "qwen4_exp", mask=mpath)
            seen = [set() for _ in range(L)]
            hooks = [layer.mlp.gate.register_forward_hook(
                lambda _m, _a, out, i=i: seen[i].update(out[2].reshape(-1).tolist()))
                for i, layer in enumerate(r1.ref.model.layers)]
            out1 = {}
            og.greedy_batched(r1.step_many, r1.new_state, [("s", prompt, 6)], 2, set(), r1.argmax,
                              on_done=lambda n, o: out1.__setitem__(n, list(o)))
            for h in hooks:
                h.remove()
        finally:
            os.remove(mpath)
        assert out0["s"] != out1["s"], (out0, out1)
        for l in range(L):
            assert not seen[l] & {e for e in range(E) if not keep[l, e]}, f"L{l}: a dropped expert was routed"
        print(f"    unmasked {out0['s']} vs masked {out1['s']}: the mask reached the batched reference")



def test_args_qwen():
    """2026-11-07: --model qwen (Qwen3.8, a4_ref.sh's qwen mode) names the dense path's model_type."""
    import oracle_generate as og
    a = parse_args(["--model", "qwen", "/snap", "set", "out", "--batch", "auto", "--resident", "auto"])
    assert a["model"] == "qwen" and a["keep"] == "auto"
    assert og.MODELS["qwen"] == ("qwen3_5",) and "qwen3_5" not in MOE_TYPES


def test_dense_keep():
    """The dense path's memory plan: layers kept = (cap - resident - 2 layers - the head's fp32 copy - slack) / a
    layer, clamped to [0, n]; none / an unknown cap keep none. Qwen3.8-27B's numbers: 64 layers of ~0.77 GB, a 2.5 GB
    head and a 2.5 GB embedding, the box's 56g cap -> most layers kept."""
    import oracle_generate as og
    G = og.GiB
    p = og.dense_keep([G] * 10, 2 * G, G, 20 * G)
    assert p["keep_layers"] == (20 - 2 - 2 - 2 - 6), p          # 8
    assert og.dense_keep([G] * 10, 2 * G, G, 100 * G)["keep_layers"] == 10
    assert og.dense_keep([G] * 10, 2 * G, G, 5 * G)["keep_layers"] == 0
    assert og.dense_keep([G] * 10, 2 * G, G, 100 * G, "none")["keep_layers"] == 0
    assert og.dense_keep([G] * 10, 2 * G, G, None)["keep_layers"] == 0
    layer, head = int(0.77e9), 248320 * 5120 * 2
    q = og.dense_keep([layer] * 64, 2 * head, head, 56 * G)
    assert 50 <= q["keep_layers"] < 64, q
    print(f"    {q['text']}")


def _tiny_qwen38(root):
    """A tiny random Qwen3.8-layout checkpoint (test_dump_taps.py's shape: config.json with a text_config,
    `model.language_model.*` + lm_head over two shards and an index, bf16), a word-level tokenizer.json and a
    generation_config.json."""
    import json
    import torch
    from safetensors.torch import save_file
    from tokenizers import Tokenizer, models
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
    text = dict(hidden_size=64, intermediate_size=96, num_hidden_layers=8, num_attention_heads=4,
                num_key_value_heads=2, head_dim=32, linear_num_key_heads=2, linear_num_value_heads=4,
                linear_key_head_dim=16, linear_value_head_dim=16, linear_conv_kernel_dim=4,
                full_attention_interval=4, vocab_size=320, rms_norm_eps=1e-6, attn_output_gate=True,
                rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25,
                                 "mrope_section": [2, 1, 1], "mrope_interleaved": True},
                partial_rotary_factor=0.25, tie_word_embeddings=False, model_type="qwen3_5_text",
                layer_types=(["linear_attention"] * 3 + ["full_attention"]) * 2)
    tc = Qwen3_5TextConfig(**text)
    torch.manual_seed(0)
    model = Qwen3_5ForCausalLM(tc).to(torch.bfloat16).eval()
    with torch.no_grad():
        for n, p in model.named_parameters():
            if p.dim() == 2 and "embed" not in n:
                p.copy_(torch.randn_like(p.float()) / p.shape[1] ** 0.5)
    shards = ({}, {})
    for n, t in model.state_dict().items():
        name = "lm_head.weight" if n == "lm_head.weight" else "model.language_model." + n[len("model."):]
        layer = int(name.split(".layers.")[1].split(".")[0]) if ".layers." in name else 99
        shards[0 if layer < 4 else 1][name] = t.contiguous()
    shards[1]["mtp.fc.weight"] = torch.zeros(4, 4, dtype=torch.bfloat16)   # skipped by name
    wm = {}
    for k, sd in enumerate(shards):
        fn = f"model-0000{k + 1}-of-00002.safetensors"
        save_file(sd, os.path.join(root, fn))
        wm.update({n: fn for n in sd})
    with open(os.path.join(root, "model.safetensors.index.json"), "w") as f:
        json.dump({"weight_map": wm}, f)
    with open(os.path.join(root, "config.json"), "w") as f:
        json.dump({"architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5",
                   "text_config": text}, f)
    with open(os.path.join(root, "generation_config.json"), "w") as f:
        json.dump({"eos_token_id": [319]}, f)
    Tokenizer(models.WordLevel({f"t{i}": i for i in range(320)}, unk_token="t0")).save(
        os.path.join(root, "tokenizer.json"))
    per = [0] * 8
    rest = head = 0
    for name, t in (kv for sd in shards for kv in sd.items()):
        if name.startswith("mtp."):
            continue
        b = t.numel() * 2
        if ".layers." in name:
            per[int(name.split(".layers.")[1].split(".")[0])] += b
        else:
            rest += b
            head += b if name.startswith("lm_head.") else 0
    return per, rest, head


def test_dense_streamed_tiny():
    """The dense path (Qwen3.8, a4_ref.sh qwen): on a tiny random checkpoint, the layer-streamed run (every layer
    streamed: --resident none; some kept: a small ORACLE_MEM_GB; all kept: a large one) writes the same ids as a
    RESIDENT Qwen3_5ForCausalLM under the same cpu_bf16 shim, HF generate - bitwise. Needs torch + transformers with
    qwen3_5 (agnes-ref-img / the box's reference image): skipped (printed) otherwise."""
    try:
        import torch
        from transformers.models.qwen3_5 import modeling_qwen3_5  # noqa: F401
    except ImportError:
        print("    skipped: no torch / transformers' qwen3_5 here (run it in the reference image)")
        return
    import tempfile
    import oracle_generate as og
    from transformers import AutoConfig, GenerationConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
    prompt = [5, 17, 42, 99, 3, 250, 7, 8, 120, 64, 33, 2]
    runs = {}
    with tempfile.TemporaryDirectory() as snap:
        per, rest, head = _tiny_qwen38(snap)
        some = (rest + 2 * max(per) + 2 * head + og.DENSE_SLACK + 3.5 * max(per)) / og.GiB   # 3 layers kept
        assert og.dense_keep(per, rest, head, int(some * og.GiB))["keep_layers"] == 3
        for tag, resident, mem in (("none", "none", None), ("some", "auto", repr(some)), ("all", "auto", "1000")):
            with tempfile.TemporaryDirectory() as st, tempfile.TemporaryDirectory() as out:
                _fake_set(st, prompt)
                saved, env = sys.argv, os.environ.get("ORACLE_MEM_GB")
                if mem:
                    os.environ["ORACLE_MEM_GB"] = mem
                sys.argv = ["oracle_generate.py", "--model", "qwen", snap, st, out, "--new-tokens", "8",
                            "--batch", "auto", "--resident", resident]
                try:
                    og.main()
                finally:
                    sys.argv = saved
                    if env is None:
                        os.environ.pop("ORACLE_MEM_GB", None)
                    else:
                        os.environ["ORACLE_MEM_GB"] = env
                runs[tag] = open(os.path.join(out, "s1.bf16.ids")).read().split()
                assert open(os.path.join(out, "s1.bf16.txt")).read().startswith("t"), tag
        # the resident reference, the shim installed (dense_build installed it, process-wide)
        tc = AutoConfig.from_pretrained(snap).get_text_config()
        tc._attn_implementation = "eager"
        model = Qwen3_5ForCausalLM(tc).to(torch.bfloat16).eval()
        from safetensors.torch import load_file
        sd = {}
        for fn in ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"):
            for k, v in load_file(os.path.join(snap, fn)).items():
                if not k.startswith("mtp."):
                    sd["model." + k[len("model.language_model."):] if k.startswith("model.language_model.") else k] = v
        model.load_state_dict(sd, strict=True)
        model.model.rotary_emb = type(model.model.rotary_emb)(tc)
        cfg = GenerationConfig(max_new_tokens=8, do_sample=False, eos_token_id=[319], pad_token_id=319)
        with torch.no_grad():
            seq = model.generate(torch.tensor([prompt]), generation_config=cfg)
    want = [str(int(x)) for x in seq[0, len(prompt):].tolist()]
    assert runs["none"] == runs["some"] == runs["all"] == want, (runs, want)
    print(f"    streamed (none / some / all kept) == resident: {want}")

if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for n, f in tests:
        f()
        print(f"ok  {n}")
    print(f"{len(tests)} tests passed")
