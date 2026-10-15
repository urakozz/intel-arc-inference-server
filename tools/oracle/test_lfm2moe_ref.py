#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""F1 for lfm2moe_ref.py (spec 24a Task 3): the streamed reference against transformers 5.19.0 un-streamed.

    PYTHONPATH=<5.19.0 site> python3 tools/oracle/test_lfm2moe_ref.py [test_name ...]   (pytest collects it too)

The downloaded tiny model is yujiepan/lfm2-moe-tiny-random ($LFM_TINY, else the HF cache - the container's /hf or
~/.cache); its tests SKIP when it is absent. The variants (pattern8, pattern8-fp32, ties, int4-g64 in both expert
spellings) are written by lfm2moe_make_tiny.py under $LFM_TINY_DIR (default <repo>/oracle-out-lfm2moe-tiny).
Everything is BITWISE (Review Focus 1): logits and every layer's residual, prompt and cached decode, bf16 and fp32.
One test per trap of spec 24 §2: the router's bf16 chain, planted ties, the combine's order, the conv's taps,
state and zero history, chunked = whole, the two-rounding norm, embedding_norm at the end and the tie, int4
exports, --layers N, and the port (the restated ops chained) against transformers' model.
"""
import glob
import io
import os
import sys
from contextlib import redirect_stderr

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

import importlib.util  # noqa: E402
import json  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ref = _load("lfm2moe_ref")
mk = _load("lfm2moe_make_tiny")
REPO = os.path.dirname(os.path.dirname(_HERE))
TINY_DIR = os.environ.get("LFM_TINY_DIR", os.path.join(REPO, "oracle-out-lfm2moe-tiny"))
BF, F32 = torch.bfloat16, torch.float32


class Skip(Exception):
    pass


def _skip(msg):
    if "pytest" in sys.modules:
        import pytest
        pytest.skip(msg)
    raise Skip(msg)


def downloaded_tiny() -> str:
    cands = [os.environ.get("LFM_TINY", "")]
    hf = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    for root in (hf, "/hf", os.path.expanduser("~/.cache/huggingface")):
        cands += sorted(glob.glob(os.path.join(root, "hub", "models--yujiepan--lfm2-moe-tiny-random", "snapshots", "*")))
    for c in cands:
        if c and os.path.exists(os.path.join(c, "config.json")):
            return c
    _skip("the tiny model is absent (hf download yujiepan/lfm2-moe-tiny-random, or LFM_TINY)")


def variant(name: str, names: str = "w") -> str:
    sub = name if name != "int4-g64" else f"int4-g64-{names}"
    d = os.path.join(TINY_DIR, sub)
    if not os.path.exists(os.path.join(d, "model.safetensors")):
        mk.make(d, name, 0, names=names, twin=os.path.join(TINY_DIR, "int4-g64-twin") if name == "int4-g64" else None)
    return d


_CACHE: dict = {}


def hf_model(snap, dtype, layers=None, experts=ref.DEFAULT_EXPERTS):
    key = ("hf", snap, dtype, layers, experts)
    if key not in _CACHE:
        _CACHE[key] = ref.hf_reference(snap, dtype, layers, experts)
    return _CACHE[key]


def port_model(snap, dtype, layers=None, weights="bf16"):
    key = ("port", snap, dtype, layers, weights)
    if key not in _CACHE:
        cfg = ref.config(snap, layers, dtype=dtype)
        _CACHE[key] = (ref.build_streamed(snap, cfg, weights), cfg)
    return _CACHE[key]


def vocab(snap) -> int:
    return min(json.load(open(os.path.join(snap, "config.json"), encoding="utf-8"))["vocab_size"], 124000)


def rand_ids(n, seed, v):
    return torch.randint(0, v, (n,), generator=torch.Generator().manual_seed(seed)).tolist()


def hidden_hooks(model, store: dict):
    return [layer.register_forward_hook(lambda _m, _a, out, i=i: store.setdefault(i, []).append(out.detach()[0].clone()))
            for i, layer in enumerate(model.model.layers)]


@torch.no_grad()
def drive(model, ids, steps, chunk=0):
    """The prompt (whole, or in chunks), then the decode tokens one at a time: every logits row, and the cache."""
    rows, cache = [], None
    c = chunk or len(ids)
    for a in range(0, len(ids), c):
        out = model(input_ids=torch.tensor([ids[a:a + c]]), past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        rows.append(out.logits[0])
    for t in steps:
        out = model(input_ids=torch.tensor([[t]]), past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        rows.append(out.logits[0])
    return torch.cat(rows), cache


@torch.no_grad()
def greedy_steps(model, ids, n):
    lg, cache = drive(model, ids, [])
    steps, last = [], lg[-1]
    for _ in range(n):
        t = int(last.float().argmax())
        steps.append(t)
        out = model(input_ids=torch.tensor([[t]]), past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        last = out.logits[0, -1]
    return steps


def compare(a, b, ids, steps, what, chunk_a=0, chunk_b=0):
    """Two models on the same prompt and decode tokens: logits and every layer's residual bitwise; conv states too."""
    ha, hb = {}, {}
    hooks = hidden_hooks(a, ha)
    try:
        la, ca = drive(a, ids, steps, chunk_a)
    finally:
        for h in hooks:
            h.remove()
    hooks = hidden_hooks(b, hb)
    try:
        lb, cb = drive(b, ids, steps, chunk_b)
    finally:
        for h in hooks:
            h.remove()
    assert la.shape == lb.shape, (what, la.shape, lb.shape)
    if not torch.equal(la, lb):
        d = (la.float() - lb.float()).abs()
        raise AssertionError(f"{what}: logits differ on {int((d > 0).any(-1).sum())} rows, max {d.max().item():.3e}")
    for i in ha:
        x, y = torch.cat(ha[i]), torch.cat(hb[i])
        assert torch.equal(x, y), f"{what}: h.L{i} differs (max {(x.float() - y.float()).abs().max().item():.3e})"
    for i, layer in enumerate(ca.layers):
        if getattr(layer, "conv_states", None) and layer.conv_states.get(0) is not None:
            assert torch.equal(layer.conv_states[0], cb.layers[i].conv_states[0]), f"{what}: conv state L{i}"
    return la


# ------------------------------------------------------------------------------------------------
# Review Focus 1: bitwise, not close

def _streamed_vs_hf(dtype):
    cases = [(variant("pattern8-fp32" if dtype == F32 else "pattern8"), "pattern8")]
    try:
        cases.append((downloaded_tiny(), "downloaded tiny"))
    except Skip as e:
        print(f"  ({e})")
    for snap, what in cases:
        hf = hf_model(snap, dtype)
        port, _ = port_model(snap, dtype)
        for n, seed in ((40, 1), (600, 2)):
            ids = rand_ids(n, seed, vocab(snap))
            steps = greedy_steps(hf, ids, 8)
            compare(hf, port, ids, steps, f"{what} {dtype} {n} ids")
            print(f"  {what} {str(dtype)[6:]}: {n} ids + 8 cached steps bitwise (greedy {steps[:4]}...)")


def test_streamed_equals_hf_bf16():
    _streamed_vs_hf(BF)


def test_streamed_equals_hf_fp32():
    _streamed_vs_hf(F32)


def test_chunked_equals_whole():
    """Review Focus 3: the 600-id prompt in chunks of 1, 2, 3, 64, 255 through the cache, and split at every offset
    1..5, equals one forward bitwise - logits, every residual, the conv states."""
    snap = variant("pattern8")
    port, _ = port_model(snap, BF)
    ids = rand_ids(600, 3, vocab(snap))
    for c in (1, 2, 3, 64, 255):
        compare(port, port, ids, [5, 6], f"chunk {c}", chunk_a=0, chunk_b=c)
    hf = hf_model(snap, BF)
    for o in range(1, 6):
        ha, hb = {}, {}
        hooks = hidden_hooks(hf, ha)
        try:
            whole, cw = drive(hf, ids[:o + 7], [])
        finally:
            for h in hooks:
                h.remove()
        hooks = hidden_hooks(hf, hb)
        try:
            with torch.no_grad():
                o1 = hf(input_ids=torch.tensor([ids[:o]]), use_cache=True)
                o2 = hf(input_ids=torch.tensor([ids[o:o + 7]]), past_key_values=o1.past_key_values, use_cache=True)
        finally:
            for h in hooks:
                h.remove()
        assert torch.equal(whole, torch.cat([o1.logits[0], o2.logits[0]])), f"split at {o}"
        for i in ha:
            assert torch.equal(torch.cat(ha[i]), torch.cat(hb[i])), f"split at {o}: h.L{i}"
    print("  chunks 1 / 2 / 3 / 64 / 255 and splits at 1..5 bitwise the whole forward (conv state included)")


# ------------------------------------------------------------------------------------------------
# Review Focus 2: the router's bf16 chain

def _layer_input(model, layer, ids):
    """The rows the layer's router sees (ffn_norm's output) for `ids`."""
    got = {}
    h = model.model.layers[layer].feed_forward.gate.register_forward_hook(
        lambda _m, a, _o: got.__setitem__("x", a[0].detach().clone()))
    with torch.no_grad():
        model(input_ids=torch.tensor([ids]), use_cache=False)
    h.remove()
    return got["x"]


def test_router_chain():
    snap = variant("pattern8")
    hf = hf_model(snap, BF)
    blk = hf.model.layers[2].feed_forward
    x = _layer_input(hf, 2, rand_ids(512, 4, vocab(snap)))
    logits, w_hf, ids_hf = blk.gate(x, blk.expert_bias)
    ids, w, sel, gap = ref.route(x, blk.gate.weight, blk.expert_bias, 4)
    assert torch.equal(ids, ids_hf) and torch.equal(w, w_hf) and w.dtype == BF and sel.dtype == F32
    s = logits.sigmoid()
    assert s.dtype == BF
    # the bias reorders: a selected expert with a LOWER sigmoid than an unselected one
    chosen = torch.zeros_like(s, dtype=torch.bool).scatter_(1, ids, True)
    min_chosen = s.float().masked_fill(~chosen, 9).min(-1).values
    max_other = s.float().masked_fill(chosen, -9).max(-1).values
    reordered = int((min_chosen < max_other).sum())
    assert reordered > 0, "no row where the bias selects a lower sigmoid"
    # the weights are s at the ids, renormalised in bf16: never sel
    want = torch.gather(s, 1, ids)
    want = want / (want.sum(-1, keepdim=True) + 1e-6)
    assert torch.equal(w, want)
    from_sel = torch.gather(sel, 1, ids)
    from_sel = (from_sel / (from_sel.sum(-1, keepdim=True) + 1e-6)).to(BF)
    leak = int((from_sel != w).any(-1).sum())
    assert leak > 0, "a non-zero bias leaking into the weights would change them"
    # bias zeroed: the selection changes on some rows, the weights of an unchanged selection are the same
    ids0, w0, _, _ = ref.route(x, blk.gate.weight, torch.zeros_like(blk.expert_bias), 4)
    same = (ids0 == ids).all(-1)
    assert int((~same).sum()) > 0 and torch.equal(w0[same], w[same])
    # the chain's last steps: sum in bf16 (+ 1e-6 rounded again), a bf16 division - not one fp32 renorm
    fp32_renorm = (torch.gather(s, 1, ids).float() / torch.gather(s, 1, ids).float().sum(-1, keepdim=True)).to(BF)
    print(f"  router: 512 rows bitwise; bias selects a lower sigmoid on {reordered} rows; a bias leaking into the "
          f"weights would change {leak}; zero bias changes the selection on {int((~same).sum())}; an fp32 renorm "
          f"differs on {int((fp32_renorm != w).any(-1).sum())} rows")


def test_route_ties():
    snap = variant("ties")
    port, cfg = port_model(snap, BF)
    rec = ref.Recorder(port, cfg)
    ids = rand_ids(64, 7, vocab(snap))
    with torch.no_grad():
        port(input_ids=torch.tensor([ids]), use_cache=False)
    rec.remove()
    t = rec.tensors()
    for layer, (a, c) in mk.TIES.items():
        gap = t[f"route.gap.L{layer}"]
        assert bool((gap == 0).all()), (layer, gap[:4])
        rid = t[f"route.ids.L{layer}"]
        fourth = rid[:, 3].tolist()
        won = {a: fourth.count(a), c: fourth.count(c)}
        assert won[a] + won[c] == len(fourth), (layer, won)
        # the recorded route is torch's CPU topk on the same sel (its tie order, whatever it is)
        hf = hf_model(snap, BF)
        blk = hf.model.layers[layer].feed_forward
        x = _layer_input(hf, layer, ids)
        rids, _, _, rgap = ref.route(x, blk.gate.weight, blk.expert_bias, 4)
        assert torch.equal(rids.to(torch.int32), rid) and bool((rgap == 0).all())
        print(f"  ties L{layer}: gap exactly 0 on all {len(fourth)} rows; the 4th slot went to expert {a} on {won[a]}, "
              f"to {c} on {won[c]} (torch's CPU topk order - the engine's rule is the lower id; such rows are "
              f"undetermined)")


# ------------------------------------------------------------------------------------------------
# Review Focus 4: the combine's order

def test_combine_order():
    snap = variant("pattern8")
    for impl in ("grouped_mm", "eager"):
        hf = hf_model(snap, BF, experts=impl)
        blk = hf.model.layers[3].feed_forward
        assert hf.config._experts_implementation == impl
        x = _layer_input(hf, 3, rand_ids(256, 5, vocab(snap)))
        _, w, ids = blk.gate(x, blk.expert_bias)
        with torch.no_grad():
            want = blk.experts(x, ids, w)
            proj = ref.expert_outputs(x, blk.experts.gate_up_proj, blk.experts.down_proj, ids, impl)
            got = ref.combine(proj, w, impl, ids)
            other = ref.combine(proj, w, "eager" if impl == "grouped_mm" else "grouped_mm", ids)
            torch_sum = (proj * w.unsqueeze(-1)).sum(dim=1)
        assert torch.equal(got, want), f"{impl}: restated combine != the experts module"
        n_other = int((other != want).any(-1).sum())
        assert n_other > 0, f"{impl}: the other order never differs - the test cannot tell them apart"
        if impl == "grouped_mm":
            assert torch.equal(got, torch_sum), "the sequential fp32 chain != torch's bf16 .sum(dim=1)"
        print(f"  combine {impl}: restated = the module bitwise (256 rows); the other order differs on {n_other} rows")


# ------------------------------------------------------------------------------------------------
# Review Focus 3: the short conv

def test_short_conv():
    snap = variant("pattern8")
    hf = hf_model(snap, BF)
    conv = hf.model.layers[3].conv
    m = ref.mm()
    H = hf.config.hidden_size
    x = _layer_input(hf, 3, rand_ids(40, 6, vocab(snap)))   # any bf16 rows of the right width
    cache = m.DynamicCache(config=hf.config)
    with torch.no_grad():
        want_p = conv(x[None, :37], past_key_values=cache)[0]
        want_d = conv(x[None, 37:38], past_key_values=cache)[0]
        got_p, st = ref.short_conv(x[:37], conv.in_proj.weight, conv.conv.weight, conv.out_proj.weight)
        got_d, st2 = ref.short_conv(x[37:38], conv.in_proj.weight, conv.conv.weight, conv.out_proj.weight, st)
    assert torch.equal(got_p, want_p) and torch.equal(got_d, want_d)
    assert torch.equal(st2, cache.layers[3].conv_states[0][0, :, -2:].t()), "the state is Bx of the last two"
    # zero history at positions 0 and 1: a non-zero history changes rows 0..1 only
    with torch.no_grad():
        noise, _ = ref.short_conv(x[:37], conv.in_proj.weight, conv.conv.weight, conv.out_proj.weight,
                                  torch.ones(2, H, dtype=BF))
    rows = (noise != got_p).any(-1)
    assert bool(rows[:2].all()) and not bool(rows[2:].any())
    # the taps' order is live: swapping two taps changes the output
    w2 = conv.conv.weight.clone()
    w2[..., [0, 2]] = w2[..., [2, 0]]
    with torch.no_grad():
        sw, _ = ref.short_conv(x[:37], conv.in_proj.weight, w2, conv.out_proj.weight)
    assert not torch.equal(sw, got_p)
    print("  short conv: prompt (37 rows) and a cached step bitwise; state = Bx of the last two; positions 0 / 1 "
          "read zeros; swapping taps 0 / 2 changes the output")


# ------------------------------------------------------------------------------------------------
# the two-rounding norm; the end of the model (Review Focus 5)

def test_norm_two_roundings():
    snap = variant("pattern8")
    hf = hf_model(snap, BF)
    norm = hf.model.layers[3].operator_norm
    x = (torch.randn(512, hf.config.hidden_size, generator=torch.Generator().manual_seed(8)) * 3).to(BF)
    with torch.no_grad():
        want = norm(x)
    got = ref.rms_norm(x, norm.weight, norm.variance_epsilon)
    assert torch.equal(got, want) and norm.variance_epsilon == 1e-5
    xf = x.float()
    one = (norm.weight.float() * (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-5))).to(BF)
    n1 = int((one != want).sum())
    assert n1 > 0
    q = hf.model.layers[2].self_attn.q_layernorm
    xq = x[:, :64]
    assert torch.equal(ref.rms_norm(xq, q.weight, q.variance_epsilon), q(xq)) and q.weight.shape == (64,)
    print(f"  rms_norm: bitwise (and the 64-wide q norm); a one-rounding variant differs on {n1} of {x.numel()} elements")


def test_final_norm_and_tie():
    snap = variant("pattern8")
    port, cfg = port_model(snap, BF)
    assert port.lm_head.weight is port.model.embed_tokens.weight
    hf = hf_model(snap, BF)
    assert hf.lm_head.weight.data_ptr() == hf.model.embed_tokens.weight.data_ptr()
    ids = rand_ids(16, 9, vocab(snap))
    step = ref.layer_major(port)
    st = {"cache": None}
    R = step([(ids, 0, st)])[0]
    with torch.no_grad():
        logits = ref.final_logits(port, R)
        full = port(input_ids=torch.tensor([ids]), use_cache=False).logits[0]
        no_norm = F.linear(R, port.model.embed_tokens.weight)
    assert torch.equal(logits, full), "layer_major + final_logits != the model's forward"
    assert not torch.equal(no_norm, full), "dropping embedding_norm changed nothing"
    assert logits.shape[-1] == 128000
    print("  embedding_norm after the last layer (dropping it changes the logits); the head IS the embedding "
          "(128000 rows); layer_major + final_logits = the forward bitwise")


# ------------------------------------------------------------------------------------------------
# int4 exports, --layers N, the port

def test_int4_export():
    twin = None
    for names in ("w", "gate_up_down"):
        snap = variant("int4-g64", names)
        twin = os.path.join(TINY_DIR, "int4-g64-twin")
        hf = hf_model(twin, BF)
        port, _ = port_model(snap, BF, weights="int4")
        ids = rand_ids(40, 10, 124000)
        compare(hf, port, ids, greedy_steps(hf, ids, 4), f"int4 {names}")
        print(f"  int4-g64 ({names}): the export dequantised = transformers on the twin, bitwise")
    cfg = ref.config(variant("int4-g64"), dtype=BF)
    try:
        with redirect_stderr(io.StringIO()):
            ref.build_streamed(variant("int4-g64"), cfg, "bf16")
        raise AssertionError("an int4 export accepted as --weights bf16")
    except SystemExit:
        pass


def test_layers_truncation():
    snap = variant("pattern8")
    hf = hf_model(snap, BF, layers=3)
    port, cfg = port_model(snap, BF, layers=3)
    assert cfg.num_hidden_layers == 3 and cfg.layer_types == ["conv", "conv", "full_attention"]
    ids = rand_ids(40, 11, vocab(snap))
    compare(hf, port, ids, greedy_steps(hf, ids, 4), "--layers 3")
    err = io.StringIO()
    try:
        with redirect_stderr(err):
            ref.config(snap, 2)
        raise AssertionError("--layers 2 accepted")
    except SystemExit:
        assert "attention layer" in err.getvalue() and "layer 2" in err.getvalue(), err.getvalue()
    print("  --layers 3 bitwise transformers at num_hidden_layers 3; --layers 2 refused naming the attention layer")


def test_port_equals_hf():
    """hfcheck on the tiny models: the restated ops chained = transformers' model, bitwise (logits, every h / mixer /
    ffn row, the routes, the conv states) - prompt whole and in chunks, then cached steps."""
    for snap, dtype in ((variant("pattern8"), BF), (variant("pattern8-fp32"), F32)):
        cfg = ref.config(snap, dtype=dtype)
        for chunk in (0, 64):
            (ht, htok), (pt, ptok) = ref.hfcheck_arms(snap, cfg, rand_ids(200, 12, 124000), 4, "bf16", chunk)
            assert htok == ptok
            bad = [k for k in sorted(ht) if k not in pt or not torch.equal(ht[k], pt[k])]
            assert not bad and set(ht) == set(pt), (snap, chunk, bad[:5], sorted(set(ht) ^ set(pt))[:5])
        print(f"  port = transformers ({str(dtype)[6:]}): {len(ht)} tensors bitwise, whole and chunk 64")


def test_expected_names():
    from safetensors import safe_open
    for name, form, spell in (("pattern8", "bf16", "w"), ("int4-g64", "int4", "w")):
        snap = variant(name)
        cfg = ref.config(snap)
        with safe_open(os.path.join(snap, "model.safetensors"), framework="pt") as f:
            assert set(f.keys()) == set(ref.expected_names(cfg, form, spell)), name
    snap = variant("int4-g64", "gate_up_down")
    with safe_open(os.path.join(snap, "model.safetensors"), framework="pt") as f:
        assert set(f.keys()) == set(ref.expected_names(ref.config(snap), "int4", "proj"))
    real = mk.default_tokenizer()
    if real and os.path.exists(os.path.join(real, "model.safetensors.index.json")):
        wm = json.load(open(os.path.join(real, "model.safetensors.index.json"), encoding="utf-8"))["weight_map"]
        want = ref.expected_names(ref.config(real), "bf16")
        assert set(wm) == set(want) and len(want) == 7610, (len(wm), len(want))
        print(f"  expected_names: the variants' tensors, and the real index's {len(wm)} names")


def test_cli_run():
    import subprocess
    import tempfile
    from safetensors import safe_open
    snap = variant("pattern8")
    d = tempfile.mkdtemp(prefix="lfm_cli_")
    p = os.path.join(d, "p.ids")
    ids = rand_ids(30, 13, 124000)
    with open(p, "w") as f:
        f.write(" ".join(map(str, ids)) + "\n")
    out = os.path.join(d, "p.golden.safetensors")
    r = subprocess.run([sys.executable, os.path.join(_HERE, "lfm2moe_ref.py"), "run", snap, "--prompt", p, "--out", out,
                        "--gen", "3"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "MoE 4th/5th sel gap" in r.stdout
    with safe_open(out, framework="pt") as f:
        keys = set(f.keys())
        assert f.get_tensor("logits").shape == (33, 128000) and f.get_tensor("tokens").numel() == 3
        assert f.get_tensor("route.ids.L2").shape == (33, 4) and f.get_tensor("route.sel.L7").shape == (33, 5)
        assert f.get_tensor("conv.L0").shape == (2, 256) and "conv.L2" not in keys and "route.ids.L1" not in keys
        assert f.get_tensor("h.L7").shape == (33, 256) and f.get_tensor("nll").shape == (29,)
        meta = f.metadata()
    assert meta["experts_implementation"] == ref.DEFAULT_EXPERTS and meta["attn_implementation"] == "eager"
    print(f"  run: {len(keys)} tensors, the golden layout")


if __name__ == "__main__":
    want = sys.argv[1:]
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)
             and (not want or n in want)]
    skipped = 0
    for n, f in tests:
        try:
            f()
            print(f"ok  {n}", flush=True)
        except Skip as e:
            skipped += 1
            print(f"SKIP {n}: {e}", flush=True)
    print(f"{len(tests) - skipped} tests passed, {skipped} skipped")
