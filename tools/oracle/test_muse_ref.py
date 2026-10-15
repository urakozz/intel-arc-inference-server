#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""MG1 for muse_ref.py (spec 23a Task 3): the streamed port against transformers 5.19.0 un-streamed.

    PYTHONPATH=<5.19.0 site> python3 tools/oracle/test_muse_ref.py [test_name ...]   (pytest collects it too)

The tiny model is optimum-intel-internal-testing/tiny-random-muse-glimmer: $MUSE_TINY, else the HF cache
(`$HF_HOME/hub/models--optimum-intel-internal-testing--tiny-random-muse-glimmer`, or /hf/hub in the container);
the variants (muse_make_tiny.py) are written to a temp dir from it and the original's small files. Every test that
needs them SKIPs when they are absent. Everything is BITWISE (Review Focus 1): raw and capped logits and every
layer's residual, prompt and cached decode, bf16 and fp32, at lengths that cross the sliding window. One test per
trap of spec 23 §1 / §1.1 (Review Focus 2-5).
"""
import glob
import json
import os
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

import importlib.util  # noqa: E402

import torch  # noqa: E402


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ref = _load("muse_ref")
mk = _load("muse_make_tiny")
BF, F32 = torch.bfloat16, torch.float32
LM = "model.language_model."


class Skip(Exception):
    pass


def _skip(msg):
    if "pytest" in sys.modules:
        import pytest
        pytest.skip(msg)
    raise Skip(msg)


def _hf(repo):
    org, name = repo.split("/")
    hubs = [os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub"), "/hf/hub"]
    for hub in hubs:
        for c in sorted(glob.glob(os.path.join(hub, f"models--{org}--{name}", "snapshots", "*"))):
            if os.path.exists(os.path.join(c, "config.json")):
                return c
    return None


def tiny_snapshot() -> str:
    c = os.environ.get("MUSE_TINY") or _hf("optimum-intel-internal-testing/tiny-random-muse-glimmer")
    if not c or not os.path.exists(os.path.join(c, "model.safetensors")):
        _skip("the tiny model is absent (hf download optimum-intel-internal-testing/tiny-random-muse-glimmer, or MUSE_TINY)")
    return c


def original_snapshot() -> str:
    c = _hf("meta-models/Muse-Glimmer-30B")
    if not c or not os.path.exists(os.path.join(c, "chat_template.jinja")):
        _skip("the original's small files are absent (hf download meta-models/Muse-Glimmer-30B --include '*.json' '*.jinja')")
    return c


def gptq_snapshot() -> str:
    c = _hf("Vishva007/Muse-Glimmer-30B-W4A16-AutoRound-GPTQ")
    if not c or not os.path.exists(os.path.join(c, "model.safetensors.index.json")):
        _skip("the GPTQ repo's small files are absent")
    return c


TMP = tempfile.mkdtemp(prefix="muse_test_")
_CACHE: dict = {}


def variant(name: str) -> str:
    """A muse_make_tiny variant (built once per process), or "tiny" for the downloaded model itself."""
    if name == "tiny":
        return tiny_snapshot()
    if name == "int4-twin":
        variant("int4")
        return os.path.join(TMP, "int4-twin")
    if name == "nope0":                      # untied with layer 0 made full / NoPE (test_nope_switch)
        d = os.path.join(TMP, "nope0")
        if not os.path.exists(d):
            import shutil
            shutil.copytree(variant("untied"), d)
            cfg = json.load(open(os.path.join(d, "config.json")))
            cfg["text_config"]["layer_types"][0] = "full_attention"
            cfg["text_config"]["layer_rope_theta"][0] = 0
            json.dump(cfg, open(os.path.join(d, "config.json"), "w"), indent=2)
        return d
    d = os.path.join(TMP, name)
    if not os.path.exists(os.path.join(d, "model.safetensors")):
        gq = gptq_snapshot() if name == "int4" else None
        mk.make(tiny_snapshot(), d, name, original_snapshot(), gq,
                twin=os.path.join(TMP, "int4-twin") if name == "int4" else None)
    return d


def hf_model(snap, dtype):
    key = ("hf", snap, dtype)
    if key not in _CACHE:
        _CACHE[key] = ref.hf_reference(snap, dtype)
    return _CACHE[key]


def port_model(snap, dtype, layers=None):
    key = ("port", snap, dtype, layers)
    if key not in _CACHE:
        tc = ref.text_config(snap, layers, dtype=dtype)
        model, pf = ref.build_streamed(snap, tc)
        _CACHE[key] = (model, pf, tc)
    return _CACHE[key]


def layers_of(model):
    return model.model.language_model.layers


def hidden_hooks(model, store: dict):
    hs = []
    for i, layer in enumerate(layers_of(model)):
        hs.append(layer.register_forward_hook(
            lambda _m, _a, out, i=i: store.setdefault(i, []).append(out.detach()[0].clone())))
    return hs


def raw_hook(model, store: list):
    return model.lm_head.register_forward_hook(lambda _m, _a, out: store.append(out.detach()[0].clone()))


@torch.no_grad()
def drive(model, ids, steps, chunk=0, keep=48):
    """Prompt (whole, or in chunks), then the given decode tokens one at a time: the capped logits of each prompt
    chunk's last `keep` rows and of every decode step [*][V] (the raw rows through raw_hook)."""
    rows, cache = [], None
    c = chunk or len(ids)
    for a in range(0, len(ids), c):
        out = model(input_ids=torch.tensor([ids[a:a + c]]), past_key_values=cache, use_cache=True, logits_to_keep=keep)
        cache = out.past_key_values
        rows.append(out.logits[0])
    for t in steps:
        out = model(input_ids=torch.tensor([[t]]), past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        rows.append(out.logits[0])
    return torch.cat(rows), cache


def rand_ids(n, seed=0, vocab=200000):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (n,), generator=g).tolist()


def greedy_steps(hf, ids, gen):
    rows, cache = drive(hf, ids, [], keep=1)
    steps, last = [], rows[-1]
    with torch.no_grad():
        for _ in range(gen):
            t = int(last.float().argmax())
            steps.append(t)
            out = hf(input_ids=torch.tensor([[t]]), past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            last = out.logits[0, -1]
    return steps


def run_both(model, ids, steps, chunk=0):
    hh, raw = {}, []
    hooks = hidden_hooks(model, hh) + [raw_hook(model, raw)]
    try:
        cap, _ = drive(model, ids, steps, chunk)
    finally:
        for h in hooks:
            h.remove()
    return cap, torch.cat(raw), hh


def compare(hf, port, ids, gen=8, what="", chunk=0):
    steps = greedy_steps(hf, ids, gen)
    cap_h, raw_h, hh = run_both(hf, ids, steps, chunk)
    cap_p, raw_p, hp = run_both(port, ids, steps, chunk)
    for name, a, b in (("capped logits", cap_p, cap_h), ("raw logits", raw_p, raw_h)):
        assert a.shape == b.shape, (what, name, a.shape, b.shape)
        if not torch.equal(a, b):
            d = (a.float() - b.float()).abs()
            raise AssertionError(f"{what}: {name} differ, max |d| {d.max():.3e}")
    assert sorted(hh) == sorted(hp)
    for i in hh:
        a, b = torch.cat(hp[i]), torch.cat(hh[i])
        assert torch.equal(a, b), f"{what}: H.L{i} differs (max |d| {(a.float() - b.float()).abs().max():.3e})"
    return steps


# ------------------------------------------------------------------------------------------------
# MG1: the streamed port equals transformers, bitwise

def _streamed_equals_hf(dtype):
    for v in ("untied", "l8", "gqa16", "win16"):
        snap = variant(v)
        hf, port = hf_model(snap, dtype), port_model(snap, dtype)[0]
        for n in ((40, 100) if v == "win16" else (40,)):
            steps = compare(hf, port, rand_ids(n, seed=n), 8, what=f"{v} {dtype} {n} ids")
            print(f"  {v} {dtype}: prompt {n} + 8 cached decode steps {steps[:3]}...: raw / capped logits and every "
                  f"H.L bitwise")
    if dtype == BF:                               # once at the real window: 2100 ids cross 2048
        snap = variant("l8")
        hf, port = hf_model(snap, dtype), port_model(snap, dtype)[0]
        compare(hf, port, rand_ids(2100, seed=2100), 4, what="l8 bf16 2100 ids")
        print("  l8 bf16: prompt 2100 (window 2048 crossed) + 4 steps: bitwise")


def test_streamed_equals_hf_bf16():
    _streamed_equals_hf(BF)


def test_streamed_equals_hf_fp32():
    _streamed_equals_hf(F32)


def test_chunked_prefill_equals_hf():
    """run's prefill in chunks (2048 on long prompts; 32 here on win16) is transformers' chunked forward bitwise."""
    snap = variant("win16")
    hf, port = hf_model(snap, BF), port_model(snap, BF)[0]
    compare(hf, port, rand_ids(100, seed=7), 4, what="win16 chunk 32", chunk=32)
    print("  win16 bf16: 100 ids in chunks of 32 (the ring of 15 cached keys + the chunk) + 4 steps: bitwise")


# ------------------------------------------------------------------------------------------------
# the traps of spec 23 §1.1

def _pow_norm(x, w=None, eps=1e-5):
    xf = x.float()
    y = xf * torch.pow(xf.pow(2).mean(-1, keepdim=True) + eps, -0.5)
    return (y * w.float() if w is not None else y).type_as(x)


def _rsqrt_norm(x, w=None, eps=1e-5):
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (y * w.float() if w is not None else y).type_as(x)


def _centered(x, w, eps, use_pow=False):
    xf = x.float()
    ms = xf.pow(2).mean(-1, keepdim=True) + eps
    y = xf * (torch.pow(ms, -0.5) if use_pow else torch.rsqrt(ms))
    return (y * (1.0 + w.float())).type_as(x)


def _apart(f, g, dim=64, draws=10**6, seed=3, w=None):
    """A bf16 row where f and g round apart, searched over `draws` random rows (None if none)."""
    gen = torch.Generator().manual_seed(seed)
    for b0 in range(0, draws, 50000):
        x = (torch.randn(50000, dim, generator=gen) * torch.rand(50000, 1, generator=gen) * 8).to(BF)
        d = (f(x) != g(x)).any(-1)
        if bool(d.any()):
            return x[int(d.nonzero()[0])]
    return None


def test_norm_forms():
    """Review Focus 2: the embed norm and the final norm ARE MuseGlimmerRMSNorm (pow) without / with scale, bitwise
    on the port's own rows; the centered norms are rsqrt with (1 + w) and eps 1e-5 (input / pre-ff) / 1e-8 (post);
    each restatement in the other form rounds apart on a crafted row."""
    m = ref.mm()
    snap = variant("untied")
    port, _, tc = port_model(snap, BF)
    lm = port.model.language_model
    ids = rand_ids(40, seed=11)
    cap = {}
    hk = [lm.embed_tokens.register_forward_hook(lambda _m, a, o: cap.update(emb_in=a[0], emb=o.detach())),
          lm.norm.register_forward_hook(lambda _m, a, o: cap.update(fin_in=a[0].detach(), fin=o.detach()))]
    with torch.no_grad():
        port(input_ids=torch.tensor([ids]), use_cache=False)
    for h in hk:
        h.remove()
    rows = lm.embed_tokens.weight[torch.tensor(ids)].unsqueeze(0)
    assert torch.equal(cap["emb"], m.MuseGlimmerRMSNorm(eps=tc.rms_norm_eps, with_scale=False)(rows))
    assert torch.equal(cap["emb"], _pow_norm(rows, eps=tc.rms_norm_eps)), "the embed norm is not pow(ms + eps, -0.5)"
    assert torch.equal(cap["fin"], _pow_norm(cap["fin_in"], lm.norm.weight, tc.rms_norm_eps)), "final norm != pow * w"
    assert lm.norm.with_scale and not lm.embed_tokens.embed_norm.with_scale
    layer = layers_of(hf_model(snap, BF))[0]           # the same class, materialised (the port's are back on meta)
    assert layer.input_layernorm.eps == 1e-5 and layer.pre_feedforward_layernorm.eps == 1e-5
    assert layer.post_attention_layernorm.eps == 1e-8 and layer.post_feedforward_layernorm.eps == 1e-8
    x = (torch.randn(1, 40, tc.hidden_size) * 3).to(BF)
    assert torch.equal(layer.input_layernorm(x), _centered(x, layer.input_layernorm.weight, 1e-5))
    assert torch.equal(layer.post_attention_layernorm(x), _centered(x, layer.post_attention_layernorm.weight, 1e-8))
    lines = []
    w = lm.norm.weight
    for what, f, g in (("embed norm: pow vs rsqrt", lambda r: _pow_norm(r), lambda r: _rsqrt_norm(r)),
                       ("final norm: pow * w vs rsqrt * w", lambda r: _pow_norm(r, w), lambda r: _rsqrt_norm(r, w)),
                       ("centered: rsqrt vs pow", lambda r: _centered(r, layer.input_layernorm.weight, 1e-5),
                        lambda r: _centered(r, layer.input_layernorm.weight, 1e-5, use_pow=True))):
        row = _apart(f, g)
        if row is None:
            lines.append(f"{what}: indistinguishable in 10^6 rows")
            continue
        if what.startswith("embed"):
            mod = m.MuseGlimmerRMSNorm(eps=1e-5, with_scale=False)
            assert torch.equal(mod(row), f(row)) and not torch.equal(mod(row), g(row))
        elif what.startswith("final"):
            assert torch.equal(lm.norm(row), f(row)) and not torch.equal(lm.norm(row), g(row))
        else:
            mod = layer.input_layernorm
            assert torch.equal(mod(row), f(row)) and not torch.equal(mod(row), g(row))
        lines.append(f"{what}: a crafted row rounds apart; the module matches the first form")
    for s in lines:
        print("  " + s)
    # why "indistinguishable": on torch's CPU, pow(x, -0.5), rsqrt(x) and 1 / sqrt(x) are the same fp32 value (the
    # correctly rounded 1 / sqrt); a device kernel must reproduce that, not an approximate native rsqrt (23c's fixture)
    v = torch.rand(10**7, generator=torch.Generator().manual_seed(5)) * 1e3 + 1e-6
    same = int((torch.pow(v, -0.5) == torch.rsqrt(v)).sum())
    inv = int((1.0 / torch.sqrt(v) == torch.rsqrt(v)).sum())
    print(f"  fp32: torch.pow(x, -0.5) == torch.rsqrt(x) on {same} of {v.numel()} values, 1 / sqrt(x) == rsqrt(x) on "
          f"{inv} (torch {torch.__version__} CPU)")


class AttnSpy:
    """Wraps the module-level eager_attention_forward (looked up at call time, M:341-343) to record each call's
    query / key / output by layer."""

    def __init__(self):
        self.m = ref.mm()
        self.orig = self.m.eager_attention_forward
        self.calls: dict = {}

        def spy(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
            out, w = self.orig(module, query, key, value, attention_mask, scaling, dropout, **kw)
            self.calls.setdefault(module.layer_idx, []).append((query.detach(), key.detach(), out.detach()))
            return out, w
        self.m.eager_attention_forward = spy

    def close(self):
        self.m.eager_attention_forward = self.orig


def test_qk_scale_after_norm():
    """q = bf16(bf16(norm(q)) * 3.87) on the full (NoPE) layer, where nothing else touches q; scaling before the
    weight-less norm (a port that does) would leave no scale at all - it differs."""
    snap = variant("untied")
    port, _, tc = port_model(snap, BF)
    attn = port.model.language_model.layers[3].self_attn
    assert tc.layer_types[3] == "full_attention"
    got = {}
    h = attn.q_proj.register_forward_hook(lambda _m, _a, o: got.update(q=o.detach()))
    spy = AttnSpy()
    try:
        with torch.no_grad():
            port(input_ids=torch.tensor([rand_ids(40, seed=12)]), use_cache=False)
    finally:
        spy.close()
        h.remove()
    q = got["q"].view(1, 40, -1, tc.head_dim).transpose(1, 2)
    query = spy.calls[3][0][0]
    want = (_pow_norm(q) * 3.87)
    wrong = _pow_norm(q * 3.87)
    assert want.dtype == BF and torch.equal(query, want), "q is not bf16(bf16(norm(q)) * 3.87)"
    assert not torch.equal(query, wrong)
    print(f"  L3 query = bf16(bf16(pow-norm(q)) * {tc.qk_scale_factor}) bitwise; norm(q * 3.87) differs")


def test_nope_switch():
    """The full layer gets position_embeddings None (M:512); with layer 0 made full (`nope0`), a +100 position shift of
    the whole prompt leaves layer 0's attention output bitwise equal, while on `untied` (layer 0 sliding, RoPE) it moves."""
    ids = rand_ids(40, seed=13)
    pe_seen = {}
    port, _, tc = port_model(variant("untied"), BF)
    hooks = [l.self_attn.register_forward_pre_hook(
        lambda _m, _a, kw, i=i: pe_seen.__setitem__(i, kw.get("position_embeddings") is not None), with_kwargs=True)
        for i, l in enumerate(layers_of(port))]
    with torch.no_grad():
        port(input_ids=torch.tensor([ids]), use_cache=False)
    for h in hooks:
        h.remove()
    assert pe_seen == {0: True, 1: True, 2: True, 3: False}, pe_seen
    res = {}
    for v in ("nope0", "untied"):
        model = port_model(variant(v), BF)[0]
        outs = []
        for shift in (0, 100):
            got = {}
            h = layers_of(model)[0].self_attn.register_forward_hook(lambda _m, _a, o: got.update(a=o[0].detach()))
            with torch.no_grad():
                model(input_ids=torch.tensor([ids]), position_ids=torch.arange(40).unsqueeze(0) + shift, use_cache=False)
            h.remove()
            outs.append(got["a"])
        res[v] = torch.equal(outs[0], outs[1])
    assert res["nope0"] and not res["untied"], res
    print("  position_embeddings None exactly on the full layers; a +100 shift: NoPE layer 0 bitwise equal, RoPE layer 0 moves")


def test_gate():
    """a = bf16(attn * bf16(sigmoid(gate(xn)))) before o_proj (M:358) on the port; with gate_proj zeroed (transformers'
    own model) o_proj's input is exactly half the attention output (sigmoid(0) = 0.5)."""
    snap = variant("untied")
    port, _, tc = port_model(snap, BF)
    for model, zero in ((port, False), (hf_model(snap, BF), True)):
        attn = layers_of(model)[1].self_attn
        saved = None
        if zero:
            saved = attn.gate_proj.weight.data.clone()
            attn.gate_proj.weight.data.zero_()
        got = {}
        hk = [attn.gate_proj.register_forward_hook(lambda _m, _a, o: got.update(g=o.detach())),
              attn.o_proj.register_forward_pre_hook(lambda _m, a: got.update(a=a[0].detach()))]
        spy = AttnSpy()
        try:
            with torch.no_grad():
                model(input_ids=torch.tensor([rand_ids(40, seed=14)]), use_cache=False)
        finally:
            spy.close()
            for h in hk:
                h.remove()
            if saved is not None:
                attn.gate_proj.weight.data.copy_(saved)
        att = spy.calls[1][0][2].reshape(1, 40, -1)
        if zero:
            assert torch.equal(got["a"], att * 0.5), "zeroed gate: o_proj input != attention / 2"
        else:
            assert torch.equal(got["a"], att * torch.sigmoid(got["g"])), "a != bf16(attn * bf16(sigmoid(g)))"
    print("  o_proj input = bf16(attn * bf16(sigmoid(gate_proj(xn)))) on the port; gate zeroed: exactly attn / 2")


def _tie_pair():
    """Two bf16 raw logits r_lo < r_hi whose capped values are equal (the four roundings)."""
    m, c = 0.19611613513818404, 20.0
    x = torch.arange(100, 400, 0.5).to(BF).unique()
    y = ref.softcap(x, m, c)
    for i in range(len(x) - 1):
        if y[i] == y[i + 1] and x[i] != x[i + 1]:
            return x[i], x[i + 1]
    raise AssertionError("no bf16 pair caps to the same value in [100, 400)")


def test_softcap_ties():
    """Review Focus 3: the port's capped logits are softcap(lm_head's raw rows) with the four bf16 roundings
    (M:1178-1181), which differ from one fp32 rounding on some rows; a planted pair of raw logits that cap to the
    same value sets cap_tie and torch's argmax picks the LOWER id while the raw argmax is the higher one."""
    snap = variant("untied")
    port, _, tc = port_model(snap, BF)
    rec = ref.Recorder(port, tc, acts=False)
    with torch.no_grad():
        out = port(input_ids=torch.tensor([rand_ids(40, seed=15)]), use_cache=False)
    rec.remove()
    raw = rec.raw[0]
    assert torch.equal(out.logits[0], ref.softcap(raw, tc.output_multiplier, tc.final_logit_softcapping))
    one = (20.0 * torch.tanh(raw.float() * tc.output_multiplier / 20.0)).to(BF)
    n_diff = int((one != out.logits[0]).sum())
    assert n_diff > 0, "four roundings never differ from one on these rows"
    lo, hi = _tie_pair()
    row = (torch.randn(1, tc.vocab_size, generator=torch.Generator().manual_seed(0)) * 4).to(BF)
    a, b = 1000, 2000                                   # the lower id gets the SMALLER raw logit
    row[0, a], row[0, b] = lo, hi
    capped = ref.softcap(row, tc.output_multiplier, tc.final_logit_softcapping)
    assert capped[0, a] == capped[0, b]
    assert bool(ref.cap_tie(row, capped)[0]) and int(capped[0].argmax()) == a and int(row[0].argmax()) == b
    print(f"  logits = softcap(raw) bitwise; {n_diff} of {raw.numel()} elements differ from a one-rounding fp32 cap; "
          f"planted raw {float(lo)} < {float(hi)} cap to {float(capped[0, a])}: cap_tie set, argmax {a} (lower id), raw "
          f"argmax {b}")


def test_untied():
    """Review Focus 4: on `untied` lm_head is its own tensor and the logits differ from the same weights with the head
    tied (the downloaded tiny = untied minus lm_head); the port reads lm_head, not embed_tokens."""
    pu = port_model(variant("untied"), BF)[0]
    pt = port_model(variant("tiny"), BF)[0]
    assert pu.lm_head.weight.data_ptr() != pu.model.language_model.embed_tokens.weight.data_ptr()
    assert pt.lm_head.weight.data_ptr() == pt.model.language_model.embed_tokens.weight.data_ptr()
    ids = torch.tensor([rand_ids(40, seed=16)])
    with torch.no_grad():
        a, b = pu(input_ids=ids, use_cache=False).logits, pt(input_ids=ids, use_cache=False).logits
    assert not torch.equal(a, b)
    hf = hf_model(variant("untied"), BF)
    assert torch.equal(hf.lm_head.weight, pu.lm_head.weight) and not torch.equal(hf.lm_head.weight,
                                                                                 hf.model.language_model.embed_tokens.weight)
    print("  untied: lm_head separate (= transformers' loaded head), logits differ from the tied twin")


def test_taps():
    """Review Focus 5 on l8: the recorder's taps (the residual after layers 1 and 5) equal hidden_states[1] / [5] of
    transformers' own output_hidden_states=[1, 5] call, bitwise; [0, 2, 3, 4] are None there."""
    snap = variant("l8")
    hf = hf_model(snap, BF)
    port, _, tc = port_model(snap, BF)
    ids = torch.tensor([rand_ids(40, seed=17)])
    with torch.no_grad():
        hs = hf(input_ids=ids, use_cache=False, output_hidden_states=[1, 5]).hidden_states
        rec = ref.Recorder(port, tc, acts=False, taps=(1, 5))
        port(input_ids=ids, use_cache=False)
        rec.remove()
    taps = rec.tensors()["taps"]
    assert [i for i, x in enumerate(hs) if x is not None] == [1, 5]
    assert taps.shape == (40, 2, tc.hidden_size)
    assert torch.equal(taps[:, 0], hs[1][0]) and torch.equal(taps[:, 1], hs[5][0])
    print("  taps [1, 5] on l8 = transformers' hidden_states[1] / [5] bitwise")


def test_int4():
    """On `int4`: each linear dequantises to (q - 8) x scale (stream.dequant_t, g64), and the port's forward equals
    transformers on the dequantised twin bitwise; a qzeros word that is not 0x77777777 is refused."""
    snap = variant("int4")
    ck = ref.Ckpt(snap)
    b = f"{LM}layers.2.mlp.down_proj"
    qw, sc = ck.get(b + ".qweight"), ck.get(b + ".scales")
    q = torch.stack([(qw >> (4 * i)) & 0xF for i in range(8)], 1).reshape(qw.shape[0] * 8, qw.shape[1])   # [K][N]
    want = ((q.float() - 8) * sc.float().repeat_interleave(64, 0)).to(BF).t()
    assert torch.equal(ref.dequant(ck, b, set()), want)
    twin = variant("int4-twin")
    for dtype in (BF, F32):
        compare(hf_model(twin, dtype), port_model(snap, dtype)[0], rand_ids(40, seed=18), 4, what=f"int4 {dtype}")
    bad = os.path.join(TMP, "int4-bad")
    if not os.path.exists(bad):
        import shutil
        from safetensors.torch import load_file, save_file
        shutil.copytree(snap, bad)
        sd = load_file(os.path.join(bad, "model.safetensors"))
        sd[f"{LM}layers.0.self_attn.q_proj.qzeros"][0, 0] = 0x77777778
        save_file(sd, os.path.join(bad, "model.safetensors"), metadata={"format": "pt"})
    try:
        ref.dequant(ref.Ckpt(bad), f"{LM}layers.0.self_attn.q_proj", set())
        raise AssertionError("a bad qzeros word was accepted")
    except SystemExit:
        pass
    print("  int4: dequant = (q - 8) x scale; forward = transformers on the dequantised twin (bf16, fp32) bitwise; "
          "a non-0x77777777 qzeros refused")


def test_layers_truncation():
    """text_config(..., layers=4) on l8 equals transformers with num_hidden_layers=4 (layer_types / layer_rope_theta
    cut to 4) bitwise."""
    import shutil
    snap = variant("l8")
    d = os.path.join(TMP, "l8-cut4")
    if not os.path.exists(d):
        shutil.copytree(snap, d)
        cfg = json.load(open(os.path.join(d, "config.json")))
        t = cfg["text_config"]
        t["num_hidden_layers"], t["layer_types"], t["layer_rope_theta"] = 4, t["layer_types"][:4], t["layer_rope_theta"][:4]
        json.dump(cfg, open(os.path.join(d, "config.json"), "w"), indent=2)
    hf = hf_model(d, BF)
    port, _, tc = port_model(snap, BF, layers=4)
    assert tc.num_hidden_layers == 4 and len(tc.layer_rope_theta) == 4
    compare(hf, port, rand_ids(40, seed=19), 4, what="l8 --layers 4")
    print("  --layers 4 on l8 = transformers' 4-layer model bitwise")


def test_layer_major():
    """layer_major: two sequences batched one layer at a time (stream.py's hold mode), prompt then cached steps, equal
    each sequence's own sequential forward bitwise (raw and capped)."""
    snap = variant("win16")
    port, pf, tc = port_model(snap, BF)
    seqs = [rand_ids(37, seed=20), rand_ids(60, seed=21)]
    step = ref.layer_major(port, pf)
    sts = [{}, {}]
    got = step([(seqs[0], 0, sts[0]), (seqs[1], 0, sts[1])])
    nxt = [int(c[-1].argmax()) for _, c in got]
    got2 = step([([nxt[0]], 37, sts[0]), ([nxt[1]], 60, sts[1])])
    for j, ids in enumerate(seqs):
        rec = ref.Recorder(port, tc, acts=False)
        with torch.no_grad():
            out = port(input_ids=torch.tensor([ids]), use_cache=True)
            r1 = rec.raw[0].clone()
            out2 = port(input_ids=torch.tensor([[nxt[j]]]), past_key_values=out.past_key_values, use_cache=True)
        rec.remove()
        assert torch.equal(got[j][1], out.logits[0]) and torch.equal(got[j][0], r1), f"seq {j}: prompt rows"
        assert torch.equal(got2[j][1], out2.logits[0]) and torch.equal(got2[j][0], rec.raw[0]), f"seq {j}: step"
    print("  layer_major: 2 sequences (37 / 60 ids, window 16) + 1 cached step = each alone, bitwise")


def test_expected_names():
    """expected_names(tc, "gptq") = the real GPTQ index's 1875 text names and vision_names its other 809; "bf16" = the
    original's 627; on the variants: int4 is the gptq form, untied the bf16 form (small files only)."""
    for snap, form, n_text in ((gptq_snapshot(), "gptq", 1875), (original_snapshot(), "bf16", 627)):
        idx = json.load(open(os.path.join(snap, "model.safetensors.index.json")))
        tc = ref.text_config(snap)
        names = {k for k in idx["weight_map"] if ref.is_text(k)}
        assert set(ref.expected_names(tc, form)) == names and len(names) == n_text, (form, len(names))
        assert len(ref.vision_names(idx)) == 809 and len(names) + 809 == len(idx["weight_map"])
    for v, form in (("int4", "gptq"), ("untied", "bf16"), ("l8", "bf16"), ("gqa16", "bf16")):
        ck = ref.Ckpt(variant(v))
        tc = ref.text_config(variant(v))
        assert set(ref.expected_names(tc, form)) == {k for k in ck.keys() if ref.is_text(k)}, v
    print("  gptq: 1875 text + 809 vision; bf16: 627 text; the variants' names match their forms")


def test_assist_loads():
    """The `assist` variant is a valid MuseGlimmerAssistantModel (strict load) whose DFlash drafting over the l8 target
    is lossless under greedy: generate(assistant_model=..., speculation_type="dflash") = plain greedy. A 5.19.0 fact
    for 23f: without `speculation_type="dflash"` generate does not pick DFlashTokenCandidateGenerator
    (generation/utils.py:1343) and fails on the assistant's config, which has no `vocab_size` (utils.py:1882) - the
    REAL assistant's config.json has none either."""
    from transformers.models.muse_glimmer_assistant import modeling_muse_glimmer_assistant as am
    d = variant("assist")
    asst, info = am.MuseGlimmerAssistantModel.from_pretrained(d, dtype=BF, output_loading_info=True)
    assert not any(info.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys")), info
    target = hf_model(variant("l8"), BF)
    ids = torch.tensor([rand_ids(24, seed=22)])
    kw = dict(max_new_tokens=12, min_new_tokens=12, do_sample=False, top_k=None, top_p=None, temperature=None)
    with torch.no_grad():
        plain = target.generate(ids, **kw)
        try:
            target.generate(ids, assistant_model=asst.eval(), **kw)
            raise AssertionError("5.19.0 took the assistant without speculation_type='dflash': record the change")
        except AttributeError as e:
            assert "vocab_size" in str(e), e
        spec = target.generate(ids, assistant_model=asst, speculation_type="dflash", **kw)
    assert torch.equal(plain, spec), (plain, spec)
    print("  assist: loads strictly; DFlash greedy over l8 (speculation_type='dflash') = plain greedy (12 tokens); "
          "without speculation_type 5.19.0 raises on the missing vocab_size")


def test_cli_run_and_hfcheck():
    """`run` writes the golden layout (raw / capped logits, cap_tie, gap.raw, taps, tokens / tokens.raw, nll);
    `hfcheck --layers 4` on l8 (streamed = resident = layer-major) is BITWISE at 2100 ids; `facts` matches the names."""
    snap = variant("l8")
    d = os.path.join(TMP, "cli")
    os.makedirs(d, exist_ok=True)
    ids = rand_ids(300, seed=23)
    open(os.path.join(d, "p.ids"), "w").write(" ".join(map(str, ids)) + "\n")
    py = [sys.executable, os.path.join(_HERE, "muse_ref.py")]
    out = os.path.join(d, "p.golden.safetensors")
    r = subprocess.run(py + ["run", snap, "--prompt", os.path.join(d, "p.ids"), "--out", out, "--gen", "4",
                             "--act-tail", "16", "--logits-tail", "8", "--taps", "--chunk", "128"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:] + r.stdout[-2000:]
    from safetensors import safe_open
    with safe_open(out, "pt") as h:
        meta = h.metadata()
        t = {k: h.get_tensor(k) for k in h.keys()}
    T, G, tc = 300, 4, ref.text_config(snap)
    assert t["logits.raw"].shape == (8 + G, tc.vocab_size) and t["logits.cap"].shape == (8 + G, tc.vocab_size)
    assert meta["logits_row0"] == str(T - 8) and meta["act_row0"] == str(T - 16) and json.loads(meta["tap_layers"]) == [1]
    assert t["H.L7"].shape == (16 + G, tc.hidden_size) and t["H.L7"].dtype == BF
    assert t["taps"].shape == (T + G, 1, tc.hidden_size) and t["cap_tie"].shape == (T + G,)
    assert t["gap.raw"].shape == (T + G,) and t["nll"].shape == (T - 1,) and t["tokens"].shape == (G,)
    assert torch.equal(t["logits.cap"].to(BF), ref.softcap(t["logits.raw"].to(BF), tc.output_multiplier,
                                                            tc.final_logit_softcapping))
    assert t["tokens"][0] == int(t["logits.cap"][7].argmax())
    assert "cap_tie:" in r.stdout and "raw top-1 - top-2 gap" in r.stdout, r.stdout[-1500:]
    r = subprocess.run(py + ["hfcheck", snap, "--layers", "4", "--n", "2100", "--gen", "2"], capture_output=True,
                       text=True)
    assert r.returncode == 0 and "BITWISE" in r.stdout, r.stdout[-1500:] + r.stderr[-1500:]
    r = subprocess.run(py + ["facts", gptq_snapshot()], capture_output=True, text=True)
    assert r.returncode == 0 and "== expected_names(tc, 'gptq')" in r.stdout, r.stdout[-1500:] + r.stderr[-1500:]
    print("  run: the golden layout (chunks of 128); hfcheck --layers 4 at 2100 ids: BITWISE; facts: the gptq names")


TESTS = [test_streamed_equals_hf_bf16, test_streamed_equals_hf_fp32, test_chunked_prefill_equals_hf, test_norm_forms,
         test_qk_scale_after_norm, test_nope_switch, test_gate, test_softcap_ties, test_untied, test_taps, test_int4,
         test_layers_truncation, test_layer_major, test_expected_names, test_assist_loads, test_cli_run_and_hfcheck]


def main() -> None:
    import traceback
    torch.manual_seed(0)
    only = sys.argv[1:]
    skipped, failed = 0, []
    for t in TESTS:
        if only and t.__name__ not in only:
            continue
        print(t.__name__, flush=True)
        try:
            t()
        except Skip as s:
            skipped += 1
            print(f"SKIP {t.__name__}: {s}")
        except Exception:      # noqa: BLE001 - report every test, then fail
            traceback.print_exc()
            failed.append(t.__name__)
            print(f"FAIL {t.__name__}", flush=True)
    if failed:
        print(f"{len(failed)} FAILED: {failed}")
        sys.exit(1)
    print(f"ok ({skipped} skipped)")


if __name__ == "__main__":
    main()
