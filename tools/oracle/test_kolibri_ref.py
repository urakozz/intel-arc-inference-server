#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""KL0 checks for kolibri_ref.py and the transformers port (spec 20a). CPU, tiny random weights.

    test_kolibri_ref.py [test_name ...]     (pytest also collects it, where installed)

Each semantic trap of spec 20 §1 has its own test - sandwich norms, NoPE full layers vs RoPE sliding
layers, the 513-key window edge (the query counts), router selection on logit + bias with
sigmoid(logit) weights unrenormalised, ties to the lower id, the ungated shared expert, q/k RMSNorm
per head, plain-w RMSNorm, the ascending-id fp32 combine, the sliding ring cache - then the port
(third_party/kolibri1/modeling_kolibri1.py) is held against the reference on the same weights: bf16
eager BITWISE (prompt forward and cached decode, step by step, across the window), fp32 within
1e-5 relative, greedy generate() token for token, sdpa close to eager; then the checkpoint readers
(bf16 shards, an int4 GPTQ export, stream.py's layer streaming), run_batch against forward, the CLI.
`test_real_index` reads a real model.safetensors.index.json when KOLIBRI_INDEX names its
directory (with config.json), else SKIPs.
"""
import json
import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

import importlib.util  # noqa: E402
import math  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

_spec = importlib.util.spec_from_file_location("kolibri_ref", os.path.join(_HERE, "kolibri_ref.py"))
ref = importlib.util.module_from_spec(_spec)
sys.modules["kolibri_ref"] = ref          # dataclasses resolve annotations through sys.modules
_spec.loader.exec_module(ref)

REAL_CONFIG = os.path.join(ref.THIRD_PARTY, "config.json")
S, FULL = "sliding_attention", "full_attention"


class Skip(Exception):
    pass


def _skip(msg):
    if "pytest" in sys.modules:
        import pytest
        pytest.skip(msg)
    raise Skip(msg)


# --------------------------------------------------------------------------------------------
# tiny configs and weights, built from the real config.json's structure

def tiny_dict(**over) -> dict:
    with open(REAL_CONFIG, encoding="utf-8") as f:
        d = json.load(f)
    d.update(hidden_size=64, num_hidden_layers=5, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
             num_experts=12, num_experts_per_tok=6, moe_intermediate_size=32, shared_expert_intermediate_size=24,
             vocab_size=256, max_position_embeddings=4096, sliding_window=5,
             layer_types=[S, S, FULL, S, FULL], eos_token_id=255, pad_token_id=254)
    d.update(over)
    return d


def shape_of(c, name: str):
    D, H, Hkv, d = c.hidden_size, c.num_attention_heads, c.num_key_value_heads, c.head_dim
    if name in ("lm_head.weight", "model.embed_tokens.weight"):
        return c.vocab_size, D
    tail = name.split(".", 3)[-1]
    if name.endswith(("layernorm.weight", "post_attn_norm.weight", "post_ffn_norm.weight")) or name == "model.norm.weight":
        return (D,)
    if name.endswith(("q_norm.weight", "k_norm.weight")):
        return (d,)
    if name.endswith("expert_bias"):
        return (c.num_experts,)
    if tail == "mlp.gate.weight":
        return c.num_experts, D
    proj = {"self_attn.q_proj.weight": (H * d, D), "self_attn.k_proj.weight": (Hkv * d, D),
            "self_attn.v_proj.weight": (Hkv * d, D), "self_attn.o_proj.weight": (D, H * d)}
    if tail in proj:
        return proj[tail]
    width = c.shared_expert_intermediate_size if "shared_experts" in name else c.moe_intermediate_size
    return (D, width) if name.endswith("down_proj.weight") else (width, D)


def random_sd(c, seed: int = 0, router_scale: float = 3.0, bias_scale: float = 0.5) -> dict:
    g = torch.Generator().manual_seed(seed)
    sd = {}
    for name in ref.expected_names(c):
        shp = shape_of(c, name)
        if len(shp) == 1 and name.endswith("expert_bias"):
            t = bias_scale * torch.randn(shp, generator=g)
        elif len(shp) == 1:
            t = 1.0 + 0.5 * torch.randn(shp, generator=g)
        elif name == "model.embed_tokens.weight":
            t = torch.randn(shp, generator=g)
        else:
            s = router_scale if name.endswith("mlp.gate.weight") else 1.0
            t = torch.randn(shp, generator=g) * (s / math.sqrt(shp[1]))
        sd[name] = t.to(torch.bfloat16)
    return sd


def tiny(seed=0, **over):
    d = tiny_dict(**over)
    c = ref.KConfig.from_dict(d)
    return d, c, random_sd(c, seed)


def hf_model(d: dict, sd: dict, dtype, attn="eager"):
    model, _ = ref.build_hf(d, device="cpu", dtype=dtype, attn=attn)
    model.load_state_dict({k: v.to(dtype) for k, v in sd.items()}, strict=True)
    return model


def one_layer(c, sd, i, mode="f32"):
    r = ref.KolibriRef(c, ref.DictSource(sd), mode=mode, prefetch=False)
    return r, r.layer(i)


# --------------------------------------------------------------------------------------------
# the real config

def test_real_config():
    c = ref.KConfig.from_file(REAL_CONFIG)
    full = [i for i in range(c.num_hidden_layers) if not c.is_sliding(i)]
    assert full == list(range(4, 50, 5)), full                    # 4:1, the 5th of every five is full
    assert (c.hidden_size, c.rms_norm_eps, c.num_hidden_layers) == (2560, 1e-6, 50)
    assert (c.num_attention_heads, c.num_key_value_heads, c.head_dim, c.n_rep) == (48, 4, 128, 12)
    assert (c.sliding_window, c.rope_theta, c.head_dtype) == (513, 1e4, "float32")
    assert (c.num_experts, c.num_experts_per_tok, c.moe_intermediate_size, c.shared_expert_intermediate_size) == \
        (384, 6, 512, 512)
    assert (c.vocab_size, c.bos_token_id, c.eos_token_ids, c.tie_word_embeddings) == (128000, None, [127906], False)
    names = ref.expected_names(c)
    assert len(names) == 58353 == len(set(names)), len(names)       # the bf16 index's entry count
    cfg = ref.hf_config(json.load(open(REAL_CONFIG, encoding="utf-8")))
    assert cfg.layer_types == c.layer_types and cfg.sliding_window == 513 and cfg.rope_parameters["rope_theta"] == 1e4
    print(f"  real config: full (NoPE) layers {full}; {len(names)} tensor names (= the bf16 index)")


def test_real_index():
    dname = os.environ.get("KOLIBRI_INDEX")
    if not dname:
        _skip("KOLIBRI_INDEX not set")
    c = ref.KConfig.from_file(dname)
    with open(os.path.join(dname, "model.safetensors.index.json"), encoding="utf-8") as f:
        wm = json.load(f)["weight_map"]
    have = {n[:-len(".qweight")] + ".weight" if n.endswith(".qweight") else n for n in wm
            if not n.endswith((".scales", ".qzeros", ".g_idx"))}
    want = set(ref.expected_names(c))
    assert have == want, (sorted(want - have)[:5], sorted(have - want)[:5])
    print(f"  {dname}: {len(wm)} entries, every expected name present")


def test_port_names_and_modules():
    """The port's state dict IS the checkpoint's names; the experts are nn.Linear (AutoRound quantises
    those) and the router is not."""
    d, c, sd = tiny()
    model = hf_model(d, sd, torch.float32)
    assert set(model.state_dict()) == set(ref.expected_names(c))
    mods = dict(model.named_modules())
    assert isinstance(mods["model.layers.0.mlp.experts.11.down_proj"], torch.nn.Linear)
    assert isinstance(mods["model.layers.0.mlp.shared_experts.gate_proj"], torch.nn.Linear)
    assert not isinstance(mods["model.layers.0.mlp.gate"], torch.nn.Linear)
    linears = sorted({n.split(".", 3)[-1].replace("experts.", "experts.E.", 1).split(".E.")[0]
                      for n, m in mods.items() if isinstance(m, torch.nn.Linear) and n.startswith("model.layers.0.")})
    print(f"  {len(model.state_dict())} names = expected; layer 0 nn.Linear groups {linears}")


# --------------------------------------------------------------------------------------------
# semantic traps

def test_norm_plain_weight():
    ops = ref.Ops("f32")
    x = torch.randn(3, 64)
    w = torch.randn(64)
    got = ref.rms_norm(ops, x, w, 1e-6)
    want = w * (x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6))
    assert torch.allclose(got, want, rtol=1e-6, atol=1e-6)
    assert not torch.allclose(got, (1 + w) * want / w, atol=1e-3)
    # bf16: x_hat rounded once, then the bf16 product (vLLM's x.to(w.dtype) * w)
    xb, wb = x.bfloat16(), w.bfloat16()
    gb = ref.rms_norm(ref.Ops("bf16"), xb, wb, 1e-6)
    h = xb.float()
    assert torch.equal(gb, wb * (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)).bfloat16())
    print("  w * x_hat, plain w; bf16 rounds x_hat once then multiplies in bf16")


def test_sandwich_norms():
    """post_attn_norm and post_ffn_norm scale what each sublayer ADDS: zero weights remove the
    sublayer, doubled weights double its contribution (f32), the residual stream is untouched."""
    d, c, sd = tiny(seed=1)
    r, lw = one_layer(c, sd, 0)
    h = torch.randn(1, 7, c.hidden_size)
    pos = torch.arange(7)
    cos, sin = ref.rope_cos_sin(r.ops, pos, c.head_dim, c.rope_theta)

    def run(**w):
        lw2 = ref.LayerWeights(lw.src, c, 0, {**lw.d, **w})
        lw2.cache = lw.cache
        return r.decoder_layer(0, lw2, h, pos, cos, sin, None)

    base = run()
    zero_both = run(post_attn_norm=torch.zeros(c.hidden_size), post_ffn_norm=torch.zeros(c.hidden_size))
    assert torch.equal(zero_both, h)
    z_ffn = run(post_ffn_norm=torch.zeros(c.hidden_size))            # h + post_attn_norm(attn)
    z_attn = run(post_attn_norm=torch.zeros(c.hidden_size))           # h + post_ffn_norm(moe(norm(h)))
    a = z_ffn - h
    two = run(post_attn_norm=2 * lw["post_attn_norm"].float(), post_ffn_norm=torch.zeros(c.hidden_size))
    assert torch.allclose(two - h, 2 * a, rtol=1e-5, atol=1e-5)
    m2 = run(post_attn_norm=torch.zeros(c.hidden_size), post_ffn_norm=2 * lw["post_ffn_norm"].float())
    assert torch.allclose(m2 - h, 2 * (z_attn - h), rtol=1e-5, atol=1e-5)
    # without the sandwich, MoE would see norm(h + attn) and add moe(...) unscaled: not what we compute
    assert not torch.allclose(base, z_ffn + (z_attn - h), atol=1e-4)
    print("  zero post norms -> identity; scaling them scales each sublayer's contribution")


def _attn_out(r, lw, i, x, pos):
    cos, sin = ref.rope_cos_sin(r.ops, pos, r.c.head_dim, r.c.rope_theta)
    return r.attn(i, lw, x, pos, cos, sin, None)


def test_nope_full_rope_sliding():
    """Full layers have no positional encoding: permuting the earlier tokens leaves the last query's
    output unchanged (attention is a set operation). Sliding layers rotate q/k: it changes. And a full
    layer's output does not depend on the absolute position offset at all, a sliding one's (RoPE is
    relative) only through rounding."""
    d, c, sd = tiny(seed=2, sliding_window=64)
    T = 9
    x = torch.randn(1, T, c.hidden_size)
    perm = torch.cat([torch.randperm(T - 1), torch.tensor([T - 1])])
    pos = torch.arange(T)
    for i, kind in ((2, "full"), (0, "sliding")):
        r, lw = one_layer(c, sd, i)
        o = _attn_out(r, lw, i, x, pos)[0, -1]
        op = _attn_out(r, lw, i, x[:, perm], pos)[0, -1]
        if kind == "full":
            assert torch.allclose(o, op, rtol=1e-5, atol=1e-6), float((o - op).abs().max())
            o_shift = _attn_out(r, lw, i, x, pos + 1000)[0, -1]
            assert torch.equal(o, o_shift)
        else:
            assert not torch.allclose(o, op, atol=1e-3), "sliding layer ignored position"
    print("  full layer: order- and offset-free (NoPE); sliding layer: RoPE")


def test_window_513_edge():
    """Window 513 counts the query: query i sees keys i-512 .. i, not i-513 (model card: "512
    preceding tokens plus the current"; vLLM window (W - 1, 0); transformers' sliding cache keeps
    W - 1). Checked on the real window by perturbing one key's input."""
    d, c, sd = tiny(seed=3, sliding_window=513, num_hidden_layers=2, layer_types=[S, FULL])
    T = 520
    r, lw = one_layer(c, sd, 0)
    pos = torch.arange(T)
    x = torch.randn(1, T, c.hidden_size)
    q = T - 1
    base = _attn_out(r, lw, 0, x, pos)[0, q]
    for p, seen in ((q - 513, False), (q - 512, True), (q, True)):
        x2 = x.clone()
        x2[0, p] += 3.0
        o = _attn_out(r, lw, 0, x2, pos)[0, q]
        assert (not torch.equal(o, base)) == seen, (p, seen)
    vis = ref.visible(torch.tensor([q]), torch.arange(T), 513)[0]
    assert int(vis.sum()) == 513 and bool(vis[q - 512]) and not bool(vis[q - 513])
    print("  key i-512 visible, i-513 not: 513 keys including the query")


def test_router_logit_plus_bias_sigmoid_weights():
    """Selection on logits + bias (not sigmoid(logits) + bias, not logits alone); the weights are
    sigmoid(logits) of the selected, NOT renormalised (plugin test_routing_semantics, 384 experts)."""
    torch.manual_seed(0)
    N, E, k = 64, 384, 6
    logits = torch.randn(N, E) * 3
    bias = torch.randn(E) * 5
    rt = ref.route(logits, bias, k)
    want = torch.topk(logits + bias, k, dim=-1, sorted=True).indices
    assert torch.equal(rt.sel_order, want)
    assert torch.equal(rt.ids, torch.sort(want, dim=-1).values)
    assert torch.equal(rt.weights, torch.sigmoid(logits.gather(1, rt.ids)))
    assert not torch.allclose(rt.weights.sum(-1), torch.ones(N))           # not renormalised
    alt = torch.topk(torch.sigmoid(logits) + bias, k, dim=-1).indices
    assert not torch.equal(torch.sort(alt, -1).values, rt.ids), "vacuous: score-add selects the same"
    assert not torch.equal(torch.sort(torch.topk(logits, k).indices, -1).values, rt.ids), "vacuous: bias inert"
    assert bool((rt.gap > 0).all())
    print("  ids = top-6(logit + bias); w = sigmoid(logit), sums", f"{float(rt.weights.sum(-1).min()):.2f}..",
          f"{float(rt.weights.sum(-1).max()):.2f}")


def test_router_ties_lower_id():
    logits = torch.zeros(3, 10)
    bias = torch.zeros(10)
    bias[[7, 8, 9]] = 1.0
    rt = ref.route(logits, bias, 6)
    assert rt.ids.tolist() == [[0, 1, 2, 7, 8, 9]] * 3, rt.ids
    assert float(rt.gap[0]) == 0.0                                          # a tie at the cut
    # the port selects the same way
    m = ref.hf_module()
    ids, w = m.route(logits, bias, 6)
    assert torch.equal(torch.sort(ids, -1).values, rt.ids) and torch.equal(w, torch.full((3, 6), 0.5))
    print("  equal scores -> the lower ids; gap 0 marks the tie; the port agrees")


def test_shared_expert_ungated():
    """MoE(x) = routed + shared(x): zero every routed down_proj -> exactly the shared expert (no
    sigmoid gate on it); zero the shared down_proj -> exactly the routed sum."""
    d, c, sd = tiny(seed=4)
    sd0 = dict(sd)
    for e in range(c.num_experts):
        sd0[f"model.layers.1.mlp.experts.{e}.down_proj.weight"] = torch.zeros_like(
            sd[f"model.layers.1.mlp.experts.{e}.down_proj.weight"])
    x = torch.randn(10, c.hidden_size)
    r, lw = one_layer(c, sd0, 1)
    y, _ = r.ffn(1, lw, x)
    shared = ref.mlp(r.ops, x, lw["shared.gate"], lw["shared.up"], lw["shared.down"])
    assert torch.equal(y, shared)
    sd1 = dict(sd)
    sd1["model.layers.1.mlp.shared_experts.down_proj.weight"] = torch.zeros_like(
        sd["model.layers.1.mlp.shared_experts.down_proj.weight"])
    r, lw = one_layer(c, sd1, 1)
    y1, rt = r.ffn(1, lw, x)
    routed = torch.zeros_like(x)
    for n in range(x.shape[0]):
        for j, e in enumerate(rt.ids[n].tolist()):
            routed[n] += rt.weights[n, j] * ref.mlp(r.ops, x[n:n + 1], *lw.expert(e))[0]
    assert torch.allclose(y1, routed, rtol=1e-6, atol=1e-6)
    print("  shared expert added as is; routed sum = sum w_e y_e")


def test_qk_norm_per_head():
    """q and k are RMS-normalised per head over head_dim: scaling one head's q or k projection rows
    changes nothing (up to rounding); scaling a v head does."""
    d, c, sd = tiny(seed=5)
    T = 6
    x = torch.randn(1, T, c.hidden_size)
    pos = torch.arange(T)
    r, lw = one_layer(c, sd, 0)
    base = _attn_out(r, lw, 0, x, pos)
    dh = c.head_dim
    for name, head, changes in (("q_proj", 3, False), ("k_proj", 1, False), ("v_proj", 1, True)):
        w = lw[name].float().clone()
        w[head * dh:(head + 1) * dh] *= 7.0
        lw2 = ref.LayerWeights(lw.src, c, 0, {**lw.d, name: w})
        o = _attn_out(r, lw2, 0, x, pos)
        assert torch.allclose(o, base, rtol=1e-5, atol=1e-5) != changes, name
    print("  q/k heads normalised independently; v not")


def test_combine_ascending_fp32():
    """The routed sum is accumulated in fp32 in ascending expert id, then + shared, one rounding."""
    d, c, sd = tiny(seed=6)
    r, lw = one_layer(c, sd, 3, mode="bf16")
    x = torch.randn(40, c.hidden_size).bfloat16()
    y, rt = r.ffn(3, lw, x)
    shared = ref.mlp(r.ops, x, lw["shared.gate"], lw["shared.up"], lw["shared.down"])
    out = {}                                   # (row, expert) -> bf16 y_e, computed on the same row sets
    for e in torch.unique(rt.ids).tolist():
        rows = torch.where((rt.ids == e).any(-1))[0]
        ye = ref.mlp(r.ops, x[rows], *lw.expert(e))
        out.update({(int(n), e): ye[j] for j, n in enumerate(rows)})
    want = []
    for n in range(x.shape[0]):
        a = torch.zeros(c.hidden_size)
        for j, e in sorted(enumerate(rt.ids[n].tolist()), key=lambda z: z[1]):
            a = a + rt.weights[n, j] * out[(n, e)].float()
        want.append((a + shared[n].float()).bfloat16())
    assert torch.equal(y, torch.stack(want))
    print("  bitwise = sum in ascending id (fp32) + shared, one bf16 rounding")


def test_ring_cache_and_decode():
    """The sliding layers' cache holds W - 1 positions; a cached decode equals the full forward."""
    d, c, sd = tiny(seed=7)
    ids = torch.randint(0, 250, (14,)).tolist()
    for mode, tol in (("f32", 1e-5), ("bf16", 0.05)):
        r = ref.KolibriRef(c, ref.DictSource(sd), mode=mode, prefetch=False)
        full = r.forward(ids, 0, r.new_cache())
        cache = r.new_cache()
        rows = [r.forward(ids[:6], 0, cache)] + [r.forward([t], 6 + j, cache) for j, t in enumerate(ids[6:])]
        inc = torch.cat(rows)
        assert torch.allclose(inc, full, rtol=tol, atol=tol), (mode, float((inc - full).abs().max()))
        for i in range(c.num_hidden_layers):
            want = c.sliding_window - 1 if c.is_sliding(i) else len(ids)
            assert cache[i][0].shape[2] == want and cache[i][2].tolist() == list(range(len(ids) - want, len(ids)))
    print("  ring: W - 1 = 4 kept in sliding layers, all 14 in full ones; decode == prefill")


def test_determinism():
    d, c, sd = tiny(seed=8)
    ids = torch.randint(0, 250, (9,)).tolist()
    a = ref.KolibriRef(c, ref.DictSource(sd), prefetch=False).generate(ids, 3)
    b = ref.KolibriRef(c, ref.DictSource(sd), prefetch=False).generate(ids, 3)
    assert torch.equal(a[0], b[0]) and a[1] == b[1]


# --------------------------------------------------------------------------------------------
# the port against the reference

def _port_steps(model, ids, split):
    with torch.no_grad():
        out = model(torch.tensor([ids[:split]]), use_cache=True)
        cache, rows = out.past_key_values, [out.logits[0].float()]
        for t in ids[split:]:
            out = model(torch.tensor([[t]]), past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            rows.append(out.logits[0].float())
    return torch.cat(rows)


def _ref_steps(r, ids, split):
    cache = r.new_cache()
    return torch.cat([r.forward(ids[:split], 0, cache)] + [r.forward([t], split + j, cache)
                                                            for j, t in enumerate(ids[split:])])


def test_hf_bf16():
    """bf16, eager: the port == the reference bit for bit - full forward and cached decode across
    the window (window 5, 16 positions), also on the real window 513 with 530 positions."""
    for over, n, split in (({}, 16, 7), ({"sliding_window": 513, "num_hidden_layers": 2,
                                          "layer_types": [S, FULL]}, 530, 520)):
        d, c, sd = tiny(seed=9, **over)
        ids = torch.randint(0, 250, (n,), generator=torch.Generator().manual_seed(1)).tolist()
        model = hf_model(d, sd, torch.bfloat16)
        r = ref.KolibriRef(c, ref.DictSource(sd), mode="bf16", prefetch=False)
        with torch.no_grad():
            hf_full = model(torch.tensor([ids]), use_cache=False).logits[0].float()
        assert torch.equal(hf_full, r.forward(ids, 0, r.new_cache())), "full forward"
        assert torch.equal(_port_steps(model, ids, split), _ref_steps(r, ids, split)), "cached decode"
    print("  bitwise: prompt forward and cached decode (window 5 x 16; window 513 x 530)")


def test_hf_f32():
    d, c, sd = tiny(seed=10)
    ids = torch.randint(0, 250, (16,)).tolist()
    model = hf_model(d, sd, torch.float32)
    r = ref.KolibriRef(c, ref.DictSource(sd), mode="f32", prefetch=False)
    with torch.no_grad():
        hf = model(torch.tensor([ids]), use_cache=False).logits[0]
    ours = r.forward(ids, 0, r.new_cache())
    err = float(((hf - ours).abs() / (ours.abs() + 1)).max())
    assert err < 1e-5, err
    print(f"  fp32: max rel diff {err:.2e} (bitwise: {torch.equal(hf, ours)})")


def test_hf_generate_and_sdpa():
    d, c, sd = tiny(seed=11)
    ids = torch.randint(0, 250, (8,)).tolist()
    model = hf_model(d, sd, torch.bfloat16)
    r = ref.KolibriRef(c, ref.DictSource(sd), mode="bf16", prefetch=False)
    with torch.no_grad():
        g = model.generate(torch.tensor([ids]), max_new_tokens=10, do_sample=False, eos_token_id=None,
                           pad_token_id=254)[0, len(ids):].tolist()
    assert g == r.generate(ids, 10)[1], g
    m32 = hf_model(d, sd, torch.float32)
    ms = hf_model(d, sd, torch.float32, attn="sdpa")
    with torch.no_grad():
        a = m32(torch.tensor([ids]), use_cache=False).logits
        b = ms(torch.tensor([ids]), use_cache=False).logits
    assert torch.allclose(a, b, rtol=1e-4, atol=1e-4), float((a - b).abs().max())
    # batched generation with left padding: the padded row decodes as it would alone
    am = torch.tensor([[1] * 8, [0] * 3 + [1] * 5])
    batch = torch.tensor([ids, [254] * 3 + ids[:5]])
    with torch.no_grad():
        gb = model.generate(batch, attention_mask=am, max_new_tokens=4, do_sample=False, eos_token_id=None,
                            pad_token_id=254)
        g1 = model.generate(torch.tensor([ids[:5]]), max_new_tokens=4, do_sample=False, eos_token_id=None,
                            pad_token_id=254)
    assert gb[1, 8:].tolist() == g1[0, 5:].tolist()
    print(f"  generate() == reference greedy ({g[:5]}...); sdpa ~ eager; left padding ok")


# --------------------------------------------------------------------------------------------
# checkpoints, streaming, batch, CLI

def write_checkpoint(tmp: str, d: dict, sd: dict) -> None:
    from safetensors.torch import save_file
    names = sorted(sd)
    half = len(names) // 2
    wm = {}
    for k, part in enumerate((names[:half], names[half:])):
        fn = f"model-0000{k + 1}-of-00002.safetensors"
        save_file({n: sd[n].contiguous() for n in part}, os.path.join(tmp, fn))
        wm.update({n: fn for n in part})
    with open(os.path.join(tmp, "model.safetensors.index.json"), "w", encoding="utf-8") as f:
        json.dump({"metadata": {}, "weight_map": wm}, f)
    with open(os.path.join(tmp, "config.json"), "w", encoding="utf-8") as f:
        json.dump(d, f)


def gptq_pack(w: torch.Tensor, g: int = 64):
    """[N, K] float -> (qweight [K/8, N] i32, scales [K/g, N] f16, qzeros, w_deq [N, K] bf16), symmetric."""
    N, K = w.shape
    wt = w.float().t()                                          # [K, N]
    s = (wt.reshape(K // g, g, N).abs().amax(1) / 7.0).clamp(min=1e-4).to(torch.float16)
    q = torch.clamp(torch.round(wt / s.float().repeat_interleave(g, 0)) + 8, 0, 15).to(torch.int64)
    q = q.reshape(K // 8, 8, N)
    qw = torch.zeros(K // 8, N, dtype=torch.int64)
    for i in range(8):
        qw |= q[:, i, :] << (4 * i)
    qw = torch.where(qw >= 2**31, qw - 2**32, qw).to(torch.int32)
    qz = torch.full((K // g, N // 8), 0x77777777, dtype=torch.int64).to(torch.int32)
    deq = ref._dequant.dequant_gptq(qw, s, g).t().contiguous()
    return qw, s, qz, deq


def test_checkpoint_and_stream():
    d, c, sd = tiny(seed=12)
    ids = torch.randint(0, 250, (10,)).tolist()
    want = ref.KolibriRef(c, ref.DictSource(sd), prefetch=False).forward(ids, 0, [None] * c.num_hidden_layers)
    with tempfile.TemporaryDirectory() as tmp:
        write_checkpoint(tmp, d, sd)
        src = ref.Checkpoint(tmp)
        got = ref.KolibriRef(src.cfg, src).forward(ids, 0, [None] * c.num_hidden_layers)
        assert torch.equal(got, want)
        model, pf = ref.hf_streamed(src, d)
        with torch.no_grad():
            hf = model(torch.tensor([ids]), use_cache=False).logits[0].float()
        assert torch.equal(hf, want)
    print("  sharded bf16 checkpoint == in-memory; the streamed port == the reference")


def test_gptq_checkpoint():
    """An int4 GPTQ export (experts packed, the rest bf16): read by dequant.py's rule."""
    d, c, sd = tiny(seed=13, hidden_size=64, moe_intermediate_size=64)
    q, deq = {}, dict(sd)
    for name in list(sd):
        if ".mlp.experts." in name:
            base = name[:-len(".weight")]
            qw, s, qz, w = gptq_pack(sd[name])
            q[base + ".qweight"], q[base + ".scales"], q[base + ".qzeros"] = qw, s, qz
            q[base + ".g_idx"] = (torch.arange(sd[name].shape[1]) // 64).to(torch.int32)
            deq[name] = w
        else:
            q[name] = sd[name]
    d2 = dict(d, quantization_config={"bits": 4, "group_size": 64, "sym": True, "quant_method": "auto-round",
                                      "packing_format": "auto_round:auto_gptq"})
    ids = torch.randint(0, 250, (8,)).tolist()
    want = ref.KolibriRef(c, ref.DictSource(deq), prefetch=False).forward(ids, 0, [None] * c.num_hidden_layers)
    with tempfile.TemporaryDirectory() as tmp:
        write_checkpoint(tmp, d2, q)
        src = ref.Checkpoint(tmp, fast=False)
        assert src.cfg.group_size == 64
        got = ref.KolibriRef(src.cfg, src).forward(ids, 0, [None] * c.num_hidden_layers)
    assert torch.equal(got, want)
    print("  int4 g64 experts dequantised by the (q - 8) * scale rule == the bf16 dequantised weights")


def test_run_batch():
    """Layer-major over several sequences == per-sequence forward (logits within bf16 noise, routes
    identical where the selection gap is not tiny); on_route sees every token of every row."""
    d, c, sd = tiny(seed=14)
    seqs = torch.randint(0, 250, (3, 12))
    for mode in ("f32", "bf16"):
        r = ref.KolibriRef(c, ref.DictSource(sd), mode=mode, prefetch=False)
        seen = {}

        def on_route(layer, seq_ix, rt):
            seen.setdefault(layer, []).append((seq_ix.clone(), rt.ids.clone()))
        h = r.run_batch(seqs, rows=2, on_route=on_route)
        for n in range(3):
            want = r.forward(seqs[n].tolist(), 0, r.new_cache())
            got = r.head(h[n])
            if mode == "f32":
                assert torch.allclose(got, want, rtol=1e-4, atol=1e-4), float((got - want).abs().max())
            else:
                assert float(F.cosine_similarity(got, want, dim=-1).min()) > 0.99
        assert sorted(seen) == list(range(c.num_hidden_layers))
        assert sum(s.numel() for s, _ in seen[0]) == 36
    print("  run_batch (rows 2) == forward per sequence; routes reported per layer")


def test_cli_run_and_hfcheck():
    from safetensors import safe_open
    d, c, sd = tiny(seed=15)
    with tempfile.TemporaryDirectory() as tmp:
        write_checkpoint(tmp, d, sd)
        with open(os.path.join(tmp, "p.ids"), "w", encoding="utf-8") as f:
            f.write("0 5 9 17 33 65 2 3\n")
        out = os.path.join(tmp, "out", "p.golden.safetensors")
        argv = sys.argv
        try:
            sys.argv = ["kolibri_ref.py", "run", tmp, "--prompt", os.path.join(tmp, "p.ids"), "--out", out, "--gen", "4"]
            ref.main()
            sys.argv = ["kolibri_ref.py", "hfcheck", tmp, "--prompt", os.path.join(tmp, "p.ids"), "--against", out]
            ref.main()
            sys.argv = ["kolibri_ref.py", "ppl", tmp, "--ids", os.path.join(tmp, "p.ids")]
            ref.main()
            sys.argv = ["kolibri_ref.py", "facts", tmp]
            ref.main()
        finally:
            sys.argv = argv
        with safe_open(out, framework="pt") as h:
            keys, meta = set(h.keys()), h.metadata()
            t = {k: h.get_tensor(k) for k in keys}
    T, G, L, k = 8, 4, c.num_hidden_layers, c.num_experts_per_tok
    assert t["logits"].shape == (T + G, c.vocab_size) and t["tokens"].shape == (G,) and t["nll"].shape == (T - 1,)
    for i in range(L):
        assert t[f"resid.L{i}"].shape == (T, c.hidden_size) and t[f"resid.L{i}"].dtype == torch.bfloat16
        assert t[f"route.moe.ids.L{i}"].shape == (T + G, k) and t[f"route.moe.ids.L{i}"].dtype == torch.int32
        assert t[f"route.moe.w.L{i}"].dtype == torch.float32 and t[f"route.moe.gap.L{i}"].shape == (T + G,)
        assert bool((t[f"route.moe.ids.L{i}"][:, 1:] > t[f"route.moe.ids.L{i}"][:, :-1]).all())
    want = ref.KolibriRef(c, ref.DictSource(sd), prefetch=False).generate([0, 5, 9, 17, 33, 65, 2, 3], G)
    assert torch.equal(t["logits"], want[0]) and t["tokens"].tolist() == want[1]
    assert meta["n_prompt"] == "8" and meta["mode"] == "bf16" and float(meta["ppl"]) > 0
    print(f"  run wrote {len(keys)} tensors; hfcheck, ppl and facts ran")


TESTS = [test_real_config, test_real_index, test_port_names_and_modules, test_norm_plain_weight,
         test_sandwich_norms, test_nope_full_rope_sliding, test_window_513_edge,
         test_router_logit_plus_bias_sigmoid_weights, test_router_ties_lower_id, test_shared_expert_ungated,
         test_qk_norm_per_head, test_combine_ascending_fp32, test_ring_cache_and_decode, test_determinism,
         test_hf_bf16, test_hf_f32, test_hf_generate_and_sdpa, test_checkpoint_and_stream, test_gptq_checkpoint,
         test_run_batch, test_cli_run_and_hfcheck]


def main() -> None:
    torch.manual_seed(0)
    only = sys.argv[1:]
    skipped = 0
    for t in TESTS:
        if only and t.__name__ not in only:
            continue
        print(t.__name__, flush=True)
        try:
            t()
        except Skip as s:
            skipped += 1
            print(f"SKIP {t.__name__}: {s}")
    print(f"ok ({skipped} skipped)")


if __name__ == "__main__":
    main()
