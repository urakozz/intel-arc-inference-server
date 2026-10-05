#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Checks for k2_ref.py (spec 18a). CPU, tiny random weights, seconds.

    test_k2_ref.py [test_name ...]     (pytest also collects it, where installed)

Each semantic trap of spec 18 §3 has its own test (plain-w grouped norm, the two independent norm
groups, the softplus threshold, the router's selection-only bias, x 2.5 after normalising, ties to
the lower id, the ascending-id bf16 combine, MoVA's routed V in the KV cache, partial RoPE, GQA),
then the whole port is held against the checkpoint's own modeling_k2_horizon.py (vendored under
third_party/k2_horizon/) on the same tiny random weights - bf16 and fp32, prompt and cached decode -
and the checkpoint readers (bf16 shards, int4 GPTQ, stream.py's layer streaming) against the
in-memory weights. `test_real_index` reads the real model.safetensors.index.json files when
K2_INDEX_BF16 / K2_INDEX_INT4 name their directories (each with its config.json), else SKIPs.
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

_spec = importlib.util.spec_from_file_location("k2_ref", os.path.join(_HERE, "k2_ref.py"))
ref = importlib.util.module_from_spec(_spec)
sys.modules["k2_ref"] = ref          # dataclasses resolve annotations through sys.modules
_spec.loader.exec_module(ref)

REAL_CONFIG = os.path.join(ref.THIRD_PARTY, "config.json")


class Skip(Exception):
    pass


def _skip(msg):
    if "pytest" in sys.modules:
        import pytest
        pytest.skip(msg)
    raise Skip(msg)


def bf(x):
    return x.to(torch.bfloat16).float()


# --------------------------------------------------------------------------------------------
# tiny configs and weights, built from the real config.json's structure

def tiny_dict(**over) -> dict:
    with open(REAL_CONFIG, encoding="utf-8") as f:
        d = json.load(f)
    d.update(hidden_size=64, intermediate_size=96, num_hidden_layers=4, num_attention_heads=4,
             num_key_value_heads=2, head_dim=16, rope_head_dim=16, mlp_only_layers=[0], num_experts=6,
             num_experts_per_tok=3, moe_intermediate_size=32, num_shared_experts=1, mova_num_experts=5,
             mova_num_experts_per_tok=2, vocab_size=128, max_position_embeddings=256)
    d.update(over)
    return d


def random_sd(c, seed: int = 0, router_scale: float = 3.0) -> dict:
    g = torch.Generator().manual_seed(seed)
    sd = {}
    for name in ref.expected_names(c):
        if name.endswith("norm.weight"):
            n = c.hidden_size if "layernorm" in name or name == "model.norm.weight" else \
                (c.num_attention_heads if "q_norm" in name else c.num_key_value_heads) * c.head_dim
            t = 1.0 + 0.5 * torch.randn(n, generator=g)
        elif name.endswith(".bias"):
            n = c.mova_num_experts if "v_router" in name else c.num_experts
            t = 0.1 * torch.randn(n, generator=g)
        elif name == "model.embed_tokens.weight":
            t = torch.randn(c.vocab_size, c.hidden_size, generator=g)
        else:
            out, inp = shape_of(c, name)
            s = router_scale if name.endswith(("v_router.weight", "mlp.gate.weight")) else 1.0
            t = torch.randn(out, inp, generator=g) * (s / math.sqrt(inp))
        sd[name] = t.to(torch.bfloat16)
    return sd


def shape_of(c, name: str):
    H, d, Hkv, D = c.num_attention_heads, c.head_dim, c.num_key_value_heads, c.hidden_size
    tail = name.split("self_attn.")[-1] if "self_attn." in name else name.split("mlp.")[-1]
    if name == "lm_head.weight":
        return c.vocab_size, D
    if "self_attn." in name:
        return {"q_proj.weight": (H * d, D), "k_proj.weight": (Hkv * d, D), "v_proj.weight": (Hkv * d, D),
                "o_proj.weight": (D, H * d), "gate_proj.weight": (H * d, D),
                "v_router.weight": (c.mova_num_experts, D)}.get(tail, (Hkv * d, D))   # v_experts.E.weight
    if tail == "gate.weight":
        return c.num_experts, D
    width = c.moe_intermediate_size * (c.num_shared_experts if "shared_experts" in name else 1) \
        if ("experts." in name) else c.intermediate_size
    return (D, width) if tail.endswith("down_proj.weight") else (width, D)


def tiny(seed=0, **over):
    d = tiny_dict(**over)
    c = ref.K2Config.from_dict(d)
    return d, c, random_sd(c, seed)


def hf_model(d: dict, sd: dict, dtype):
    model, _ = ref.build_hf(d, device="cpu", dtype=dtype)
    model.load_state_dict({k: v.to(dtype) for k, v in sd.items()}, strict=True)
    return model


# --------------------------------------------------------------------------------------------
# the real config

def test_real_config():
    c = ref.K2Config.from_file(REAL_CONFIG)
    kinds = [c.layer_kind(i) for i in range(c.num_hidden_layers)]
    assert kinds == ["dense"] * 3 + ["mova+moe"] * 45, kinds
    assert (c.hidden_size, c.layernorm_num_groups, c.rms_norm_eps) == (2560, 2, 1e-6)
    assert (c.num_attention_heads, c.num_key_value_heads, c.head_dim, c.n_rep) == (32, 8, 128, 4)
    assert (c.rope_dim, c.rope_theta, c.query_key_norm, c.attention_gate_func) == (128, 1e7, False, "softplus")
    assert (c.num_experts, c.num_experts_per_tok, c.moe_intermediate_size, c.num_shared_experts) == (100, 8, 768, 1)
    assert (c.mova_num_experts, c.mova_num_experts_per_tok, c.router_scaling_factor) == (64, 4, 2.5)
    assert c.norm_topk_prob and c.moe_gate_bias and c.router_score_func == "sigmoid"
    assert (c.vocab_size, c.bos_token_id, c.eos_token_ids, c.intermediate_size) == (250624, 0, [1], 6144)
    names = ref.expected_names(c)
    assert len(names) == 16998 == len(set(names)), len(names)   # the bf16 index's entry count
    print(f"  real config: 0-2 dense, 3-47 MoVA+MoE; {len(names)} tensor names (= the bf16 index)")


def test_real_index():
    """The real indexes, when K2_INDEX_BF16 / K2_INDEX_INT4 name directories holding them."""
    dirs = [os.environ.get(k) for k in ("K2_INDEX_BF16", "K2_INDEX_INT4")]
    if not any(dirs):
        _skip("K2_INDEX_BF16 / K2_INDEX_INT4 not set")
    for dname in filter(None, dirs):
        c = ref.K2Config.from_file(dname)
        with open(os.path.join(dname, "model.safetensors.index.json"), encoding="utf-8") as f:
            wm = json.load(f)["weight_map"]
        have = {n[:-len(".qweight")] + ".weight" if n.endswith(".qweight") else n
                for n in wm if not n.endswith((".scales", ".qzeros", ".g_idx"))}
        want = set(ref.expected_names(c))
        assert have == want, (sorted(want - have)[:5], sorted(have - want)[:5])
        nq = sum(n.endswith(".qweight") for n in wm)
        print(f"  {dname}: {len(have)} weights match expected_names; {nq} int4 (group {c.group_size})")


# --------------------------------------------------------------------------------------------
# the traps, one at a time

def test_norm_plain_weight():
    m = ref.hf_module()
    g = torch.Generator().manual_seed(1)
    x = bf(torch.randn(5, 64, generator=g))
    w = bf(0.3 * torch.randn(64, generator=g))              # far from 1: w and 1 + w differ a lot
    for mode in ("bf16", "f32"):
        ar = ref.Arith(mode)
        y = ref.grouped_rms_norm(ar, x, w, 2, 1e-6)
        xg = x.view(5, 2, 32)
        xhat = (xg * torch.rsqrt(xg.pow(2).mean(-1, keepdim=True) + 1e-6)).view(5, 64)
        assert torch.equal(y, ar.r(w * xhat))
        assert not torch.allclose(y, ar.r((1 + w) * xhat), atol=1e-2)
    hf = m.K2HorizonRMSNorm(64, n_groups=2, eps=1e-6)
    hf.weight.data = w.to(torch.bfloat16)
    assert torch.equal(ref.grouped_rms_norm(ref.Arith("bf16"), x, w, 2, 1e-6),
                       hf.to(torch.bfloat16)(x.to(torch.bfloat16)).float())
    print("  grouped norm is w * x_hat (bitwise = K2HorizonRMSNorm in bf16), not (1 + w) * x_hat")


def test_norm_two_groups():
    g = torch.Generator().manual_seed(2)
    ar = ref.Arith("f32")
    x = torch.randn(3, 64, generator=g)
    w = torch.ones(64)
    y = ref.grouped_rms_norm(ar, x, w, 2, 1e-6)
    x2 = x.clone()
    x2[:, 32:] *= 100.0                                      # only the second group grows
    y2 = ref.grouped_rms_norm(ar, x2, w, 2, 1e-6)
    assert torch.equal(y[:, :32], y2[:, :32]), "group 1 must not see group 2's mean of squares"
    assert torch.allclose(y[:, 32:], y2[:, 32:], atol=1e-5)  # scale-invariant within its group
    for h in (y[:, :32], y[:, 32:]):                         # each group has unit RMS
        assert torch.allclose(h.pow(2).mean(-1), torch.ones(3), atol=1e-4)
    one = ref.grouped_rms_norm(ar, x2, w, 1, 1e-6)
    assert not torch.allclose(one, y2, atol=1e-2)
    print("  two groups of 1280 (here 32), each on its own mean of squares; one group differs")


def test_softplus_threshold():
    beta = math.log(2)
    ar = ref.Arith("f32")
    x = torch.cat([torch.linspace(-40, 40, 8001), torch.tensor([28.0, 28.85, 28.86, 29.0, 100.0, 200.0, 1e4])])
    y = ref.attn_gate(ar, x, "softplus")
    lin = beta * x > 20
    assert torch.equal(y[lin], x[lin]), "above beta*x = 20 the gate is x itself"
    naive = torch.log1p(torch.exp(beta * x)) / beta
    ok = ~lin
    assert torch.allclose(y[ok], naive[ok], rtol=1e-6, atol=1e-7)
    assert float(ref.attn_gate(ar, torch.tensor([0.0]), "softplus")) == 1.0   # log2(1 + 2^0), beta = ln 2
    assert math.isinf(float(naive[x == 200.0][0])) and float(y[x == 200.0][0]) == 200.0
    # the threshold is on beta * x, not on x: 20 < x < 20 / ln 2 takes the log1p branch
    mid = (x > 20) & (x < 20 / beta)
    assert mid.any() and torch.allclose(y[mid], naive[mid], rtol=1e-6)
    yb = ref.attn_gate(ref.Arith("bf16"), bf(x), "softplus")
    assert torch.equal(yb, F.softplus(x.to(torch.bfloat16), beta=beta).float())
    print("  softplus beta = ln 2: x itself where beta*x > 20 (x = 200 stays 200, the naive formula is inf); "
          "log1p branch below, incl. 20 < x < 28.85; bf16 = F.softplus on bf16")


def test_router_bias_selects_only():
    ar = ref.Arith("bf16")
    logits = bf(torch.tensor([[2.0, 1.5, 1.0, 0.5, 0.0, -0.5]]))
    bias = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.9, 0.0])       # pulls expert 4 into the top 3
    rt = ref.route(ar, logits, bias, 3, 2.5, normalise=True)
    assert rt.ids.tolist() == [[0, 1, 4]], rt.ids
    s = torch.sigmoid(logits[0])
    want = bf(s[[0, 1, 4]] / s[[0, 1, 4]].sum() * 2.5)
    assert torch.allclose(rt.weights[0], want, atol=1e-2), (rt.weights, want)
    sb = s + bias                                             # the trap: weights from s + bias
    assert not torch.allclose(rt.weights[0], bf(sb[[0, 1, 4]] / sb[[0, 1, 4]].sum() * 2.5))
    # the modeling code's own function, its ids sorted ascending
    m = ref.hf_module()
    w_hf, ids_hf = m.calc_router_weights(logits.to(torch.bfloat16), bias.to(torch.bfloat16), "sigmoid", 3, 2.5)
    o = ids_hf.sort(-1)
    assert torch.equal(o.values, rt.ids) and torch.equal(bf(w_hf.gather(-1, o.indices)), rt.weights)
    print(f"  bias picks expert 4 into the top 3, weights from sigmoid alone: {rt.weights[0].tolist()}")


def test_router_scale_after_normalise():
    g = torch.Generator().manual_seed(3)
    logits = torch.randn(16, 100, generator=g) * 2
    bias = 0.05 * torch.randn(100, generator=g)
    rt = ref.route(ref.Arith("f32"), logits, bias, 8, 2.5, normalise=True)
    assert torch.allclose(rt.weights.sum(-1), torch.full((16,), 2.5), atol=1e-5)   # not 1: x 2.5 after
    rtb = ref.route(ref.Arith("bf16"), bf(logits), bias, 8, 2.5, normalise=True)
    assert torch.equal(rtb.weights, bf(rtb.weights)) and torch.allclose(rtb.weights.sum(-1), torch.full((16,), 2.5), atol=0.05)
    assert bool((rt.ids[:, 1:] > rt.ids[:, :-1]).all()), "ids ascending"
    assert bool((rt.gap >= 0).all())
    print("  weights sum to 2.5 (scaling after normalising; before it would cancel), ids ascending, bf16-rounded")


def test_router_ties_lower_id():
    ar = ref.Arith("f32")
    rt = ref.route(ar, torch.zeros(1, 100), None, 8, 2.5, normalise=True)
    assert rt.ids.tolist() == [list(range(8))] and float(rt.gap[0]) == 0.0
    th = torch.topk(torch.sigmoid(torch.zeros(1, 100)), 8).indices.sort().values.tolist()
    # a tie exactly at the cut: experts 3 and 7 equal for the last place
    logits = torch.full((1, 10), -5.0)
    logits[0, [0, 1]] = 3.0
    logits[0, [3, 7]] = 1.0
    rt = ref.route(ar, logits, None, 3, 2.5, normalise=True)
    assert rt.ids.tolist() == [[0, 1, 3]] and float(rt.gap[0]) == 0.0
    # with the bias, ties are on s + bias: equal sums tie even when s differs
    logits = torch.tensor([[1.0, 0.0, -1.0]])
    s = torch.sigmoid(logits)
    bias = torch.tensor([0.0, float(s[0, 0] - s[0, 1]), 0.0])
    rt = ref.route(ar, logits, bias, 1, 1.0, normalise=False)
    assert float(s[0, 1] + bias[1]) == float(s[0, 0]) and rt.ids.tolist() == [[0]]
    print(f"  ties go to the lower id (100 equal scores -> 0..7; torch.topk here -> {th})")


def test_combine_ascending_order():
    """bf16: 1 + 2^-8 + 2^-8 is 1 in ascending order, 1 + 2^-7 when the small ones add first."""
    ar = ref.Arith("bf16")
    vals = {2: 1.0, 5: 2.0 ** -8, 9: 2.0 ** -8}
    rt = ref.Route(ids=torch.tensor([[2, 5, 9]]), weights=torch.ones(1, 3), gap=torch.zeros(1),
                   sel_order=torch.tensor([[5, 9, 2]]))
    calls = []

    def expert(e, xs):
        calls.append(e)
        return torch.full((xs.shape[0], 4), vals[e])
    y = ref.combine_ascending(ar, rt, torch.zeros(1, 4), expert, 4)
    assert calls == [2, 5, 9] and float(y[0, 0]) == 1.0
    sel_order_sum = ar.r(ar.r(torch.tensor(vals[5]) + vals[9]) + vals[2])
    assert float(sel_order_sum) == 1.0078125, "the other order gives a different bf16 result"
    y32 = ref.combine_ascending(ref.Arith("f32"), rt, torch.zeros(1, 4), expert, 4)
    assert float(y32[0, 0]) == 1.0078125
    print("  combine runs experts 2, 5, 9 (ascending, not selection order 5, 9, 2): bf16 1.0 vs 1.0078125")


def test_partial_rope_and_gqa():
    m = ref.hf_module()
    g = torch.Generator().manual_seed(4)
    H, Hkv, T, d, rd = 4, 2, 5, 16, 8
    pos = torch.arange(3, 3 + T)
    for mode in ("bf16", "f32"):
        ar = ref.Arith(mode)
        q = ar.r(torch.randn(H, T, d, generator=g))
        cos, sin = ref.rope_cos_sin(ar, pos, rd, 1e7)
        ours = ref.apply_rope(ar, q, cos, sin)
        dt = torch.bfloat16 if mode == "bf16" else torch.float32
        qq, cc, ss = q.to(dt)[None], cos.to(dt)[None], sin.to(dt)[None]
        a, b = torch.split(m.split_to_interleaved(qq), [rd, d - rd], dim=-1)    # :270-289
        rot, _ = m.apply_rotary_pos_emb(m.interleaved_to_split(a), m.interleaved_to_split(a), cc, ss)
        hf = m.interleaved_to_split(torch.cat([m.split_to_interleaved(rot), b], dim=-1))[0].float()
        assert torch.equal(ours, hf), (mode, float((ours - hf).abs().max()))
        passthru = torch.cat([torch.arange(rd // 2, d // 2), torch.arange(d // 2 + rd // 2, d)])
        assert torch.equal(ours[..., passthru], q[..., passthru])
    # the rotated pair (i, i + d/2) turns by pos * theta^(-2i/rd)
    ar = ref.Arith("f32")
    x = torch.zeros(1, 1, d)
    x[0, 0, 1] = 1.0
    cos, sin = ref.rope_cos_sin(ar, torch.tensor([5]), rd, 1e7)
    y = ref.apply_rope(ar, x, cos, sin)[0, 0]
    ang = 5 * 1e7 ** (-2 * 1 / rd)
    assert abs(float(y[1]) - math.cos(ang)) < 1e-6 and abs(float(y[1 + d // 2]) - math.sin(ang)) < 1e-6
    # GQA: q head h reads kv head h // (H / Hkv)
    q = torch.randn(H, T, d, generator=g)
    k, v = torch.randn(Hkv, T, d, generator=g), torch.randn(Hkv, T, d, generator=g)
    o = ref.attention(ar, q, k, v, pos, pos, H // Hkv)
    for h in range(H):
        s = (q[h] @ k[h // 2].t()) * d ** -0.5
        s = s.masked_fill(torch.ones(T, T, dtype=torch.bool).triu(1), float("-inf"))
        assert torch.allclose(o[:, h], torch.softmax(s, -1) @ v[h // 2], atol=1e-6)
    o_wrong = ref.attention(ar, q, k[[1, 0]], v[[1, 0]], pos, pos, H // Hkv)
    assert not torch.allclose(o, o_wrong)
    print("  partial RoPE (8 of 16) bitwise = the modeling code's interleave dance; pass-through unchanged; "
          "GQA head h -> kv h // 2")


def test_mova_v_is_the_cache():
    d, c, sd = tiny(seed=5)
    r = ref.K2Ref(c, ref.DictSource(sd), mode="f32", prefetch=False)
    ids = torch.randint(0, c.vocab_size, (7,), generator=torch.Generator().manual_seed(5))
    cache, rec = r.new_cache(), ref.Recorder(7)
    r.forward(ids, 0, cache, rec)
    i = 2
    assert c.is_mova(i) and not c.is_mova(0)
    h_in = rec.t[f"resid.L{i - 1}"][0]
    p = f"model.layers.{i}."
    x = ref.grouped_rms_norm(r.ar, h_in, sd[p + "input_layernorm.weight"], 2, c.rms_norm_eps)
    s = torch.sigmoid(x @ sd[p + "self_attn.v_router.weight"].float().t())
    sel = s + sd[p + "self_attn.v_router.bias"].float()
    want = torch.zeros(7, c.num_key_value_heads * c.head_dim)
    for t in range(7):
        top = sorted(torch.topk(sel[t], 2).indices.tolist())
        w = s[t, top] / s[t, top].sum() * 2.5
        for e, we in zip(top, w):
            want[t] += we * F.silu(x[t] @ sd[p + f"self_attn.v_experts.{e}.weight"].float().t())
        assert rec.t[f"route.mova.ids.L{i}"][0][t].tolist() == top
    got = cache[i][1].transpose(0, 1).reshape(7, -1)
    assert torch.allclose(got, want, atol=1e-5), float((got - want).abs().max())
    # attention reads that V: a never-selected value expert changes nothing, a selected one does
    used = set(rec.t[f"route.mova.ids.L{i}"][0].flatten().tolist())
    unused = sorted(set(range(c.mova_num_experts)) - used)
    base = r.forward(ids, 0, r.new_cache())
    for e, expect_change in ([(unused[0], False)] if unused else []) + [(sorted(used)[0], True)]:
        sd2 = dict(sd)
        sd2[p + f"self_attn.v_experts.{e}.weight"] = sd[p + f"self_attn.v_experts.{e}.weight"] * 3
        r2 = ref.K2Ref(c, ref.DictSource(sd2), mode="f32", prefetch=False)
        out = r2.forward(ids, 0, r2.new_cache())
        assert torch.equal(out, base) != expect_change, (e, expect_change)
    print(f"  layer {i}: cached V = sum_e w_e SiLU(V_e x) over the routed 2 of 5; unused experts {unused} inert")


def test_decode_matches_prefill():
    d, c, sd = tiny(seed=6)
    for mode in ("bf16", "f32"):
        r = ref.K2Ref(c, ref.DictSource(sd), mode=mode, prefetch=False)
        ids = torch.randint(0, c.vocab_size, (9,), generator=torch.Generator().manual_seed(6)).tolist()
        full = r.forward(ids, 0, r.new_cache())
        cache = r.new_cache()
        r.forward(ids[:6], 0, cache)
        steps = torch.cat([r.forward([t], 6 + j, cache) for j, t in enumerate(ids[6:])])
        diff = float((steps - full[6:]).abs().max())
        assert torch.equal(steps.argmax(-1), full[6:].argmax(-1)) and diff < (0.1 if mode == "bf16" else 1e-4), diff
        print(f"  {mode}: 3 cached decode steps vs one 9-id forward, max |diff| {diff:.3g}")


def test_determinism():
    d, c, sd = tiny(seed=7)
    r = ref.K2Ref(c, ref.DictSource(sd), prefetch=False)
    a = r.generate(list(range(5)), 4)
    b = r.generate(list(range(5)), 4)
    assert torch.equal(a[0], b[0]) and a[1] == b[1]
    print(f"  two runs bitwise equal, greedy {a[1]}")


# --------------------------------------------------------------------------------------------
# against the checkpoint's own modeling code

def _hf_vs_ref(d, c, sd, dtype, mode, ids, gen):
    model = hf_model(d, sd, dtype)
    caps = {}

    def cap(i):
        def fn(_m, _a, out):          # returns None: a hook's return value replaces the output
            caps.setdefault(i, out[0].float().clone())
        return fn
    hooks = [layer.register_forward_hook(cap(i)) for i, layer in enumerate(model.model.layers)]
    with torch.no_grad():
        out = model(input_ids=torch.tensor([ids]), use_cache=True)
    for h in hooks:
        h.remove()
    rows, cache = [out.logits[0].float()], out.past_key_values
    toks = []
    with torch.no_grad():
        for _ in range(gen):
            toks.append(int(rows[-1][-1].argmax()))
            out = model(input_ids=torch.tensor([[toks[-1]]]), past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            rows.append(out.logits[0].float())
    hf = torch.cat(rows)
    r = ref.K2Ref(c, ref.DictSource(sd), mode=mode, prefetch=False)
    rec = ref.Recorder(len(ids))
    ours, our_toks = r.generate(ids, gen, rec)
    resid = [float((rec.t[f"resid.L{i}"][0] - caps[i]).abs().max()) for i in range(c.num_hidden_layers)]
    return hf, ours, toks, our_toks, resid, rec


def test_hf_bf16():
    for label, over in (("full RoPE", {}), ("partial RoPE + qk norm", dict(rope_head_dim=8, query_key_norm=True)),
                        ("no shared expert, MoE without norm", dict(num_shared_experts=0, norm_topk_prob=False))):
        d, c, sd = tiny(seed=8, **over)
        ids = torch.randint(0, c.vocab_size, (11,), generator=torch.Generator().manual_seed(8)).tolist()
        hf, ours, toks, our_toks, resid, rec = _hf_vs_ref(d, c, sd, torch.bfloat16, "bf16", ids, 5)
        diff = (hf - ours).abs()
        nbit = int((diff.amax(-1) == 0).sum())
        assert toks == our_toks, (toks, our_toks)
        assert torch.equal(hf, ours), (label, f"{nbit}/{hf.shape[0]} rows bitwise, max {float(diff.max())}, resid {resid}")
        n_routes = sum(v[0].numel() for k, v in rec.t.items() if ".ids." in k)
        print(f"  {label}: HF eager bf16 vs port bf16 - logits bitwise on all {hf.shape[0]} rows "
              f"(11 prompt + 5 cached decode), residuals bitwise ({max(resid)}), greedy {toks}; {n_routes} routes")


def test_hf_f32():
    d, c, sd = tiny(seed=9, rope_head_dim=8)
    ids = torch.randint(0, c.vocab_size, (11,), generator=torch.Generator().manual_seed(9)).tolist()
    hf, ours, toks, our_toks, resid, _ = _hf_vs_ref(d, c, sd, torch.float32, "f32", ids, 5)
    diff = float((hf - ours).abs().max())
    assert toks == our_toks and diff < 1e-4, (diff, resid)
    print(f"  HF fp32 vs port f32: max |logit diff| {diff:.3g}, max |resid diff| {max(resid):.3g}, greedy {toks}")


# --------------------------------------------------------------------------------------------
# the readers: bf16 shards, int4 GPTQ, layer streaming

def write_checkpoint(dirname: str, d: dict, tensors: dict) -> None:
    """One shard per layer plus one for the rest, an index, the config (the real layout)."""
    from safetensors.torch import save_file
    shards: dict[str, dict] = {}
    for k, v in tensors.items():
        fn = f"model-L{k.split('.')[2]}.safetensors" if k.startswith("model.layers.") else "model-rest.safetensors"
        shards.setdefault(fn, {})[k] = v.contiguous()
    wm = {}
    for fn, t in shards.items():
        save_file(t, os.path.join(dirname, fn))
        wm.update({k: fn for k in t})
    with open(os.path.join(dirname, "model.safetensors.index.json"), "w", encoding="utf-8") as f:
        json.dump({"metadata": {}, "weight_map": wm}, f)
    with open(os.path.join(dirname, "config.json"), "w", encoding="utf-8") as f:
        json.dump(d, f)


def test_checkpoint_and_stream():
    d, c, sd = tiny(seed=10)
    ids = torch.randint(0, c.vocab_size, (8,), generator=torch.Generator().manual_seed(10)).tolist()
    mem = ref.K2Ref(c, ref.DictSource(sd), prefetch=False).generate(ids, 3)
    with tempfile.TemporaryDirectory() as tmp:
        write_checkpoint(tmp, d, sd)
        ck = ref.Checkpoint(tmp)
        assert ck.cfg == c
        disk = ref.K2Ref(c, ck, prefetch=True).generate(ids, 3)
        assert torch.equal(mem[0], disk[0]) and mem[1] == disk[1]
        # the vendored model, layer-streamed by stream.attach, against the resident one
        streamed, pf = ref.hf_streamed(ck, d)
        with torch.no_grad():
            a = streamed(input_ids=torch.tensor([ids]), use_cache=False).logits
            b = hf_model(d, sd, torch.bfloat16)(input_ids=torch.tensor([ids]), use_cache=False).logits
        assert torch.equal(a, b)
        assert all(p.is_meta for n, p in streamed.named_parameters() if n.startswith("model.layers."))
    print(f"  bf16 shards (prefetched) = in-memory weights bitwise, greedy {disk[1]}; HF streamed = HF resident")


def gptq_pack(w_q: torch.Tensor, scales: torch.Tensor, group: int):
    """[N, K] ints in 0..15 + [K/group, N] f16 scales -> (qweight [K/8, N], qzeros, g_idx)."""
    N, K = w_q.shape
    q = w_q.t().to(torch.int64).reshape(K // 8, 8, N)
    packed = torch.zeros(K // 8, N, dtype=torch.int64)
    for i in range(8):
        packed |= q[:, i, :] << (4 * i)
    qweight = (packed - ((packed >> 31) << 32)).to(torch.int32)          # wrap to int32
    qzeros = torch.full((K // group, N // 8), ref.Checkpoint.QZ, dtype=torch.int32)
    return qweight, qzeros, torch.arange(K, dtype=torch.int32) // group


def test_gptq_checkpoint():
    over = dict(hidden_size=128, intermediate_size=128, num_attention_heads=4, num_key_value_heads=2, head_dim=32,
                rope_head_dim=32, moe_intermediate_size=64, mova_num_experts=8)
    d, c, sd = tiny(seed=11, **over)
    d["quantization_config"] = {"bits": 4, "group_size": 64, "sym": True, "desc_act": False,
                                "quant_method": "gptq", "provider": "auto-round"}
    g = torch.Generator().manual_seed(11)
    files, deq = {}, dict(sd)
    for name, t in sd.items():
        quantise = (t.dim() == 2 and name.startswith("model.layers.") and not name.endswith("mlp.gate.weight"))
        if not quantise:            # embedding, head, norms, the MoE router stay bf16 (the real split)
            files[name] = t.to(torch.float16) if name.endswith("v_router.bias") else t
            deq[name] = files[name]
            continue
        N, K = t.shape
        q = torch.randint(0, 16, (N, K), generator=g)
        sc = (torch.rand(K // 64, N, generator=g) * 0.05 + 0.01).to(torch.float16)
        qw, qz, gi = gptq_pack(q, sc, 64)
        base = name[:-len(".weight")]
        files.update({base + ".qweight": qw, base + ".scales": sc, base + ".qzeros": qz, base + ".g_idx": gi})
        deq[name] = ((q.float() - 8) * sc.float().repeat_interleave(64, 0).t()).to(torch.bfloat16)
    ids = list(range(3, 11))
    want = ref.K2Ref(c, ref.DictSource(deq), prefetch=False).generate(ids, 2)
    with tempfile.TemporaryDirectory() as tmp:
        write_checkpoint(tmp, d, files)
        ck = ref.Checkpoint(tmp, fast=False)
        assert ck.cfg.group_size == 64
        got = ref.K2Ref(ck.cfg, ck).generate(ids, 2)
        assert torch.equal(want[0], got[0]) and want[1] == got[1]
        fast_note = "C++ dequant_t not tried (no g++/ninja)"
        import shutil
        if shutil.which("g++") and shutil.which("ninja"):
            name = "model.layers.3.self_attn.v_experts.1"
            assert torch.equal(ref.Checkpoint(tmp, fast=True).linear(name), ck.linear(name))
            fast_note = "C++ dequant_t equal"
        bad = dict(files)
        bad["model.layers.1.self_attn.q_proj.g_idx"] = torch.flip(files["model.layers.1.self_attn.q_proj.g_idx"], [0])
        tmp_bad = os.path.join(tmp, "bad")
        os.makedirs(tmp_bad)
        write_checkpoint(tmp_bad, d, bad)
        try:
            ref.Checkpoint(tmp_bad, fast=False).linear("model.layers.1.self_attn.q_proj")
            raise AssertionError("a permuted g_idx must be refused")
        except ValueError:
            pass
    print(f"  int4 GPTQ shards (f16 v_router bias, bf16 MoE router) = the dequantised weights bitwise; "
          f"{fast_note}; permuted g_idx refused")


def test_cli_run_and_hfcheck():
    """The box command's path: `run` on a checkpoint directory, then `hfcheck --against` its output."""
    from safetensors import safe_open
    d, c, sd = tiny(seed=12)
    with tempfile.TemporaryDirectory() as tmp:
        write_checkpoint(tmp, d, sd)
        with open(os.path.join(tmp, "p.ids"), "w", encoding="utf-8") as f:
            f.write("0 5 9 17 33 65 2 3\n")
        out = os.path.join(tmp, "out", "p.golden.safetensors")
        argv = sys.argv
        try:
            sys.argv = ["k2_ref.py", "run", tmp, "--prompt", os.path.join(tmp, "p.ids"), "--out", out, "--gen", "4"]
            ref.main()
            sys.argv = ["k2_ref.py", "hfcheck", tmp, "--prompt", os.path.join(tmp, "p.ids"), "--against", out]
            ref.main()
        finally:
            sys.argv = argv
        with safe_open(out, framework="pt") as h:
            keys, meta = set(h.keys()), h.metadata()
            t = {k: h.get_tensor(k) for k in keys}
    T, G, L = 8, 4, c.num_hidden_layers
    assert t["logits"].shape == (T + G, c.vocab_size) and t["tokens"].shape == (G,)
    for i in range(L):
        assert t[f"resid.L{i}"].shape == (T, c.hidden_size) and t[f"resid.L{i}"].dtype == torch.bfloat16
        if c.is_mova(i):
            assert t[f"route.mova.ids.L{i}"].shape == (T + G, c.mova_num_experts_per_tok)
            assert t[f"route.moe.ids.L{i}"].dtype == torch.int32 and t[f"route.moe.w.L{i}"].dtype == torch.bfloat16
            assert t[f"route.moe.gap.L{i}"].shape == (T + G,)
            assert bool((t[f"route.moe.ids.L{i}"][:, 1:] > t[f"route.moe.ids.L{i}"][:, :-1]).all())
        else:
            assert f"route.moe.ids.L{i}" not in keys
    assert t["rope.cos"].shape == (T + G, c.rope_dim)
    want = ref.K2Ref(c, ref.DictSource(sd), prefetch=False).generate([0, 5, 9, 17, 33, 65, 2, 3], G)
    assert torch.equal(t["logits"], want[0]) and t["tokens"].tolist() == want[1]
    assert meta["n_prompt"] == "8" and meta["mode"] == "bf16"
    print(f"  run wrote {len(keys)} tensors (the plan 8d layout + gaps); hfcheck read them back")


TESTS = [test_real_config, test_real_index, test_norm_plain_weight, test_norm_two_groups, test_softplus_threshold,
         test_router_bias_selects_only, test_router_scale_after_normalise, test_router_ties_lower_id,
         test_combine_ascending_order, test_partial_rope_and_gqa, test_mova_v_is_the_cache,
         test_decode_matches_prefill, test_determinism, test_hf_bf16, test_hf_f32, test_checkpoint_and_stream,
         test_gptq_checkpoint, test_cli_run_and_hfcheck]


def main() -> None:
    torch.manual_seed(0)
    only = sys.argv[1:]
    skipped = 0
    for t in TESTS:
        if only and t.__name__ not in only:
            continue
        print(t.__name__)
        try:
            t()
        except Skip as s:
            skipped += 1
            print(f"SKIP {t.__name__}: {s}")
    print(f"ok ({skipped} skipped)")


if __name__ == "__main__":
    main()
