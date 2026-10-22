#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Checks for dflash_ref.py (spec 19a Task 1). CPU, tiny random weights, seconds.

    test_dflash_ref.py [test_name ...]     (pytest also collects it, where installed)

Synthetic DFlash 2 / DFlash (v1) drafters cover: the configs' attention resolution, shapes, the
tap-offset convention at the API, context K/V (RoPE, window, ignored positions >= p), the
full-attention layer, non-causal vs causal block attention, the conv's one-row look-back
(incl. K < block - 1), the selector's edge scores (anchor = row 0's predecessor), the walk's tie
rule, Gumbel keying / determinism, d2t, the mask embedding, the W4A16 unpack, and a load from
checkpoint-shaped files. The last test runs the real z-lab/Qwen3.8-27B-DFlash2 (+ the W4A16
arm) from the HF cache on random taps and SKIPS when the files are not there.
"""
import json
import math
import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

import importlib.util  # noqa: E402

import torch  # noqa: E402

_spec = importlib.util.spec_from_file_location("dflash_ref", os.path.join(_HERE, "dflash_ref.py"))
ref = importlib.util.module_from_spec(_spec)
sys.modules["dflash_ref"] = ref          # dataclasses resolve annotations through sys.modules
_spec.loader.exec_module(ref)


class Skip(Exception):
    pass


def _skip(msg):
    if "pytest" in sys.modules:
        import pytest
        pytest.skip(msg)
    raise Skip(msg)


# --------------------------------------------------------------------------------------------
# synthetic drafters

QWEN38_DFLASH2 = {  # z-lab/Qwen3.8-27B-DFlash2 config.json (2026-10-05), trimmed to what is read
    "architectures": ["DFlash2DraftModel"], "attention_bias": False, "is_causal": False,
    "dflash_config": {"block_size": 8, "conv_group_size": 16, "conv_kernel_size": 2,
                      "mask_token_id": 248070, "selector_rank": 256, "selector_top_k": 16,
                      "target_layer_ids": [5, 19, 33, 47, 61]},
    "head_dim": 128, "hidden_act": "silu", "hidden_size": 5120, "intermediate_size": 17408,
    "layer_types": ["sliding_attention"] * 5, "num_attention_heads": 32, "num_hidden_layers": 5,
    "num_key_value_heads": 8, "rms_norm_eps": 1e-06,
    "rope_parameters": {"rope_theta": 10000000, "rope_type": "default"}, "sliding_window": 2048,
    "tie_word_embeddings": False, "use_sliding_window": True, "vocab_size": 248320}

ORNITH_DFLASH = {  # ornith-ai/Ornith-1.5-35B-A3B-DFlash config.json (2026-10-05), trimmed
    "architectures": ["DFlashDraftModel"], "attention_bias": False,
    "dflash_config": {"block_size": 16, "mask_token_id": 248077,
                      "target_layer_ids": [1, 6, 11, 16, 22, 27, 32, 37]},
    "head_dim": 128, "hidden_act": "silu", "hidden_size": 2048, "intermediate_size": 6144,
    "layer_types": ["sliding_attention"] * 5 + ["full_attention"], "num_attention_heads": 32,
    "num_hidden_layers": 6, "num_key_value_heads": 8, "rms_norm_eps": 1e-06,
    "rope_parameters": {"rope_theta": 10000000, "rope_type": "default"}, "sliding_window": 4096,
    "tie_word_embeddings": False, "use_sliding_window": True, "vocab_size": 248320}

VOCAB, HID, HT = 300, 64, 48      # HT: the target's hidden (fc input per tap) != drafter hidden


def v2_dict(window=6, causal=False, layers=2):
    return {"architectures": ["DFlash2DraftModel"], "is_causal": causal,
            "dflash_config": {"block_size": 8, "conv_group_size": 16, "conv_kernel_size": 2,
                              "mask_token_id": 297, "selector_rank": 8, "selector_top_k": 16,
                              "target_layer_ids": [1, 3]},
            "head_dim": 16, "hidden_act": "silu", "hidden_size": HID, "intermediate_size": 96,
            "layer_types": ["sliding_attention"] * layers, "num_attention_heads": 4,
            "num_hidden_layers": layers, "num_key_value_heads": 2, "rms_norm_eps": 1e-6,
            "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
            "sliding_window": window, "vocab_size": VOCAB}


def v1_dict(layer_types=("sliding_attention", "sliding_attention", "full_attention"), window=4,
            is_causal=None, draft_vocab=None):
    d = {"architectures": ["DFlashDraftModel"],
         "dflash_config": {"block_size": 16, "mask_token_id": 298, "target_layer_ids": [0, 2, 4]},
         "head_dim": 16, "hidden_act": "silu", "hidden_size": HID, "intermediate_size": 96,
         "layer_types": list(layer_types), "num_attention_heads": 4,
         "num_hidden_layers": len(layer_types), "num_key_value_heads": 2, "rms_norm_eps": 1e-6,
         "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
         "sliding_window": window, "vocab_size": VOCAB}
    if is_causal is not None:
        d["is_causal"] = is_causal
    if draft_vocab is not None:
        d["draft_vocab_size"] = draft_vocab
    return d


def random_tensors(cfg, seed=0, dtype=torch.bfloat16):
    """Checkpoint-named tensors of a drafter (bf16 like the real ones), scaled so activations
    stay O(1), plus the target-shared embedding and head."""
    g = torch.Generator().manual_seed(seed)

    def lin(o, i):
        return (torch.randn(o, i, generator=g) / i ** 0.5).to(dtype)

    def norm(n):
        return (1 + 0.1 * torch.randn(n, generator=g)).to(dtype)

    H, hd, nh, nkv = cfg.hidden, cfg.head_dim, cfg.n_heads, cfg.n_kv_heads
    t = {"fc.weight": lin(H, HT * len(cfg.target_layer_ids)), "hidden_norm.weight": norm(H),
         "norm.weight": norm(H)}
    for i in range(cfg.n_layers):
        L = f"layers.{i}."
        t.update({L + "input_layernorm.weight": norm(H), L + "post_attention_layernorm.weight": norm(H),
                  L + "self_attn.q_proj.weight": lin(nh * hd, H), L + "self_attn.k_proj.weight": lin(nkv * hd, H),
                  L + "self_attn.v_proj.weight": lin(nkv * hd, H), L + "self_attn.o_proj.weight": lin(H, nh * hd),
                  L + "self_attn.q_norm.weight": norm(hd), L + "self_attn.k_norm.weight": norm(hd),
                  L + "mlp.gate_proj.weight": lin(cfg.ffn, H), L + "mlp.up_proj.weight": lin(cfg.ffn, H),
                  L + "mlp.down_proj.weight": lin(H, cfg.ffn)})
        if cfg.version == 2:
            G = H // cfg.conv_group_size
            for cv in ("attention_conv", "mlp_conv"):
                base = 0.3 * torch.randn(2, cfg.conv_kernel_size, H, generator=g)
                base[:, 0] += 1.0
                t[L + cv + ".base_kernel"] = base.to(dtype)
                t[L + cv + ".kernel_projection.weight"] = (0.3 * lin(2 * cfg.conv_kernel_size * G, H).float()).to(dtype)
    if cfg.version == 2:
        t["candidate_selector.hidden_projection.weight"] = lin(cfg.selector_rank, H)
        t["candidate_selector.predecessor_codebook"] = torch.randn(cfg.vocab, cfg.selector_rank, generator=g).to(dtype)
        t["candidate_selector.successor_codebook"] = torch.randn(cfg.vocab, cfg.selector_rank, generator=g).to(dtype)
    embed = torch.randn(cfg.vocab, H, generator=g).to(dtype)
    head = (torch.randn(cfg.draft_vocab, H, generator=g) * 2.0).to(dtype)
    return t, embed, head


def make(cdict, seed=0, **kw):
    cfg = ref.config_from_dict(cdict)
    t, e, h = random_tensors(cfg, seed)
    if cfg.draft_vocab != cfg.vocab:
        t["d2t"] = kw.pop("d2t")
    return ref.Drafter(cfg, t, e, h, **kw)


def random_taps(d, n, seed=1):
    g = torch.Generator().manual_seed(seed)
    return {i: torch.randn(n, d.target_hidden, generator=g) for i in d.cfg.target_layer_ids}


def ctx_of(d, n, seed=1):
    return d.context_kv(random_taps(d, n, seed), torch.arange(n))


# --------------------------------------------------------------------------------------------
# tests


def test_real_configs():
    q = ref.config_from_dict(QWEN38_DFLASH2)
    assert q.version == 2 and q.max_k == 7 and q.target_layer_ids == [5, 19, 33, 47, 61]
    assert q.windows == [2048] * 5 and q.causal == [False] * 5
    assert (q.hidden, q.n_heads, q.n_kv_heads, q.head_dim, q.ffn, q.vocab) == (5120, 32, 8, 128, 17408, 248320)
    assert (q.conv_group_size, q.conv_kernel_size, q.selector_rank, q.selector_top_k) == (16, 2, 256, 16)
    assert q.mask_token_id == 248070 and q.rope_theta == 1e7 and q.neox
    o = ref.config_from_dict(ORNITH_DFLASH)
    assert o.version == 1 and o.max_k == 15 and len(o.target_layer_ids) == 8
    # vLLM's rule: no is_causal -> layer_types decides, sliding layers are CAUSAL, full is not
    assert o.windows == [4096] * 5 + [None] and o.causal == [True] * 5 + [False]
    print("real configs: DFlash2 all-sliding non-causal; Ornith 5 causal sliding + 1 full")


def test_shapes_and_determinism():
    for d, K in ((make(v2_dict()), 7), (make(v1_dict(is_causal=False)), 5)):
        c = d.cfg
        ctx = ctx_of(d, 10)
        assert len(ctx.k) == c.n_layers and ctx.k[0].shape == (10, c.n_kv_heads, c.head_dim)
        assert ctx.fc.shape == (10, c.hidden)
        o = d.draft_block(17, 10, K, ctx)
        assert o.ids.shape == (K,) and o.cand_ids.shape == (K, 16) and o.unary.shape == (K, 16)
        assert o.q.shape == (K, c.vocab) and o.hidden.shape == (K + 1, c.hidden)
        assert torch.equal(o.positions, torch.arange(10, 11 + K))
        assert torch.isfinite(o.hidden).all() and torch.isfinite(o.unary).all()
        assert torch.allclose(o.q.sum(-1), torch.ones(K), atol=1e-5)
        assert (o.unary[:, :-1] >= o.unary[:, 1:]).all()
        if c.version == 2:
            assert o.edges.shape == (K, 16, 16)
            assert torch.equal(o.ids, o.cand_ids[torch.arange(K), o.chosen])
            off = torch.ones_like(o.q, dtype=torch.bool)
            off.scatter_(1, o.cand_ids, False)
            assert (o.q[off] == 0).all()
        else:
            assert torch.equal(o.ids, o.cand_ids[:, 0])       # greedy v1 = the row argmax
        o2 = d.draft_block(17, 10, K, ctx)
        assert torch.equal(o.ids, o2.ids) and torch.equal(o.hidden, o2.hidden) and torch.equal(o.q, o2.q)
        for bad in (0, c.max_k + 1):
            try:
                d.draft_block(17, 10, bad, ctx)
                raise AssertionError("K out of range accepted")
            except ValueError:
                pass
    print("shapes, q normalised, sorted top-16, deterministic, K range enforced")


def test_tap_offset_at_api():
    """taps[i] = the residual AFTER target layer i. A toy residual target pins both helpers."""
    g = torch.Generator().manual_seed(5)
    n_layers, N = 6, 5
    Ws = [torch.randn(HT, HT, generator=g) / HT ** 0.5 for _ in range(n_layers)]
    x = torch.randn(N, HT, generator=g)
    hs = [x]                                  # transformers: [embeddings, out_0, ..., out_{L-1}]
    outs = []
    for W in Ws:
        x = x + torch.tanh(x @ W)
        outs.append(x)
        hs.append(x)
    d = make(v1_dict(is_causal=False))         # target_layer_ids [0, 2, 4]
    ids = d.cfg.target_layer_ids
    t_hf = ref.taps_from_hf_hidden_states(hs, ids)
    t_dump = ref.taps_from_resid({f"resid.L{i}": outs[i] for i in range(n_layers)}, ids)
    for i in ids:
        assert torch.equal(t_hf[i], outs[i]) and torch.equal(t_dump[i], outs[i])
        assert not torch.equal(t_hf[i], hs[i])                       # not the INPUT of layer i
    ctx = d.context_kv(t_hf, torch.arange(N))
    man = ref.rms_norm(torch.cat([outs[i] for i in ids], -1) @ d.w["fc.weight"].t(),
                       d.w["hidden_norm.weight"], d.cfg.eps)
    assert torch.allclose(ctx.fc, man, atol=1e-5)
    for bad in (torch.cat([outs[i] for i in ids], -1), {i + 1: outs[i] for i in ids}, {0: outs[0], 2: outs[2]}):
        try:
            d.context_kv(bad, torch.arange(N))
            raise AssertionError("context_kv accepted taps not keyed by target_layer_ids")
        except ValueError:
            pass
    try:
        ref.taps_from_hf_hidden_states(hs, [0, 5])                  # hs[6] would be post-norm
        raise AssertionError("tap at the last layer accepted")
    except ValueError:
        pass
    print("tap offset: taps[i] = out of layer i = hf hidden_states[i+1] = resid.L{i}; API rejects others")


def test_context_kv_and_rope():
    d = make(v2_dict())
    c = d.cfg
    taps = random_taps(d, 9)
    pos = torch.arange(9)
    ctx = d.context_kv(taps, pos)
    L = "layers.1.self_attn."
    k = (ctx.fc @ d.w[L + "k_proj.weight"].t()).view(9, c.n_kv_heads, c.head_dim)
    k = ref.rms_norm(k, d.w[L + "k_norm.weight"], c.eps)
    hd = c.head_dim
    inv = 1.0 / (c.rope_theta ** (torch.arange(0, hd, 2, dtype=torch.float64) / hd))
    z = torch.complex(k[..., : hd // 2].double(), k[..., hd // 2:].double())
    z = z * torch.polar(torch.ones(9, hd // 2, dtype=torch.float64), pos[:, None].double() * inv)[:, None]
    man = torch.cat([z.real, z.imag], -1).float()
    assert torch.allclose(ctx.k[1], man, atol=1e-5)
    v = (ctx.fc @ d.w[L + "v_proj.weight"].t()).view(9, c.n_kv_heads, c.head_dim)
    assert torch.allclose(ctx.v[1], v, atol=1e-5)
    # relative: <rope(a, i), rope(b, j)> depends on i - j only
    a, b = torch.randn(1, 1, hd), torch.randn(1, 1, hd)
    for neox in (True, False):
        def dot(i, j):
            return (ref.rope(a, torch.tensor([i]), 1e4, neox) * ref.rope(b, torch.tensor([j]), 1e4, neox)).sum()
        assert torch.allclose(dot(7, 3), dot(107, 103), atol=1e-4)
    # positions >= p in the context are ignored
    big = d.context_kv(random_taps(d, 14), torch.arange(14))
    small = ref.ContextKV(pos=big.pos[:9], k=[k[:9] for k in big.k], v=[v[:9] for v in big.v], fc=big.fc[:9])
    assert torch.equal(d.draft_block(5, 9, 7, big).hidden, d.draft_block(5, 9, 7, small).hidden)
    print("context K/V = RoPE(k_norm(k_proj(hidden_norm(fc(taps))))); rope relative; ctx >= p ignored")


def test_context_window():
    W, N = 6, 14
    d = make(v2_dict(window=W))
    taps = random_taps(d, N)
    p, K = N, 7
    base = d.draft_block(5, p, K, d.context_kv(taps, torch.arange(N))).hidden

    def bumped(c):
        t = {i: v.clone() for i, v in taps.items()}
        for v in t.values():
            v[c] += 3.0
        return d.draft_block(5, p, K, d.context_kv(t, torch.arange(N))).hidden

    for c in range(0, p - W + 1):               # p - c >= W: outside every row's window
        assert torch.equal(bumped(c), base), c
    assert not torch.allclose(bumped(p - W + 1), base)   # p - c = W - 1: row 0 sees it
    assert not torch.allclose(bumped(p - 1), base)
    # per-row window: row j at p + j sees c iff p + j - c <= W - 1
    m = d.attention_mask(0, torch.arange(p, p + K + 1), torch.arange(0, p + K + 1))
    for j in range(K + 1):
        for c in range(p):
            assert bool(m[j, c]) == (p + j - c <= W - 1)
    print(f"sliding window {W}: positions <= p - {W} invisible, p - {W - 1} visible; per-row edge")


def test_full_attention_layer():
    N, p = 12, 12
    mixed = make(v1_dict(window=4, is_causal=False))          # sliding, sliding, full
    slid = make(v1_dict(("sliding_attention",) * 3, window=4, is_causal=False))
    for d, sees in ((mixed, True), (slid, False)):
        taps = random_taps(d, N)
        a = d.draft_block(5, p, 5, d.context_kv(taps, torch.arange(N))).hidden
        t = {i: v.clone() for i, v in taps.items()}
        for v in t.values():
            v[0] += 3.0
        b = d.draft_block(5, p, 5, d.context_kv(t, torch.arange(N))).hidden
        assert (not torch.allclose(a, b)) == sees
    print("full-attention layer sees position 0 at p = 12; all-sliding (window 4) does not")


def test_block_attention_causality():
    for mk in (lambda causal: make(v2_dict(causal=causal)),
               lambda causal: make(v1_dict(is_causal=causal))):
        for causal in (False, True):
            d = mk(causal)
            K, p = 6, 9
            ctx = ctx_of(d, p)
            x = d.embed_block(11, K)
            a = d.forward_block(x, p, ctx)
            x2 = x.clone()
            x2[K] += 1.0                         # the last mask row's input
            b = d.forward_block(x2, p, ctx)
            if causal:
                assert torch.equal(a[:K], b[:K])
            else:
                assert not torch.allclose(a[1], b[1])
                assert not torch.allclose(a[0], b[0])
            assert not torch.allclose(a[K], b[K])
    print("block attention: non-causal rows see later rows; causal (is_causal true) do not")


def test_conv_lookback():
    g = torch.Generator().manual_seed(3)
    H, grp = 32, 16
    for taps in (2, 3):
        for L in (8, 4, 2):                      # block of 1 + K rows: K = 7, 3, 1
            R = 2 * L                            # two blocks stacked
            x = torch.randn(R, H, generator=g)
            delta = torch.randn(R, taps, H // grp, generator=g)
            base = torch.randn(taps, H, generator=g)
            out = ref.grouped_conv(x, delta, base, L, grp)
            coef = base[None] + delta.repeat_interleave(grp, -1)
            for r in range(R):
                man = coef[r, 0] * x[r]
                for t in range(1, taps):
                    if r % L >= t:
                        man = man + coef[r, t] * x[r - t]
                assert torch.allclose(out[r], man, atol=1e-5)
            for r in range(R):                   # dependence: exactly row r and rows r + t in-block
                x2 = x.clone()
                x2[r] += 1.0
                ch = (ref.grouped_conv(x2, delta, base, L, grp) - out).abs().amax(-1) > 0
                exp = torch.tensor([rr == r or (0 < rr - r < taps and rr % L >= rr - r) for rr in range(R)])
                assert torch.equal(ch, exp), (taps, L, r)
    print("conv: out[r] = sum_t (base_t + delta[r,t]) x[r-t] for r % (1+K) >= t; no reach across blocks")


def test_v2_layer_wiring():
    """One DFlash 2 layer written out by hand (qwen3_dflash2.py:267-287): coefficients from the
    normed input, side 0 before attention / the MLP, side 1 after, residual adds outside."""
    d = make(v2_dict(layers=1))
    c, w = d.cfg, d.w
    K, p = 5, 9
    ctx = ctx_of(d, p)
    x = d.embed_block(21, K)
    R, G = K + 1, c.hidden // c.conv_group_size

    def conv(h, which):
        co = (h @ w[f"layers.0.{which}.kernel_projection.weight"].t()).view(R, 2, c.conv_kernel_size, G)
        base = w[f"layers.0.{which}.base_kernel"]
        return ref.grouped_conv(h, co[:, 0], base[0], R, c.conv_group_size), co[:, 1], base[1]

    h = ref.rms_norm(x, w["layers.0.input_layernorm.weight"], c.eps)
    h, co, b1 = conv(h, "attention_conv")
    h = d._attention(0, h, torch.arange(p, p + R), ctx)
    h = ref.grouped_conv(h, co, b1, R, c.conv_group_size)
    r = x + h
    h = ref.rms_norm(r, w["layers.0.post_attention_layernorm.weight"], c.eps)
    h, co, b1 = conv(h, "mlp_conv")
    g, u = h @ w["layers.0.mlp.gate_proj.weight"].t(), h @ w["layers.0.mlp.up_proj.weight"].t()
    h = (torch.nn.functional.silu(g) * u) @ w["layers.0.mlp.down_proj.weight"].t()
    h = ref.grouped_conv(h, co, b1, R, c.conv_group_size)
    man = ref.rms_norm(r + h, w["norm.weight"], c.eps)
    assert torch.allclose(d.forward_block(x, p, ctx), man, atol=1e-5)
    print("DFlash 2 layer: conv side 0 on the normed input, side 1 on the output, both coefficient sets from the input")


def test_short_K():
    d = make(v2_dict(causal=True))       # causal: row j depends only on rows <= j
    ctx = ctx_of(d, 9)
    full = d.draft_block(7, 9, 7, ctx)
    for K in (1, 3, 5):
        o = d.draft_block(7, 9, K, ctx)
        assert torch.allclose(o.hidden, full.hidden[:K + 1], atol=1e-5), K
        assert torch.equal(o.cand_ids, full.cand_ids[:K]) and torch.equal(o.ids, full.ids[:K])
    dn = make(v2_dict(causal=False))
    for K in (1, 3):
        o = dn.draft_block(7, 9, K, ctx_of(dn, 9))
        assert o.ids.shape == (K,) and torch.isfinite(o.hidden).all()
    print("K < block - 1: the block is 1 + K rows; causal drafts are the K = 7 prefix")


def test_score_edges_and_anchor():
    g = torch.Generator().manual_seed(4)
    V, r, K, k = 50, 4, 3, 5
    pred, succ = torch.randn(V, r, generator=g), torch.randn(V, r, generator=g)
    cand = torch.stack([torch.randperm(V, generator=g)[:k] for _ in range(K)])
    unary, hid = torch.randn(K, k, generator=g), torch.randn(K, r, generator=g)
    S = ref.score_edges(pred, succ, cand, unary, hid, anchor_id=42)
    for j in range(K):
        for a in range(k):
            pv = 42 if j == 0 else int(cand[j - 1, a])
            for c in range(k):
                man = unary[j, c] + (pred[pv] * hid[j] * succ[cand[j, c]]).sum()
                assert torch.allclose(S[j, a, c], man, atol=1e-5)
    assert all(torch.equal(S[0, a], S[0, 0]) for a in range(k))       # row 0: every prev is the anchor
    S2 = ref.score_edges(pred, succ, cand, unary, hid, anchor_id=7)
    assert not torch.allclose(S2[0], S[0]) and torch.equal(S2[1:], S[1:])
    # in the drafter: the anchor token is row 0's predecessor
    d = make(v2_dict())
    o = d.draft_block(33, 9, 4, ctx_of(d, 9))
    cs = d.w["candidate_selector.predecessor_codebook"]
    hidp = o.hidden[1:2] @ d.w["candidate_selector.hidden_projection.weight"].t()
    succ_t = d.w["candidate_selector.successor_codebook"][o.cand_ids[0]].float()
    man = o.unary[0] + (cs[33].float() * hidp[0] * succ_t).sum(-1)
    assert torch.allclose(o.scores[0], man, atol=1e-4)
    print("edge scores = unary + <pred[prev] * h, succ[c]>; row 0's prev is the anchor")


def test_walk_tie_rule():
    k, K = 4, 3
    cand = torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33]])
    S = torch.zeros(K, k, k)
    S[0, 0] = torch.tensor([1.0, 3.0, 3.0, 0.0])        # tie between index 1 and 2 -> 1
    S[0, 1:] = 100.0                                     # never read: row 0 starts at prev 0
    S[1, 1] = torch.tensor([5.0, 0.0, 0.0, 5.0])        # prev = 1; tie 0 / 3 -> 0
    S[1, 2] = torch.tensor([0.0, 0.0, 9.0, 0.0])        # would be read if prev were 2
    S[2, 0] = torch.tensor([0.0, 2.0, 2.0, 2.0])        # prev = 0; three-way tie -> 1
    idx, ids, real = ref.selector_walk(S, cand, p=5)
    assert idx.tolist() == [1, 0, 1] and ids.tolist() == [11, 20, 31]
    assert torch.equal(real[1], S[1, 1]) and torch.equal(real[0], S[0, 0])
    print("walk: argmax over S[j, prev], ties to the lower candidate index, prev starts at 0")


def _murmur_py(seed, pos, key):
    """An independent int-arithmetic murmur3_hash32 (gumbel.py:96) for cross-checking."""
    M = 0xFFFFFFFF

    def rotl(v, s):
        return ((v << s) | (v >> (32 - s))) & M

    def mix(h, kk):
        kk = (kk * 0xCC9E2D51) & M
        kk = rotl(kk, 15)
        kk = (kk * 0x1B873593) & M
        h ^= kk
        h = rotl(h, 13)
        return (h * 5 + 0xE6546B64) & M

    h = 0
    for kk in (seed & M, (seed >> 32) & M, pos & M, key & M):
        h = mix(h, kk)
    h ^= 16
    h ^= h >> 16
    h = (h * 0x85EBCA6B) & M
    h ^= h >> 13
    h = (h * 0xC2B2AE35) & M
    return h ^ (h >> 16)


def test_gumbel():
    keys = torch.tensor([0, 1, 7, 248069, 2**31 + 5])
    for seed, pos in ((0, 0), (12345678901234, 1 << 30), (2**63 - 1, 77)):
        u = ref.murmur3_uniform32(seed, pos, keys)
        for kk, uu in zip(keys.tolist(), u.tolist()):
            h = _murmur_py(seed, pos, kk)
            exp = float(torch.tensor((h >> 16) * 2.0 ** -16 + ((h & 0xFFFF) + 0.5) * 2.0 ** -32,
                                     dtype=torch.float64).float())
            assert uu == exp and 0.0 < uu < 1.0
    scores = torch.tensor([1.0, 0.5, 0.0, -0.5, 2.0, 1.5])
    cand = torch.tensor([100, 101, 102, 103, 104, 105])
    S = scores.view(1, 1, -1)
    a = [int(ref.selector_walk(S, cand.view(1, -1), 9, 1.0, s)[1][0]) for s in range(50)]
    b = [int(ref.selector_walk(S, cand.view(1, -1), 9, 1.0, s)[1][0]) for s in range(50)]
    assert a == b and len(set(a)) > 1
    # keyed by token id: the same (id, score) pairs in another order draw the same winner
    perm = torch.tensor([3, 0, 5, 1, 4, 2])
    c = [int(ref.selector_walk(scores[perm].view(1, 1, -1), cand[perm].view(1, -1), 9, 1.0, s)[1][0])
         for s in range(50)]
    assert a == c
    # keyed by position: another p gives another sequence
    e = [int(ref.selector_walk(S, cand.view(1, -1), 10, 1.0, s)[1][0]) for s in range(50)]
    assert a != e
    # tiny temperature = greedy; T > 0 without a seed is refused
    assert int(ref.selector_walk(S, cand.view(1, -1), 9, 1e-6, 3)[1][0]) == 104
    try:
        ref.selector_walk(S, cand.view(1, -1), 9, 1.0, None)
        raise AssertionError("sampling without a seed accepted")
    except ValueError:
        pass
    # the draw itself: u from the int reference at P - 1 + salt, g = -log(-log(1 - u))
    # (gumbel.py fp32 path: -log(-_log1p_neg_stable(u)))
    for s in range(20):
        u = torch.tensor([float(torch.tensor(((h >> 16) * 2.0 ** -16 + ((h & 0xFFFF) + 0.5) * 2.0 ** -32),
                                             dtype=torch.float64).float())
                          for h in (_murmur_py(s, 9 + (1 << 30), int(k)) for k in cand)], dtype=torch.float64)
        exp = int(cand[torch.argmax(scores.double() / 0.9 - torch.log(-torch.log1p(-u)))])
        assert int(ref.selector_walk(S, cand.view(1, -1), 9, 0.9, s)[1][0]) == exp
    # Gumbel-max samples softmax(scores / T)
    T, n = 0.7, 6000
    hits = torch.zeros(6)
    for s in range(n):
        hits[int(ref.selector_walk(S, cand.view(1, -1), 9, T, s)[0][0])] += 1
    assert (hits / n - torch.softmax(scores / T, 0)).abs().max() < 0.025
    # the whole drafter: same seed -> same draft; q is softmax(realized / T)
    d = make(v2_dict())
    ctx = ctx_of(d, 9)
    o1 = d.draft_block(3, 9, 7, ctx, 0.8, 11)
    o2 = d.draft_block(3, 9, 7, ctx, 0.8, 11)
    assert torch.equal(o1.ids, o2.ids)
    qs = torch.gather(o1.q, 1, o1.cand_ids)
    assert torch.allclose(qs, torch.softmax(o1.scores / 0.8, -1), atol=1e-6)
    print("gumbel: murmur3 matches an int reference; keyed by (seed, P-1, token id); samples softmax")


def test_d2t():
    Vd = 100
    d2t = 2 * torch.arange(Vd) + 1                # draft id d -> target id 3d + 1 (offsets, EAGLE-3)
    d = make(v1_dict(is_causal=False, draft_vocab=Vd), d2t=d2t)
    o = d.draft_block(4, 9, 5, ctx_of(d, 9))
    lg = d.head_logits(o.hidden[1:])
    assert torch.equal(o.ids, 3 * lg.argmax(-1) + 1)
    image = torch.zeros(VOCAB, dtype=torch.bool)
    image[3 * torch.arange(Vd) + 1] = True
    assert (o.q[:, ~image] == 0).all() and torch.allclose(o.q.sum(-1), torch.ones(5), atol=1e-5)
    cfg = ref.config_from_dict(v1_dict(is_causal=False, draft_vocab=Vd))
    t, e, h = random_tensors(cfg)
    try:
        ref.Drafter(cfg, t, e, h)                 # no d2t
        raise AssertionError("missing d2t accepted")
    except ValueError:
        pass
    try:
        ref.config_from_dict({**v2_dict(), "draft_vocab_size": 100})
        raise AssertionError("DFlash 2 + d2t accepted")
    except NotImplementedError:
        pass
    print("d2t: target id = draft id + d2t[draft id]; q zero off its image")


def test_mask_embedding_and_scale():
    d = make(v2_dict())
    base = d.embed_block(5, 3)
    assert torch.equal(base[0], d.embed[5].float()) and torch.equal(base[1], d.embed[297].float())
    me = torch.randn(HID)
    dm = make(v2_dict(), mask_embedding=me)
    x = dm.embed_block(5, 3)
    assert torch.equal(x[0], base[0]) and all(torch.equal(x[j], me) for j in (1, 2, 3))
    cd = v2_dict()
    cd["dflash_config"] = {**cd["dflash_config"], "input_embedding_scale": 2.5}
    ds = make(cd)
    assert torch.allclose(ds.embed_block(5, 3), 2.5 * base)
    print("embedding: anchor + mask rows from the target table; mask_embedding.pt; input scale")


def _pack_int4(q):
    """compressed-tensors pack_to_int32 (symmetric): u = q + 8, 8 consecutive columns per int32,
    first column in the low nibble."""
    u = (q + 8).to(torch.int64)
    o, i = u.shape
    u = u.view(o, i // 8, 8)
    packed = (u << (torch.arange(8) * 4)).sum(-1)
    packed = torch.where(packed >= 2**31, packed - 2**32, packed)
    return packed.to(torch.int32)


def test_dequant_pack_quantized():
    g = torch.Generator().manual_seed(9)
    q = torch.randint(-8, 8, (6, 256), generator=g)
    s = torch.rand(6, 2, generator=g).half()
    w = ref.dequant_pack_quantized(_pack_int4(q), s, [6, 256], 128)
    assert torch.equal(w, q.float() * s.float().repeat_interleave(128, 1))
    print("W4A16 unpack: low nibble first, q - 8, per-128 group scale")


def _write_checkpoint(root, cfg_dict, tensors, embed, head, packed_names=()):
    from safetensors.torch import save_file
    dd, td = os.path.join(root, "drafter"), os.path.join(root, "target")
    os.makedirs(dd)
    os.makedirs(td)
    cd = dict(cfg_dict)
    out = {}
    for n, v in tensors.items():
        if n in packed_names:
            w = v.float()
            s = (w.abs().view(w.shape[0], -1, 32).amax(-1) / 7).clamp_min(1e-8)
            q = torch.clamp(torch.round(w / s.repeat_interleave(32, 1)), -8, 7).to(torch.int64)
            b = n[: -len(".weight")]
            out[b + ".weight_packed"] = _pack_int4(q)
            out[b + ".weight_scale"] = s.half()
            out[b + ".weight_shape"] = torch.tensor(list(w.shape))
        else:
            out[n] = v.contiguous()
    if packed_names:
        cd["quantization_config"] = {
            "quant_method": "compressed-tensors", "format": "pack-quantized",
            "config_groups": {"group_0": {"weights": {"num_bits": 4, "group_size": 32, "symmetric": True,
                                                      "type": "int", "strategy": "group"}}}}
    save_file(out, os.path.join(dd, "model.safetensors"))
    with open(os.path.join(dd, "config.json"), "w") as f:
        json.dump(cd, f)
    shards = {"model-00001-of-00002.safetensors": {"model.language_model.embed_tokens.weight": embed,
                                                   "mtp.embed_tokens.weight": torch.zeros_like(embed)},
              "model-00002-of-00002.safetensors": {"lm_head.weight": head}}
    wm = {}
    for fn, ts in shards.items():
        save_file({k: v.contiguous() for k, v in ts.items()}, os.path.join(td, fn))
        wm.update({k: fn for k in ts})
    with open(os.path.join(td, "model.safetensors.index.json"), "w") as f:
        json.dump({"weight_map": wm}, f)
    return dd, td


def test_load_from_files():
    cd = v2_dict()
    cfg = ref.config_from_dict(cd)
    t, e, h = random_tensors(cfg, 2)
    with tempfile.TemporaryDirectory() as root:
        dd, td = _write_checkpoint(root, cd, t, e, h)
        d = ref.load_drafter(dd, td)
        mem = ref.Drafter(cfg, t, e, h)
        ctx, ctx_m = ctx_of(d, 9), ctx_of(mem, 9)
        assert torch.equal(d.draft_block(8, 9, 7, ctx).hidden, mem.draft_block(8, 9, 7, ctx_m).hidden)
        torch.save({"embedding": torch.ones(HID), "mask_token_id": 297}, os.path.join(dd, "mask_embedding.pt"))
        assert torch.equal(ref.load_drafter(dd, td).embed_block(1, 2)[1], torch.ones(HID))
    pn = ("fc.weight", "layers.0.mlp.down_proj.weight", "layers.1.self_attn.q_proj.weight")
    with tempfile.TemporaryDirectory() as root:
        dd, td = _write_checkpoint(root, cd, t, e, h, packed_names=pn)
        dq = ref.load_drafter(dd, td)
        for n in pn:
            a, b = dq.w[n].flatten(), t[n].float().flatten()
            assert torch.nn.functional.cosine_similarity(a, b, 0) > 0.99, n
        assert dq.w["layers.0.mlp.up_proj.weight"].equal(t["layers.0.mlp.up_proj.weight"].float())
        o = dq.draft_block(8, 9, 7, ctx_of(dq, 9))
        assert torch.isfinite(o.hidden).all()
    print("load: drafter safetensors + target index (language_model.embed_tokens, lm_head); W4A16 dequant")


def test_real_checkpoint():
    """z-lab/Qwen3.8-27B-DFlash2 on the Qwen/Qwen3.8-27B embedding / head, random taps, one
    block. Plus the W4A16 drafter when present. ~16 GB RSS (estimated); skipped when not in the HF cache."""
    dd = ref.find_snapshot("z-lab/Qwen3.8-27B-DFlash2", ("config.json", "model.safetensors"))
    td = ref.find_snapshot("Qwen/Qwen3.8-27B", ("config.json", "model.safetensors.index.json"))
    if dd is None or td is None:
        _skip(f"real checkpoints not in {ref.hf_hub_dir()} (drafter {dd}, target {td})")
    with open(os.path.join(td, "model.safetensors.index.json")) as f:
        wm = json.load(f)["weight_map"]
    need = {wm[n] for n in wm if n.endswith(("embed_tokens.weight", "lm_head.weight")) and not n.startswith("mtp.")}
    if not all(os.path.isfile(os.path.join(td, s)) for s in need):
        _skip(f"target shards {sorted(need)} not downloaded yet")
    import time
    t0 = time.time()
    e, h = ref.read_target_embed_head(td)
    d = ref.load_drafter(dd, embed=e, lm_head=h)
    c = d.cfg
    assert c.version == 2 and d.target_hidden == 5120 and c.max_k == 7
    t1 = time.time()
    N = 24
    taps = {i: torch.randn(N, 5120, generator=torch.Generator().manual_seed(i)) * 4 for i in c.target_layer_ids}
    ctx = d.context_kv(taps, torch.arange(N))
    o = d.draft_block(9707, N, 7, ctx)
    t2 = time.time()
    assert torch.isfinite(o.hidden).all() and torch.isfinite(o.scores).all()
    assert o.ids.shape == (7,) and ((o.ids >= 0) & (o.ids < c.vocab)).all()
    assert torch.equal(o.ids, d.draft_block(9707, N, 7, ctx).ids)
    print(f"real DFlash2: load {t1 - t0:.0f} s, ctx + block {t2 - t1:.1f} s, ids {o.ids.tolist()}")
    qd = ref.find_snapshot("syvai/Qwen3.8-27B-DFlash2-W4A16", ("config.json", "model.safetensors"))
    if qd is None:
        print("  (W4A16 drafter not in the cache: arm skipped)")
        return
    fc_bf16 = d.w["fc.weight"]
    del d
    dq = ref.load_drafter(qd, embed=e, lm_head=h)
    cos = torch.nn.functional.cosine_similarity(dq.w["fc.weight"].flatten(), fc_bf16.flatten(), 0)
    oq = dq.draft_block(9707, N, 7, dq.context_kv(taps, torch.arange(N)))
    assert torch.isfinite(oq.hidden).all() and cos > 0.98, float(cos)
    print(f"  W4A16: fc cosine vs bf16 {float(cos):.4f}, ids {oq.ids.tolist()}")


# --------------------------------------------------------------------------------------------
# spec 23f: the Muse-Glimmer assistant form, against transformers 5.19.0 on 23a's tiny pair

MUSE_ASSIST = {  # meta-models/Muse-Glimmer-30B-assistant config.json @ e8192f3a (2026-10-15), unmodified fields
    "architectures": ["MuseGlimmerAssistantModel"], "attention_dropout": 0, "block_size": 16, "bos_token_id": 200000,
    "dtype": "bfloat16", "eos_token_id": 200001, "head_dim": 128, "hidden_act": "silu", "hidden_size": 6656,
    "intermediate_size": 19968, "layer_types": ["sliding_attention"] * 5, "mask_token_id": 201818,
    "max_position_embeddings": 131072, "model_type": "muse_glimmer_assistant", "num_attention_heads": 32,
    "num_hidden_layers": 5, "num_key_value_heads": 8, "pad_token_id": 200018, "rms_norm_eps": 1e-05,
    "rope_parameters": {"rope_theta": 500000.0, "rope_type": "default"}, "sliding_window": 2048,
    "target_layer_ids": [1, 13, 25, 37, 49], "transformers_version": "5.15.0.dev0"}

_MUSE: dict = {}


def _muse_env():
    """test_muse_ref's tiny-variant builder and transformers 5.19.0 (spec 21a's site): SKIP without them."""
    if "t" in _MUSE:
        return _MUSE["t"]
    try:
        import transformers
    except ImportError:
        _skip("transformers is not importable (run with spec 21a's 5.19.0 site first on PYTHONPATH)")
    if transformers.__version__ != "5.19.0":
        _skip(f"transformers {transformers.__version__}: the Muse form is held to 5.19.0 (spec 21a's site)")
    spec = importlib.util.spec_from_file_location("test_muse_ref", os.path.join(_HERE, "test_muse_ref.py"))
    t = importlib.util.module_from_spec(spec)
    sys.modules["test_muse_ref"] = t
    spec.loader.exec_module(t)
    try:
        t.variant("l8")
    except t.Skip as s:
        _skip(str(s))
    _MUSE["t"] = t
    return t


def _muse_pair(dtype, assist="assist"):
    """(the l8 target - transformers' own -, transformers' assistant, the port) in `dtype` (cached)."""
    key = (dtype, assist)
    if key in _MUSE:
        return _MUSE[key]
    t = _muse_env()
    from transformers.models.muse_glimmer_assistant import modeling_muse_glimmer_assistant as am
    target = t.hf_model(t.variant("l8"), dtype)
    ad = t.variant(assist)
    asst = am.MuseGlimmerAssistantModel.from_pretrained(ad, dtype=dtype, attn_implementation="eager").eval()
    e, h = ref.read_target_embed_head(t.variant("l8"))
    port = ref.load_drafter(ad, embed=e, lm_head=h, dtype=dtype)
    _MUSE[key] = (target, asst, port)
    return _MUSE[key]


def _muse_taps(target, ids):
    """transformers' own taps: output_hidden_states=[1, 5] gives hidden_states[i] = decoder layer i's OUTPUT
    (OC:112-131; no embedding prepended) - {1: [N, H], 5: [N, H]}."""
    with torch.no_grad():
        out = target(input_ids=torch.tensor([ids]), output_hidden_states=[1, 5])
    return {i: out.hidden_states[i][0] for i in (1, 5)}


class _Outputs:
    def __init__(self, hs):
        self.hidden_states = hs


def _hf_generator(asst, target):
    from transformers import GenerationConfig
    from transformers.generation.candidate_generator import DFlashTokenCandidateGenerator
    gc = GenerationConfig(max_length=100000, do_sample=False)
    return DFlashTokenCandidateGenerator(asst, target.get_input_embeddings(), target.get_output_embeddings(), gc, None)


def _hf_candidates(g, ids_upto_anchor, hs_rows, n_last):
    """One get_candidates call: input_ids = the committed ids + the anchor (the last), hidden_states rows = the
    main model's last forward (the first n rows are the newly committed positions), n_last = accepted drafts."""
    T = len(ids_upto_anchor)
    hs = [None] * 8
    for i, r in hs_rows.items():
        hs[i] = r[None]
    mk = {"position_ids": torch.arange(T)[None], "attention_mask": torch.ones(1, T, dtype=torch.long)}
    with torch.no_grad():
        cand, logits = g.get_candidates(torch.tensor([ids_upto_anchor]), mk, _Outputs(tuple(hs)),
                                        is_first_iteration=False, n_last_matches=n_last)
    return cand[0, T:], logits[0].float()


def test_muse_config():
    """The assistant's config reads as the Muse form (AC:62-89): top-level block / mask / taps, window 2048 INCLUSIVE
    (decision 13), the two-rounding norm, theta 5e5, eps 1e-5, no vocab (the target's), no causal layer; the cached
    real config.json when present is the vendored dict."""
    c = ref.config_from_dict(MUSE_ASSIST)
    assert (c.form, c.version, c.block_size, c.max_k, c.mask_token_id) == ("muse", 1, 16, 15, 201818)
    assert c.target_layer_ids == [1, 13, 25, 37, 49] and c.windows == [2048] * 5 and c.causal == [False] * 5
    assert c.window_inclusive and c.norm_two_roundings and not c.embed_norm and not c.head_cap and c.q_scale == 1.0
    assert (c.hidden, c.n_heads, c.n_kv_heads, c.head_dim, c.ffn, c.eps, c.rope_theta, c.vocab) == \
        (6656, 32, 8, 128, 19968, 1e-5, 500000.0, 0)
    bad = dict(MUSE_ASSIST, layer_types=["sliding_attention"] * 4 + ["full_attention"])
    try:
        ref.config_from_dict(bad)
        raise AssertionError("a full-attention assistant layer was accepted")
    except NotImplementedError:
        pass
    snap = ref.find_snapshot("meta-models/Muse-Glimmer-30B-assistant", ("config.json",))
    if snap:
        with open(os.path.join(snap, "config.json")) as f:
            assert json.load(f) == MUSE_ASSIST, "the real assistant's config.json moved from the vendored dict"
        print("  the cached meta-models/Muse-Glimmer-30B-assistant config.json == the vendored dict")


def _bitwise(a, b, what):
    if not torch.equal(a, b):
        d = (a.float() - b.float()).abs()
        raise AssertionError(f"{what}: differ, max |d| {d.max():.3e} at {int(d.argmax())}")


def test_muse_equals_transformers():
    """Review Focus 1-3. The l8 target's own taps (output_hidden_states=[1, 5]) into both transformers'
    DFlashTokenCandidateGenerator and the port (MuseSession): candidate logits and ids BITWISE, in bf16 and fp32,
    - at anchors 20..59 of a 60-id prompt (a fresh generator per anchor: the prefill call);
    - on the window-16 assistant (assist16) across its edge at the same anchors;
    - a continuation (Review Focus 3): prefill at anchor 20, 3 drafts accepted (4 committed rows: the anchor + 3;
      the verify's 12 rejected rows are random and must never matter), the next block at 24, then 0 accepted, then
      the block at 25: logits / ids and every layer's cache bitwise transformers' after its crop;
    and the stateless draft_block (the dflash_accept form) gives the same ids with logits to 1e-5 (bf16: 2 ulps)."""
    _muse_env()
    ids = _rand_muse_ids(61, seed=31)
    for dtype in (torch.float32, torch.bfloat16):
        for assist in ("assist", "assist16"):
            target, asst, port = _muse_pair(dtype, assist)
            taps = _muse_taps(target, ids[:60])
            worst = 0.0
            for p in range(20, 60):
                g = _hf_generator(asst, target)
                cand, logits = _hf_candidates(g, ids[:p + 1], {i: taps[i][:p] for i in taps}, 0)
                s = ref.MuseSession(port)
                pids, plog = s.draft({i: taps[i][:p] for i in taps}, ids[p])
                _bitwise(plog, logits, f"{dtype} {assist} anchor {p}: candidate logits")
                assert torch.equal(pids, cand), (p, pids, cand)
                ctx = port.context_kv({i: taps[i][:p] for i in taps}, torch.arange(p))
                o = port.draft_block(ids[p], p, 15, ctx)
                assert torch.equal(o.ids, cand) or _tie_ok(logits, o.ids), (dtype, assist, p, o.ids, cand)
                worst = max(worst, float((o.unary[:, 0] - logits.max(-1).values).abs().max()))
            # the continuation
            g = _hf_generator(asst, target)
            s = ref.MuseSession(port)
            _hf_candidates(g, ids[:21], {i: taps[i][:20] for i in taps}, 0)
            s.draft({i: taps[i][:20] for i in taps}, ids[20])
            junk = torch.Generator().manual_seed(7)
            for (c0, n, nxt) in ((20, 4, 24), (24, 1, 25)):
                rows = {i: torch.cat([taps[i][c0:c0 + n], torch.randn(16 - n, taps[i].shape[1], generator=junk)
                                      .to(dtype)]) for i in taps}
                cand, logits = _hf_candidates(g, ids[:nxt + 1], rows, n - 1)
                pids, plog = s.draft({i: taps[i][c0:c0 + n] for i in taps}, ids[nxt])
                _bitwise(plog, logits, f"{dtype} {assist} continuation at {nxt}")
                assert torch.equal(pids, cand)
                for li, layer in enumerate(g.cache.layers):
                    _bitwise(s.k[li], layer.keys, f"{dtype} {assist} cache keys L{li} at {nxt}")
                    _bitwise(s.v[li], layer.values, f"{dtype} {assist} cache values L{li} at {nxt}")
            g.cache.crop(-16)
            s.crop()
            for li, layer in enumerate(g.cache.layers):
                _bitwise(s.k[li], layer.keys, f"{dtype} {assist} cache after crop L{li}")
            print(f"  {str(dtype)[6:]} {assist}: anchors 20..59 + the continuation BITWISE; stateless draft_block "
                  f"ids equal, top logit within {worst:.2e}")


def _rand_muse_ids(n, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 200000, (n,), generator=g).tolist()


def _tie_ok(logits, ids):
    """ids are argmaxes of `logits` up to ties within 1e-5 (the stateless form's summation order)."""
    best = logits.max(-1).values
    return bool(((logits.gather(1, ids[:, None])[:, 0] - best).abs() <= 1e-5 * best.abs().clamp_min(1)).all())


def test_muse_window_edge():
    """Review Focus 2 on the REAL window (2048): anchors p = 2047, 2048, 2049, 3000 over a 3001-id prompt, fp32 -
    port == transformers bitwise; row 0 (at p) sees context p - 2048 (inclusive) and not p - 2049: perturbing the taps
    at p - 2049 leaves the drafts bitwise unchanged, perturbing p - 2048 moves them; the vLLM edge (window_inclusive
    False, <= 2047) differs from transformers at p >= 2048 (decision 13: one key at the edge)."""
    _muse_env()
    target, asst, port = _muse_pair(torch.float32, "assist")
    ids = _rand_muse_ids(3002, seed=33)
    taps = _muse_taps(target, ids[:3001])
    for p in (2047, 2048, 2049, 3000):
        g = _hf_generator(asst, target)
        cand, logits = _hf_candidates(g, ids[:p + 1], {i: taps[i][:p] for i in taps}, 0)
        pids, plog = ref.MuseSession(port).draft({i: taps[i][:p] for i in taps}, ids[p])
        _bitwise(plog, logits, f"window edge, anchor {p}")
        if p >= 2049:
            for off, moves in ((2049, False), (2048, True)):
                pt = {i: taps[i][:p].clone() for i in taps}
                for i in pt:
                    pt[i][p - off] += 3.0
                _, l2 = ref.MuseSession(port).draft(pt, ids[p])
                assert torch.equal(l2, plog) != moves, (p, off, moves)
        if p >= 2048:
            port.cfg.window_inclusive = False
            try:
                _, lv = ref.MuseSession(port).draft({i: taps[i][:p] for i in taps}, ids[p])
            finally:
                port.cfg.window_inclusive = True
            assert not torch.equal(lv, plog), f"the <= 2047 edge equals transformers' at {p}"
    print("  p 2047 / 2048 / 2049 / 3000: BITWISE; p - 2048 visible, p - 2049 not; the vLLM edge differs")


def test_muse_review_focus_4():
    """Review Focus 4, bf16, anchors 20..59: each of the drafter's own rules, removed, changes the drafts against
    transformers - the norm's two roundings (one rounding instead), the raw embedding (the target's weight-less norm
    applied), no q scale (the target's 3.87 applied), its own RoPE table (theta 1e6 instead of 5e5). Each must move
    the logits somewhere and flip at least one draft id."""
    _muse_env()
    target, asst, port = _muse_pair(torch.bfloat16, "assist")
    ids = _rand_muse_ids(61, seed=31)
    taps = _muse_taps(target, ids[:60])
    base = {}
    for p in range(20, 60):
        base[p] = ref.MuseSession(port).draft({i: taps[i][:p] for i in taps}, ids[p])
    c = port.cfg
    muts = (("one-rounding norm", "norm_two_roundings", False), ("embedding norm", "embed_norm", True),
            ("q x 3.87", "q_scale", 3.87), ("theta 1e6", "rope_theta", 1e6))
    for what, attr, val in muts:
        old = getattr(c, attr)
        setattr(c, attr, val)
        try:
            moved = flips = 0
            for p in range(20, 60):
                pids, plog = ref.MuseSession(port).draft({i: taps[i][:p] for i in taps}, ids[p])
                moved += int(not torch.equal(plog, base[p][1]))
                flips += int((pids != base[p][0]).sum())
        finally:
            setattr(c, attr, old)
        assert moved > 0 and flips > 0, (what, moved, flips)
        print(f"  {what}: logits moved at {moved} / 40 anchors, {flips} / 600 draft ids flipped")


# --------------------------------------------------------------------------------------------
# spec 26a: the LiLiCorr form (Review Focus 1-5 each a mutation that must move the drafts or the lattice)

def _lmt():
    if "lilicorr_make_tiny" in sys.modules:
        return sys.modules["lilicorr_make_tiny"]
    spec = importlib.util.spec_from_file_location("lilicorr_make_tiny", os.path.join(_HERE, "lilicorr_make_tiny.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["lilicorr_make_tiny"] = m
    spec.loader.exec_module(m)
    return m


LIL_VOCAB = 2048        # the synthetic target's vocabulary (the mask id 990 a row); the tiny pair uses Bolt's 131072


def _lil(seed=0, vocab=LIL_VOCAB, window=16, head_scale=0.15):
    """A tiny LiLiCorr (lilicorr_make_tiny's tensors, decoded by the reference's NVFP4 reader) over a synthetic target
    embedding / head (logits ~N(0, 2.4): log-probs spread over a few nats, not one-hot)."""
    mt = _lmt()
    cd = mt.tiny_config(vocab=vocab, window=window)
    cfg = ref.config_from_dict(cd)
    t = ref._read_nvfp4_tensors(mt.make_tensors(cd, seed), cd["quantization_config"])
    g = torch.Generator().manual_seed(100 + seed)
    emb = torch.randn(vocab, cfg.hidden, generator=g).to(torch.bfloat16)
    head = (torch.randn(vocab, cfg.hidden, generator=g) * head_scale).to(torch.bfloat16)
    return ref.Drafter(cfg, t, emb, head)


def _lil_ctx(d, n, seed=1):
    g = torch.Generator().manual_seed(seed)
    taps = {i: torch.randn(n, d.cfg.hidden, generator=g) * 2.0 for i in d.cfg.target_layer_ids}
    return d.context_kv(taps, torch.arange(n)), taps


def _lil_flips(d, ctx, n, mut, anchors=range(20, 40), K=7, T=0.0):
    """(anchors whose lattice moved, draft ids flipped of len(anchors) x K) under the mutation set `mut`."""
    base = [d.draft_block(int(a) % LIL_VOCAB, p, K, ctx, T, 5 if T else None) for p, a in zip(anchors, range(n))]
    d.mut = set(mut)
    try:
        alt = [d.draft_block(int(a) % LIL_VOCAB, p, K, ctx, T, 5 if T else None) for p, a in zip(anchors, range(n))]
    finally:
        d.mut = set()
    moved = sum(int(not torch.equal(x.lattice, y.lattice)) for x, y in zip(base, alt))
    flips = sum(int((x.ids != y.ids).sum()) for x, y in zip(base, alt))
    return moved, flips


REAL_LILICORR = {   # nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr @ 449d2414 config.json, trimmed to what is read
    "architectures": ["LiLiCorrDraftModel"], "model_type": "qwen3", "block_size": 8,
    "dflash_config": {"mask_token_id": 990, "target_layer_ids": [1, 18, 35, 51, 68, 85], "use_swa": True,
                      "swa_window_size": 1024, "causal": False, "attention_sink_bias": True, "projector_type": "lilicorr",
                      "lilicorr_enabled": True, "lilicorr_candidate_topk": 8, "lilicorr_hidden_size": 1024,
                      "lilicorr_num_layers": 2, "lilicorr_num_heads": 8, "lilicorr_mlp_ratio": 2.0,
                      "lilicorr_factor_dim": 1024, "lilicorr_vector_eps": 0.0001, "lilicorr_logit_scale": 8.0,
                      "conv_kernel_size": 2, "conv_group_size": 16},
    "hidden_size": 4096, "num_hidden_layers": 5, "num_attention_heads": 32, "num_key_value_heads": 2, "head_dim": 128,
    "intermediate_size": 12288, "hidden_act": "silu", "rms_norm_eps": 1e-06, "vocab_size": 131072,
    "attention_bias": False, "rope_theta": 10000.0, "rope_scaling": None, "tie_word_embeddings": False,
    "layer_types": ["sliding_attention"] * 5, "sliding_window": 1024, "attention_sink_bias": True,
    "use_sliding_window": True, "max_window_layers": 0,
    "quantization_config": {"quant_algo": "W4A16_NVFP4", "kv_cache_quant_algo": None, "group_size": 16,
                            "quant_method": "modelopt_fp4"}}


def test_lilicorr_config():
    c = ref.config_from_dict(REAL_LILICORR)
    assert (c.form, c.version, c.block_size, c.max_k, c.mask_token_id) == ("lilicorr", 2, 8, 7, 990)
    assert c.windows == [1024] * 5 and c.causal == [False] * 5 and not c.window_inclusive and c.sinks
    assert (c.cand_topk, c.lil_hidden, c.lil_layers, c.lil_heads, c.lil_mlp, c.factor_dim) == (8, 1024, 2, 8, 2048, 1024)
    assert (c.vector_eps, c.lil_logit_scale, c.rope_theta, c.n_kv_heads, c.ffn) == (1e-4, 8.0, 10000.0, 2, 12288)
    assert c.target_layer_ids == [1, 18, 35, 51, 68, 85] and (c.conv_kernel_size, c.conv_group_size) == (2, 16)
    shapes = ref.lilicorr_head_shapes(c)
    n = sum(math.prod(s) for s in shapes.values())
    assert 36.5e6 < n < 36.9e6, n                     # the card's "37M" (spec 26 §1: ~36.7 M)
    snap = ref.find_snapshot("nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr")
    if snap:                                          # the cached config.json itself (the weights may be absent)
        with open(os.path.join(snap, "config.json")) as f:
            rc = ref.config_from_dict(json.load(f))
        assert (rc.windows, rc.cand_topk, rc.lil_mlp, rc.block_size) == (c.windows, c.cand_topk, c.lil_mlp, 8)
    bad = [("lilicorr_enabled", False, "lilicorr_enabled"), ("lilicorr_candidate_topk", 12, "power of two"),
           ("causal", True, "causal"), ("conv_kernel_size", 0, "convolutions")]
    for key, val, what in bad:
        cd = json.loads(json.dumps(REAL_LILICORR))
        cd["dflash_config"][key] = val
        try:
            ref.config_from_dict(cd)
            raise AssertionError(f"{key}={val} accepted")
        except (ValueError, NotImplementedError) as e:
            assert what in str(e), (key, str(e))
    for top, val, what in (("has_own_lm_head", True, "has_own_lm_head"), ("draft_vocab_size", 32768, "vocabulary"),
                           ("quantization_config", {"quant_method": "compressed-tensors"}, "NVFP4")):
        cd = json.loads(json.dumps(REAL_LILICORR))
        cd[top] = val
        try:
            ref.config_from_dict(cd)
            raise AssertionError(f"{top} accepted")
        except (ValueError, NotImplementedError) as e:
            assert what in str(e), (top, str(e))
    print(f"lilicorr config: window 1024 (|q - k| <= 1023), sinks, block 8 at the top level, head {n / 1e6:.2f} M "
          f"parameters; {len(bad) + 3} refusals by name")


def test_lilicorr_nvfp4_enumeration():
    """Review Focus 3: every (e2m1 code, e4m3 scale) product is exactly a bf16 and an f16; the NaN scales refused."""
    tbl = ref._e4m3_table()
    via_torch = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).double()
    fin = torch.tensor([c not in ref.E4M3_NAN for c in range(256)])
    assert torch.equal(tbl[fin], via_torch[fin]) and via_torch[~fin].isnan().all()
    n, lo, hi = 0, float("inf"), 0.0
    for code in range(16):
        for sc in range(256):
            if sc in ref.E4M3_NAN:
                w = torch.tensor([[code | (code << 4)] * 8], dtype=torch.uint8)
                s = torch.tensor([[sc]], dtype=torch.uint8)
                try:
                    ref.nvfp4_decode(w, s, 1.0, name="t")
                    raise AssertionError(f"scale 0x{sc:02x} accepted")
                except ValueError as e:
                    assert "weight_scale[0][0]" in str(e) and "NaN" in str(e)
                continue
            v = ref.e2m1_value(code) * ref.e4m3_value(sc)                 # python float = fp64, exact here
            t = torch.tensor(v, dtype=torch.float64)
            assert t.to(torch.bfloat16).double().item() == v, (code, sc, v)
            assert t.to(torch.float16).double().item() == v, (code, sc, v)
            if v != 0.0:
                lo, hi = min(lo, abs(v)), max(hi, abs(v))
            n += 1
    assert n == 16 * 254 and lo == 2.0 ** -10 and hi == 2688.0
    # the layout: k = 2i in the LOW nibble; one scale per 16 consecutive k; g kept apart
    codes = torch.arange(32) % 16
    w = (codes[0::2] | (codes[1::2] << 4)).to(torch.uint8).view(1, 16)
    s = torch.tensor([[0x38, 0x40]], dtype=torch.uint8)                 # 1.0 and 2.0
    wp, g = ref.nvfp4_decode(w, s.view(torch.float8_e4m3fn), torch.tensor(0.37, dtype=torch.float32))
    exp = torch.tensor([ref.e2m1_value(int(c)) for c in codes]) * torch.tensor([1.0] * 16 + [2.0] * 16)
    assert torch.equal(wp[0], exp.float()) and g == float(torch.tensor(0.37, dtype=torch.float32))
    try:
        ref.nvfp4_decode(w, s[:, :1], 1.0, name="t")
        raise AssertionError("a short scale tensor accepted")
    except ValueError:
        pass
    print(f"nvfp4: {n} (code, scale) products exact in bf16 and f16, |W'| in [2^-10, 2688]; 2 x 16 NaN pairs refused; "
          f"low nibble = even k, one scale per 16 k")


def test_lilicorr_g_on_the_output():
    """Review Focus 3: y = g x (W' x) (+ b) bitwise; folding g into W' is a different evaluation."""
    d = _lil()
    assert len(d.g) == 16 and all(not math.log2(v).is_integer() for v in d.g.values())
    x = torch.randn(9, 256, generator=torch.Generator().manual_seed(3))
    for name in ("fc.weight", "layers.1.mlp.down_proj.weight", "layers.0.mlp.gate_proj.weight"):
        k = d.w[name].shape[1]
        xx = x.repeat(1, k // 256 + 1)[:, :k]
        y = d._lin(xx, name)
        assert torch.equal(y, ref.linear(xx, d.w[name]) * torch.tensor(d.g[name], dtype=torch.float32)), name
        assert torch.equal(d.w[name], d.w[name].to(torch.bfloat16).float()), name      # W' bf16-exact
    ctx, _ = _lil_ctx(d, 40)
    moved, flips = _lil_flips(d, ctx, 20, {"fold_g"})
    d.mut = {"fold_g"}
    y2 = d._lin(x.repeat(1, 6), "fc.weight")
    d.mut = set()
    diff = int((y2 != d._lin(x.repeat(1, 6), "fc.weight")).sum())
    assert diff > 0 and moved > 0, (diff, moved)
    print(f"g on the output: bitwise g x (W' x); folded into W' {diff} / {y2.numel()} fc outputs differ, the lattice "
          f"moves at {moved} / 20 anchors ({flips} draft ids flip)")


def test_lilicorr_anchor():
    """Review Focus 1: the anchor is hidden_norm(fc(taps)) at p - 1, anchor_valid true; the embedding, the pre-norm row
    and the row at p - 2 each move the lattice."""
    d = _lil()
    ctx, taps = _lil_ctx(d, 40)
    p = 30
    o = d.draft_block(5, p, 7, ctx)
    assert torch.equal(o.anchor, ctx.fc[p - 1].float())
    a, valid = d.lilicorr_anchor(5, p, ctx)
    assert valid and torch.equal(a, ctx.fc[p - 1])
    for mut in ("anchor_embed", "anchor_prenorm", "anchor_p2", "anchor_invalid"):
        moved, flips = _lil_flips(d, ctx, 20, {mut})
        assert moved == 20, (mut, moved)
        print(f"  anchor mutation {mut}: the lattice moves at {moved} / 20 anchors, {flips} / 140 draft ids flip")
    # the session: commit the prompt rows, draft, commit the anchor + 3 accepted drafts, draft again
    s = ref.LiLiCorrSession(d)
    s.commit({i: taps[i][:25] for i in taps}, range(25))
    o1 = s.draft(7, 7)          # (a context projected over 25 rows vs 40: torch's fp32 GEMM is row-count dependent)
    assert torch.equal(o1.anchor, s.ctx.fc[24]) and torch.equal(o1.ids, d.draft_block(7, 25, 7, ctx).ids)
    s.commit({i: taps[i][25:29] for i in taps}, range(25, 29))
    o2 = s.draft(9, 7)
    assert int(s.ctx.pos[-1]) == 28 and torch.allclose(o2.anchor, ctx.fc[28], atol=1e-5)
    assert torch.allclose(o2.lattice, d.draft_block(9, 29, 7, ctx).lattice, atol=1e-4)
    try:
        s.commit({i: taps[i][30:31] for i in taps}, [30])
        raise AssertionError("a gap accepted")
    except ValueError:
        pass
    print("anchor: the normed context row at p - 1 (session: the newest committed row)")


def test_lilicorr_lattice_and_walk():
    """Review Focus 2: start broadcast for slot 0, pair[s - 1] for slots 1..6, scores x 8.0, the slot bias at offset 7,
    the walk over the lattice alone from previous 0, the log-probs only as features."""
    d = _lil()
    ctx, _ = _lil_ctx(d, 40)
    o = d.draft_block(11, 30, 7, ctx)
    L = o.lattice
    assert L.shape == (7, 8, 8) and L.dtype == torch.float32
    assert all(torch.equal(L[0, 0], L[0, i]) for i in range(8))           # slot 0: broadcast over the previous index
    prev, ids = 0, []
    for s in range(7):                                                    # the walk reads the lattice alone
        row = L[s, prev]
        j = int(torch.nonzero(row == row.max())[0, 0])
        assert torch.equal(o.scores[s], row) and int(o.chosen[s]) == j
        ids.append(int(o.cand_ids[s, j]))
        prev = j
    assert o.ids.tolist() == ids
    d.mut = {"no_scale"}
    L1 = d.draft_block(11, 30, 7, ctx).lattice
    d.mut = set()
    assert torch.equal(L1 * 8.0, L)                                       # x 8.0 (exact: a power of two)
    # the bias at the trained offset 7: a short lattice (S = 3, vLLM's prefix form) moves under offset S, S = 7 does not
    hm, cand, lp = o.hidden[1:], o.cand_ids, o.log_probs
    args = lambda S: (cand[None, :S], lp[None, :S], hm[None, :S], o.anchor[None], torch.tensor([True]))  # noqa: E731
    base7, base3 = d.lilicorr_lattice(*args(7)), d.lilicorr_lattice(*args(3))
    d.mut = {"rel_offset_s"}
    alt7, alt3 = d.lilicorr_lattice(*args(7)), d.lilicorr_lattice(*args(3))
    d.mut = set()
    assert torch.equal(base7, alt7) and not torch.equal(base3, alt3)
    b = d.lilicorr_slot_bias(7)
    rel, same = d.w["lilicorr.relative_slot_bias"].float(), d.w["lilicorr.same_slot_bias"].float()
    assert torch.equal(b[:, 9, 2], rel[:, 1 - 0 + 7]) and torch.equal(b[:, 3, 5], rel[:, 7] + same)
    for mut in ("start_transposed", "pair_shift", "rel_sign", "no_same_slot", "lp_in_walk", "lp_topk",
                "cross_no_anchor"):
        moved, flips = _lil_flips(d, ctx, 20, {mut})
        assert flips > 0 or (mut != "lp_in_walk" and moved > 0), (mut, moved, flips)
        print(f"  lattice mutation {mut}: the lattice moves at {moved} / 20 anchors, {flips} / 140 draft ids flip")
    # the walk's start index cannot matter (slot 0 is a broadcast): every start gives the same drafts
    for start in range(8):
        pv, got = start, []
        for s in range(7):
            j = int(torch.argmax(L[s, pv]))
            got.append(int(o.cand_ids[s, j]))
            pv = j
        assert got == ids
    print("lattice: slot 0 = start broadcast, x 8.0, offset 7 (S = 3 moves under offset S), the walk = the lattice's "
          "argmax chain from previous 0")


def test_lilicorr_candidates():
    """The candidates: top 8 by logit (ties to the lower id) with full-vocabulary log-probs; torch.topk's tie order
    recorded (24a's lesson)."""
    lg = torch.randn(5, 1000, generator=torch.Generator().manual_seed(4))
    lg[2, [10, 400, 900]] = 9.0                         # an exact tie at the top
    lg[3, [5, 6]] = lg[3].max() + 1.0
    cand, lp = ref.lilicorr_candidates(lg, 8)
    assert cand[2, :3].tolist() == [10, 400, 900] and cand[3, :2].tolist() == [5, 6]
    assert torch.allclose(lp, torch.log_softmax(lg, -1).gather(1, cand), atol=1e-6)
    assert torch.equal(lp, lg.gather(1, cand) - torch.logsumexp(lg, -1, keepdim=True))
    tv, ti = torch.topk(lg, 8, dim=-1)
    agree = sum(int(torch.equal(ti[r], cand[r])) for r in range(5))
    print(f"candidates: top 8 descending, ties to the lower id, lp = logit - lse(all); torch.topk agrees on {agree} / 5 "
          f"rows (2 with planted exact ties) - recorded, not required")


def test_lilicorr_sinks():
    """Review Focus 4: a per-head logit in the denominator, unscaled, no value row; dropping it moves the drafts."""
    d = _lil()
    s = torch.randn(4, 3, 6, generator=torch.Generator().manual_seed(2))
    pr = d._softmax_sink(0, s, 3)
    sink = d.w["layers.0.self_attn.attention_sink_bias"].float()
    exp = torch.exp(s) / (torch.exp(s).sum(-1, keepdim=True) + torch.exp(sink)[:, None, None])
    assert torch.allclose(pr, exp, rtol=1e-6) and (pr.sum(-1) < 1).all()
    ctx, _ = _lil_ctx(d, 40)
    for mut in ("no_sink", "sink_scaled"):
        moved, flips = _lil_flips(d, ctx, 20, {mut})
        assert moved == 20 and flips > 0, (mut, moved, flips)
        print(f"  sink mutation {mut}: the lattice moves at {moved} / 20 anchors, {flips} / 140 draft ids flip")
    print("sinks: softmax(cat(scores, sink))[:-1], the sink unscaled and valueless")


def test_lilicorr_window_edge():
    """The window |q - k| <= 15 at window 16 (vLLM's; spec 26 Decision 11), rows 0 and 7 at p = 15, 16, 17."""
    d = _lil()
    ctx, taps = _lil_ctx(d, 60)
    for p in (15, 16, 17):
        kpos = torch.arange(p)
        qpos = torch.arange(p, p + 8)
        m = d.attention_mask(0, qpos, kpos)
        assert m[0].nonzero().flatten().tolist() == [k for k in range(p) if p - k <= 15]
        assert m[7].nonzero().flatten().tolist() == [k for k in range(p) if p + 7 - k <= 15]
        # end to end: positions older than p - 15 are never read; p - 15 is
        # end to end: positions older than p - 15 are never read (masked keys add exact zeros - equal to rounding), p - 15
        # is
        full = d.draft_block(3, p, 7, ctx).hidden
        lo = max(0, p - 15)
        keep = ref.ContextKV(pos=ctx.pos[lo:], k=[k[lo:] for k in ctx.k], v=[v[lo:] for v in ctx.v], fc=ctx.fc[lo:])
        assert (d.draft_block(3, p, 7, keep).hidden - full).abs().max() <= 1e-5, p
        cut = ref.ContextKV(pos=ctx.pos[lo + 1:], k=[k[lo + 1:] for k in ctx.k], v=[v[lo + 1:] for v in ctx.v],
                            fc=ctx.fc[lo + 1:])
        assert (d.draft_block(3, p, 7, cut).hidden - full).abs().max() > 1e-3, p
    base = d.draft_block(3, 17, 7, ctx).hidden
    old = list(d.cfg.windows)
    d.cfg.windows = [17] * d.cfg.n_layers           # the inclusive reading (|q - k| <= 16): row 0 at 17 also sees 1
    try:
        alt = d.draft_block(3, 17, 7, ctx).hidden
    finally:
        d.cfg.windows = old
    assert (alt - base).abs().max() > 1e-3
    print("window: row 0 sees p - 15 .. p - 1, row 7 sees p - 8 .. p - 1 at window 16; older rows unread; the "
          "inclusive reading moves the block")


def test_lilicorr_k_prefix():
    """K = 1..7 from one block are the first K of the K = 7 walk at T 0 (the block keeps 8 rows, the head 7 slots)."""
    d = _lil()
    ctx, _ = _lil_ctx(d, 40)
    for p in (20, 27, 39):
        o7 = d.draft_block(p, p, 7, ctx)
        for K in range(1, 8):
            o = d.draft_block(p, p, K, ctx)
            assert torch.equal(o.ids, o7.ids[:K]) and torch.equal(o.lattice, o7.lattice)
            assert torch.equal(o.scores, o7.scores[:K]) and o.q.shape == (K, LIL_VOCAB)
    print("K = 1..7: prefixes of the K = 7 walk (one 8-row block, the lattice over 7 slots)")


def test_lilicorr_sampled():
    """The sampled walk: reproducible from its seed, keyed like vLLM 3901d56800 (seed ^ (slot << 32), position p + s,
    the candidate id); q = softmax(realized / T) on the 8 candidates."""
    d = _lil()
    ctx, _ = _lil_ctx(d, 40)
    o1 = d.draft_block(3, 30, 7, ctx, 1.0, 1234)
    o2 = d.draft_block(3, 30, 7, ctx, 1.0, 1234)
    assert torch.equal(o1.ids, o2.ids) and torch.equal(o1.chosen, o2.chosen)
    prev, ids = 0, []
    for s in range(7):
        row = o1.lattice[s, prev]
        u = ref.murmur3_uniform32(1234 ^ (s << 32), 30 + s + ref.DRAFT_NOISE_SALT, o1.cand_ids[s])
        j = int(torch.argmax(row.double() / 1.0 - torch.log(-torch.log1p(-u))))
        ids.append(int(o1.cand_ids[s, j]))
        prev = j
    assert o1.ids.tolist() == ids
    qs = torch.gather(o1.q, 1, o1.cand_ids[:7])
    assert torch.allclose(qs, torch.softmax(o1.scores.double() / 1.0, -1).float(), atol=1e-7)
    assert torch.allclose(o1.q.sum(-1), torch.ones(7), atol=1e-6)
    seeds = [d.draft_block(3, 30, 7, ctx, 1.0, s).ids for s in range(12)]
    assert len({tuple(x.tolist()) for x in seeds}) > 1
    moved, flips = _lil_flips(d, ctx, 20, {"no_step_seed"}, T=1.0)
    assert flips > 0
    print(f"sampled: seed-reproducible, step-seeded (seed ^ (s << 32): without it {flips} / 140 ids move), q = "
          f"softmax(realized / T) on the candidates")


def test_lilicorr_taps_order():
    """Review Focus 5 (the API half): the taps concatenate in target_layer_ids order before fc; another order moves
    every context row (test_dflash_accept.py's tiny pair checks the taps ARE the residual after those blocks)."""
    d = _lil()
    ctx, taps = _lil_ctx(d, 40)
    ids = d.cfg.target_layer_ids
    rev = {a: taps[b] for a, b in zip(ids, reversed(ids))}
    ctx2 = d.context_kv(rev, torch.arange(40))
    assert not torch.equal(ctx.fc, ctx2.fc)
    cat = torch.cat([taps[i] for i in ids], -1)
    assert torch.equal(ctx.fc_raw, d._lin(cat, "fc.weight"))
    o, o2 = d.draft_block(5, 30, 7, ctx), d.draft_block(5, 30, 7, ctx2)
    assert not torch.equal(o.lattice, o2.lattice)
    print(f"taps: cat in {ids} order -> fc ({cat.shape[1]} = {len(ids)} x {d.cfg.hidden}); reversed moves the lattice")


def test_lilicorr_load_from_files():
    """lilicorr_make_tiny.py's checkpoint read back (NVFP4 decoded) = the in-memory drafter; stray quantised tensors and
    NaN scales refused naming them."""
    mt = _lmt()
    from safetensors.torch import load_file, save_file
    with tempfile.TemporaryDirectory() as root:
        dd = mt.make(os.path.join(root, "lil"), seed=0, vocab=LIL_VOCAB)
        with open(os.path.join(dd, "config.json")) as f:
            cd = json.load(f)
        raw = load_file(os.path.join(dd, "model.safetensors"))
        nq = sum(1 for n in raw if n.endswith(".weight_scale_2"))
        assert nq == len(mt.nvfp4_names(cd)) == 16 and len(raw) == 16 * 3 + 37 + 13 * 2 + 2
        mem = _lil()
        g = torch.Generator().manual_seed(100)
        emb = torch.randn(LIL_VOCAB, 256, generator=g).to(torch.bfloat16)
        head = (torch.randn(LIL_VOCAB, 256, generator=g) * 0.15).to(torch.bfloat16)
        d = ref.load_drafter(dd, embed=emb, lm_head=head)
        assert set(d.w) == set(mem.w) and all(torch.equal(d.w[n], mem.w[n]) for n in d.w) and d.g == mem.g
        bad = dict(raw)
        sc = bad["fc.weight_scale"].view(torch.uint8).clone()
        sc[3, 5] = 0x7F
        bad["fc.weight_scale"] = sc.view(torch.float8_e4m3fn)
        save_file(bad, os.path.join(dd, "model.safetensors"))
        try:
            ref.load_drafter(dd, embed=emb, lm_head=head)
            raise AssertionError("a NaN scale accepted")
        except ValueError as e:
            assert "fc: weight_scale[3][5]" in str(e), str(e)
        bad = dict(raw)
        bad["layers.0.self_attn.q_proj.weight_scale"] = raw["fc.weight_scale"].clone()
        save_file(bad, os.path.join(dd, "model.safetensors"))
        try:
            ref.load_drafter(dd, embed=emb, lm_head=head)
            raise AssertionError("a stray scale accepted")
        except ValueError as e:
            assert "q_proj.weight_scale" in str(e), str(e)
    print(f"load: {len(raw)} tensors ({nq} NVFP4 linears x 3 + {len(raw) - 3 * nq} BF16), decoded = in memory; NaN / "
          "stray scales refused")


def test_lilicorr_real_checkpoint():
    """nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr @ 449d2414 from the HF cache (SKIP without its weights): the
    header's 179 tensors (25 NVFP4 linears x 3 + 104 BF16), every NVFP4 linear decoded exactly (nvfp4_decode asserts W'
    bf16-exact; no NaN scale), the parameters (the card's 1.113 B), the global scales, the three formats' bytes and their
    weight error against g x W' (fp64), and one draft on random taps over a random stand-in target embedding / head
    (Bolt's 41.9 GB checkpoint is the box's) - finite, slot 0 a broadcast, the drafts among the candidates."""
    import time
    snap = ref.find_snapshot("nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr", ("config.json", "model.safetensors"))
    if snap is None:
        _skip("the LiLiCorr drafter's weights are not in the HF cache (hf download nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-"
              "A9B-24GB-LiLiCorr --revision 449d24140160b8e51871bfc0f0a7338abf640aa1)")
    cfg = ref.read_config(snap)
    wm = ref._weight_map(snap)
    raw = ref._read(wm, list(wm))
    kinds = {}
    for n, v in raw.items():
        kinds[str(v.dtype)] = kinds.get(str(v.dtype), 0) + 1
    nq = sum(1 for n in raw if n.endswith(".weight_scale_2"))
    assert len(raw) == 179 and nq == 25 and kinds.get("torch.bfloat16") == 104, (len(raw), nq, kinds)
    zero_blocks = sub_blocks = blocks = 0
    codes = torch.zeros(16, dtype=torch.long)
    for n in raw:
        if n.endswith(".weight_scale"):
            s = raw[n].view(torch.uint8)
            blocks += s.numel()
            zero_blocks += int((s == 0).sum())
            sub_blocks += int(((s & 0x78) == 0).sum()) - int((s == 0).sum())
            w = raw[n[: -len("_scale")]]
            codes += torch.bincount(torch.cat([(w & 0xF).flatten(), (w >> 4).flatten()]).long(), minlength=16)
    t0 = time.time()
    t = ref.read_drafter_tensors(snap, cfg)
    t_dec = time.time() - t0
    gs = {n[: -len(ref.G_SUFFIX)]: float(v) for n, v in t.items() if n.endswith(ref.G_SUFFIX)}
    assert len(gs) == 25 and all(math.isfinite(g) and g > 0 for g in gs.values())
    params = sum(v.numel() for n, v in t.items() if not n.endswith(ref.G_SUFFIX))
    assert 1.112e9 < params < 1.114e9, params                          # the card's "1.113B draft parameters"
    head_params = sum(v.numel() for n, v in t.items() if n.startswith("lilicorr.") and not n.endswith(ref.G_SUFFIX))
    pow2 = sum(1 for g in gs.values() if math.log2(g).is_integer())
    # the formats (spec 26 §3): bytes of the backbone + fc, and the weight error against g x W' (fp64)
    lin = [n for n in t if n.endswith(".weight") and not n.startswith("lilicorr.") and t[n].dim() == 2]
    b_exact = sum(t[n].numel() * 2 for n in lin)
    b_int8 = sum(t[n].numel() + t[n].shape[0] * 4 for n in lin)
    b_int4 = sum(t[n].numel() // 2 + t[n].numel() // 64 * 2 for n in lin)
    b_head = sum(v.numel() * 2 for n, v in t.items() if n.startswith("lilicorr.") and not n.endswith(ref.G_SUFFIX))
    da = importlib.util.spec_from_file_location("dflash_accept_rt", os.path.join(_HERE, "dflash_accept.py"))
    DA = importlib.util.module_from_spec(da)
    da.loader.exec_module(DA)
    DA._load_deps()
    e8, e4 = [], []
    for n in lin:
        if n not in gs and n[: -len(".weight")] not in gs:
            continue
        g = gs[n[: -len(".weight")]]
        w64 = t[n].double() * g
        q, st = DA.int8_rows_g(t[n].float(), g)
        e8.append(float((q.double() * st.double()[:, None] - w64).norm() / w64.norm()))
        q4, s16 = DA.rtn_int4_g64_f64(w64)
        N, K = q4.shape
        d4 = (q4.double().reshape(N, K // 64, 64) - 8.0) * s16.double()[..., None]
        e4.append(float((d4.reshape(N, K) - w64).norm() / w64.norm()))
    e8.sort()
    e4.sort()
    # one draft over a random stand-in target (the drafter's own arithmetic on random taps)
    gen = torch.Generator().manual_seed(26)
    emb = (torch.randn(cfg.vocab, cfg.hidden, generator=gen, dtype=torch.bfloat16) * 0.02).to(torch.bfloat16)
    head = (torch.randn(cfg.vocab, cfg.hidden, generator=gen, dtype=torch.bfloat16) * 0.02).to(torch.bfloat16)
    d = ref.Drafter(cfg, t, emb, head)
    del raw, t
    taps = {i: torch.randn(40, cfg.hidden, generator=gen) for i in cfg.target_layer_ids}
    ctx = d.context_kv(taps, torch.arange(40))
    t1 = time.time()
    o = d.draft_block(11, 40, 7, ctx)
    t_draft = time.time() - t1
    assert torch.isfinite(o.lattice).all() and torch.isfinite(o.hidden).all()
    assert all(torch.equal(o.lattice[0, 0], o.lattice[0, i]) for i in range(8))
    assert all(int(o.ids[s]) in o.cand_ids[s].tolist() for s in range(7))
    print(f"real LiLiCorr: 179 tensors ({kinds}), 25 NVFP4 linears decoded exactly in {t_dec:.0f} s; {params:,} parameters "
          f"({head_params:,} in the head); g in [{min(gs.values()):.3g}, {max(gs.values()):.3g}], {pow2} a power of two; "
          f"scale blocks {blocks:,}: {zero_blocks:,} zero, {sub_blocks:,} subnormal; e2m1 codes {codes.tolist()}")
    print(f"  formats (backbone + fc, {len(lin)} linears): exact {b_exact / 1e9:.3f} GB, int8 {b_int8 / 1e9:.3f}, int4 g64 "
          f"{b_int4 / 1e9:.3f}; the head {b_head / 1e9:.3f} GB; weight error vs g x W' over the {len(e8)} NVFP4 backbone "
          f"linears: int8 {e8[0]:.4f} / {e8[len(e8) // 2]:.4f} / {e8[-1]:.4f}, int4 g64 {e4[0]:.4f} / {e4[len(e4) // 2]:.4f} / "
          f"{e4[-1]:.4f} (min / median / max relative)")
    print(f"  one draft on random taps (a random stand-in target): {t_draft:.2f} s on this CPU (float32), ids {o.ids.tolist()}")


TESTS = [test_real_configs, test_shapes_and_determinism, test_tap_offset_at_api, test_context_kv_and_rope,
         test_context_window, test_full_attention_layer, test_block_attention_causality, test_conv_lookback, test_v2_layer_wiring,
         test_short_K, test_score_edges_and_anchor, test_walk_tie_rule, test_gumbel, test_d2t,
         test_mask_embedding_and_scale, test_dequant_pack_quantized, test_load_from_files, test_real_checkpoint,
         test_muse_config, test_muse_equals_transformers, test_muse_window_edge, test_muse_review_focus_4,
         test_lilicorr_config, test_lilicorr_nvfp4_enumeration, test_lilicorr_g_on_the_output, test_lilicorr_anchor,
         test_lilicorr_lattice_and_walk, test_lilicorr_candidates, test_lilicorr_sinks, test_lilicorr_window_edge,
         test_lilicorr_k_prefix, test_lilicorr_sampled, test_lilicorr_taps_order, test_lilicorr_load_from_files,
         test_lilicorr_real_checkpoint]


def main() -> None:
    torch.manual_seed(0)
    only = sys.argv[1:]
    skipped = 0
    for t in TESTS:
        if only and t.__name__ not in only:
            continue
        try:
            t()
        except Skip as s:
            skipped += 1
            print(f"SKIP {t.__name__}: {s}")
    print(f"ok ({skipped} skipped)")


if __name__ == "__main__":
    main()
