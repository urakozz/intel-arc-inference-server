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


TESTS = [test_real_configs, test_shapes_and_determinism, test_tap_offset_at_api, test_context_kv_and_rope,
         test_context_window, test_full_attention_layer, test_block_attention_causality, test_conv_lookback, test_v2_layer_wiring,
         test_short_K, test_score_edges_and_anchor, test_walk_tie_rule, test_gumbel, test_d2t,
         test_mask_embedding_and_scale, test_dequant_pack_quantized, test_load_from_files, test_real_checkpoint]


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
