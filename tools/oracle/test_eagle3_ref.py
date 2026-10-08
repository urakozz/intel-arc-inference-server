#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Checks for eagle3_ref.py (the EAGLE3 K2 P0). CPU, tiny random drafters, seconds.

    test_eagle3_ref.py [test_name ...]

- the drafter against an INDEPENDENT re-implementation of the pinned semantics: transformers'
  own LlamaAttention / LlamaMLP / LlamaRMSNorm / LlamaRotaryEmbedding (eager, a DynamicCache)
  wired as speculators 0.8.0's Eagle3DraftModel.forward + Eagle3FirstLayerMixin.forward wire
  them (the training TTT loop over the whole sequence with its block masks: step 0 causal +
  sliding window, then one diagonal block per step), feeding the drafter's OWN greedy ids
  back instead of the teacher's - so row r's step j IS anchor r + 1's draft j. Both window
  readings ("anchor" = training's mask, "query" = vLLM's), with a window that binds;
- propose_batch == propose, anchor by anchor; vLLM's position convention (s - 1) drafts the
  same ids (S5);
- the checkpoint reader (speculators names, a `model.` prefix, `midlayer.`), refusals;
- d2t / t2d: vocab_mapping.py's construction round-trips, a direct-id d2t and a short t2d are
  refused;
- the int8 / int4 g64 RTN arms against plain loops of their C++ rules, and the derived bytes.
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


def _load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


E3 = _load("eagle3_ref")
REAL_CONFIG = os.path.join(_HERE, "third_party", "eagle3_k2", "config.json")
VOCAB, DRAFT_VOCAB, HID, TH = 100, 40, 64, 64


def tiny_dict(window=6, **over) -> dict:
    with open(REAL_CONFIG, encoding="utf-8") as f:
        d = json.load(f)
    d["transformer_layer_config"].update(hidden_size=HID, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                                         intermediate_size=128, vocab_size=VOCAB, sliding_window=window,
                                         max_position_embeddings=512)
    if window is None:
        d["transformer_layer_config"]["layer_types"] = ["full_attention"]
    d.update(draft_vocab_size=DRAFT_VOCAB, eagle_aux_hidden_state_layer_ids=[1, 2, 3])
    d.update(over)
    return d


def vocab_maps(seed=0):
    """speculators/train/vocab_mapping.py:64-96, re-written: the draft_vocab most frequent ids
    (ties to the lower id), sorted; d2t = selected - arange; t2d marks them."""
    g = torch.Generator().manual_seed(seed)
    freq = {i: int(c) for i, c in enumerate(torch.randint(0, 50, (VOCAB,), generator=g).tolist()) if c > 0}
    sel = sorted(freq, key=lambda t: (-freq[t], t))[:DRAFT_VOCAB]
    sel.sort()
    d2t = torch.tensor(sel, dtype=torch.long) - torch.arange(DRAFT_VOCAB)
    t2d = torch.zeros(VOCAB, dtype=torch.bool)
    t2d[sel] = True
    return d2t, t2d, sel


def random_tensors(c, seed=0) -> dict:
    g = torch.Generator().manual_seed(seed)
    t = {}
    for n, s in E3.expected_shapes(c).items():
        if n in ("d2t", "t2d"):
            continue
        if n.endswith("norm.weight"):
            t[n] = (1.0 + 0.3 * torch.randn(s, generator=g)).to(torch.bfloat16)
        else:
            t[n] = (torch.randn(s, generator=g) / math.sqrt(s[1])).to(torch.bfloat16)
    t["embed_tokens.weight"] = torch.randn(VOCAB, c.target_hidden, generator=g).to(torch.bfloat16)
    t["d2t"], t["t2d"], _ = vocab_maps(seed)
    return t


def make(seed=0, **kw):
    d = tiny_dict(**kw)
    c = E3.config_from_dict(d)
    return d, c, random_tensors(c, seed)


def seq(c, n, seed=1):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, VOCAB, (n,), generator=g)
    aux = (2.0 * torch.randn(n, c.n_aux * c.target_hidden, generator=g)).to(torch.bfloat16)
    return ids, aux


# --------------------------------------------------------------------------------------------
# the independent re-implementation (transformers modules, speculators' wiring)

def hf_drafts(d: dict, t: dict, ids, aux, K: int, window: str) -> torch.Tensor:
    """speculators' Eagle3DraftModel.forward (core.py:193-357) with the drafter's own greedy ids
    fed back. Rows r = 0..N-2: token ids[r + 1], aux[r], position r + 1 (data.py's shift).
    Returns [N - 1, K] target ids: row r's step j."""
    from transformers import DynamicCache, LlamaConfig
    from transformers.models.llama import modeling_llama as m
    tl = {k: v for k, v in d["transformer_layer_config"].items() if k != "model_type"}
    lc = LlamaConfig(**tl)
    lc._attn_implementation = "eager"
    H, W = lc.hidden_size, lc.sliding_window if lc.layer_types[0] == "sliding_attention" else None
    attn = m.LlamaAttention(lc, layer_idx=0)
    # Eagle3FirstLayerMixin._patch_eagle3_projections (model_definitions.py:38-52)
    attn.q_proj = torch.nn.Linear(2 * H, lc.num_attention_heads * lc.head_dim, bias=False)
    attn.k_proj = torch.nn.Linear(2 * H, lc.num_key_value_heads * lc.head_dim, bias=False)
    attn.v_proj = torch.nn.Linear(2 * H, lc.num_key_value_heads * lc.head_dim, bias=False)
    mlp = m.LlamaMLP(lc)
    norm = lambda n: m.LlamaRMSNorm(n, eps=lc.rms_norm_eps)  # noqa: E731
    in_ln, hid_ln, post_ln, fin = norm(H), norm(H), norm(H), norm(H)
    input_norm = norm(t["fc.weight"].shape[1])
    fc = torch.nn.Linear(t["fc.weight"].shape[1], H, bias=False)
    lm_head = torch.nn.Linear(H, t["lm_head.weight"].shape[0], bias=False)
    rot = m.LlamaRotaryEmbedding(lc)
    L = "layers.0."
    with torch.no_grad():
        for mod, name in ((attn.q_proj, L + "self_attn.q_proj"), (attn.k_proj, L + "self_attn.k_proj"),
                          (attn.v_proj, L + "self_attn.v_proj"), (attn.o_proj, L + "self_attn.o_proj"),
                          (mlp.gate_proj, L + "mlp.gate_proj"), (mlp.up_proj, L + "mlp.up_proj"),
                          (mlp.down_proj, L + "mlp.down_proj"), (in_ln, L + "input_layernorm"),
                          (hid_ln, L + "hidden_norm"), (post_ln, L + "post_attention_layernorm"), (fin, "norm"),
                          (input_norm, "input_norm"), (fc, "fc"), (lm_head, "lm_head")):
            mod.weight.copy_(t[name + ".weight"].float())
    embed = t["embed_tokens.weight"].float()
    d2t = t["d2t"].long()
    N = len(ids)
    T = N - 1
    input_ids = torch.as_tensor(ids[1:], dtype=torch.long)
    pos = torch.arange(1, N)[None]
    hs = fc(input_norm(aux[:-1].float()))[None]                              # core.py:236-244
    q, kv = torch.arange(T)[:, None], torch.arange(T)[None, :]
    cache = DynamicCache()
    out = torch.empty(T, K, dtype=torch.long)
    for j in range(K):
        # the block mask at step j: [step-0 block | diagonal x j] (attention.py:16-34, 115-132)
        ref = q + (j if window == "query" else 0)
        blk0 = (kv <= q) & ((ref - kv <= W - 1) if W is not None else True)
        vis = torch.cat([blk0] + [kv == q] * j, 1)
        mask = torch.zeros(vis.shape).masked_fill(~vis, float("-inf"))[None, None]
        emb = embed[input_ids][None]
        # Eagle3FirstLayerMixin.forward (model_definitions.py:73-110), norm_before_residual
        e, h = in_ln(emb), hid_ln(hs)
        residual = h if d["norm_before_residual"] else hs
        x = torch.cat([e, h], -1)
        pe = rot(x, pos)
        a, _ = attn(hidden_states=x, position_embeddings=pe, attention_mask=mask, past_key_values=cache)
        h2 = residual + a
        h2 = h2 + mlp(post_ln(h2))
        o = fin(h2)                                                            # core.py:302-306
        hs = o if d["norm_output"] else h2
        di = lm_head(o)[0].argmax(-1)
        input_ids = di + d2t[di]                                               # the own drafts, as target ids
        out[:, j] = input_ids
        pos = pos + 1                                                          # core.py:357
    return out


# --------------------------------------------------------------------------------------------
# tests

def test_real_config():
    c = E3.config_from_dict(json.load(open(REAL_CONFIG, encoding="utf-8")))
    assert (c.hidden, c.heads, c.kv_heads, c.head_dim, c.inter) == (2560, 32, 8, 128, 6144)
    assert (c.window, c.rope_theta, c.eps, c.vocab, c.draft_vocab) == (2048, 1e4, 1e-6, 250624, 32768)
    assert c.aux_ids == [2, 24, 45] and c.norm_before_fc and not c.fc_norm
    assert c.norm_before_residual and c.norm_output and c.speculative_tokens == 3
    sh = E3.expected_shapes(c)
    assert sh["fc.weight"] == (2560, 7680) and sh["input_norm.weight"] == (7680,)
    assert sh["layers.0.self_attn.q_proj.weight"] == (4096, 5120) and sh["layers.0.self_attn.o_proj.weight"] == (2560, 4096)
    assert sh["lm_head.weight"] == (32768, 2560) and sh["t2d"] == (250624,)
    print(f"  real config: {len(sh)} tensors + embed_tokens; window {c.window}, aux {c.aux_ids}, verifier {c.verifier}")


def test_against_independent():
    for window_cfg, mode, seed in ((6, "anchor", 0), (6, "query", 0), (64, "query", 3), (None, "query", 4)):
        d, c, t = make(seed=seed, window=window_cfg)
        dr = E3.Drafter(c, t)
        n, K = 20, 4
        ids, aux = seq(c, n, seed=seed + 1)
        want = hf_drafts(d, t, ids, aux, K, mode)
        ctx = dr.context(ids, aux, 0)
        got = torch.stack([dr.propose(ctx, p, K, window=mode) for p in range(1, n)])
        assert torch.equal(got, want), (window_cfg, mode, (got != want).nonzero()[:5].tolist())
        print(f"  window {window_cfg}, {mode}: {got.numel()} drafts == transformers' modules in speculators' "
              f"TTT wiring (rows 1..{n - 1}, K {K}); distinct ids {len(set(got.flatten().tolist()))}")
    # the window option is not vacuous: a binding window drafts differently somewhere
    d, c, t = make(seed=0, window=6)
    dr = E3.Drafter(c, t)
    ids, aux = seq(c, 20, seed=1)
    ctx = dr.context(ids, aux, 0)
    qa = torch.stack([dr.propose(ctx, p, 4, window="query") for p in range(1, 20)])
    an = torch.stack([dr.propose(ctx, p, 4, window="anchor") for p in range(1, 20)])
    assert not torch.equal(qa, an)
    assert torch.equal(qa[:, 0], an[:, 0])           # step 0 is the same row either way
    print(f"  query vs anchor window at W = 6: {int((qa != an).sum())} of {qa.numel()} drafts differ (steps >= 1 only)")


def test_batch_equals_propose():
    for window_cfg, mode in ((6, "query"), (6, "anchor"), (None, "query")):
        _, c, t = make(seed=5, window=window_cfg)
        dr = E3.Drafter(c, t)
        ids, aux = seq(c, 30, seed=6)
        ctx = dr.context(ids, aux[2:], 2)
        anchors = [3, 4, 9, 15, 22, 29]
        b = dr.propose_batch(ctx, anchors, 5, window=mode)
        for i, p in enumerate(anchors):
            r = dr.propose(ctx, p, 5, window=mode)
            assert torch.equal(b[i], r), (mode, p, b[i].tolist(), r.tolist())
        assert torch.equal(dr.propose_batch(ctx, anchors, 1)[:, 0], b[:, 0])
    print("  propose_batch == propose anchor by anchor (binding window both readings, full attention), K = 5")


def test_context_chunking_and_aux_from():
    _, c, t = make(seed=7, window=6)
    dr = E3.Drafter(c, t)
    ids, aux = seq(c, 40, seed=8)
    a = dr.context(ids, aux, 0, chunk=256)
    b = dr.context(ids, aux, 0, chunk=5)
    assert torch.allclose(a.hid, b.hid, atol=1e-5) and torch.equal(a.d0, b.d0)
    # a later aux_from with the window still covered gives the same rows
    cut = dr.context(ids, aux[10:], 10)
    i = cut.row(30)
    assert torch.allclose(cut.hid[i], a.hid[a.row(30)], atol=1e-5) and int(cut.d0[i]) == int(a.d0[a.row(30)])
    assert torch.equal(dr.propose(cut, 30, 4), dr.propose(a, 30, 4))
    try:
        dr.context(ids, aux[10:], 10, row_from=5)
        raise AssertionError("row_from before the aux was accepted")
    except ValueError:
        pass
    print("  context: query chunking invariant; aux_from 10 reproduces rows >= 10 + window")


def test_position_convention():
    _, c, t = make(seed=9, window=6)
    a = E3.Drafter(c, t)
    b = E3.Drafter(c, t, pos_offset=-1)            # vLLM: the target positions, s - 1 (S5)
    ids, aux = seq(c, 24, seed=10)
    ca, cb = a.context(ids, aux, 0), b.context(ids, aux, 0)
    assert torch.allclose(ca.hid, cb.hid, atol=1e-4), float((ca.hid - cb.hid).abs().max())
    pa = a.propose_batch(ca, list(range(1, 24)), 4)
    pb = b.propose_batch(cb, list(range(1, 24)), 4)
    assert torch.equal(pa, pb)
    print(f"  RoPE at s (training) vs s - 1 (vLLM): same {pa.numel()} drafts, hidden within "
          f"{float((ca.hid - cb.hid).abs().max()):.1e}")


def test_checkpoint_reader():
    from safetensors.torch import save_file
    d, c, t = make(seed=11)
    with tempfile.TemporaryDirectory() as tmp:
        names = {}
        for k, v in t.items():
            k2 = k.replace("layers.0.", "midlayer.") if k.startswith("layers.0.mlp") else k
            k2 = "model." + k2 if k2.startswith("fc") else k2
            names[k2] = v.contiguous()
        save_file(names, os.path.join(tmp, "model.safetensors"))
        with open(os.path.join(tmp, "config.json"), "w") as f:
            json.dump(d, f)
        dr = E3.load_drafter(tmp)
        ref = E3.Drafter(c, t)
        ids, aux = seq(c, 16, seed=12)
        x, y = dr.context(ids, aux, 0), ref.context(ids, aux, 0)
        assert torch.equal(x.hid, y.hid) and torch.equal(x.d0, y.d0)
        assert dr.extra == []
        # an embedding-less drafter takes K2's
        del names["embed_tokens.weight"]
        save_file(names, os.path.join(tmp, "model.safetensors"))
        try:
            E3.load_drafter(tmp)
            raise AssertionError("an embedding-less drafter loaded without one")
        except ValueError:
            pass
        assert torch.equal(E3.load_drafter(tmp, embed=t["embed_tokens.weight"]).context(ids, aux, 0).d0, y.d0)
        names.pop("layers.0.hidden_norm.weight")
        save_file(names, os.path.join(tmp, "model.safetensors"))
        try:
            E3.load_drafter(tmp, embed=t["embed_tokens.weight"])
            raise AssertionError("a missing tensor was accepted")
        except ValueError as e:
            assert "hidden_norm" in str(e)
    for bad in ({"num_hidden_layers": 2}, {"model_type": "qwen3"}):
        dd = tiny_dict()
        dd["transformer_layer_config"].update(bad)
        try:
            E3.config_from_dict(dd)
            raise AssertionError(f"{bad} accepted")
        except NotImplementedError:
            pass
    print("  reader: speculators names, model. prefix, midlayer. -> layers.0.; refuses a missing tensor, "
          "2 layers, qwen3")


def test_vocab_maps():
    d2t, t2d, sel = vocab_maps(3)
    tgt = E3.check_vocab_maps(d2t, t2d)
    assert tgt.tolist() == sel
    for i in range(DRAFT_VOCAB):                     # round trip: draft -> target -> t2d rank -> draft
        assert bool(t2d[int(tgt[i])]) and int(t2d[: int(tgt[i])].sum()) == i
    for bad, why in ((torch.tensor(sel), "d2t holding target ids directly"),
                     (d2t.flip(0), "a permuted d2t")):
        try:
            E3.check_vocab_maps(bad, t2d)
            raise AssertionError(why + " accepted")
        except ValueError:
            pass
    short = t2d.clone()
    short[sel[5]] = False
    try:
        E3.check_vocab_maps(d2t, short)
        raise AssertionError("t2d missing an id accepted")
    except ValueError:
        pass
    # greedy maps to targets and ties go to the lower draft id
    lg = torch.zeros(2, DRAFT_VOCAB)
    lg[0, 7] = lg[0, 3] = 1.0
    lg[1, DRAFT_VOCAB - 1] = 2.0
    g = E3.greedy_ids(lg)
    assert g.tolist() == [3, DRAFT_VOCAB - 1]
    print(f"  d2t: target = draft + d2t[draft] round-trips {DRAFT_VOCAB} ids; direct ids, a permutation and "
          f"a short t2d refused; ties -> lower draft id")


def _rtn4_loop(w):
    """src/loader/rtn.h, scalar, then dequantised."""
    N, K = w.shape
    out = torch.empty(N, K)
    for n in range(N):
        for g0 in range(0, K, 64):
            vals = [float(x) for x in w[n, g0:g0 + 64].float()]
            amax = max(abs(v) for v in vals)
            s = float(torch.tensor(2.0 * amax / 15.0, dtype=torch.float32).to(torch.float16).float())
            for k, v in enumerate(vals):
                q = (float(torch.round(torch.tensor(v, dtype=torch.float32) / torch.tensor(s, dtype=torch.float32))) + 8.0) \
                    if s > 0 else 8.0
                q = min(max(q, 0.0), 15.0)
                out[n, g0 + k] = (q - 8.0) * s
    return out


def test_rtn_arms():
    g = torch.Generator().manual_seed(13)
    w = torch.randn(6, 128, generator=g).to(torch.bfloat16)
    w[1, :64] = 0                                         # an all-zero group
    w[2, 0] = 7.0
    w[2, 1:64] = 0.5                                      # 0.5 / (14/15) = 0.5357.. and ties near .5
    w4 = E3.rtn_int4_g64(w)
    assert torch.equal(w4, _rtn4_loop(w)), float((w4 - _rtn4_loop(w)).abs().max())
    assert bool((w4[1, :64] == 0).all())
    DA = _load("dflash_accept")
    DA._load_deps()
    q, s = DA.rtn_int8_rows(w)
    assert torch.equal(E3.rtn_int8_rows(w), q.float() * s[:, None])
    _, c, t = make(seed=14)
    for arm in E3.ARMS:
        dr = E3.Drafter(c, t)
        n = dr.quantize(arm)
        assert n == {"bf16": 0, "int8": 8, "int8h": 9, "int4h": 9}[arm], (arm, n)
        ids, aux = seq(c, 12, seed=15)
        dr.context(ids, aux, 0)
    c_real = E3.config_from_dict(json.load(open(REAL_CONFIG, encoding="utf-8")))
    b = E3.drafter_bytes(c_real, "int8", "int8", kv_positions=4096)
    body_params = 4096 * 5120 + 2 * 1024 * 5120 + 2560 * 4096 + 3 * 6144 * 2560
    assert b["body"] == body_params + 4 * (4096 + 2 * 1024 + 2560 + 2 * 6144 + 2560)
    assert b["head"] == 32768 * 2560 + 4 * 32768 and b["kv"] == 2048 * 8 * 128 * 2 * 2
    b4 = E3.drafter_bytes(c_real, "int4", "int8")
    assert b4["body"] == body_params // 2 + 2 * (4096 * 80 + 2 * 1024 * 80 + 2560 * 64 + 2 * 6144 * 40 + 2560 * 96)
    print(f"  RTN: int4 g64 == rtn.h's loop (zero group, ties), int8 == dflash_accept's; arms apply 0/8/9/9 linears; "
          f"derived step bytes int8h {b['step'] / 1e6:.1f} MB (KV at 4k: {b['kv'] / 1e6:.1f}), int4h {b4['step'] / 1e6:.1f} MB")


TESTS = [test_real_config, test_vocab_maps, test_rtn_arms, test_against_independent, test_batch_equals_propose,
         test_context_chunking_and_aux_from, test_position_convention, test_checkpoint_reader]


def main() -> None:
    torch.set_grad_enabled(False)
    only = sys.argv[1:]
    for t in TESTS:
        if not only or t.__name__ in only:
            print(t.__name__)
            t()
    print("ok")


if __name__ == "__main__":
    main()
