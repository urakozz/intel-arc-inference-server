#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""F1 for qwen4exp_ref.py (spec 21a Task 3): the streamed port against transformers 5.19.0 un-streamed.

    PYTHONPATH=<5.19.0 site> python3 tools/oracle/test_qwen4exp_ref.py [test_name ...]   (pytest collects it too)

The tiny model is qikp/tiny-random-Qwen4-Exp_Qwen3.8-Flash-Next: $Q4EXP_TINY, else oracle-out-q4exp/tiny,
else the HF cache (`$HF_HOME/hub/models--qikp--...`); every test that needs it SKIPs when it is absent.
Everything is BITWISE (Review Focus 2): logits and every layer's 4-stream residual, prompt and cached
decode, bf16 and fp32, at prompt lengths that cross the QSA cut (2051). One test per trap of spec 21 §2:
the HC rounding chain, the PLE ids (EOS history, padding, a decode boundary, the I64 constants), the
indexer cache at 2047..2060 and every (p + 1) % 4 == 0 there, the sigmoid gate, the v-head map h // 3,
router ties, --layers N, int4 experts at g64 / g128, the int8 PLE table, the original's fused experts,
the trace (Task 7), the CLI.
"""
import glob
import io
import os
import sys
import tempfile
from contextlib import redirect_stderr

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

import importlib.util  # noqa: E402
import math  # noqa: E402

import torch  # noqa: E402


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ref = _load("qwen4exp_ref")
mk = _load("qwen4exp_make_tiny")
REPO = os.path.dirname(os.path.dirname(_HERE))
BF, F32 = torch.bfloat16, torch.float32


class Skip(Exception):
    pass


def _skip(msg):
    if "pytest" in sys.modules:
        import pytest
        pytest.skip(msg)
    raise Skip(msg)


def tiny_snapshot() -> str:
    cands = [os.environ.get("Q4EXP_TINY", ""), os.path.join(REPO, "oracle-out-q4exp", "tiny")]
    hf = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    cands += sorted(glob.glob(os.path.join(hf, "hub", "models--qikp--tiny-random-Qwen4-Exp_Qwen3.8-Flash-Next",
                                           "snapshots", "*")))
    cands += sorted(glob.glob("/hf/hub/models--qikp--tiny-random-Qwen4-Exp_Qwen3.8-Flash-Next/snapshots/*"))
    for c in cands:
        if c and os.path.exists(os.path.join(c, "config.json")):
            return c
    _skip("the tiny model is absent (hf download qikp/tiny-random-Qwen4-Exp_Qwen3.8-Flash-Next, or Q4EXP_TINY)")


TMP = tempfile.mkdtemp(prefix="q4exp_test_")
_CACHE: dict = {}


def hf_model(dtype, snap=None, layers=None, **cfg_over):
    snap = snap or tiny_snapshot()
    key = ("hf", snap, dtype, layers, tuple(sorted(cfg_over.items())))
    if key not in _CACHE:
        m = ref.hf_reference(snap, dtype, layers)
        for k, v in cfg_over.items():
            setattr(m.config, k, v)
        _CACHE[key] = m
    return _CACHE[key]


def port_model(dtype, snap=None, layers=None, ple=None):
    snap = snap or tiny_snapshot()
    key = ("port", snap, dtype, layers, ple)
    if key not in _CACHE:
        tc = ref.text_config(snap, layers, dtype=dtype)
        table = ref.PleTable(ple or snap, tc)
        _CACHE[key] = ref.build_streamed(snap, tc, table) + (tc,)
    return _CACHE[key]


def hidden_hooks(model, store: dict):
    hs = []
    for i, layer in enumerate(model.model.layers):
        hs.append(layer.register_forward_hook(
            lambda _m, _a, out, i=i: store.setdefault(i, []).append(out.detach()[0].clone())))
    return hs


@torch.no_grad()
def drive(model, ids, steps, chunk=0, keep=48):
    """Prompt (whole, or in chunks), then the given decode tokens one at a time: the logits of each prompt
    chunk's last `keep` rows and of every decode step, bf16 [*][V] (a 2100-row f32 logits block is 2 GB -
    the container's 8 GB cap; both sides of a comparison make the same call, and every row's residual is
    compared through the H.L* hooks)."""
    rows, cache = [], None
    c = chunk or len(ids)
    for a in range(0, len(ids), c):
        out = model(input_ids=torch.tensor([ids[a:a + c]]), past_key_values=cache, use_cache=True,
                    logits_to_keep=keep)
        cache = out.past_key_values
        rows.append(out.logits[0])
    for t in steps:
        out = model(input_ids=torch.tensor([[t]]), past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        rows.append(out.logits[0])
    return torch.cat(rows), cache


def rand_ids(n, seed=0, vocab=248320):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (n,), generator=g).tolist()


def compare_runs(hf, port, ids, gen=8, chunk=0, what=""):
    """Both models on the same prompt and the same decode tokens (HF's greedy, decided in the same run - one
    2100-row bf16 eager forward is minutes on a shared CPU): logits and H.L* bitwise."""
    hh = {}
    hooks = hidden_hooks(hf, hh)
    with torch.no_grad():
        lg, cache = drive(hf, ids, [], chunk)
        rows, steps, last = [lg], [], lg[-1]
        for _ in range(gen):
            t = int(last.float().argmax())
            steps.append(t)
            out = hf(input_ids=torch.tensor([[t]]), past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            last = out.logits[0, -1]
            rows.append(out.logits[0])
    for h in hooks:
        h.remove()
    lg_hf = torch.cat(rows)
    model = port[0]
    hp = {}
    hooks = hidden_hooks(model, hp)
    lg_p, _ = drive(model, ids, steps, chunk)
    for h in hooks:
        h.remove()
    assert lg_p.shape == lg_hf.shape, (lg_p.shape, lg_hf.shape)
    if not torch.equal(lg_p, lg_hf):
        d = (lg_p.float() - lg_hf.float()).abs()
        raise AssertionError(f"{what}: logits differ, max |d| {d.max():.3e} first row {int(d.amax(-1).nonzero()[0])}")
    for i in hh:
        a, b = torch.cat(hp[i]), torch.cat(hh[i])
        assert torch.equal(a, b), f"{what}: H.L{i} differs (max |d| {(a.float() - b.float()).abs().max():.3e})"
    return steps


# ------------------------------------------------------------------------------------------------
# F1: the streamed port equals transformers, bitwise

def _streamed_equals_hf(dtype):
    hf, port = hf_model(dtype), port_model(dtype)
    for n in (40, 2100):
        steps = compare_runs(hf, port, rand_ids(n, seed=n), 8, what=f"{dtype} {n} ids")
        print(f"  {dtype} prompt {n} + 8 cached decode steps {steps[:4]}...: logits and H.L0-3 bitwise")


def test_streamed_equals_hf_bf16():
    _streamed_equals_hf(BF)


def test_streamed_equals_hf_fp32():
    _streamed_equals_hf(F32)


def test_chunked_prefill_equals_hf():
    """The reference's chunked prefill (forward_chunks, --chunk) is transformers' own chunked forward."""
    hf, port = hf_model(BF), port_model(BF)
    compare_runs(hf, port, rand_ids(2100, seed=7), 4, chunk=512, what="chunk 512")
    print("  2100 ids in chunks of 512 + 4 steps: bitwise")


def test_indexer_cache_bitwise():
    """Review Focus 3: the port's cached block keys vs transformers' per-query recomputation - the selection
    masks, the QSA attention output and the recorded 512th / 513th gap, rows 2047..2060 (each (p + 1) % 4
    == 0 there: 2047, 2051, 2055, 2059), in one prefill and as cached decode steps across the boundary."""
    hf, (pm, _, _, tc) = hf_model(BF), port_model(BF)
    L = tc.layer_types.index("indexed_attention")
    ids = rand_ids(2061, seed=3)

    def capture(model):
        st = {"mask": [], "attn": [], "in": []}
        a = model.model.layers[L].self_attn
        h1 = a.indexer.register_forward_hook(lambda _m, _a, o: st["mask"].append(o.detach()[0, 0].clone()))
        h2 = a.register_forward_hook(lambda _m, _a, o: st["attn"].append(o[0].detach()[0].clone()))
        h3 = a.indexer.register_forward_pre_hook(lambda _m, args: st["in"].append((args[0].detach().clone(), args[1])))
        return st, (h1, h2, h3)
    for mode in ("prefill", "decode"):
        res = {}
        for name, model in (("hf", hf), ("port", pm)):
            st, hs = capture(model)
            rec = ref.Recorder(model, tc) if name == "port" else None
            if mode == "prefill":
                drive(model, ids, [])
            else:
                drive(model, ids[:2045], ids[2045:])
            for h in hs:
                h.remove()
            if rec is not None:
                rec.remove()
                res["gap"] = rec.tensors()[f"qsa.gap.L{L}"]
                res["sel"] = rec.tensors()[f"qsa.sel.L{L}"]
            if mode == "prefill":
                masks, attn = st["mask"][0], st["attn"][0]
            else:
                masks = [m[0] for m in st["mask"][1:]]
                attn = torch.cat(st["attn"][1:])
                width = max(m.shape[0] for m in masks)
                masks = torch.stack([torch.cat([m, m.new_full((width - m.shape[0],), torch.finfo(m.dtype).min)])
                                     for m in masks])
                masks = torch.cat([torch.zeros(2045, width, dtype=masks.dtype), masks])
                attn = torch.cat([torch.zeros(2045, attn.shape[-1], dtype=attn.dtype), attn])
            res[name] = (masks, attn)
            if name == "hf" and mode == "prefill":
                res["hf_in"] = st["in"][0]
        rows = list(range(2047, 2061))
        mh, ah = res["hf"]
        mp, ap = res["port"]
        for r in rows:
            assert torch.equal(mp[r, : r + 1], mh[r, : r + 1]), f"{mode}: row {r}'s selection differs"
            assert torch.equal(ap[r], ah[r]), f"{mode}: row {r}'s attention output differs"
        sel_rows = (mh[:, :2061] == 0).sum(-1)
        assert int(sel_rows[2050]) == 2051 and int(sel_rows[2051]) == 2048, (int(sel_rows[2050]), int(sel_rows[2051]))
        gap = res["gap"]
        g_rows = gap[-(2061 - 2045):] if mode == "decode" else gap
        assert all(math.isinf(float(x)) for x in (g_rows[:2051] if mode == "prefill" else g_rows[:6]))
        if mode == "prefill":
            want = _gaps_from_transformers(hf.model.layers[L].self_attn.indexer, *res["hf_in"], range(2051, 2061))
            got = [float(gap[r]) for r in range(2051, 2061)]
            assert got == want, (got, want)
            sel = res["sel"]
            assert sel.shape == (2061 - 2051, 512)
            for j, r in enumerate(range(2051, 2061)):
                chosen = sorted(set((mh[r, : r + 1] == 0).nonzero().flatten().tolist()) - set(range(4 * ((r + 1) // 4), r + 1)))
                blocks = sorted({c // 4 for c in chosen})
                assert sel[j].tolist() == blocks, f"row {r}: recorded blocks differ from the mask's"
        print(f"  {mode}: rows 2047..2060 selections and attention bitwise; row 2050 sees 2051, row 2051 2048; "
              f"gaps {[f'{float(x):.3g}' for x in gap[-10:]]}")


def _gaps_from_transformers(ix, x, pe, rows):
    """The 512th - 513th block score per row, computed the way M:723-747 computes scores (per query, every
    block from the raw keys) on transformers' own indexer inputs - an independent restatement of the gap
    the port records."""
    m = ref.mq()
    cos, sin = pe
    D, C = ix.index_head_dim, ix.compress_ratio
    with torch.no_grad():
        qk = ix.index_qk_proj(x)
        q, k = torch.split(qk, [ix.index_n_heads * D, D], dim=-1)
        q = ix.q_layernorm(q.reshape(1, x.shape[1], -1, D))
        q = m.apply_rotary_pos_emb(q, cos=cos, sin=sin, unsqueeze_dim=2)
        raw = k.reshape(1, x.shape[1], D)
    out = []
    for r in rows:
        n = (r + 1) // C
        idx = torch.arange(n * C).view(n, C)
        groups = raw[0].index_select(0, idx.flatten()).view(n, C, D)
        pooled = ix.k_layernorm(groups.float().mean(dim=1).to(raw.dtype))
        keys = m.apply_rotary_pos_emb(pooled.unsqueeze(1), cos=cos[0].index_select(0, idx[:, 0]),
                                      sin=sin[0].index_select(0, idx[:, 0])).squeeze(1)
        s = torch.matmul(q[0, r].float(), keys.float().transpose(-1, -2)).transpose(-1, -2)
        s = torch.relu(s).sum(dim=-1) / math.sqrt(D)
        v = s.detach().topk(ix.block_topk + 1).values
        out.append(float(v[-2] - v[-1]))
    return out


# ------------------------------------------------------------------------------------------------
# the traps, one test each

def test_hc_chain():
    """The gated residual's rounding chain restated (ref.gated_residual / hc_combine) equals transformers'
    module bitwise, with block_inject (a layer's HC) and without (the final mixer); and a whole decoder
    layer recomposed from the restated chain equals the layer's forward."""
    hf = hf_model(BF)
    tc = hf.config
    g = torch.Generator().manual_seed(5)
    H = (torch.randn(3, tc.hc_count * tc.hidden_size, generator=g) * 2).to(BF)
    hc = hf.model.layers[0].attn_hyper_connection
    a = hc(H)
    b = ref.gated_residual(H, hc.hc_norm.weight, hc.input_mix_weight_down.weight, hc.input_mix_weight_up.weight,
                           hc.block_inject_weight.weight, tc.hc_count, tc.hidden_size, tc.rms_norm_eps)
    assert all(torch.equal(x, y) for x, y in zip(a, b)), "the HC with block_inject differs"
    mx = hf.model.hyper_connection_mixer
    assert mx.block_inject_weight is None
    assert torch.equal(mx(H), ref.gated_residual(H, mx.hc_norm.weight, mx.input_mix_weight_down.weight,
                                                 mx.input_mix_weight_up.weight, None, tc.hc_count, tc.hidden_size,
                                                 tc.rms_norm_eps)), "the final mixer differs"
    layer = hf.model.layers[0]            # GDN, no PLE
    with torch.no_grad():
        want = layer(H[None], position_embeddings=None, attention_mask=None, past_key_values=None)
        x, H0, inj = ref.gated_residual(H[None], *_hcw(layer.attn_hyper_connection), tc.hc_count, tc.hidden_size,
                                        tc.rms_norm_eps)
        Hn = ref.hc_combine(H0, layer.linear_attn(x), inj)
        x, H0, inj = ref.gated_residual(Hn, *_hcw(layer.mlp_hyper_connection), tc.hc_count, tc.hidden_size,
                                        tc.rms_norm_eps)
        got = ref.hc_combine(H0, layer.mlp(x), inj)
    assert torch.equal(want, got), "a decoder layer recomposed from the restated chain differs"
    print("  attn-side HC, the final mixer and a whole GDN layer: bitwise")


def _hcw(hc):
    return (hc.hc_norm.weight, hc.input_mix_weight_down.weight, hc.input_mix_weight_up.weight,
            hc.block_inject_weight.weight)


def test_ple_ids():
    """Review Focus 5: PleTable.ids (ref.ple_hash_ids) equals Qwen4ExpTextNGramEmbedding's ids exactly -
    EOS at 0, 1, 5 and twice, left padding, across a cached decode boundary - and the multipliers / primes /
    offsets recomputed from the formula equal the tiny checkpoint's I64 tensors."""
    snap = tiny_snapshot()
    hf = hf_model(F32)
    ple = ref.PleTable(snap, ref.text_config(snap, dtype=F32))
    E = 248044
    emb = hf.model.layers[1].ple.ple_embedding
    ck = ref.Ckpt(snap)
    for nm, want in (("layer_multipliers", ple.multipliers), ("ngram_heads_vocab_sizes", ple.sizes),
                     ("ngram_heads_offsets", ple.offsets)):
        t = ck.get(f"model.layers.1.ple.ple_embedding.{nm}")
        assert t.tolist() == list(want) and getattr(emb, nm).tolist() == list(want), nm
    got = {}
    h = emb.ngram_embedding.register_forward_hook(lambda _m, a, _o: got.setdefault("ids", []).append(a[0].detach().clone()))
    base = rand_ids(12, seed=11, vocab=1000)
    cases = {"eos@0": [E] + base, "eos@1": base[:1] + [E] + base[1:], "eos@5": base[:5] + [E] + base[5:],
             "eos twice": base[:3] + [E, E] + base[3:7] + [E] + base[7:], "plain": base}
    try:
        for name, seq in cases.items():
            got.clear()
            with torch.no_grad():
                hf(input_ids=torch.tensor([seq]), use_cache=False)
            assert torch.equal(got["ids"][0][0], ple.ids(torch.tensor(seq))), name
            # a cached decode boundary: 7 ids, then one at a time
            got.clear()
            drive(hf, seq[:7], seq[7:])
            dec = torch.cat([x[0] for x in got["ids"]])
            assert torch.equal(dec, ple.ids(torch.tensor(seq))), f"{name}: decode"
            parts = [ple.ids(torch.tensor(seq[:7]))] + [ple.ids(torch.tensor([t]), prev=torch.tensor(seq[:7 + j]))
                                                      for j, t in enumerate(seq[7:])]
            assert torch.equal(torch.cat(parts), dec), f"{name}: prev="
        got.clear()
        seq = base[:10]
        mask = torch.tensor([[0, 0, 0] + [1] * 7])
        with torch.no_grad():
            hf(input_ids=torch.tensor([seq]), attention_mask=mask, use_cache=False)
        assert torch.equal(got["ids"][0][0], ple.ids(torch.tensor(seq), mask=mask[0])), "padding"
    finally:
        h.remove()
    print(f"  {len(cases)} sequences (prompt and decode) + left padding: ids exact; I64 constants = formula "
          f"{ple.multipliers}")


def test_sigmoid_gate():
    """output_gate_type sigmoid (M:492) is live: the GDN layer's output differs under silu; the restated
    gated norm equals the module for both activations."""
    hf = hf_model(BF)
    tc = hf.config
    g = torch.Generator().manual_seed(9)
    x = torch.randn(1, 6, tc.hidden_size, generator=g).to(BF)
    gdn = hf.model.layers[0].linear_attn
    with torch.no_grad():
        a = gdn(x)
        gdn.norm.activation = "silu"
        b = gdn(x)
        gdn.norm.activation = "sigmoid"
    assert not torch.equal(a, b), "the GDN output did not change with the gate"
    v = torch.randn(12, 128, generator=g).to(BF)
    z = torch.randn(12, 128, generator=g).to(BF)
    for act in ("sigmoid", "silu"):
        gdn.norm.activation = act
        assert torch.equal(gdn.norm(v, z), ref.gdn_gated_norm(v, z, gdn.norm.weight, gdn.norm.variance_epsilon, act))
    gdn.norm.activation = "sigmoid"
    print(f"  sigmoid vs silu outputs differ (max |d| {(a.float() - b.float()).abs().max():.3g}); gated norm restated bitwise")


def test_v_head_map():
    """v head h reads k head h // 3 (repeat_interleave, M:575-576): zeroing k head kh's projection rows
    silences exactly v heads 3kh..3kh+2 of the core attention output."""
    hf = hf_model(F32)
    gdn = hf.model.layers[0].linear_attn
    kd = gdn.key_dim
    g = torch.Generator().manual_seed(4)
    x = torch.randn(1, 9, hf.config.hidden_size, generator=g)
    w0 = gdn.in_proj_qkv.weight.data.clone()
    try:
        for kh in (0, 5, 15):
            gdn.in_proj_qkv.weight.data.copy_(w0)
            gdn.in_proj_qkv.weight.data[kd + 128 * kh: kd + 128 * (kh + 1)] = 0
            cap = {}
            h = gdn.norm.register_forward_pre_hook(lambda _m, a: cap.update(o=a[0].detach().clone()))
            with torch.no_grad():
                gdn(x)
            h.remove()
            o = cap["o"].view(9, gdn.num_v_heads, 128)
            dead = [v for v in range(gdn.num_v_heads) if bool((o[:, v] == 0).all())]
            assert dead == [3 * kh, 3 * kh + 1, 3 * kh + 2], (kh, dead)
    finally:
        gdn.in_proj_qkv.weight.data.copy_(w0)
    print("  k heads 0, 5, 15 zeroed -> exactly v heads 3k..3k+2 silent")


def test_route_ties():
    """A 16-expert top-4 variant with expert 9's router row a copy of expert 3's: the recorded route is
    torch.topk's (the port = transformers bitwise), route.gap is exactly 0 on the rows where the pair
    straddles the cut, and on those rows the chosen expert follows the tie rule the facts sheet records."""
    d, twin = os.path.join(TMP, "ties"), os.path.join(TMP, "ties-hf")
    if not os.path.exists(d):
        mk.make(d, hidden=128, layer_types="lllq", experts=16, topk=4, form="tiny", tie=(3, 9), twin=twin)
    hf = hf_model(BF, snap=twin)
    port = port_model(BF, snap=d)
    compare_runs(hf, port, rand_ids(64, seed=2), 2, what="ties")
    pm, _, _, tc = port
    rec = ref.Recorder(pm, tc)
    drive(pm, rand_ids(256, seed=8), [])
    rec.remove()
    t = rec.tensors()
    straddle = lower = 0
    for i in range(tc.num_hidden_layers):
        ids, gap = t[f"route.ids.L{i}"], t[f"route.gap.L{i}"]
        for r in range(ids.shape[0]):
            s = set(ids[r].tolist())
            if (3 in s) != (9 in s):
                # a straddle at the cut is exactly the case where p(4th) == p(5th)
                if float(gap[r]) == 0.0:
                    straddle += 1
                    lower += 3 in s
    assert straddle > 0, "no row had the tied pair at the cut"
    print(f"  {straddle} (row, layer) with the tied pair at the cut, gap == 0 there; the lower id (3) won "
          f"{lower}/{straddle}")


def test_indexer_gaps_small_budget():
    """The recorded 512th / 513th gap (here the 4th / 5th: an indexer budget of 16 tokens = 4 blocks, so the
    selection acts from position 19) equals the per-query recomputation from transformers' own indexer inputs
    on every row past the cut - on the downloaded tiny every gap at 2051+ is an exact 0 (relu zeros the
    scores of most blocks, so the cut falls inside a tie), which makes test_indexer_cache_bitwise's gap check
    weak; here the gaps are mostly non-zero. The forward is transformers' bitwise as well."""
    d = os.path.join(TMP, "budget16")
    if not os.path.exists(d):
        mk.make(d, hidden=128, layer_types="lllq", experts=8, topk=2, form="tiny", indexer_budget=16, seed=5)
    hf, port = hf_model(BF, snap=d), port_model(BF, snap=d)
    ids = rand_ids(96, seed=61)
    compare_runs(hf, port, ids, 4, what="budget 16")
    pm, _, _, tc = port
    L = tc.layer_types.index("indexed_attention")
    rec = ref.Recorder(pm, tc)
    drive(pm, ids, [])
    rec.remove()
    gap = rec.tensors()[f"qsa.gap.L{L}"]
    cap = []
    ix = hf.model.layers[L].self_attn.indexer
    h = ix.register_forward_pre_hook(lambda _m, args: cap.append((args[0].detach().clone(), args[1])))
    with torch.no_grad():
        hf(input_ids=torch.tensor([ids]), use_cache=False)
    h.remove()
    rows = [r for r in range(len(ids)) if (r + 1) // 4 > ix.block_topk]
    want = _gaps_from_transformers(ix, cap[0][0], cap[0][1], rows)
    got = [float(gap[r]) for r in rows]
    assert got == want, (got[:6], want[:6])
    assert all(math.isinf(float(gap[r])) for r in range(len(ids)) if (r + 1) // 4 <= ix.block_topk)
    nz = sum(1 for g in got if g > 0)
    print(f"  {len(rows)} rows past the cut: gaps = the recomputation exactly ({nz} non-zero, min "
          f"{min(got):.3g}, median {sorted(got)[len(got) // 2]:.3g})")


def test_layers_truncation():
    """--layers N (spec 21's development mode): on an 8-layer tiny (lllqlllq), text_config(layers=4) equals
    transformers built with num_hidden_layers=4 bitwise (prompt + cached decode); layers=2 equals it on an
    uncached forward (transformers 5.19.0's DynamicCache needs an indexed_attention layer for the sequence
    length - cache_utils.py:1566 - so a cached run needs N >= 4, which run / hfcheck / trace require); layers=1
    is refused naming ple_layer_ids (one-indexed 2 = layer_idx 1 must stay inside)."""
    d, twin = os.path.join(TMP, "l8"), os.path.join(TMP, "l8-hf")
    if not os.path.exists(d):
        mk.make(d, hidden=128, layer_types="lllqlllq", experts=8, topk=2, form="tiny", twin=twin)
    hf4, port4 = hf_model(BF, snap=twin, layers=4), port_model(BF, snap=d, layers=4)
    compare_runs(hf4, port4, rand_ids(48, seed=12), 4, what="layers=4")
    hf2, (pm2, _, _, tc2) = hf_model(BF, snap=twin, layers=2), port_model(BF, snap=d, layers=2)
    ids = torch.tensor([rand_ids(40, seed=13)])
    with torch.no_grad():
        a = hf2(input_ids=ids, use_cache=False, logits_to_keep=8).logits
        b = pm2(input_ids=ids, use_cache=False, logits_to_keep=8).logits
    assert torch.equal(a, b), "layers=2 uncached differs"
    err = io.StringIO()
    with redirect_stderr(err):
        try:
            ref.require_cacheable(tc2)
            raise AssertionError("layers=2 was accepted for a cached run")
        except SystemExit:
            pass
    assert "indexed_attention" in err.getvalue(), err.getvalue()
    err = io.StringIO()
    with redirect_stderr(err):
        try:
            ref.text_config(d, layers=1)
            raise AssertionError("layers=1 was accepted")
        except SystemExit:
            pass
    assert "ple_layer_ids" in err.getvalue(), err.getvalue()
    print(f"  layers=4 bitwise (cached); layers=2 bitwise (uncached; refused for a cached run); layers=1 refused: "
          f"{err.getvalue().strip()[:80]}")


def test_int4_experts():
    """Intel's / our per-expert int4 (auto_round:auto_gptq bytes, qzeros 0x77777777) at g128 and g64: the
    dequant is (q - 8) x scale (stream.py's rule, held to dequant.py), and the port on the int4 checkpoint
    equals transformers on its dequantised twin, bitwise; `ours` also quantises the dense linears."""
    for form, g in (("intel", 128), ("ours", 64)):
        d, twin = os.path.join(TMP, f"i4-{form}"), os.path.join(TMP, f"i4-{form}-hf")
        if not os.path.exists(d):
            mk.make(d, hidden=128, layer_types="lllq", experts=8, topk=2, form=form, group=g, twin=twin)
        ck = ref.Ckpt(d)
        tc = ref.text_config(d)
        lazy = ref.LazyExperts(ck, tc, "model.language_model.layers.")
        gu, dn = lazy.expert(2, 5)
        b = "model.language_model.layers.2.mlp.experts.5."
        want_g = mk.unpack_int4(ck.get(b + "gate_proj.qweight"), ck.get(b + "gate_proj.scales"), g)
        want_d = mk.unpack_int4(ck.get(b + "down_proj.qweight"), ck.get(b + "down_proj.scales"), g)
        assert torch.equal(gu[: tc.moe_intermediate_size], want_g) and torch.equal(dn, want_d), f"{form}: dequant"
        assert set(ck.keys()) == set(ref.expected_names(tc, form, mtp=False)), f"{form}: names"
        compare_runs(hf_model(BF, snap=twin), port_model(BF, snap=d), rand_ids(40, seed=g), 3, what=form)
        print(f"  {form} g{g}: dequant = (q - 8) x scale; names = expected_names; forward = transformers on the "
              f"dequantised twin, bitwise")


def test_fused_experts():
    """The original's storage (`model.language_model.*`, fused gate_up_proj [E][2I][H] gate rows first,
    down_proj [E][H][I], `mtp.*` beside): the port reads it per expert and equals transformers on the same
    weights; its names are expected_names(tc, "bf16")."""
    d, twin = os.path.join(TMP, "fused"), os.path.join(TMP, "fused-hf")
    if not os.path.exists(d):
        mk.make(d, hidden=128, layer_types="lllq", experts=8, topk=2, form="bf16", mtp=True, twin=twin)
    tc = ref.text_config(d)
    names = set(ref.Ckpt(d).keys())
    assert names == set(ref.expected_names(tc, "bf16")), sorted(names ^ set(ref.expected_names(tc, "bf16")))[:6]
    compare_runs(hf_model(BF, snap=twin), port_model(BF, snap=d), rand_ids(40, seed=21), 3, what="fused")
    print(f"  fused experts + model.language_model.* + mtp.*: {len(names)} names = expected_names('bf16'); bitwise")


def test_ple_int8():
    """21b's int8 file: the rows equal clamp(rne(w / s), -127, 127) with s = max|w| / 127 and dequantise to
    bf16(q x s); the forward on it equals transformers with the dequantised table, bitwise."""
    snap = tiny_snapshot()
    hf = ref.hf_reference(snap, BF)
    tc = ref.text_config(snap)
    emb = hf.model.layers[1].ple.ple_embedding.ngram_embedding
    table = ref.PleTable(snap, tc)
    full = torch.from_numpy(__import__("numpy").concatenate([s for s in table.shards]))
    d = os.path.join(TMP, "ple-int8")
    mk.write_ple_int8(full, tc, d)
    t8 = ref.PleTable("int8:" + d, tc)
    ids = torch.stack([torch.tensor([t8.offsets[h] + j for h in range(16)]) for j in (0, 1, 7, 100)])
    rows = t8.rows(ids)
    for h in range(16):
        w = full[ids[:, h]].float()
        s = w.abs().amax(-1) / 127.0
        q = torch.round(w / s.unsqueeze(-1)).clamp(-127, 127)
        assert torch.equal(rows[:, h], (q * s.unsqueeze(-1)).to(BF)), h
    deq = torch.zeros_like(emb.weight)
    for h in range(16):
        deq[t8.offsets[h]:t8.offsets[h] + t8.sizes[h]] = _int8_rows_head(t8, h)
    emb.weight.data.copy_(deq)
    compare_runs(hf, port_model(BF, ple="int8:" + d), rand_ids(40, seed=33), 3, what="int8 PLE")
    print("  int8 rows = rne(w / s) at s = max|w| / 127; forward on the int8 file = transformers on the "
          "dequantised table, bitwise")


def _int8_rows_head(t8, h):
    import numpy as np
    q = torch.from_numpy(np.array(t8.q[h])).float()
    s_arr, s_dt = t8.s[h]
    s = torch.from_numpy(np.array(s_arr))
    s = s.view(BF).float() if s_dt == "BF16" else s.float()
    return (q * s.unsqueeze(-1)).to(BF)


def test_expected_names_tiny():
    """The downloaded tiny's names are expected_names(tc, "tiny") exactly."""
    snap = tiny_snapshot()
    tc = ref.text_config(snap)
    names = set(ref.Ckpt(snap).keys())
    want = set(ref.expected_names(tc, "tiny"))
    assert names == want, (sorted(names - want)[:5], sorted(want - names)[:5])
    print(f"  {len(names)} names")


def test_real_names():
    """expected_names(tc, "bf16") / ("intel") equal the real checkpoints' index (vision excluded) - read from
    directories holding their config.json + model.safetensors.index.json: Q4EXP_ORIG_INDEX / Q4EXP_INTEL_INDEX,
    default oracle-out-q4exp/meta/{orig,intel} (small files only; SKIP when absent)."""
    import json
    done = 0
    for form, env, sub in (("bf16", "Q4EXP_ORIG_INDEX", "orig"), ("intel", "Q4EXP_INTEL_INDEX", "intel")):
        d = os.environ.get(env) or os.path.join(REPO, "oracle-out-q4exp", "meta", sub)
        if not os.path.exists(os.path.join(d, "model.safetensors.index.json")):
            continue
        tc = ref.text_config(d)
        wm = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
        names = {k for k in wm if not k.startswith("model.visual.")}
        want = set(ref.expected_names(tc, form))
        assert names == want, (form, sorted(names - want)[:5], sorted(want - names)[:5])
        print(f"  {form}: {len(names)} text tensors = expected_names (+ {len(wm) - len(names)} vision)")
        done += 1
    if not done:
        _skip("no real index (Q4EXP_ORIG_INDEX / Q4EXP_INTEL_INDEX)")


def test_trace_equals_run():
    """Task 7: the trace's routes are the sequential forward's (route.ids sorted ascending), and three
    sources batched layer-major equal each alone; p is the router's pre-cast fp32 renormalised value and
    onorm the routed expert's output norm."""
    pm, pf, _, tc = port_model(BF)
    srcs = [("a", rand_ids(70, seed=41)), ("b", rand_ids(33, seed=42)), ("c", rand_ids(90, seed=43))]
    batched = {c: ref.trace_routes(pm, pf, tc, srcs, chunk=c) for c in (0, 32)}
    for name, ids in srcs:
        for c in (32, 0):                     # chunked rounds, then whole (the run below is whole)
            alone = ref.trace_routes(pm, pf, tc, [(name, ids)], chunk=c)[name]
            for k in ("ids", "p", "onorm"):
                assert torch.equal(alone[k], batched[c][name][k]), f"{name}: {k} batched != alone (chunk {c})"
        rec = ref.Recorder(pm, tc, acts=False)
        drive(pm, ids, [])
        rec.remove()
        t = rec.tensors()
        for i in range(tc.num_hidden_layers):
            want = t[f"route.ids.L{i}"].sort(-1).values
            assert torch.equal(alone["ids"][:, i], want), f"{name} L{i}: trace ids != run's"
            w = t[f"route.w.L{i}"]
            order = t[f"route.ids.L{i}"].argsort(-1)
            assert torch.equal(alone["p"][:, i].to(BF).float(), w.gather(-1, order)), f"{name} L{i}: p"
        assert bool((alone["onorm"] > 0).all())
    print("  3 sources (70 / 33 / 90 ids) batched = each alone (whole, and in chunks of 32); whole = run's routes")


def test_cli_run_and_hfcheck():
    """`run` writes the golden layout; `hfcheck` (transformers' own indexer against the cache) is bitwise
    on the tiny at 2060 ids; `facts` prints the tables and the expected-names match."""
    import subprocess
    snap = tiny_snapshot()
    d = os.path.join(TMP, "cli")
    os.makedirs(d, exist_ok=True)
    ids = rand_ids(2060, seed=51)
    open(os.path.join(d, "p.ids"), "w").write(" ".join(map(str, ids)) + "\n")
    env = dict(os.environ)
    py = [sys.executable, os.path.join(_HERE, "qwen4exp_ref.py")]
    out = os.path.join(d, "p.golden.safetensors")
    r = subprocess.run(py + ["run", snap, "--prompt", os.path.join(d, "p.ids"), "--out", out, "--gen", "4",
                             "--act-tail", "16", "--logits-tail", "8"], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr[-2000:] + r.stdout[-2000:]
    from safetensors import safe_open
    with safe_open(out, "pt") as h:
        meta = h.metadata()
        t = {k: h.get_tensor(k) for k in h.keys()}
    T, G, tc = 2060, 4, ref.text_config(snap)
    assert t["logits"].shape == (8 + G, tc.vocab_size) and meta["logits_row0"] == str(T - 8)
    assert t["H.L3"].shape == (16 + G, tc.hc_count * tc.hidden_size) and t["H.L3"].dtype == BF
    assert t["route.ids.L0"].shape == (T + G, tc.num_experts_per_tok) and t["ple.ids"].shape == (T + G, 16)
    assert t["qsa.sel.L3"].shape == (T + G - 2051, 512) and t["qsa.gap.L3"].shape == (T + G,)
    assert t["nll"].shape == (T - 1,) and t["tokens"].shape == (G,)
    assert "QSA 512th/513th gap" in r.stdout and "MoE 2th/3th gap" in r.stdout, r.stdout[-1500:]
    r = subprocess.run(py + ["hfcheck", snap, "--layers", "4", "--n", "2060", "--gen", "2"], capture_output=True,
                       text=True, env=env)
    assert r.returncode == 0 and "BITWISE" in r.stdout, r.stdout[-1500:] + r.stderr[-1500:]
    r = subprocess.run(py + ["facts", snap], capture_output=True, text=True, env=env)
    assert r.returncode == 0 and "expected_names(tc, 'tiny')" in r.stdout and "lower index wins" in r.stdout, \
        r.stdout[-1500:] + r.stderr[-1500:]
    print("  run: the golden layout; hfcheck: BITWISE; facts: names and the topk tie rule")


TESTS = [test_expected_names_tiny, test_real_names, test_hc_chain, test_ple_ids, test_sigmoid_gate, test_v_head_map,
         test_streamed_equals_hf_bf16, test_streamed_equals_hf_fp32, test_chunked_prefill_equals_hf,
         test_indexer_cache_bitwise, test_indexer_gaps_small_budget, test_layers_truncation, test_route_ties, test_int4_experts, test_fused_experts,
         test_ple_int8, test_trace_equals_run, test_cli_run_and_hfcheck]


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
