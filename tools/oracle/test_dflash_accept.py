#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Checks for dflash_accept.py (spec 19a Task 3). CPU, tiny random drafters, seconds.

    test_dflash_accept.py [test_name ...]

- the batched drafter (B blocks at once, the head over all mask rows) equals
  dflash_ref.Drafter.draft_block anchor by anchor - ids, and the hidden rows - for DFlash 2 and
  DFlash (v1, incl. a full-attention and a causal layer), K = 1..7, with anchors before and past
  the sliding window and context positions >= the anchor present (Review Focus 2-4);
- top_k_fast == top_k_stable, ties across the k-th value included;
- select_draft_vocab == src/loader/draft_vocab.cc on tests/loader/draft_vocab_test.cc's cases
  and Qwen3.8's shape; a masked head drafts only V' ids and the full mask changes nothing
  (Review Focus 5);
- the int8 RTN (loader/lm_head_int8.cc's rule) and the int8 head;
- the acceptance accounting: lengths, censoring where the text leaves the greedy path, the
  per-depth counts and E_K;
- a run over tap dumps written by dump_taps.save_dump, resumed.
"""
import argparse
import glob
import json
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


T = _load("test_dflash_ref")          # its synthetic drafters (and dflash_ref as T.ref)
ref = T.ref
DA = _load("dflash_accept")
DA._load_deps()
DT = DA.dt


def _ctx(d, n, seed=1):
    taps = T.random_taps(d, n, seed)
    return d.context_kv(taps, torch.arange(n)), taps


def _check_batch_equals_reference(d, n_ctx, anchors_pos, label):
    ctx, _ = _ctx(d, n_ctx)
    g = torch.Generator().manual_seed(3)
    anchor_ids = torch.randint(0, T.VOCAB - 4, (len(anchors_pos),), generator=g).tolist()
    head = DA.Head(d.lm_head)
    worst = 0.0
    for K in range(1, d.cfg.max_k + 1):
        ps = torch.tensor(anchors_pos)
        h = DA.forward_blocks(d, anchor_ids, ps, K, ctx)
        ids = DA.draft_batch(d, head, anchor_ids, ps, K, ctx, {"full": None})["full"]
        for b, p in enumerate(anchors_pos):
            o = d.draft_block(anchor_ids[b], p, K, ctx)
            assert torch.equal(ids[b], o.ids), (label, K, p, ids[b].tolist(), o.ids.tolist())
            worst = max(worst, (h[b] - o.hidden).abs().max().item())
    assert worst < 1e-4, (label, worst)
    return worst


def test_batch_equals_draft_block():
    d2 = T.make(T.v2_dict(window=6))                               # 2 sliding non-causal layers
    w2 = _check_batch_equals_reference(d2, 20, [1, 2, 5, 9, 13, 16], "v2")
    d1 = T.make(T.v1_dict(window=4, is_causal=None))                # Ornith-like: causal sliding + full
    w1 = _check_batch_equals_reference(d1, 20, [0, 3, 7, 12, 15], "v1")
    print(f"batched drafts == draft_block per anchor (v2, v1 causal+full), K = 1..max; hidden within "
          f"{max(w1, w2):.1e}")


def test_top_k_fast():
    g = torch.Generator().manual_seed(0)
    x = torch.randint(0, 6, (50, 40), generator=g).float()           # many ties
    x[3] = 1.0                                                       # all tied
    x[4, :] = float("-inf")
    x[4, 7] = 2.0
    for k in (1, 4, 16):
        i1, v1 = DA.top_k_fast(x, k)
        i2, v2 = ref.top_k_stable(x, k)
        assert torch.equal(i1, i2) and torch.equal(v1, v2), k
    print("top_k_fast == top_k_stable (ties across the boundary, -inf rows)")


def test_select_draft_vocab():
    s = DA.select_draft_vocab
    v = s([95, 96], [97, 98], [50, 50, 99, 120, 3, 95, 10], 100, 32)
    assert len(v) == 32 and v == sorted(set(v)) and max(v) < 100
    assert all(i in v for i in (95, 96, 97, 98, 50, 99, 3, 10)) and 120 not in v
    assert all(i in v for i in range(26)) and not any(i in v for i in range(26, 95) if i != 50)
    assert s([1], [2], [60, 61, 62, 63, 64, 65, 66, 67], 100, 5) == [1, 2, 60, 61, 62]
    m = s([150, 7], [99, 100], [], 100, 16)
    assert 7 in m and 99 in m and 100 not in m and len(m) == 16
    assert s([5], [], [9], 64, 64) == list(range(64))
    for bad in (lambda: s([], [], [], 100, 0), lambda: s([], [], [], 100, 101), lambda: s([1, 2, 3], [4], [], 100, 3)):
        try:
            bad()
            raise AssertionError("no throw")
        except ValueError:
            pass
    assert s([3, 1, 2], [], [], 100, 3) == [1, 2, 3]
    added = list(range(248044, 248077))
    for size in (32768, 65536, 131072):
        v = s(added, [248046, 248044], [], 248077, size)
        assert len(v) == size and v[size - 34] == size - 34 and v[size - 33] == 248044
    print("select_draft_vocab == loader::select_draft_vocab on draft_vocab_test.cc's cases + Qwen3.8's shape")


def test_vocab_mask():
    d = T.make(T.v2_dict(window=6))
    ctx, _ = _ctx(d, 16)
    head = DA.Head(d.lm_head)
    keep_ids = DA.select_draft_vocab([290, 291], [292], [], 300, 40)
    keep = torch.zeros(T.VOCAB, dtype=torch.bool)
    keep[torch.tensor(keep_ids)] = True
    full = torch.ones(T.VOCAB, dtype=torch.bool)
    ps = torch.tensor([4, 9, 15])
    out = DA.draft_batch(d, head, [1, 2, 3], ps, 7, ctx, {"none": None, "v": keep, "all": full})
    assert torch.equal(out["none"], out["all"])
    assert keep[out["v"].flatten()].all()
    assert not torch.equal(out["none"], out["v"])
    print(f"V' mask: drafts only V' ids ({len(keep_ids)} of {T.VOCAB}); an all-true mask changes nothing")


def test_int8():
    g = torch.Generator().manual_seed(1)
    w = (torch.randn(37, 70, generator=g) * torch.rand(37, 1, generator=g) * 3).to(torch.bfloat16)
    w[5] = 0
    q, s = DA.rtn_int8_rows(w)
    assert q.dtype == torch.int8 and q.abs().max() <= 127 and s[5] == 0 and (q[5] == 0).all()
    err = (q.float() * s[:, None] - w.float()).abs()
    assert (err <= s[:, None] / 2 + 1e-6).all()
    assert (q.abs().amax(1)[s > 0] == 127).all()
    d = T.make(T.v2_dict())
    before = {n: v.clone() for n, v in d.w.items()}
    n = DA.int8_simulate(d)
    lin = [k for k in before if k.endswith(DA.LINEAR_SUFFIXES)]
    assert n == len(lin) and n == 1 + 2 * (4 + 3 + 2) + 1, n
    for k, v in d.w.items():
        if k in lin:
            assert not torch.equal(v, before[k]) and len(torch.unique(v[0])) <= 255
        else:
            assert torch.equal(v, before[k]), k
    h = torch.randn(3, T.HID, generator=g)
    hb, hq = DA.Head(d.lm_head), DA.Head(d.lm_head, int8=True, chunk=64)
    q, s = DA.rtn_int8_rows(d.lm_head)
    assert torch.allclose(hq.logits(h), h @ (q.float() * s[:, None]).t(), atol=1e-4)
    cos = torch.nn.functional.cosine_similarity(hq.logits(h), hb.logits(h), dim=-1).min()
    assert cos > 0.999, cos
    print(f"int8 RTN per row (s = amax/127, |err| <= s/2); {n} drafter linears quantised, the rest untouched; "
          f"int8 head cos {float(cos):.5f}")


def test_accept_accounting():
    #           0  1  2  3  4  5  6  7  8  9
    x = torch.tensor([9, 9, 1, 2, 3, 4, 5, 6, 7, 8])                   # text
    g = torch.tensor([9, 1, 2, 3, 4, 0, 6, 7, 8, 5])                   # greedy after t (g_from 0)
    # text agrees with greedy except x[6]=5 vs g[5]=0
    ps = torch.tensor([1, 2, 4, 5])
    drafts = torch.tensor([[2, 3, 4, 9],      # p=1: d0=2 vs g[1]=1, a miss at depth 0
                           [2, 3, 4, 0],      # p=2: g[2..5] = 2,3,4,0 -> all 4 accepted
                           [4, 0, 6, 7],      # p=4: g[4]=4 ok, g[5]=0 ok, then x[6]=5 != g[5]=0 -> censored
                           [0, 6, 7, 8]])     # p=5: g[5]=0 ok; depth 1 needs x[6]==g[5] -> censored
    length, cens, counted, accepted = DA.accept_rows(drafts, ps, x, g, 0)
    assert length.tolist() == [0, 4, 2, 1] and cens.tolist() == [False, False, True, True], (length, cens)
    assert counted == [4, 2, 1, 1] and accepted == [3, 2, 1, 1], (counted, accepted)
    # g_from offsets the greedy array
    l2, c2, _, _ = DA.accept_rows(drafts, ps, x, g[1:], 1)
    l3, c3, _, _ = DA.accept_rows(drafts, ps, x, g, 0)
    assert torch.equal(l2, l3) and torch.equal(c2, c3)
    e = DA.pool([{"4": {"rows": 4, "censored": 2, "hist": [1, 0, 0, 0, 1], "counted": counted,
                        "accepted": accepted}}], [4])["4"]
    assert abs(e["E"] - (1 + 0.75 + 0.75 + 0.75 + 0.75)) < 1e-9 and e["mean_len_plus1"] == 3.0
    assert DA.anchors_of(10, 20) == [10, 11, 12, 13]
    print("acceptance: lengths, censoring off the greedy path, per-depth counts, E_K, anchors")


def _fake_dump(path, name, d, n_prompt, n, seed):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, T.VOCAB - 4, (n,), generator=g)
    taps = T.random_taps(d, n, seed)
    greedy = torch.cat([ids[1:], ids[:1]])                     # text = greedy everywhere
    res = {"taps": {i: t.to(torch.bfloat16) for i, t in taps.items()}, "greedy": greedy,
           "top_ids": greedy[:, None].repeat(1, 4), "top_logits": torch.zeros(n, 4), "lse": torch.zeros(n)}
    DT.save_dump(path, name, ids.tolist(), n_prompt, 0, res, {"vocab_used": 300})


def test_run_over_dumps():
    cd = T.v2_dict(window=6)
    cfg = ref.config_from_dict(cd)
    t, e, h = T.random_tensors(cfg, 2)
    with tempfile.TemporaryDirectory() as root:
        dd, td = T._write_checkpoint(root, cd, t, e, h)
        with open(os.path.join(td, "tokenizer.json"), "w") as f:
            json.dump({"added_tokens": [{"id": 290}, {"id": 291}]}, f)
        with open(os.path.join(td, "generation_config.json"), "w") as f:
            json.dump({"eos_token_id": [291, 292]}, f)
        d = ref.load_drafter(dd, td)
        dumps = os.path.join(root, "dumps")
        os.makedirs(dumps)
        _fake_dump(os.path.join(dumps, "prose__a.taps.safetensors"), "prose/a", d, 8, 20, 1)
        _fake_dump(os.path.join(dumps, "code__b.taps.safetensors"), "code/b", d, 5, 15, 2)
        out = os.path.join(root, "out")
        rk = os.path.join(root, "ranked.ids")
        with open(rk, "w") as f:
            f.write("# rank.py\n" + "\n".join(str(i) for i in range(299, 200, -1)) + "\n")
        ns = argparse.Namespace(dumps=dumps, out_dir=out, arms="bf16,int8", vocab="", vocab_on="bf16",
                                draft_vocab_ids=rk, vocab_used=300, k="1-7", batch=4, drafter=dd,
                                w4a16=None, target=td, sources=None, dry_run=False)
        DA.VOCAB_SIZES["tiny"] = 64
        ns.vocab = "tiny"
        DA.cmd_run(ns)
        r = json.load(open(os.path.join(out, "accept.bf16.json")))
        assert sorted(r["variants"]) == ["bf16", "bf16-vtiny", "bf16-vtiny-r"] and sorted(r["variants"]["bf16"]) == ["code/b", "prose/a"]
        st = r["variants"]["bf16"]["prose/a"]
        assert st["7"]["rows"] == len(DA.anchors_of(8, 20)) == 6 and st["7"]["censored"] == 0
        assert sum(st["3"]["hist"]) == 6 and len(st["3"]["hist"]) == 4
        assert r["agree"]["prose/a"] == [12, 12]
        assert st["3"]["counted"][0] == 6
        mt = os.path.getmtime(os.path.join(out, "accept.bf16.json"))
        DA.cmd_run(ns)                                   # resumed: nothing left to do
        assert os.path.getmtime(os.path.join(out, "accept.bf16.json")) == mt
        assert os.path.isfile(os.path.join(out, "summary.md"))
        summ = DA.summarize([os.path.join(out, "accept.bf16.json"), os.path.join(out, "accept.int8.json")])
        assert set(summ) == {"bf16", "bf16-vtiny", "bf16-vtiny-r", "int8"} and "all" in summ["bf16"]
        assert summ["bf16"]["all"]["sources"] == 2
        ns.dry_run, ns.out_dir = True, os.path.join(root, "dry")
        DA.cmd_run(ns)
        assert not os.listdir(ns.out_dir)
        del DA.VOCAB_SIZES["tiny"]
    print("run: two dumps x bf16 / int8 (+ a V' arm), JSON per arm, resumed, summary, dry run writes nothing")


def test_positive_alignment():
    """End to end with a known answer: a dump whose greedy ids ARE draft_block's K = 1 and K = 2
    drafts (taps from position 3, so the greedy array is offset) accepts every depth-0 draft
    (E_1 = 2) and, where the text follows the greedy path, K = 2 accepts both."""
    d = T.make(T.v2_dict(window=6))
    n, n_prompt, f = 22, 9, 3
    g = torch.Generator().manual_seed(4)
    ids = torch.randint(0, T.VOCAB - 4, (n,), generator=g)
    taps = {i: t[f:] for i, t in T.random_taps(d, n, 5).items()}
    ctx = d.context_kv(taps, torch.arange(f, n))
    greedy = torch.randint(0, T.VOCAB - 4, (n - f,), generator=g)
    for p in DA.anchors_of(n_prompt, n):
        greedy[p - f] = d.draft_block(int(ids[p]), p, 1, ctx).ids[0]
    res = {"taps": {i: t.to(torch.bfloat16) for i, t in taps.items()}, "greedy": greedy,
           "top_ids": greedy[:, None], "top_logits": torch.zeros(n - f, 1), "lse": torch.zeros(n - f)}
    with tempfile.TemporaryDirectory() as root:
        path = os.path.join(root, "x__y.taps.safetensors")
        # bf16 taps in the file: the drafts above came from the float32 taps, so recompute
        DT.save_dump(path, "x/y", ids.tolist(), n_prompt, f, res, {})
        dump = DT.load_dump(path)
        ctx = d.context_kv({i: dump["taps"][i] for i in d.cfg.target_layer_ids}, torch.arange(f, n))
        for p in DA.anchors_of(n_prompt, n):
            dump["greedy"][p - f] = d.draft_block(int(ids[p]), p, 1, ctx).ids[0]
        head = DA.Head(d.lm_head)
        r, _ = DA.run_source(d, head, dump, [1], {"a": None}, batch=4)
        e = DA.pool([r["a"]], [1])["1"]
        assert e["E"] == 2.0 and e["alpha"] == [1.0] and e["hist"] == [0, e["rows"]], e
        # K = 2 on a text that IS the greedy path: ids[p + 1] = greedy[p] and greedy[p + 1] = d_1
        x = dump["ids"].clone()
        p = DA.anchors_of(n_prompt, n)[0]
        o = d.draft_block(int(x[p]), p, 2, ctx)
        x[p + 1] = o.ids[0]
        dump["ids"] = x
        dump["greedy"][p - f], dump["greedy"][p + 1 - f] = o.ids[0], o.ids[1]
        r, _ = DA.run_source(d, head, dump, [2], {"a": None}, batch=4)
        st = r["a"]["2"]
        assert st["accepted"][1] >= 1 and st["hist"][2] >= 1, st
    print("alignment: greedy[p] = draft_block's d_0 -> E_1 = 2 exactly (taps from 3); a greedy-path text accepts K = 2")


# ------------------------------------------------------------------------------ spec 23f: the Muse-Glimmer form

MUSE_V = 300


def _muse_cfg_dict(window=16, layers=2):
    return {"architectures": ["MuseGlimmerAssistantModel"], "model_type": "muse_glimmer_assistant", "block_size": 16,
            "head_dim": 16, "hidden_act": "silu", "hidden_size": 64, "intermediate_size": 128,
            "layer_types": ["sliding_attention"] * layers, "mask_token_id": 297, "num_attention_heads": 4,
            "num_hidden_layers": layers, "num_key_value_heads": 2, "rms_norm_eps": 1e-5,
            "rope_parameters": {"rope_theta": 500000.0, "rope_type": "default"}, "sliding_window": window,
            "target_layer_ids": [1, 5]}


def _muse_tensors(cfg, seed=0):
    """Checkpoint-named (the assistant's) random bf16 tensors + a target embedding / head."""
    g = torch.Generator().manual_seed(seed)
    H, I, hd, nq, nkv = cfg.hidden, cfg.ffn, cfg.head_dim, cfg.n_heads, cfg.n_kv_heads
    t = {"encoder.fc.weight": (H, 2 * H), "encoder.output_norm_enc.weight": (H,), "norm.weight": (H,)}
    for i in range(cfg.n_layers):
        b = f"layers.{i}."
        t.update({b + "input_layernorm.weight": (H,), b + "post_attention_layernorm.weight": (H,),
                  b + "self_attn.q_proj.weight": (nq * hd, H), b + "self_attn.k_proj.weight": (nkv * hd, H),
                  b + "self_attn.v_proj.weight": (nkv * hd, H), b + "self_attn.o_proj.weight": (H, nq * hd),
                  b + "self_attn.q_norm.weight": (hd,), b + "self_attn.k_norm.weight": (hd,),
                  b + "mlp.gate_proj.weight": (I, H), b + "mlp.up_proj.weight": (I, H), b + "mlp.down_proj.weight": (H, I)})
    out = {}
    for n, s in t.items():
        if len(s) == 2:
            out[n] = (torch.randn(s, generator=g) * s[1] ** -0.5).to(torch.bfloat16)
        else:
            out[n] = (1.0 + torch.randn(s, generator=g) * 0.1).to(torch.bfloat16)
    embed = torch.randn(MUSE_V, H, generator=g).to(torch.bfloat16)
    head = torch.randn(MUSE_V, H, generator=g).to(torch.bfloat16)
    return out, embed, head


def _muse_drafter(window=16, seed=0):
    cfg = ref.config_from_dict(_muse_cfg_dict(window))
    t, e, h = _muse_tensors(cfg, seed)
    return ref.Drafter(cfg, {ref._canon(n): v for n, v in t.items()}, e, h, dtype=torch.float32)


def test_muse_batch_equals_draft_block():
    """forward_blocks_muse (B blocks stacked, one mask per block) == Drafter.draft_block per anchor - drafts of every K
    = 1..15 equal (one 16-row block serves every K), the hidden rows within 1e-5 - at anchors before, at and past the
    window (16: |q - k| <= 16), with context positions >= the anchor present (never read)."""
    d = _muse_drafter(window=16)
    n = 60
    taps = {i: torch.randn(n, 64, generator=torch.Generator().manual_seed(i)) * 2 for i in (1, 5)}
    ctx = d.context_kv(taps, torch.arange(n))
    anchors_pos = [3, 15, 16, 17, 31, 44]
    g = torch.Generator().manual_seed(3)
    anchor_ids = torch.randint(0, MUSE_V - 4, (len(anchors_pos),), generator=g).tolist()
    head = DA.Head(d.lm_head)
    ps = torch.tensor(anchors_pos)
    h = DA.forward_blocks_muse(d, anchor_ids, ps, ctx)
    lg = DA.muse_draft_logits(d, head, anchor_ids, ps, ctx)
    ids = DA.greedy_ids(lg)
    worst = 0.0
    for b, p in enumerate(anchors_pos):
        o = d.draft_block(anchor_ids[b], p, 15, ctx)
        assert torch.equal(ids[b], o.ids), (p, ids[b].tolist(), o.ids.tolist())
        for K in (1, 4, 15):
            assert torch.equal(d.draft_block(anchor_ids[b], p, K, ctx).ids, o.ids[:K]), (p, K)
        worst = max(worst, (h[b] - o.hidden).abs().max().item())
    assert worst < 1e-5, worst
    # a context position at the anchor or later never moves a draft; p - 16 does, p - 17 does not
    p = 31
    h1 = DA.forward_blocks_muse(d, [anchor_ids[4]], torch.tensor([p]), ctx)   # the same shapes as the perturbed runs
    for off, moves in ((0, False), (5, False), (-16, True), (-17, False)):
        t2 = {i: v.clone() for i, v in taps.items()}
        for i in t2:
            t2[i][p + off] += 5.0
        c2 = d.context_kv(t2, torch.arange(n))
        h2 = DA.forward_blocks_muse(d, [anchor_ids[4]], torch.tensor([p]), c2)
        assert torch.equal(h2[0], h1[0]) != moves, (off, moves)
    print(f"Muse: batched blocks == draft_block per anchor (K 1..15 from one block), hidden within {worst:.1e}; "
          "the window |q - k| <= 16 at the anchor's row; positions >= p never read")


def test_int4g64_equals_rtn_h():
    """The int4g64 arm's quantiser == loader/rtn.h's rtn_int4_g64 bit for bit on tools/oracle/fixtures/
    rtn_int4_g64.json (written by the C++ function itself: ties, a zero group, an f16-subnormal scale, clamps)."""
    with open(os.path.join(_HERE, "fixtures", "rtn_int4_g64.json")) as f:
        fx = json.load(f)
    N, K = fx["N"], fx["K"]
    w = torch.tensor(fx["w_bf16"], dtype=torch.int32).to(torch.int16).view(torch.bfloat16).reshape(N, K)
    q, s16 = DA.rtn_int4_g64(w)
    want_s = torch.tensor(fx["scales_f16"], dtype=torch.int32).to(torch.int16).view(torch.float16).reshape(K // 64, N).t()
    assert torch.equal(s16.view(torch.int16), want_s.contiguous().view(torch.int16)), "scales differ"
    words = torch.tensor(fx["qweight"], dtype=torch.int64).reshape(K // 8, N)       # word (r, n): k = 8r..8r+7
    want_q = torch.stack([(words >> (4 * j)) & 0xF for j in range(8)], 1).reshape(K, N).t()
    assert torch.equal(q.long(), want_q), "nibbles differ"
    assert (s16[1, 0] == 0) and bool((q[1, :64] == 8).all()) and bool((s16[2, 0].float() < 6.1e-5))
    dq = DA.int4g64_dequant(w)
    assert torch.equal(dq, (q.float() - 8).reshape(N, K // 64, 64).mul(s16.float()[..., None]).reshape(N, K))
    d = _muse_drafter()
    before = {k: v.clone() for k, v in d.w.items()}
    nq = DA.int4g64_simulate(d)
    assert nq == 1 + 2 * 7, nq
    for k, v in d.w.items():
        assert torch.equal(v, before[k]) != k.endswith(DA.LINEAR_SUFFIXES), k
    print(f"int4 g64 RTN == loader/rtn.h on the fixture ({N} x {K}: ties, a zero group, a subnormal scale); "
          f"{nq} Muse drafter linears quantised, the norms untouched")


def test_muse_sampled():
    """p_rows = mtp_accept.filt on the dump's capped top-64; q_rows = the top 16 at T, raw or capped (20 tanh(z m /
    20)); overlap = sum min(p, q): 1 when equal, 0 when disjoint, symmetric; capped q is flatter than raw."""
    g = torch.Generator().manual_seed(9)
    lg = torch.randn(5, MUSE_V, generator=g) * 30
    qi, qp = DA.q_rows(lg, "raw", 0.19611613513818404, 20.0, 1.0)
    ci, cp = DA.q_rows(lg, "capped", 0.19611613513818404, 20.0, 1.0)
    assert torch.equal(qi, ci) and torch.allclose(qp.sum(-1), torch.ones(5, dtype=torch.float64))
    assert bool((cp.max(-1).values < qp.max(-1).values).all()), "the capped q is not flatter"
    cap = torch.tanh(lg * 0.19611613513818404 / 20.0) * 20.0
    tv, ti = torch.sort(cap, dim=-1, descending=True, stable=True)
    pi, pp = DA.p_rows(ti[:, :64], tv[:, :64], 1.0, 64, 0.95)
    assert torch.allclose(pp.sum(-1), torch.ones(5, dtype=torch.float64))
    one = DA.overlap(qi, qp, qi, qp)
    assert torch.allclose(one, torch.ones(5, dtype=torch.float64))
    zero = DA.overlap(qi, qp, qi + MUSE_V, qp)
    assert torch.equal(zero, torch.zeros(5, dtype=torch.float64))
    assert torch.allclose(DA.overlap(pi, pp, ci, cp), DA.overlap(ci, cp, pi, pp))
    print(f"sampled: p filtered (T 1, top-k 64, top-p 0.95), q top-16 raw / capped; overlap raw "
          f"{float(DA.overlap(pi, pp, qi, qp).mean()):.3f}, capped {float(DA.overlap(pi, pp, ci, cp).mean()):.3f}")


def _muse_fake_dump(path, name, d, n_prompt, n, seed, greedy_of=None):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, MUSE_V - 4, (n,), generator=g)
    taps = {i: (torch.randn(n, 64, generator=g) * 2).to(torch.bfloat16) for i in (1, 5)}
    greedy = torch.cat([ids[1:], ids[:1]])
    if greedy_of is not None:
        greedy = greedy_of(ids, taps)
    top = torch.sort(torch.randn(n, MUSE_V, generator=g) * 3, dim=-1, descending=True, stable=True)
    res = {"taps": taps, "greedy": greedy, "top_ids": top.indices[:, :64].contiguous(),
           "top_logits": top.values[:, :64].contiguous(),
           "lse": torch.zeros(n)}
    DT.save_dump(path, name, ids.tolist(), n_prompt, 0, res, {"vocab_used": MUSE_V})


def test_muse_run_over_dumps():
    """cmd_run --model muse over two dumps from checkpoint-shaped files: per K 1..15 from one block per anchor (the
    anchors n_prompt .. n - 15), a dump whose greedy IS the drafter's own drafts accepts every depth (E_15 = 16), the
    sampled sums per q form, resumed, the summary's sampled table, the dflash2 arm SKIPPED when absent."""
    from safetensors.torch import save_file
    cd = _muse_cfg_dict(window=16)
    cfg = ref.config_from_dict(cd)
    t, e, h = _muse_tensors(cfg, 4)
    with tempfile.TemporaryDirectory() as root:
        dd, td = os.path.join(root, "drafter"), os.path.join(root, "target")
        os.makedirs(dd)
        os.makedirs(td)
        with open(os.path.join(dd, "config.json"), "w") as f:
            json.dump(cd, f)
        save_file(t, os.path.join(dd, "model.safetensors"))
        save_file({"model.language_model.embed_tokens.weight": e, "lm_head.weight": h},
                  os.path.join(td, "model.safetensors"))
        with open(os.path.join(td, "config.json"), "w") as f:
            json.dump({"text_config": {"output_multiplier": 0.19611613513818404, "final_logit_softcapping": 20.0}}, f)
        with open(os.path.join(td, "generation_config.json"), "w") as f:
            json.dump({"temperature": 1.0, "top_k": 64, "top_p": 0.95}, f)
        d = ref.load_drafter(dd, td, dtype=torch.float32)
        dumps = os.path.join(root, "dumps")
        os.makedirs(dumps)

        def own_drafts(ids, taps):   # greedy[p + j] = the drafter's draft j at anchor p = n_prompt (text = greedy)
            ctx = d.context_kv({i: taps[i] for i in (1, 5)}, torch.arange(len(ids)))
            o = d.draft_block(int(ids[20]), 20, 15, ctx)
            gr = torch.cat([ids[1:], ids[:1]]).clone()
            gr[20:35] = o.ids
            return gr
        _muse_fake_dump(os.path.join(dumps, "code__a.taps.safetensors"), "code/a", d, 20, 36, 1, None)
        # code/a's text must follow its greedy for the depth chain: rebuild with ids[p + 1 + j] = greedy[p + j]
        dump = DT.load_dump(os.path.join(dumps, "code__a.taps.safetensors"))
        gr = own_drafts(dump["ids"], dump["taps"])
        x = dump["ids"].clone()
        x[21:36] = gr[20:35]
        gr2 = torch.cat([x[1:], x[:1]]).clone()
        gr2[20:35] = gr[20:35]
        res = {"taps": dump["taps"], "greedy": gr2, "top_ids": dump["top_ids"], "top_logits": dump["top_logits"],
               "lse": dump["lse"]}
        DT.save_dump(os.path.join(dumps, "code__a.taps.safetensors"), "code/a", x.tolist(), 20, 0, res, {})
        _muse_fake_dump(os.path.join(dumps, "prose__b.taps.safetensors"), "prose/b", d, 30, 70, 2)
        out = os.path.join(root, "out")
        ns = argparse.Namespace(dumps=dumps, out_dir=out, arms="bf16,int4g64h,dflash2", vocab="", vocab_on="",
                                draft_vocab_ids=None, vocab_used=MUSE_V, k="1-15", batch=4, drafter=dd, w4a16=None,
                                target=td, sources=None, dry_run=False, model="muse", kmax=15, q="raw,capped",
                                temperature=None, top_k=None, top_p=None, dflash2=None)
        DA.cmd_run(ns)
        r = json.load(open(os.path.join(out, "accept.bf16.json")))
        st = r["variants"]["bf16"]["code/a"]
        assert st["15"]["rows"] == len(DA.anchors_of(20, 36, 15)) == 2, st["15"]["rows"]
        e15 = DA.pool([st], [15])["15"]
        assert st["15"]["accepted"][14] >= 1 and st["15"]["hist"][15] >= 1, st["15"]
        sb = r["variants"]["bf16"]["prose/b"]
        assert sb["15"]["rows"] == len(DA.anchors_of(30, 70, 15)) == 26
        assert sorted(sb["sampled"]) == ["capped", "raw"] and sb["sampled"]["raw"]["counted"][0] == 26
        assert not os.path.exists(os.path.join(out, "accept.dflash2.json"))
        mt = os.path.getmtime(os.path.join(out, "accept.bf16.json"))
        DA.cmd_run(ns)
        assert os.path.getmtime(os.path.join(out, "accept.bf16.json")) == mt
        summ = DA.summarize(sorted(glob.glob(os.path.join(out, "accept.*.json"))), None, tuple(range(1, 16)))
        assert set(summ) == {"bf16", "int4g64h"} and "sampled" in summ["bf16"]["all"]
        assert len(summ["bf16"]["all"]["sampled"]["capped"]["E"]) == 15
    print(f"run --model muse: K 1..15 from one block, the drafter's own drafts accept to depth 15 (E_15 "
          f"{e15['E']:.2f} over the uncensored row), sampled sums raw / capped, resumed, dflash2 SKIPPED")


TESTS = [test_top_k_fast, test_select_draft_vocab, test_int8, test_accept_accounting,
         test_batch_equals_draft_block, test_vocab_mask, test_positive_alignment, test_run_over_dumps,
         test_muse_batch_equals_draft_block, test_int4g64_equals_rtn_h, test_muse_sampled, test_muse_run_over_dumps]


def main() -> None:
    torch.set_grad_enabled(False)
    only = sys.argv[1:]
    for t in TESTS:
        if not only or t.__name__ in only:
            t()
    print("ok")


if __name__ == "__main__":
    main()
